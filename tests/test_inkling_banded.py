"""CPU checks of the banded sliding attention (ticket A1): the chunk loop against the full masked computation, in
float64, outputs and gradients. The reference below is a literal copy of the installed transformers code
(modeling_inkling2_loop.py, transformers 5.24.0.dev0): ``repeat_kv``, ``eager_attention_forward`` and the gather
at the end of ``Inkling2LoopRelativePositionBias.forward``. It is not imported from the code under test."""

from __future__ import annotations

import hashlib
import inspect
import types

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from inkling_qlora import inkling_banded as ib

# ---- literal copies of the reference (do not edit to match the code under test) ----


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    position_bias: torch.Tensor | None = None,
    sinks: torch.Tensor | None = None,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if position_bias is not None:
        attn_weights = attn_weights + position_bias
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    if sinks is not None:
        # The sink is one extra logit per query with a zero value: it only absorbs probability mass
        combined_logits = torch.cat([attn_weights, sinks], dim=-1).float()
        attn_weights = F.softmax(combined_logits, dim=-1)[..., :-1].to(query.dtype)
    else:
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def eager_attention_forward_fp64(
    module, query, key, value, attention_mask, scaling, dropout=0.0, position_bias=None, sinks=None, **kwargs
):
    """``eager_attention_forward`` with its one fp32 cast (``.float()`` before the sink softmax) made fp64.

    The reference softmax runs in fp32 even for fp64 inputs, so with the literal copy banded and full differ by fp32
    rounding (about 1e-7) from the different column counts alone; the 1e-12 bar is only reachable with this
    variant. Every other line is the literal copy above."""
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if position_bias is not None:
        attn_weights = attn_weights + position_bias
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    combined_logits = torch.cat([attn_weights, sinks], dim=-1).double()  # the only change: .float() -> .double()
    attn_weights = F.softmax(combined_logits, dim=-1)[..., :-1].to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def reference_bias(rel_logits: torch.Tensor, relative_distance: torch.Tensor, rel_extent: int) -> torch.Tensor:
    gather_index = relative_distance.clamp(0, rel_extent - 1).expand(*rel_logits.shape[:2], -1, -1)
    position_bias = rel_logits.gather(-1, gather_index)
    return position_bias.masked_fill((relative_distance < 0) | (relative_distance >= rel_extent), 0.0)


# ---- setup ----

HEADS, KV_HEADS, HEAD_DIM, WINDOW, EXTENT = 6, 2, 8, 16, 16
DTYPE = torch.float64


class StubModule:
    num_key_value_groups = HEADS // KV_HEADS
    training = False


def distances(t: int) -> torch.Tensor:
    pos = torch.arange(t)
    return pos[:, None] - pos[None, :]


def additive_mask(t: int, padding: torch.Tensor | None = None, window: int | None = WINDOW) -> torch.Tensor:
    """(B, 1, T, T) additive mask as transformers builds it for eager attention: a key is visible when it is causal,
    inside the window (``kv > q - window``) and not padding."""
    q = torch.arange(t)[:, None]
    kv = torch.arange(t)[None, :]
    allowed = kv <= q
    if window is not None:
        allowed = allowed & (kv > q - window)
    allowed = allowed[None, None]
    if padding is not None:  # (B, T), 1 for real tokens
        allowed = allowed & padding.bool()[:, None, None, :]
    else:
        allowed = allowed.expand(1, 1, t, t)
    mask = torch.zeros(allowed.shape, dtype=DTYPE)
    return mask.masked_fill(~allowed, torch.finfo(DTYPE).min)


def make_inputs(batch: int, t: int, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)

    def r(*shape: int, scale: float = 1.0) -> torch.Tensor:
        return (torch.randn(*shape, generator=g, dtype=DTYPE) * scale).requires_grad_()

    return {
        "query": r(batch, HEADS, t, HEAD_DIM),
        "key": r(batch, KV_HEADS, t, HEAD_DIM),
        "value": r(batch, KV_HEADS, t, HEAD_DIM),
        "rel_logits": r(batch, HEADS, t, EXTENT),
        "sinks": r(batch, HEADS, t, 1),
    }


def full(inputs: dict[str, torch.Tensor], mask: torch.Tensor, attn_fn=eager_attention_forward_fp64) -> torch.Tensor:
    t = inputs["query"].shape[2]
    bias = reference_bias(inputs["rel_logits"], distances(t), EXTENT)
    out, _ = attn_fn(
        StubModule(),
        inputs["query"],
        inputs["key"],
        inputs["value"],
        mask,
        scaling=HEAD_DIM**-0.5,
        position_bias=bias,
        sinks=inputs["sinks"],
    )
    return out


def banded(
    inputs: dict[str, torch.Tensor], mask: torch.Tensor, chunk: int, attn_fn=eager_attention_forward_fp64
) -> torch.Tensor:
    t = inputs["query"].shape[2]
    return ib.banded_attention(
        attn_fn,
        StubModule(),
        inputs["query"],
        inputs["key"],
        inputs["value"],
        mask,
        distances(t),
        inputs["rel_logits"],
        inputs["sinks"],
        HEAD_DIM**-0.5,
        WINDOW,
        EXTENT,
        chunk,
    )


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm().clamp_min(1e-300)).item()


def compare(
    inputs: dict[str, torch.Tensor],
    mask: torch.Tensor,
    chunk: int,
    mask_ref: torch.Tensor | None = None,
    attn_fn=eager_attention_forward_fp64,
):
    """Relative differences of the output and the five gradients, banded vs full."""
    mask_ref = mask if mask_ref is None else mask_ref
    out_ref = full(inputs, mask_ref, attn_fn)
    out_band = banded(inputs, mask, chunk, attn_fn)
    assert out_band.shape == out_ref.shape
    upstream = torch.randn(out_ref.shape, generator=torch.Generator().manual_seed(1), dtype=DTYPE)
    names = list(inputs)
    g_ref = torch.autograd.grad(out_ref, [inputs[n] for n in names], upstream)
    g_band = torch.autograd.grad(out_band, [inputs[n] for n in names], upstream)
    worst = {"output": rel(out_band, out_ref)}
    worst.update({f"grad {n}": rel(gb, gr) for n, gb, gr in zip(names, g_band, g_ref, strict=True)})
    return worst, out_band.detach(), out_ref.detach()


# ---- tests ----


@pytest.mark.parametrize("t", [17, 31, 32, 33, 100])
@pytest.mark.parametrize("chunk", [1, 5, 16, 64])
def test_banded_equals_full(t: int, chunk: int) -> None:
    worst, _, _ = compare(make_inputs(1, t), additive_mask(t), chunk)
    assert max(worst.values()) <= 1e-12, worst


@pytest.mark.parametrize("t", [33, 100])
@pytest.mark.parametrize("chunk", [1, 16])
def test_banded_with_literal_reference(t: int, chunk: int) -> None:
    """The literal reference (fp32 softmax): equal up to fp32 rounding."""
    padding = torch.ones(2, t, dtype=torch.long)
    padding[0, : t // 3] = 0
    worst, _, _ = compare(make_inputs(2, t, seed=7), additive_mask(t, padding), chunk, attn_fn=eager_attention_forward)
    assert max(worst.values()) <= 1e-6, worst


@pytest.mark.parametrize("t", [17, 33, 100])
@pytest.mark.parametrize("chunk", [1, 5, 16, 64])
def test_banded_equals_full_padded_batch(t: int, chunk: int) -> None:
    padding = torch.ones(2, t, dtype=torch.long)
    padding[0, : t // 3] = 0  # left-padded: its first rows see no key at all
    padding[1, t - t // 4 :] = 0  # right-padded
    mask = additive_mask(t, padding)
    fully_masked = (mask == torch.finfo(DTYPE).min).all(-1).squeeze(1)  # (B, T)
    assert fully_masked[0].any()
    worst, out_band, out_ref = compare(make_inputs(2, t, seed=t), mask, chunk)
    assert max(worst.values()) <= 1e-12, worst
    assert torch.equal(out_band[fully_masked], out_ref[fully_masked])


@pytest.mark.parametrize("shift", [-1, 1])
def test_off_by_one_band_is_detected(monkeypatch: pytest.MonkeyPatch, shift: int) -> None:
    """With chunk 1 the band alone is the window, so a causal-only mask must reproduce the windowed reference. A
    band start moved by one key either way (one real key dropped, or one key past the window kept) breaks it."""
    t = 40
    inputs = make_inputs(1, t, seed=3)
    windowed, causal_only = additive_mask(t), additive_mask(t, window=None)
    worst, _, _ = compare(inputs, causal_only, chunk=1, mask_ref=windowed)
    assert max(worst.values()) <= 1e-12, worst

    original = ib.band_start
    monkeypatch.setattr(ib, "band_start", lambda qs, window: max(0, original(qs, window) + shift))
    worst, _, _ = compare(inputs, causal_only, chunk=1, mask_ref=windowed)
    assert worst["output"] > 1e-3, worst
    if shift == 1:  # dropping a real key also shows with the full mask and the default chunk
        worst, _, _ = compare(inputs, windowed, chunk=16)
        assert worst["output"] > 1e-3, worst


def _fallback_stub(**overrides) -> types.SimpleNamespace:
    attrs = dict(
        is_sliding=True,
        sliding_window=16,
        training=False,
        attention_dropout=0.0,
        config=types.SimpleNamespace(_attn_implementation="eager"),
        relative_position_bias=nn.Module(),
    )
    attrs.update(overrides)
    return types.SimpleNamespace(**attrs)


def test_fallback_conditions() -> None:
    t = 40
    hidden = torch.zeros(1, t, 4)
    mask, dist = torch.zeros(1, 1, t, t), distances(t)

    def uses_original(stub=None, hidden=hidden, mask=mask, dist=dist, cache=None, **kwargs) -> bool:
        return ib._use_original(stub or _fallback_stub(), hidden, mask, dist, cache, kwargs)

    assert not uses_original()
    assert uses_original(_fallback_stub(is_sliding=False))
    assert uses_original(cache=object())
    assert uses_original(past_key_values=object())
    assert uses_original(mask=None)
    assert uses_original(mask=torch.zeros(t, t))
    assert uses_original(mask=torch.zeros(1, 1, t, t + 3))
    assert uses_original(dist=distances(t)[:, :-1])
    assert uses_original(dist=distances(t + 1))
    assert uses_original(hidden=torch.zeros(1, 16, 4), mask=torch.zeros(1, 1, 16, 16), dist=distances(16))
    assert not uses_original(hidden=torch.zeros(1, 17, 4), mask=torch.zeros(1, 1, 17, 17), dist=distances(17))
    assert uses_original(_fallback_stub(training=True, attention_dropout=0.1))
    assert not uses_original(_fallback_stub(training=True))
    assert uses_original(output_attentions=True)
    assert uses_original(_fallback_stub(config=types.SimpleNamespace(_attn_implementation="sdpa")))
    assert uses_original(
        _fallback_stub(config=types.SimpleNamespace(_attn_implementation="eager", output_attentions=True))
    )
    rpb = nn.Module()
    handle = rpb.register_forward_hook(lambda *a: None)
    assert uses_original(_fallback_stub(relative_position_bias=rpb))
    handle.remove()
    assert not uses_original(_fallback_stub(relative_position_bias=rpb))
    handle = rpb.register_forward_pre_hook(lambda *a: None)
    assert uses_original(_fallback_stub(relative_position_bias=rpb))
    handle.remove()


def test_band_start() -> None:
    assert [ib.band_start(qs, 1024) for qs in (0, 256, 1023, 1024, 1280)] == [0, 0, 0, 1, 257]


def test_changed_source_hash_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each pinned function is checked: a changed source for any one of them raises, naming it."""
    names = list(ib.PINNED_SOURCE_SHA256)
    assert {"get_relative_distance", "masking_utils.sliding_window_overlay"} <= set(names)
    fake = {name: repeat_kv for name in names}
    monkeypatch.setattr(ib, "_reference_functions", lambda modeling: fake)
    digest = hashlib.sha256(inspect.getsource(repeat_kv).encode()).hexdigest()
    monkeypatch.setattr(ib, "PINNED_SOURCE_SHA256", dict.fromkeys(names, digest))
    ib.check_reference_sources(None)
    for name in names:
        monkeypatch.setitem(ib.PINNED_SOURCE_SHA256, name, "0" * 64)
        with pytest.raises(RuntimeError, match=f"source of {name} changed"):
            ib.check_reference_sources(None)
        monkeypatch.setitem(ib.PINNED_SOURCE_SHA256, name, digest)
