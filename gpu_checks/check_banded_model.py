"""Check the banded sliding attention patch on a tiny random Inkling model, on the CPU.

Ticket A1 (docs/inkling_general_speed.md section 10): the banded sliding attention patch on a tiny random
``Inkling2LoopForConditionalGeneration``, on the CPU (no GPU, no lock). Needs the Inkling transformers build.

Checks, float32: logits and the gradient of a scalar loss with respect to every parameter, patched vs unpatched, for
T = 50 and for a padded batch of two (one left-, one right-padded), rel <= 1e-4, with the query chunk at 5, 16 and the
default; ``enable`` returns the number of sliding layers; T = 16 (= the window), cached calls, output_attentions
(kwarg or config) and hooks on ``relative_position_bias`` take the fallback
(counted by wrapping the original forward), T = 50 never does; ``disable`` restores bit-identical outputs; a wrong
pinned hash for any of the pinned functions raises RuntimeError.

Run in WSL:
    cd /path/to/Inkling-2-Mini-vLLM && PYTHONPATH=. ~/inkling-venv/bin/python gpu_checks/check_banded_model.py
"""  # run as scripts, not a package

import sys
from typing import Any

import torch
import transformers.models.inkling2_loop.modeling_inkling2_loop as modeling  # pyright: ignore[reportMissingImports]
from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopConfig  # pyright: ignore[reportMissingImports]

from inkling_qlora import inkling_banded as ib

WINDOW = 16
VOCAB = 128
FAILURES: list[str] = []


def check(ok: bool, what: str) -> None:  # noqa: FBT001  # the condition is the first positional argument at every call
    """Print an ok or FAIL line for ``what`` and record a failure."""
    print(f"{'ok  ' if ok else 'FAIL'} {what}", flush=True)
    if not ok:
        FAILURES.append(what)


def tiny_model() -> torch.nn.Module:
    """Build a tiny random Inkling model for the CPU checks."""
    config = Inkling2LoopConfig(
        text_config={
            "vocab_size": VOCAB,
            "unpadded_vocab_size": VOCAB,
            "hidden_size": 32,
            "num_hidden_layers": 6,  # layers 0-4 sliding, 5 full; 2-3 are loop layers run twice
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "swa_num_attention_heads": 6,
            "swa_num_key_value_heads": 2,
            "swa_head_dim": 8,
            "sliding_window_size": WINDOW,
            "d_rel": 4,
            "rel_extent": 32,
            "moe_intermediate_size": 16,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "n_layers_per_attn_res_block": 2,
            "loop_layer_index_start": 2,
            "loop_layer_index_end": 4,
            "loop_num_passes": 2,
            "loop_gate_rank": 4,
            "pad_token_id": 0,
            "eos_token_id": 1,
        },
        vision_config={
            "model_type": "inkling2_vision",
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "projection_hidden_dim": 32,
            "text_hidden_size": 32,
        },
    )
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    model = modeling.Inkling2LoopForConditionalGeneration(config).float()
    with torch.no_grad():  # post_init leaves some parameters at zero or one; randomize so every path matters
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return model


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return the relative L2 error of ``a`` against ``b``."""
    return ((a - b).double().norm() / b.double().norm().clamp_min(1e-30)).item()


def run(model, ids, mask, weights) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run a forward and backward pass and return the logits and the gradient of every parameter."""
    model.zero_grad(set_to_none=True)
    logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
    loss = (logits * weights).sum()
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    return logits.detach(), grads


def compare_case(model, name: str, ids, mask) -> None:
    """Compare logits and gradients of the patched and unpatched model on one input."""
    weights = torch.randn(*ids.shape, VOCAB, generator=torch.Generator().manual_seed(1))
    ib.disable_banded_attention(model)
    ref_logits, ref_grads = run(model, ids, mask, weights)
    for chunk in (5, 16, ib.BAND_CHUNK):
        original_chunk, ib.BAND_CHUNK = ib.BAND_CHUNK, chunk  # pyrefly: ignore[bad-assignment]
        try:
            ib.enable_banded_attention(model)
            logits, grads = run(model, ids, mask, weights)
            ib.disable_banded_attention(model)
        finally:
            ib.BAND_CHUNK = original_chunk
        check(sorted(grads) == sorted(ref_grads), f"{name}, chunk {chunk}: same parameters get gradients")
        worst_name, worst = max(((n, rel(grads[n], ref_grads[n])) for n in ref_grads), key=lambda x: x[1])
        r_logits = rel(logits, ref_logits)
        check(r_logits <= 1e-4, f"{name}, chunk {chunk}: logits rel {r_logits:.2e} (bar 1e-4)")
        check(
            worst <= 1e-4,
            f"{name}, chunk {chunk}: worst of {len(ref_grads)} parameter gradients rel {worst:.2e} ({worst_name})",
        )


def count_fallbacks(model) -> list[int]:
    """Wrap each sliding attention's original forward to count fallback calls; return the counter."""
    counter = [0]
    for module in model.modules():
        if hasattr(module, "_banded_original_forward"):
            original = module._banded_original_forward

            def wrapped(*args, _original=original, **kwargs):
                counter[0] += 1
                return _original(*args, **kwargs)

            module._banded_original_forward = wrapped
    return counter


def main() -> None:  # noqa: PLR0915  # check script, one long flow
    """Run every banded-attention check on the tiny model."""
    model: Any = tiny_model()
    model.train()
    sliding = [m for m in model.modules() if isinstance(m, modeling.Inkling2LoopSlidingAttention) and m.is_sliding]
    full_attn = [
        m for m in model.modules() if isinstance(m, modeling.Inkling2LoopSlidingAttention) and not m.is_sliding
    ]
    n = ib.enable_banded_attention(model)
    check(n == len(sliding) == 5, f"enable returns {n} (sliding layers: {len(sliding)})")
    check(all(not hasattr(m, "_banded_original_forward") for m in full_attn), "full-attention modules unpatched")
    check(ib.enable_banded_attention(model) == n, "enabling twice patches nothing new")
    check(ib.disable_banded_attention(model) == n, "disable restores every module")
    check(all("forward" not in m.__dict__ for m in sliding), "no instance forward left after disable")

    g = torch.Generator().manual_seed(0)
    ids = torch.randint(2, VOCAB, (1, 50), generator=g)
    compare_case(model, "T=50", ids, torch.ones_like(ids))
    ids2 = torch.randint(2, VOCAB, (2, 50), generator=g)
    mask2 = torch.ones_like(ids2)
    mask2[0, :20] = 0  # left padding: its first 20 query rows see no key
    mask2[1, 37:] = 0  # right padding
    ids2 = ids2.masked_fill(mask2 == 0, 0)
    compare_case(model, "padded batch of 2", ids2, mask2)

    # Fallback: T = window (16) and cached calls run the original forward; T = 50 never does.
    model.eval()
    calls_per_forward = 7  # sliding layers 0, 1, 4 once; loop layers 2, 3 twice
    with torch.no_grad():
        ib.enable_banded_attention(model)
        counter = count_fallbacks(model)
        model(input_ids=ids[:, :WINDOW], use_cache=False)
        check(counter[0] == calls_per_forward, f"T=16: {counter[0]} fallback calls (expected {calls_per_forward})")
        counter[0] = 0
        model(input_ids=ids, use_cache=False)
        check(counter[0] == 0, f"T=50 without cache: {counter[0]} fallback calls (expected 0)")
        counter[0] = 0
        model(input_ids=ids, use_cache=True)
        check(counter[0] == calls_per_forward, f"T=50 with a cache: {counter[0]} fallback calls")
        counter[0] = 0
        model(input_ids=ids, use_cache=False, output_attentions=True)
        check(counter[0] == calls_per_forward, f"T=50, output_attentions kwarg: {counter[0]} fallback calls")
        counter[0] = 0
        model.config.output_attentions = model.config.text_config.output_attentions = True
        try:
            model(input_ids=ids, use_cache=False)
        finally:
            model.config.output_attentions = model.config.text_config.output_attentions = False
        check(counter[0] == calls_per_forward, f"T=50, config.output_attentions: {counter[0]} fallback calls")
        counter[0] = 0
        hooked = [m.relative_position_bias for m in sliding]
        handles = [rpb.register_forward_hook(lambda *a: None) for rpb in hooked]
        model(input_ids=ids, use_cache=False)
        for h in handles:
            h.remove()
        check(counter[0] == calls_per_forward, f"T=50, hooks on relative_position_bias: {counter[0]} fallback calls")
        counter[0] = 0
        model(input_ids=ids, use_cache=False)
        check(counter[0] == 0, f"T=50 after the hooks are removed: {counter[0]} fallback calls")
        ib.disable_banded_attention(model)

        # disable restores bit-identical outputs
        before = model(input_ids=ids2, attention_mask=mask2, use_cache=False).logits
        ib.enable_banded_attention(model)
        ib.disable_banded_attention(model)
        after = model(input_ids=ids2, attention_mask=mask2, use_cache=False).logits
        check(torch.equal(before, after), "disable restores bit-identical logits")

    # A wrong pinned hash for any of the pinned functions raises, naming it, and patches nothing.
    pinned = dict(ib.PINNED_SOURCE_SHA256)
    for name in pinned:
        ib.PINNED_SOURCE_SHA256[name] = "0" * 64
        try:
            ib.enable_banded_attention(model)
            check(ok=False, what=f"a wrong pinned hash for {name} raises")
        except RuntimeError as exc:
            check(f"source of {name} changed" in str(exc), f"a wrong pinned hash for {name} raises")
        finally:
            ib.PINNED_SOURCE_SHA256.clear()
            ib.PINNED_SOURCE_SHA256.update(pinned)
    check(all(not hasattr(m, "_banded_original_forward") for m in sliding), "nothing patched after the refusal")

    print(f"\n{'PASS' if not FAILURES else f'FAIL ({len(FAILURES)})'}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
