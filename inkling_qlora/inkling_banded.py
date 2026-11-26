"""Exact banded sliding-window attention for Inkling 2 Mini, as a per-model patch (docs/inkling_general_speed.md,
ticket A1).

The installed ``Inkling2LoopSlidingAttention.forward`` builds the relative position bias and the attention logits
for every (query, key) pair, T x T, although a sliding layer only lets a query see itself and the
``sliding_window - 1`` keys before it. Here the queries are taken in chunks of ``BAND_CHUNK`` positions; a chunk
``[qs, qe)`` can only see keys ``[ks, qe)`` with ``ks = max(0, qs - window + 1)``, every other key being masked
for every query of the chunk by causality or by the window. Dropping those columns removes terms that are exactly
zero after the softmax (the sink logit is always finite), so the result is the reference's up to the order of
floating-point sums. Each chunk is computed by the transformers module's own ``eager_attention_forward``, looked up
at call time, on slices of the query, key, value, mask, bias and sinks.

Use ``enable_banded_attention(model)`` / ``disable_banded_attention(model)``. The patch is per module instance (no
class or module global changes) and falls back to the original forward whenever the call is not a plain, cache-less
sliding call longer than the window, so generation and anything at or below 1,024 tokens run the original code.

``banded_attention`` is the pure chunk loop and imports only torch, so the CPU tests can drive it without the
Inkling transformers build; transformers is imported lazily in ``enable_banded_attention``.
"""

from __future__ import annotations

import hashlib
import inspect
import sys
import types
from collections.abc import Callable
from typing import Any

import torch

BAND_CHUNK = 256

# SHA-256 of ``inspect.getsource`` of the reference functions in the installed build (transformers 5.24.0.dev0 in
# ~/inkling-venv, computed 2026-11-22): the attention and bias code the patch repeats, and the code that defines the
# window the band relies on (the sliding mask: ``kv_idx > q_idx - sliding_window`` and causal; the distance
# ``q - kv``). ``enable_banded_attention`` refuses to patch when any of them has changed.
PINNED_TRANSFORMERS_VERSION = "5.24.0.dev0"
PINNED_SOURCE_SHA256 = {
    "Inkling2LoopSlidingAttention.forward": "f7ea05668f9689e26aff89a47105995a7022dc48c448a242f50d25613d21bce3",
    "Inkling2LoopRelativePositionBias.forward": "55b2f2f77712605608996a31bc5d5e36730518fdc86b3601902238ca05844934",
    "eager_attention_forward": "e356c83147edb3b79a901134e147528f1e3f44bc54c0cb016a7a9bf2b245e6ba",
    "get_relative_distance": "704e9f25a591340a16cc096f7fe72c6418746616d1af5f7f709ddc578cba88cc",
    "masking_utils.sliding_window_overlay": "d36e026d30028edb89f79f9fad37e03a225a80f53320745dc75bd6ca69273b35",
    "masking_utils.causal_mask_function": "e8566f1e2978274ae64fe4fb4aa6e34c013a333232a295bbfd9b53c535244e69",
    "masking_utils.sliding_window_causal_mask_function": (
        "6071b3da2ce1df64a68aad85f0da39e863b6785c9ee467caa693308d9d7a42de"
    ),
}

_ORIGINAL_ATTR = "_banded_original_forward"
_HAD_INSTANCE_FORWARD_ATTR = "_banded_had_instance_forward"


def band_start(qs: int, window: int) -> int:
    """First key a query chunk starting at ``qs`` can see: the window holds the query itself and ``window - 1``
    earlier keys."""
    return max(0, qs - window + 1)


def banded_attention(
    attn_fn: Callable[..., tuple[torch.Tensor, Any]],
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    relative_distance: torch.Tensor,
    rel_logits: torch.Tensor,
    sinks: torch.Tensor,
    scaling: float,
    window: int,
    extent: int,
    chunk: int = BAND_CHUNK,
    **kwargs: Any,
) -> torch.Tensor:
    """Sliding attention computed chunk by chunk over the keys each chunk can see.

    Shapes: ``query`` (B, H, T, D); ``key``, ``value`` (B, KVH, T, D); ``attention_mask`` (B, 1 or H, T, T),
    additive; ``relative_distance`` (T, T); ``rel_logits`` (B, H, T, extent), the per-distance bias before the
    gather; ``sinks`` (B, H, T, 1). Returns the attention output as ``attn_fn`` lays it out, (B, T, H, D).
    ``kwargs`` go to ``attn_fn`` unchanged (it is called with ``dropout=0.0``).
    """
    t = query.shape[2]
    outputs = []
    for qs in range(0, t, chunk):
        qe = min(qs + chunk, t)
        ks = band_start(qs, window)
        distance = relative_distance[qs:qe, ks:qe]
        index = distance.clamp(0, extent - 1).expand(*rel_logits.shape[:2], -1, -1)
        bias = rel_logits[:, :, qs:qe].gather(-1, index)
        bias = bias.masked_fill((distance < 0) | (distance >= extent), 0.0)
        out, _ = attn_fn(
            module,
            query[:, :, qs:qe],
            key[:, :, ks:qe],
            value[:, :, ks:qe],
            attention_mask[:, :, qs:qe, ks:qe],
            dropout=0.0,
            scaling=scaling,
            position_bias=bias,
            sinks=sinks[:, :, qs:qe],
            **kwargs,
        )
        outputs.append(out)
    return torch.cat(outputs, dim=1)


def _use_original(
    self: Any,
    hidden_states: torch.Tensor,
    attention_mask: Any,
    relative_distance: Any,
    cache: Any,
    kwargs: dict[str, Any],
) -> bool:
    t = hidden_states.shape[1]
    return (
        not self.is_sliding
        or cache is not None
        or kwargs.get("past_key_values") is not None
        or getattr(self.config, "_attn_implementation", "eager") not in (None, "eager")
        or not isinstance(attention_mask, torch.Tensor)
        or attention_mask.dim() != 4
        or tuple(attention_mask.shape[-2:]) != (t, t)
        or not isinstance(relative_distance, torch.Tensor)
        or relative_distance.dim() != 2
        or tuple(relative_distance.shape) != (t, t)
        or t <= self.sliding_window
        or (self.training and self.attention_dropout > 0)
        or bool(kwargs.get("output_attentions"))
        or bool(getattr(self.config, "output_attentions", False))  # attentions would be recorded
        or _has_hooks(self.relative_position_bias)  # the patch does not call that module, so its hooks would not run
    )


def _has_hooks(module: Any) -> bool:
    return bool(getattr(module, "_forward_hooks", None) or getattr(module, "_forward_pre_hooks", None))


def banded_sliding_forward(
    self: Any,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    relative_distance: torch.Tensor,
    conv_mask: torch.Tensor | None = None,
    cache: Any = None,
    key_hidden_states: torch.Tensor | None = None,
    value_hidden_states: torch.Tensor | None = None,
    cache_offset: int = 0,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Drop-in for ``Inkling2LoopSlidingAttention.forward`` (bound per instance by ``enable_banded_attention``).
    Repeats the reference body with the attention done by ``banded_attention``; returns ``(attn_output, None)``."""
    if _use_original(self, hidden_states, attention_mask, relative_distance, cache, kwargs):
        return getattr(self, _ORIGINAL_ATTR)(
            hidden_states,
            attention_mask,
            relative_distance,
            conv_mask=conv_mask,
            cache=cache,
            key_hidden_states=key_hidden_states,
            value_hidden_states=value_hidden_states,
            cache_offset=cache_offset,
            **kwargs,
        )
    kwargs.pop("past_key_values", None)  # None here (checked above); keep it out of the projections' kwargs
    key_hidden_states = hidden_states if key_hidden_states is None else key_hidden_states
    value_hidden_states = hidden_states if value_hidden_states is None else value_hidden_states
    input_shape = hidden_states.shape[:-1]
    position_ids = kwargs["position_ids"]

    query_states = self.q_proj(hidden_states, position_ids)
    key_states = self.k_proj(key_hidden_states, cache=None, conv_mask=conv_mask, cache_offset=cache_offset, **kwargs)
    value_states = self.v_proj(
        value_hidden_states, cache=None, conv_mask=conv_mask, cache_offset=cache_offset, **kwargs
    )

    # The reference's Inkling2LoopRelativePositionBias.forward up to the gather, from hidden_states (the query input).
    rpb = self.relative_position_bias
    relative_states = rpb.linear(hidden_states).view(*hidden_states.shape[:-1], rpb.num_heads, -1)
    relative_states = rpb.norm(relative_states, position_ids)
    rel_logits = (relative_states @ rpb.proj).transpose(1, 2)

    sinks = self.sink_proj(hidden_states).transpose(1, 2).unsqueeze(-1)
    attn_fn = sys.modules[type(self).__module__].eager_attention_forward
    attn_output = banded_attention(
        attn_fn,
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        relative_distance,
        rel_logits,
        sinks,
        self.scaling,
        self.sliding_window,
        rpb.rel_extent,
        BAND_CHUNK,
        sliding_window=self.sliding_window,
        **kwargs,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output, hidden_states)
    return attn_output, None


def _reference_functions(modeling: types.ModuleType) -> dict[str, Callable[..., Any]]:
    import transformers.masking_utils as masking_utils

    return {
        "Inkling2LoopSlidingAttention.forward": modeling.Inkling2LoopSlidingAttention.forward,
        "Inkling2LoopRelativePositionBias.forward": modeling.Inkling2LoopRelativePositionBias.forward,
        "eager_attention_forward": modeling.eager_attention_forward,
        "get_relative_distance": modeling.get_relative_distance,
        "masking_utils.sliding_window_overlay": masking_utils.sliding_window_overlay,
        "masking_utils.causal_mask_function": masking_utils.causal_mask_function,
        "masking_utils.sliding_window_causal_mask_function": masking_utils.sliding_window_causal_mask_function,
    }


def check_reference_sources(modeling: types.ModuleType) -> None:
    """RuntimeError unless every reference function's source hashes to its pinned value."""
    for name, fn in _reference_functions(modeling).items():
        digest = hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()
        if digest != PINNED_SOURCE_SHA256[name]:
            raise RuntimeError(
                f"banded attention: the source of {name} changed (sha256 {digest}, pinned "
                f"{PINNED_SOURCE_SHA256[name]} for transformers {PINNED_TRANSFORMERS_VERSION}); re-check the patch "
                "against the installed modeling_inkling2_loop.py / masking_utils.py and update the pin"
            )


def enable_banded_attention(model: torch.nn.Module) -> int:
    """Patch every sliding ``Inkling2LoopSlidingAttention`` in ``model``; return how many (20 for Inkling 2 Mini).
    RuntimeError if the reference code changed or no module was found. Patching twice is a no-op per module."""
    import transformers.models.inkling2_loop.modeling_inkling2_loop as modeling

    check_reference_sources(modeling)
    count = 0
    for module in model.modules():
        if not isinstance(module, modeling.Inkling2LoopSlidingAttention) or not module.is_sliding:
            continue
        count += 1
        if hasattr(module, _ORIGINAL_ATTR):
            continue
        setattr(module, _HAD_INSTANCE_FORWARD_ATTR, "forward" in module.__dict__)
        setattr(module, _ORIGINAL_ATTR, module.forward)
        module.forward = types.MethodType(banded_sliding_forward, module)
    if count == 0:
        raise RuntimeError("banded attention: no sliding Inkling2LoopSlidingAttention module in the model")
    return count


def disable_banded_attention(model: torch.nn.Module) -> int:
    """Undo ``enable_banded_attention``; return how many modules were restored."""
    count = 0
    for module in model.modules():
        if not hasattr(module, _ORIGINAL_ATTR):
            continue
        original = getattr(module, _ORIGINAL_ATTR)
        if getattr(module, _HAD_INSTANCE_FORWARD_ATTR):
            module.forward = original
        else:
            del module.forward  # back to the class's forward
        delattr(module, _ORIGINAL_ATTR)
        delattr(module, _HAD_INSTANCE_FORWARD_ATTR)
        count += 1
    return count
