"""Load Inkling 2 Mini for QLoRA on a 24 GB Ampere GPU, one layer of experts at a time.

The official checkpoint (thinkingmachines/Inkling-2-Mini) stores the MoE experts in NVFP4 and MXFP8, which Ampere cards
cannot compute in. transformers (PR #51050) dequantizes them to bf16 on load, which makes the model ~53 GB, too big for
a 3090 or for WSL's memory, and bitsandbytes does not quantize the experts' 3D tensors. So this module:

1. swaps the expert module for :class:`Quant4bitExperts` before the model is built, so the experts take no memory
   while transformers loads everything else (attention, router, embeddings: kept in bf16), and by default leaves
   out the vision tower, since judges read only text;
2. then reads the experts from the checkpoint one layer at a time. NVFP4 layers keep their packed form; MXFP8
   layers are dequantized with the PR's own op and quantized to 4-bit NF4 (bitsandbytes), or with
   ``mxfp8_storage="nvfp4"`` to NVFP4 so they can use the fused kernels; the full bf16 experts never exist
   at once.

The experts stay frozen (put LoRA on attention and the router); the forward pass is the original's eager loop over the
experts that received tokens, with each expert's weights dequantized on the fly, as bitsandbytes' Linear4bit does
(``experts_impl="fused"`` instead runs the NVFP4 layers through grouped kernels, see inkling_kernels.py).
Needs the separate environment with transformers from PR #51050 (``~/inkling-venv``), not the judges' training env.
"""

import json
import os
import re
from pathlib import Path
from typing import Any
from typing import cast

import torch
import torch.nn.functional as F  # noqa: N812  # torch's conventional alias
from torch import nn

from inkling_qlora.inkling_banded import enable_banded_attention
from inkling_qlora.inkling_kernels import SoftcappedSwiGLU
from inkling_qlora.inkling_kernels import TorchBackend
from inkling_qlora.inkling_kernels import fused_experts
from inkling_qlora.inkling_nvfp4 import E2M1_VALUES
from inkling_qlora.inkling_nvfp4 import dequantize_mxfp8
from inkling_qlora.inkling_nvfp4 import quantize_nvfp4

# LoRA on the language model's attention projections (the experts are frozen and quantized; the router is not a
# Linear). A regex, so the vision tower's layers are left alone.
INKLING_LORA_TARGETS = r".*language_model\..*\.(?:(?:q|k|v|o)_proj\.linear|o_proj\.gate_proj)"
# Inkling's chat template adds "Thinking effort level: 0.9" unless told otherwise; judges answer directly.
INKLING_CHAT_TEMPLATE_KWARGS = {"reasoning_effort": "none"}

EXPERT_KEY = re.compile(
    r"^model\.llm\.layers\.(\d+)\.mlp\.experts\.(w13|w2)_weight(_scale|_amax|_outer_scale_denominator)?$"
)

EXPERTS_IMPLS = ("loop", "fused")

# Exact banded sliding attention (inkling_banded.py) is on unless INKLING_BANDED_ATTENTION or the argument says
# otherwise. Set from A1's GPU gate (section 10.4 of docs/inkling_general_speed.md): PASS, so True.
BANDED_ATTENTION_DEFAULT = True

# The checkpoint revision everything was validated on (T5-T7). Pinned so that a new upstream revision cannot silently
# change or break what is loaded; the environment variable INKLING_REVISION overrides it. Only applies to this repo id.
INKLING_MODEL_ID = "thinkingmachines/Inkling-2-Mini"
INKLING_REVISION = "ed40e7da625f594fc9c4990f5f66be24873c22d7"


def inkling_revision(model_id: str) -> str | None:
    """Return the pinned revision (``INKLING_REVISION`` env var, else the constant) for Inkling's repo id, else None."""
    if model_id != INKLING_MODEL_ID:
        return None
    return os.environ.get("INKLING_REVISION") or INKLING_REVISION


class Quant4bitExperts(nn.Module):
    """Drop-in for Inkling2LoopExperts holding the experts quantized, each dequantized only when it is used.

    NVFP4 layers keep the checkpoint's own packed codes and scales (no added error: the same math as the PR's
    Inkling2LoopDequantizeExperts, per expert); MXFP8 layers (8-bit) are re-quantized to NF4 to save memory, or to
    NVFP4 when ``mxfp8_storage="nvfp4"``.
    """

    def __init__(self, config: Any) -> None:
        """Build empty experts sized from the model ``config``; the weights are set per layer later."""
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_cap = float(config.mlp_gate_softcap)
        self.up_cap = float(config.mlp_up_softcap)
        self.act_fn = SoftcappedSwiGLU(self.gate_cap, self.up_cap)
        self.quant_states: dict[str, list[Any]] = {"gate_up": [], "down": []}
        self.storage = ""  # "nvfp4" or "nf4" once loaded
        self.loaded = False
        # "fused" (default) only takes effect on NVFP4 storage, NF4 layers always loop; "loop" is the reference/rollback
        self.impl = "fused"
        self._backend_cache: tuple[tuple[torch.Tensor, ...], Any] | None = None

    @torch.no_grad()
    def set_nvfp4(self, parts: dict[str, torch.Tensor], device: torch.device | str) -> None:
        """Keep a layer's NVFP4 tensors (codes, block scales, amax, denominator for w13 and w2) as they are."""
        packed = {}
        for proj, name in (("w13", "gate_up"), ("w2", "down")):
            factor = parts[f"{proj}_weight_amax"].float() / parts[f"{proj}_weight_outer_scale_denominator"].float()
            packed[name] = (parts[f"{proj}_weight"], parts[f"{proj}_weight_scale"], factor)
        self.set_nvfp4_packed(packed, device)

    @torch.no_grad()
    def set_nvfp4_packed(
        self, packed: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]], device: torch.device | str
    ) -> None:
        """Register (codes, block scales, factor (E,)) for ``gate_up`` and ``down``, rows in checkpoint order."""
        for name in ("gate_up", "down"):
            codes, scale, factor = packed[name]
            self.register_buffer(f"{name}_codes", codes.to(device), persistent=False)
            self.register_buffer(f"{name}_scale", scale.to(device), persistent=False)
            self.register_buffer(f"{name}_factor", factor.to(device), persistent=False)
        self._backend_cache = None
        byte = torch.arange(256)
        values = torch.tensor(E2M1_VALUES)
        self.register_buffer(
            "byte_to_values", torch.stack((values[byte & 0xF], values[byte >> 4]), -1).to(device), persistent=False
        )
        self.storage, self.loaded = "nvfp4", True

    @torch.no_grad()
    def set_weights(self, gate_up: torch.Tensor, down: torch.Tensor, device: torch.device | str) -> None:
        """Quantize (E, 2I, H) gate_up and (E, H, I) down bf16 weights to NF4, one expert at a time, onto ``device``."""
        import bitsandbytes.functional as bnbf  # noqa: PLC0415  # optional heavy dependency

        assert gate_up.shape == (self.num_experts, 2 * self.intermediate_dim, self.hidden_dim), gate_up.shape  # noqa: S101  # shape guard
        assert down.shape == (self.num_experts, self.hidden_dim, self.intermediate_dim), down.shape  # noqa: S101  # shape guard
        for name, weights in (("gate_up", gate_up), ("down", down)):
            states = []
            for e in range(self.num_experts):
                packed, state = bnbf.quantize_4bit(
                    weights[e].to(device, torch.bfloat16).contiguous(), quant_type="nf4", compress_statistics=True
                )
                self.register_buffer(f"{name}_{e}", packed, persistent=False)
                states.append(state)
            self.quant_states[name] = states
        self.storage, self.loaded = "nf4", True

    def weight(self, name: str, e: int) -> torch.Tensor:
        """Return expert ``e``'s ``gate_up`` or ``down`` weight dequantized to bf16."""
        if self.storage == "nf4":
            import bitsandbytes.functional as bnbf  # noqa: PLC0415  # optional heavy dependency

            return bnbf.dequantize_4bit(getattr(self, f"{name}_{e}"), self.quant_states[name][e]).to(torch.bfloat16)
        codes = getattr(self, f"{name}_codes")[e]  # (rows, columns / 2) bytes, low nibble first
        scale = getattr(self, f"{name}_scale")[e].float()  # (rows, columns / 16)
        rows = codes.shape[0]
        w = cast("torch.Tensor", self.byte_to_values)[codes.long()].view(*scale.shape, 16) * scale[..., None]
        w = (w * getattr(self, f"{name}_factor")[e]).reshape(rows, -1)
        if name == "gate_up":  # the checkpoint interleaves gate and up rows; the model wants [gate; up]
            w = w.view(rows // 2, 2, -1).transpose(0, 1).reshape(rows, -1)
        return w.to(torch.bfloat16)

    def _apply(self, fn: Any, recurse: bool = True) -> Any:  # noqa: FBT001, FBT002  # mirrors nn.Module._apply
        """Handle ``.to``/``.cuda``/``.half`` etc.

        Moves the packed buffers but never changes their dtype (uint8 codes, float8 scales and float32 factors are
        not weights to cast), and drops the backend cache so the old device's tensors are not kept alive.
        """
        packed = {
            n: b for n, b in self._buffers.items() if b is not None and n.endswith(("_codes", "_scale", "_factor"))
        }
        for n in packed:
            self._buffers[n] = None  # hidden from super()._apply
        try:
            result = super()._apply(fn, recurse)
        finally:
            for n, b in packed.items():
                moved = fn(b)
                self._buffers[n] = moved if moved.dtype == b.dtype else b.to(device=moved.device)
            self._backend_cache = None
        return result

    def _backend(self, device: torch.device) -> Any:
        """Return the fused-path backend, built once over this module's own buffers.

        The buffers are not registered, so ``.to`` is safe; it is rebuilt if a buffer was replaced, e.g. by moving the
        module.
        """
        buffers = tuple(
            getattr(self, f"{name}_{part}") for name in ("gate_up", "down") for part in ("codes", "scale", "factor")
        )
        cache = self._backend_cache
        if cache is None or any(a is not b for a, b in zip(cache[0], buffers, strict=True)):
            if torch.device(device).type == "cuda":
                from inkling_qlora.inkling_triton import TritonBackend  # noqa: PLC0415  # imports triton

                backend = TritonBackend(*buffers)
            else:
                backend = TorchBackend(*buffers)
            cache = self._backend_cache = (buffers, backend)
        return cache[1]

    def _forward_loop(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        # the original's eager loop (modeling_inkling2_loop.Inkling2LoopExperts.forward), weights dequantized per expert
        final = torch.zeros_like(hidden_states, dtype=torch.float32)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for idx in hit:
            e = int(idx[0])
            if e == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(mask[e])
            x = hidden_states[token_idx]
            gate, up = F.linear(x, self.weight("gate_up", e).to(x.dtype)).chunk(2, dim=-1)
            y = F.linear(self.act_fn(gate, up), self.weight("down", e).to(x.dtype))
            y = y * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, y.to(final.dtype))
        return final.to(hidden_states.dtype)

    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        """Route the tokens through the grouped kernels (NVFP4 storage, ``fused``) or the per-expert loop."""
        if self.impl == "fused" and self.storage == "nvfp4":
            backend = self._backend(hidden_states.device)
            return fused_experts(hidden_states, top_k_index, top_k_weights, backend, self.gate_cap, self.up_cap)
        return self._forward_loop(hidden_states, top_k_index, top_k_weights)


def _snapshot(model_id: str, revision: str | None = None) -> Path:
    """Return the local snapshot of ``model_id`` at ``revision`` (default: the pinned ``inkling_revision``)."""
    from huggingface_hub import snapshot_download  # noqa: PLC0415  # optional heavy dependency

    return Path(snapshot_download(model_id, revision=revision or inkling_revision(model_id)))


def expert_tensors(snapshot: Path) -> dict[int, dict[str, str]]:
    """{layer: {checkpoint key suffix: shard file}} for every expert tensor in the checkpoint."""
    weight_map = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    out: dict[int, dict[str, str]] = {}
    for key, shard in weight_map.items():
        m = EXPERT_KEY.match(key)
        if m:
            out.setdefault(int(m.group(1)), {})[f"{m.group(2)}_weight{m.group(3) or ''}"] = shard
    return out


def read_layer(snapshot: Path, layer: int, files: dict[str, str], device: str) -> dict[str, torch.Tensor]:
    """Read one layer's raw expert tensors from the checkpoint, keyed by suffix (``w13_weight``, ...)."""
    from safetensors import safe_open  # noqa: PLC0415  # optional heavy dependency

    parts = {}
    for name, shard in files.items():
        with safe_open(snapshot / shard, "pt", device=device) as fh:
            parts[name] = fh.get_tensor(f"model.llm.layers.{layer}.mlp.experts.{name}")
    return parts


@torch.no_grad()
def requantize_mxfp8_to_nvfp4(
    raw: dict[str, torch.Tensor], device: torch.device | str
) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Re-quantize an MXFP8 layer's raw tensors (as read by ``read_layer``) to NVFP4 on ``device``.

    Per expert and projection, decoded to float32 on the device and quantized, so one expert in float32 is the
    peak. Rows stay in checkpoint order (gate/up interleaved), as the NVFP4 layers' are.
    """
    packed = {}
    for proj, name in (("w13", "gate_up"), ("w2", "down")):
        values, scales = raw[f"{proj}_weight"], raw[f"{proj}_weight_scale"]
        codes, block_scales, factors = [], [], []
        for e in range(values.shape[0]):
            w = dequantize_mxfp8(values[e].to(device), scales[e].to(device))
            c, s, f = quantize_nvfp4(w)
            del w
            codes.append(c)
            block_scales.append(s)
            factors.append(f)
        packed[name] = (torch.stack(codes), torch.stack(block_scales), torch.stack(factors))
    return packed


@torch.no_grad()
def dequantize_layer(
    snapshot: Path, layer: int, files: dict[str, str], device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one layer's experts as bf16 (gate_up (E, 2I, H), down (E, H, I)), through the PR's own conversion ops."""
    from transformers.core_model_loading import Inkling2LoopDequantizeExperts  # noqa: PLC0415  # optional heavy dependency  # pyright: ignore[reportAttributeAccessIssue]  # pyrefly: ignore[missing-module-attribute]
    from transformers.core_model_loading import Interleave  # noqa: PLC0415  # optional heavy dependency

    raw = read_layer(snapshot, layer, files, device)
    out = []
    for proj in ("w13", "w2"):
        parts = {k: v for k, v in raw.items() if k.startswith(f"{proj}_")}
        sources = [f"{proj}_weight{s}" for s in ("", "_scale", "_amax", "_outer_scale_denominator")]
        w = Inkling2LoopDequantizeExperts().convert(parts, sources, ["w"])["w"]
        if proj == "w13":  # the checkpoint interleaves gate and up rows; the model wants [gate; up]
            w = Interleave(dim=1).convert({"w": w}, ["w"], ["w"])["w"]
        out.append(w)
        del parts
    return out[0], out[1]


class _NoVisionTower(nn.Module):
    """Stands in for the vision tower when only text is judged: no parameters, so its checkpoint weights are skipped."""

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Fail: this stand-in has no vision weights."""
        msg = "the vision tower was not loaded (text_only=True); images are not supported"
        raise RuntimeError(msg)


class _TextOnlyAutoModel:
    """Replaces ``AutoModel`` inside the Inkling module while it is built; it is only used for the vision tower."""

    @staticmethod
    def from_config(*args: Any, **kwargs: Any) -> nn.Module:
        """Return the parameter-free vision tower stand-in."""
        return _NoVisionTower()


def _embeddings_to_cpu(embedding: nn.Embedding) -> None:
    """Keep the (frozen) token embedding table in CPU memory and look rows up there.

    Only the looked-up rows are sent to the GPU. Inkling's table is 201,024 x 2,304 (0.86 GiB in bf16); a lookup
    of a few thousand rows costs little.

    The table is held as a plain attribute, not a parameter, so ``model.parameters()`` (and so ``model.device``, which
    reads the first parameter) still points at the GPU. It is left out of ``state_dict``: fine for LoRA, which saves
    only the adapter, but do not save the full model from here. The norm after the lookup stays on the GPU.

    ``embedding.weight`` is set back to the same CPU tensor, but as a plain attribute (not a Parameter or buffer):
    transformers' Trainer registers a bitsandbytes 32-bit override on every ``nn.Embedding`` and bnb's
    ``check_overrides`` does ``getattr(module, "weight")``; the tensor matches no optimizer parameter, so it is ignored.
    """
    table = embedding.weight.detach().to("cpu").contiguous()
    padding_idx = embedding.padding_idx
    del embedding.weight
    embedding.weight = table  # type: ignore[assignment]  # a plain tensor attribute, see above
    embedding.cpu_table = table

    def forward(input_ids: torch.Tensor) -> torch.Tensor:
        rows = F.embedding(input_ids.to("cpu"), table, padding_idx).to(input_ids.device, non_blocking=True)
        norm: Any = getattr(embedding, "embed_norm", None)
        return norm(rows) if norm is not None else rows

    embedding.forward = forward  # type: ignore[method-assign]


def resolve_experts_impl(experts_impl: str | None) -> str:
    """Return ``experts_impl``, or ``INKLING_EXPERTS_IMPL`` (default "fused") when None; ValueError if unknown."""
    if experts_impl is None:
        experts_impl = os.environ.get("INKLING_EXPERTS_IMPL", "fused")
    if experts_impl not in EXPERTS_IMPLS:
        msg = f"experts_impl must be one of {EXPERTS_IMPLS}, got {experts_impl!r}"
        raise ValueError(msg)
    return experts_impl


def resolve_banded_attention(banded_attention: bool | None) -> bool:  # noqa: FBT001  # tests pass it positionally
    """Return ``banded_attention``, or ``INKLING_BANDED_ATTENTION`` ("1" on, "0" off) when None.

    Falls back to ``BANDED_ATTENTION_DEFAULT`` when the variable is unset; ValueError for any other value.
    """
    if banded_attention is not None:
        return bool(banded_attention)
    value = os.environ.get("INKLING_BANDED_ATTENTION")
    if value is None:
        return BANDED_ATTENTION_DEFAULT
    if value == "1":
        return True
    if value == "0":
        return False
    msg = f"INKLING_BANDED_ATTENTION must be '1' or '0', got {value!r}"
    raise ValueError(msg)


def load_inkling_4bit(  # noqa: C901, PLR0913, PLR0915  # one linear loading recipe with keyword-only options
    model_id: str = "thinkingmachines/Inkling-2-Mini",
    *,
    device: str = "cuda",
    layers: list[int] | None = None,
    text_only: bool = True,
    embeddings_on_cpu: bool = True,
    experts_impl: str | None = None,
    mxfp8_storage: str = "nvfp4",
    banded_attention: bool | None = None,
    **from_pretrained_kwargs: Any,
) -> nn.Module:
    """Return the model with every expert quantized on ``device`` and the rest in bf16.

    ``layers`` limits which layers' experts are loaded (a quick test); the others stay empty and must not be run.
    ``text_only`` (the default; judges read text) leaves out the vision tower: 0.32B parameters, about 0.6 GiB of
    GPU memory freed for training. ``embeddings_on_cpu`` (the default) keeps the token embedding table in CPU
    memory, another 0.86 GiB freed; the output layer (``lm_head``, the same size) stays on the GPU, since every
    token needs all of it.

    ``experts_impl`` is "loop" (the per-expert loop, the reference) or "fused" (the grouped kernels, NVFP4 layers
    only); None reads ``INKLING_EXPERTS_IMPL``, default "fused" (``INKLING_EXPERTS_IMPL=loop`` rolls back).
    ``mxfp8_storage`` is how the MXFP8 layers are kept: "nvfp4" (the default; re-quantized to NVFP4, so they can use
    the fused path) or "nf4" (re-quantized to NF4, looped).

    ``banded_attention`` turns on the exact banded sliding attention (inkling_banded.py) once the model is loaded.
    None reads ``INKLING_BANDED_ATTENTION`` ("1" on, "0" off), default ``BANDED_ATTENTION_DEFAULT`` (on). It changes
    only sequences longer than the 1,024-token window; shorter ones take the original path.
    """
    experts_impl = resolve_experts_impl(experts_impl)
    banded_attention = resolve_banded_attention(banded_attention)
    if mxfp8_storage not in ("nf4", "nvfp4"):
        msg = f"mxfp8_storage must be 'nf4' or 'nvfp4', got {mxfp8_storage!r}"
        raise ValueError(msg)
    import transformers.models.inkling2_loop.modeling_inkling2_loop as modeling  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
    from transformers import AutoModelForImageTextToText  # noqa: PLC0415  # optional heavy dependency

    original = modeling.Inkling2LoopExperts
    original_auto = modeling.AutoModel
    base = modeling.Inkling2LoopPreTrainedModel
    original_init = base._init_weights  # noqa: SLF001  # patched while the model is built

    def init_weights(self: Any, module: nn.Module) -> None:  # the experts are filled below, not initialized
        if not isinstance(module, Quant4bitExperts):
            original_init(self, module)

    modeling.Inkling2LoopExperts = Quant4bitExperts  # the sparse MoE block builds its experts from this name
    if text_only:
        modeling.AutoModel = _TextOnlyAutoModel  # Inkling2LoopModel builds its vision tower through AutoModel
    base._init_weights = init_weights  # noqa: SLF001  # patched while the model is built
    revision = from_pretrained_kwargs.pop("revision", None) or inkling_revision(model_id)
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device, revision=revision, **from_pretrained_kwargs
        )
    finally:
        modeling.Inkling2LoopExperts = original
        modeling.AutoModel = original_auto
        base._init_weights = original_init  # noqa: SLF001  # restore the original
    if embeddings_on_cpu and device != "cpu":
        _embeddings_to_cpu(cast("nn.Embedding", model.get_input_embeddings()))
        torch.cuda.empty_cache()
    snapshot = _snapshot(model_id, revision)
    files = expert_tensors(snapshot)
    experts = {
        int(m.group(1)): mod
        for name, mod in model.named_modules()
        if isinstance(mod, Quant4bitExperts) and (m := re.search(r"layers\.(\d+)\.mlp\.experts$", name))
    }
    assert sorted(experts) == sorted(files), (sorted(experts), sorted(files))  # noqa: S101  # consistency guard
    for mod in experts.values():
        mod.impl = experts_impl
    for layer in sorted(experts) if layers is None else layers:
        if "w13_weight_amax" not in files[layer] and mxfp8_storage == "nvfp4":
            raw = read_layer(snapshot, layer, files[layer], "cpu")
            experts[layer].set_nvfp4_packed(requantize_mxfp8_to_nvfp4(raw, device), device)
            del raw
            if device != "cpu":
                torch.cuda.empty_cache()
            continue
        if "w13_weight_amax" in files[layer]:  # NVFP4: keep the checkpoint's packed form
            experts[layer].set_nvfp4(read_layer(snapshot, layer, files[layer], device), device)
            continue
        gate_up, down = dequantize_layer(snapshot, layer, files[layer], device)
        experts[layer].set_weights(gate_up, down, device)
        del gate_up, down
        if device != "cpu":
            torch.cuda.empty_cache()
    if banded_attention:
        enable_banded_attention(model)
    return model
