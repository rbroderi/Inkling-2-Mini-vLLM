"""Compare fused and loop experts on a real Inkling layer against an fp32 golden path.

Fused vs loop experts on a real Inkling layer, against an independent fp32 golden path (ticket T5, step 1, of
docs/inkling_fused_experts_plan.md).

Run in WSL (takes the GPU lock, needs 8 GiB free, exits 75 if the GPU is busy; peaks under 2 GiB):
    cd /path/to/Inkling-2-Mini-vLLM && HF_HUB_OFFLINE=1 PYTHONPATH=. ~/inkling-venv/bin/python gpu_checks/check_real_layer.py
    ... check_real_layer.py --layer 20 --mxfp8-storage nvfp4   # an MXFP8 layer re-quantized to NVFP4

Inputs: 750 bf16 hidden rows at unit RMS, top-8 experts from a random permutation per token, float32 routing weights
summing to 8 per token, and a random bf16 upstream gradient. Compared for the output, d(hidden), d(top_k_weights):

- fused vs loop (both bf16): rel <= 5e-3 (output), <= 1e-2 (gradients), cosine >= 0.9999;
- each vs the golden path (weights decoded to fp32 per expert without bf16 rounding, everything fp32, TF32 off):
  fused error <= 1.25 x loop error + 1e-4.

Also checks, on the CPU, that ``SoftcappedSwiGLU`` equals transformers' ``Inkling2LoopSoftcappedSwiGLU`` exactly.
"""  # run as scripts, not a package

import argparse
from typing import Any

import torch

from inkling_qlora import inkling as ik
from inkling_qlora.inkling_kernels import SoftcappedSwiGLU
from inkling_qlora.inkling_nvfp4 import dequantize_nvfp4

MODEL = "thinkingmachines/Inkling-2-Mini"
MIN_FREE_MIB = 8192
TOKENS, TOP_K = 750, 8


def rel(x: torch.Tensor, ref: torch.Tensor) -> float:
    """Return the relative L2 error of ``x`` against ``ref``."""
    return ((x.double() - ref.double()).norm() / ref.double().norm()).item()


def cosine(x: torch.Tensor, ref: torch.Tensor) -> float:
    """Return the cosine similarity of ``x`` and ``ref``."""
    x, ref = x.double().flatten(), ref.double().flatten()
    return (x @ ref / (x.norm() * ref.norm())).item()


def text_config():
    """Return the model's text config."""
    from transformers import AutoConfig  # noqa: PLC0415  # heavy GPU dependency, lazy

    return AutoConfig.from_pretrained(MODEL).get_text_config()


def load_layer(layer: int, mxfp8_storage: str = "nf4", device: str = "cuda") -> ik.Quant4bitExperts:
    """One real layer's experts as ``load_inkling_4bit`` would store them (NVFP4, or MXFP8 per ``mxfp8_storage``)."""
    mod = ik.Quant4bitExperts(text_config())
    snapshot = ik._snapshot(MODEL)
    files = ik.expert_tensors(snapshot)[layer]
    if "w13_weight_amax" in files:
        mod.set_nvfp4(ik.read_layer(snapshot, layer, files, device), device)
    elif mxfp8_storage == "nvfp4":
        raw = ik.read_layer(snapshot, layer, files, "cpu")
        mod.set_nvfp4_packed(ik.requantize_mxfp8_to_nvfp4(raw, device), device)
    else:
        msg = f"layer {layer} is MXFP8; this check needs mxfp8_storage='nvfp4'"
        raise ValueError(msg)
    return mod


def make_inputs(mod: ik.Quant4bitExperts, seed: int = 0, device: str = "cuda"):
    """Build random hidden rows, routing, routing weights and an upstream gradient."""
    g = torch.Generator(device=device).manual_seed(seed)
    hidden = torch.randn(TOKENS, mod.hidden_dim, device=device, generator=g)
    hidden = (hidden / hidden.pow(2).mean(-1, keepdim=True).sqrt()).to(torch.bfloat16)
    top_k_index = torch.rand(TOKENS, mod.num_experts, device=device, generator=g).argsort(-1)[:, :TOP_K]
    weights = torch.rand(TOKENS, TOP_K, device=device, generator=g) + 0.05
    weights = weights / weights.sum(-1, keepdim=True) * TOP_K
    grad = torch.randn(TOKENS, mod.hidden_dim, device=device, generator=g).to(torch.bfloat16)
    return hidden, top_k_index, weights, grad


def run_module(mod: ik.Quant4bitExperts, impl: str, hidden, top_k_index, weights, grad):  # noqa: PLR0913, PLR0917  # check helper, many inputs
    """Run the experts module with implementation ``impl``, forward and backward."""
    mod.impl = impl
    h = hidden.clone().requires_grad_()
    w = weights.clone().requires_grad_()
    out = mod(h, top_k_index, w)
    out.backward(grad)
    return out.detach(), h.grad, w.grad


def golden(mod: Any, hidden, top_k_index, weights, grad):
    """Run an independent fp32 eager loop as the reference.

    Each expert's weight is decoded to fp32 with the codec (no bf16 rounding), gate/up de-interleaved (even rows =
    gate), the softcapped SwiGLU written out, fp32 autograd.
    """
    from torch.utils.checkpoint import checkpoint  # noqa: PLC0415  # heavy GPU dependency, lazy

    def expert(x: torch.Tensor, e: int) -> torch.Tensor:
        gu = dequantize_nvfp4(mod.gate_up_codes[e], mod.gate_up_scale[e], mod.gate_up_factor[e])
        dn = dequantize_nvfp4(mod.down_codes[e], mod.down_scale[e], mod.down_factor[e])
        gate, up = x @ gu[0::2].T, x @ gu[1::2].T
        act = mod.gate_cap * torch.tanh(gate / mod.gate_cap) * torch.sigmoid(gate)
        return (act * mod.up_cap * torch.tanh(up / mod.up_cap)) @ dn.T

    h = hidden.float().requires_grad_()
    w = weights.float().requires_grad_()
    out = torch.zeros_like(h)
    for e in range(mod.num_experts):
        tok, pos = torch.where(top_k_index == e)
        if tok.numel() == 0:
            continue
        # checkpointed so the fp32 weights are decoded again in the backward instead of saved (4 GiB for 128 experts)
        y = checkpoint(expert, h[tok], e, use_reentrant=False) * w[tok, pos, None]
        out = out.index_add(0, tok, y)
    out.backward(grad.float())
    return out.detach(), h.grad, w.grad


def check_swiglu_cpu() -> None:
    """Check that our softcapped SwiGLU equals the model's own on random bf16 input (CPU)."""
    from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopSoftcappedSwiGLU  # noqa: PLC0415  # heavy GPU dependency, lazy  # pyright: ignore[reportMissingImports]

    cfg = text_config()
    ours = SoftcappedSwiGLU(cfg.mlp_gate_softcap, cfg.mlp_up_softcap)
    theirs = Inkling2LoopSoftcappedSwiGLU(cfg)
    g = torch.Generator().manual_seed(5)
    gate = (torch.randn(512, 1152, generator=g) * 6).to(torch.bfloat16)
    up = (torch.randn(512, 1152, generator=g) * 30).to(torch.bfloat16)
    assert torch.equal(ours(gate, up), theirs(gate, up)), "SoftcappedSwiGLU differs from Inkling2LoopSoftcappedSwiGLU"
    print("SoftcappedSwiGLU == Inkling2LoopSoftcappedSwiGLU on random bf16 input (CPU, torch.equal)")


def compare(mod: ik.Quant4bitExperts, inputs, label: str) -> tuple[dict[str, float], list[str]]:
    """Fused vs loop vs golden on ``inputs`` = (hidden, top_k_index, weights, grad); returns (numbers, failed bars)."""
    peaks = {}
    base = torch.cuda.memory_allocated()
    for name, fn, extra in (("fused", run_module, ("fused",)), ("loop", run_module, ("loop",)), ("golden", golden, ())):
        torch.cuda.reset_peak_memory_stats()
        peaks[name] = (fn(mod, *extra, *inputs), (torch.cuda.max_memory_allocated() - base) / 2**30)
    (fused, p_fused), (loop, p_loop), (gold, p_gold) = peaks["fused"], peaks["loop"], peaks["golden"]
    print(label)
    results: dict[str, float] = {}
    failures = []
    for name, f, lp, gd, bar in zip(
        ("out", "d_hidden", "d_weights"), fused, loop, gold, (5e-3, 1e-2, 1e-2), strict=True
    ):
        r, c = rel(f, lp), cosine(f, lp)
        ef, el = rel(f, gd), rel(lp, gd)
        limit = 1.25 * el + 1e-4
        ok = r <= bar and c >= 0.9999 and ef <= limit
        results.update({f"{name}_rel": r, f"{name}_cos": c, f"{name}_fused_gold": ef, f"{name}_loop_gold": el})
        print(
            f"  {name:9s} fused vs loop rel {r:.2e} (bar {bar:.0e}), cosine {c:.7f} (bar 0.9999); vs fp32 golden: "
            f"fused {ef:.2e}, loop {el:.2e} (bar fused <= {limit:.2e}) {'ok' if ok else 'FAIL'}"
        )
        if not ok:
            failures.append(name)
    print(
        f"  layer weights {base / 2**30:.2f} GiB; peak above that: fused {p_fused:.2f}, loop {p_loop:.2f} "
        f"(its autograd saves every used expert's bf16 weights), golden {p_gold:.2f} GiB"
    )
    return results, failures


def check_layer(layer: int, mxfp8_storage: str = "nf4") -> dict[str, float]:
    """Run the comparison on one layer with synthetic inputs; asserts every bar and returns the measured numbers."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    mod = load_layer(layer, mxfp8_storage)
    inputs = make_inputs(mod)
    label = f"layer {layer} (storage {mod.storage}, mxfp8_storage={mxfp8_storage}), {TOKENS} tokens x top-{TOP_K}"
    results, failures = compare(mod, inputs, label)
    assert not failures, f"layer {layer}: bars failed for {failures}"
    mod = None
    torch.cuda.empty_cache()
    return results


def main() -> None:
    """Compare fused and loop experts on a real layer against the fp32 golden path."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--mxfp8-storage", choices=("nf4", "nvfp4"), default="nf4")
    args = parser.parse_args()
    check_swiglu_cpu()
    check_layer(args.layer, args.mxfp8_storage)
    print("real-layer check passed")


if __name__ == "__main__":
    from gpu_checks.gpu_guard import gpu_session
    from gpu_checks.inkling_snapshot import pin_full_snapshot

    pin_full_snapshot()
    with gpu_session(min_free_mib=MIN_FREE_MIB):
        main()
