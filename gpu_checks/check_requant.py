"""Re-quantization gate for the MXFP8 layers 20-23 (ticket T5, step 2, of docs/inkling_fused_experts_plan.md).

Run in WSL (takes the GPU lock, needs 8 GiB free, exits 75 if the GPU is busy; peaks under 3 GiB):
    cd /path/to/Inkling-2-Mini-vLLM && HF_HUB_OFFLINE=1 PYTHONPATH=. ~/inkling-venv/bin/python gpu_checks/check_requant.py

For layer 20, experts 0-15, both projections: relative error against the MXFP8 original (``dequantize_mxfp8``) of
(a) NF4 through bitsandbytes, as ``Quant4bitExperts.set_weights`` stores it (bf16 input, ``compress_statistics``), and
(b) ``quantize_nvfp4``. Gate: mean (b) <= 1.15 x mean (a). Then step 1's fused/loop/golden comparison is repeated on
layer 20 loaded with ``mxfp8_storage="nvfp4"``. The gate result is printed either way; the exit code is non-zero if
the gate or the layer comparison fails.
"""  # run as scripts, not a package

import torch

from gpu_checks.check_real_layer import MODEL
from gpu_checks.check_real_layer import check_layer
from inkling_qlora import inkling as ik
from inkling_qlora.inkling_nvfp4 import dequantize_mxfp8
from inkling_qlora.inkling_nvfp4 import dequantize_nvfp4
from inkling_qlora.inkling_nvfp4 import quantize_nvfp4

MIN_FREE_MIB = 8192
LAYER, EXPERTS = 20, range(16)
GATE = 1.15


def rel(x: torch.Tensor, ref: torch.Tensor) -> float:
    """Return the relative L2 error of ``x`` against ``ref``."""
    return ((x.double() - ref.double()).norm() / ref.double().norm()).item()


def requant_errors() -> tuple[float, float]:
    """Return the mean relative error of NF4 and of NVFP4 against the MXFP8 original."""
    import bitsandbytes.functional as bnbf  # noqa: PLC0415  # heavy GPU dependency, lazy

    snapshot = ik._snapshot(MODEL)
    files = ik.expert_tensors(snapshot)[LAYER]
    assert "w13_weight_amax" not in files, f"layer {LAYER} is not MXFP8"
    raw = ik.read_layer(snapshot, LAYER, files, "cpu")
    errs: dict[str, list[float]] = {"nf4": [], "nvfp4": []}
    for proj in ("w13", "w2"):
        values, scales = raw[f"{proj}_weight"], raw[f"{proj}_weight_scale"]
        for e in EXPERTS:
            w = dequantize_mxfp8(values[e].cuda(), scales[e].cuda())  # float32 original
            packed, state = bnbf.quantize_4bit(
                w.to(torch.bfloat16).contiguous(), quant_type="nf4", compress_statistics=True
            )
            nf4 = bnbf.dequantize_4bit(packed, state).to(torch.bfloat16)
            nvfp4 = dequantize_nvfp4(*quantize_nvfp4(w))
            errs["nf4"].append(rel(nf4, w))
            errs["nvfp4"].append(rel(nvfp4, w))
            del w, packed, nf4, nvfp4
        n = len(EXPERTS)
        print(
            f"  {proj}: mean rel error NF4 {sum(errs['nf4'][-n:]) / n:.4f}, NVFP4 {sum(errs['nvfp4'][-n:]) / n:.4f} "
            f"({n} experts)"
        )
    mean_nf4 = sum(errs["nf4"]) / len(errs["nf4"])
    mean_nvfp4 = sum(errs["nvfp4"]) / len(errs["nvfp4"])
    return mean_nf4, mean_nvfp4


def main() -> int:
    """Evaluate the re-quantization gate, then compare the layer; return the exit code."""
    print(f"layer {LAYER}, experts {EXPERTS.start}-{EXPERTS.stop - 1}, both projections, vs the MXFP8 original")
    mean_nf4, mean_nvfp4 = requant_errors()
    ratio = mean_nvfp4 / mean_nf4
    gate_ok = ratio <= GATE
    print(
        f"  overall: NF4 {mean_nf4:.4f}, NVFP4 {mean_nvfp4:.4f}, ratio {ratio:.3f} (gate <= {GATE}): "
        f"{'PASS' if gate_ok else 'FAIL'}"
    )
    print(f"  peak allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    torch.cuda.empty_cache()
    check_layer(LAYER, "nvfp4")
    if not gate_ok:
        print("re-quantization gate FAILED: keep mxfp8_storage='nf4'")
        return 1
    print("re-quantization gate passed and the layer-20 comparison passed")
    return 0


if __name__ == "__main__":
    import sys

    from gpu_checks.gpu_guard import gpu_session

    with gpu_session(min_free_mib=MIN_FREE_MIB):
        code = main()
    sys.exit(code)
