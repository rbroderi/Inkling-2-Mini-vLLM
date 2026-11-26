"""Environment probe for the Inkling fused-expert work: versions, Triton tl.dot (bf16, fp32 tf32/ieee) and the
2-D tl.interleave -> tl.trans -> tl.dot pattern the NVFP4 kernel uses.

Run in WSL: cd /path/to/Inkling-2-Mini-vLLM && PYTHONPATH=. ~/inkling-venv/bin/python gpu_checks/check_env.py
Exits 75 if the GPU is busy. Uses under 100 MB.
"""

import torch
import triton
import triton.language as tl

from gpu_checks.gpu_guard import gpu_session


@triton.jit
def _dot_kernel(a_ptr, b_ptr, c_ptr, N: tl.constexpr, PRECISION: tl.constexpr):
    i = tl.arange(0, N)
    a = tl.load(a_ptr + i[:, None] * N + i[None, :])
    b = tl.load(b_ptr + i[:, None] * N + i[None, :])
    tl.store(c_ptr + i[:, None] * N + i[None, :], tl.dot(a, b, input_precision=PRECISION, out_dtype=tl.float32))


@triton.jit
def _dot_bf16_kernel(a_ptr, b_ptr, c_ptr, N: tl.constexpr):
    i = tl.arange(0, N)
    a = tl.load(a_ptr + i[:, None] * N + i[None, :])
    b = tl.load(b_ptr + i[:, None] * N + i[None, :])
    tl.store(c_ptr + i[:, None] * N + i[None, :], tl.dot(a, b, out_dtype=tl.float32))


@triton.jit
def _interleave_dot_kernel(x_ptr, a_ptr, c_ptr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """x: uint8 [BN, BK/2] (two codes per byte); out = a[BM, BK] @ interleave(lo, hi)[BN, BK]^T."""
    n = tl.arange(0, BN)
    kh = tl.arange(0, BK // 2)
    x = tl.load(x_ptr + n[:, None] * (BK // 2) + kh[None, :]).to(tl.int32)
    w = tl.interleave(x & 0xF, (x >> 4) & 0xF)  # [BN, BK], interleaved along the last axis
    wt = tl.trans(w.to(tl.bfloat16))  # [BK, BN]
    m = tl.arange(0, BM)
    k = tl.arange(0, BK)
    a = tl.load(a_ptr + m[:, None] * BK + k[None, :])
    tl.store(c_ptr + m[:, None] * BN + n[None, :], tl.dot(a, wt, out_dtype=tl.float32))


def _rel(x: torch.Tensor, ref: torch.Tensor) -> float:
    return ((x.double() - ref).norm() / ref.norm()).item()


def main() -> None:
    print(f"torch {torch.__version__}, triton {triton.__version__}, CUDA {torch.version.cuda}")
    print(f"device {torch.cuda.get_device_name(0)}, capability {torch.cuda.get_device_capability(0)}")

    n = 64
    g = torch.Generator(device="cuda").manual_seed(0)

    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16, generator=g)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16, generator=g)
    c = torch.empty(n, n, device="cuda", dtype=torch.float32)
    _dot_bf16_kernel[(1,)](a, b, c, N=n)
    rel = _rel(c, a.double() @ b.double())
    assert rel <= 1e-5, f"bf16 tl.dot rel error {rel}"
    print(f"tl.dot bf16 64x64, fp32 accumulate: passed (rel {rel:.2e})")

    a32 = torch.randn(n, n, device="cuda", dtype=torch.float32, generator=g)
    b32 = torch.randn(n, n, device="cuda", dtype=torch.float32, generator=g)
    ref32 = a32.double() @ b32.double()
    _dot_kernel[(1,)](a32, b32, c, N=n, PRECISION="tf32")
    print(f"tl.dot fp32 inputs, default input_precision=tf32: rel {_rel(c, ref32):.2e} (informational)")
    _dot_kernel[(1,)](a32, b32, c, N=n, PRECISION="ieee")
    rel = _rel(c, ref32)
    assert rel <= 1e-5, f"fp32 ieee tl.dot rel error {rel}"
    print(f"tl.dot fp32 inputs, input_precision=ieee: passed (rel {rel:.2e})")

    bm, bn, bk = 64, 64, 64
    x = torch.randint(0, 256, (bn, bk // 2), device="cuda", dtype=torch.uint8)
    a = torch.randn(bm, bk, device="cuda", dtype=torch.bfloat16, generator=g)
    c = torch.empty(bm, bn, device="cuda", dtype=torch.float32)
    _interleave_dot_kernel[(1,)](x, a, c, BM=bm, BN=bn, BK=bk)
    xi = x.to(torch.int32)
    w = torch.stack([xi & 0xF, (xi >> 4) & 0xF], dim=-1).reshape(bn, bk)
    rel = _rel(c, a.double() @ w.double().T)
    assert rel <= 1e-5, f"2-D interleave -> trans -> dot rel error {rel}"
    print(f"2-D tl.interleave -> tl.trans -> tl.dot (uint8 codes): passed (rel {rel:.2e})")


if __name__ == "__main__":
    with gpu_session():
        main()
