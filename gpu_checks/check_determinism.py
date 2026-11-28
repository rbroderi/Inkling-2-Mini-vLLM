"""Check that the fused experts give bit-identical results on repeated runs.

GPU determinism check (ticket T6): fused_experts with TritonBackend twice on the same inputs gives bit-identical
output, d(hidden) and d(top_k_weights) in bf16.

Run in WSL (GPU lock with min_free_mib=4096, exits 75 if busy):
    cd /path/to/Inkling-2-Mini-vLLM && PYTHONPATH=. ~/inkling-venv/bin/python gpu_checks/check_determinism.py
"""  # run as scripts, not a package

import torch

from gpu_checks.gpu_guard import gpu_session
from inkling_qlora.inkling_kernels import fused_experts
from inkling_qlora.inkling_nvfp4 import quantize_nvfp4
from inkling_qlora.inkling_triton import TritonBackend

E, H, I, T, K = 8, 256, 128, 600, 4  # noqa: E741


def packed(rows: int, columns: int, seed: int):
    """Return ``E`` random expert matrices quantized to NVFP4 and stacked: codes, scales and factors."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    parts = [quantize_nvfp4(torch.randn(rows, columns, device="cuda", generator=g) * 0.05) for _ in range(E)]
    codes, scale, factor = zip(*parts, strict=True)
    return torch.stack(codes), torch.stack(scale), torch.stack(factor)


def run(backend, hidden0, index, weights0, grad):
    """Run one forward and backward pass and return the output and both gradients."""
    hidden = hidden0.clone().requires_grad_()
    top_w = weights0.clone().requires_grad_()
    out = fused_experts(hidden, index, top_w, backend, 4.0, 25.0)
    out.backward(grad)
    assert hidden.grad is not None
    assert top_w.grad is not None
    return out.detach(), hidden.grad, top_w.grad


def main() -> None:
    """Check that repeated runs give bit-identical results."""
    backend = TritonBackend(*packed(2 * I, H, 1), *packed(H, I, 2))
    g = torch.Generator(device="cuda").manual_seed(3)
    index = torch.randint(0, E + 1, (T, K), device="cuda", generator=g)  # includes the sentinel E
    index[:, 0] = 1  # many pairs on one expert and many tokens hit the same expert pairs: stresses the combine
    hidden = torch.randn(T, H, device="cuda", generator=g).bfloat16()
    weights = torch.rand(T, K, device="cuda", generator=g)
    grad = torch.randn(T, H, device="cuda", generator=g).bfloat16()
    first = run(backend, hidden, index, weights, grad)
    for _ in range(3):
        again = run(backend, hidden, index, weights, grad)
        for name, a, b in zip(("out", "dX", "d_top_k_weights"), first, again, strict=True):
            assert torch.equal(a, b), f"{name} differs between runs"
    print("fused_experts TritonBackend bf16: out, dX, d(top_k_weights) bit-identical over 4 runs")


if __name__ == "__main__":
    with gpu_session(min_free_mib=4096):
        main()
