"""Block-size sweep for the NVFP4 grouped GEMM on a real layer (ticket T5, step 3, of
docs/inkling_fused_experts_plan.md).

Run in WSL (takes the GPU lock, needs 8 GiB free, exits 75 if the GPU is busy):
    cd /path/to/Inkling-2-Mini-vLLM && HF_HUB_OFFLINE=1 PYTHONPATH=. ~/inkling-venv/bin/python gpu_checks/tune_blocks.py

Layer 0's real weights; 750 tokens x top-8 drawn from a random permutation per token (6,000 rows over 128 experts),
bf16 input. For each (projection, transpose), every combination of BLOCK_M in {32, 64, 128}, BLOCK_N in {64, 128},
BLOCK_K in {32, 64}, num_warps in {4, 8} (num_stages 3, or 2 when 3 does not fit in shared memory) is timed: 5
warm-up calls (which also exclude JIT compilation), then the median of 20 calls, each between two
``torch.cuda.synchronize()``.

The routing plan's block_m is one value per ``fused_experts`` call, shared by the four GEMMs, and each kernel's BLOCK_M
must be >= it (a larger BLOCK_M only wastes tile rows), so BLOCK_M = block_m is tuned jointly: the chosen block_m
minimises the per-example cost 2 x (forward gate_up + forward down) + transposed gate_up + transposed down (gradient
checkpointing runs the forward twice), and the other sizes are the fastest per (projection, transpose) at that block_m.
Prints the table to paste into ``inkling_triton.TUNED_BLOCK_SIZES``.
"""

from __future__ import annotations

import itertools
import statistics
import time

import torch

from gpu_checks.check_real_layer import load_layer, make_inputs
from inkling_qlora.inkling_kernels import build_route_plan
from inkling_qlora.inkling_triton import nvfp4_grouped_gemm

MIN_FREE_MIB = 8192
BLOCK_MS, BLOCK_NS, BLOCK_KS, WARPS = (32, 64, 128), (64, 128), (32, 64), (4, 8)
CALLS_PER_EXAMPLE = {("gate_up", False): 2, ("down", False): 2, ("gate_up", True): 1, ("down", True): 1}


def time_call(fn) -> float:
    for _ in range(5):
        fn()
    times = []
    for _ in range(20):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    return statistics.median(times) * 1e3


def main() -> None:
    mod = load_layer(0)
    hidden, top_k_index, _, _ = make_inputs(mod)
    lut = mod._backend(hidden.device).e4m3_lut
    weights = {
        "gate_up": (mod.gate_up_codes, mod.gate_up_scale.view(torch.uint8), mod.gate_up_factor.float().contiguous()),
        "down": (mod.down_codes, mod.down_scale.view(torch.uint8), mod.down_factor.float().contiguous()),
    }
    num_rows = top_k_index.numel()
    g = torch.Generator(device="cuda").manual_seed(7)
    inputs = {
        ("gate_up", False): torch.randn(num_rows, 2304, device="cuda", generator=g).to(torch.bfloat16),
        ("down", False): torch.randn(num_rows, 1152, device="cuda", generator=g).to(torch.bfloat16),
        ("gate_up", True): torch.randn(num_rows, 2304, device="cuda", generator=g).to(torch.bfloat16),
        ("down", True): torch.randn(num_rows, 2304, device="cuda", generator=g).to(torch.bfloat16),
    }
    results: dict[tuple[str, bool], dict[tuple, float]] = {key: {} for key in CALLS_PER_EXAMPLE}
    print(f"{num_rows} rows over {mod.num_experts} experts, bf16", flush=True)
    for block_m in BLOCK_MS:
        plan = build_route_plan(top_k_index, mod.num_experts, block_m)
        table = (plan.expert_of_block, plan.block_row0, plan.block_rows)
        for (proj, transpose), bn, bk, warps in itertools.product(CALLS_PER_EXAMPLE, BLOCK_NS, BLOCK_KS, WARPS):
            codes, scale, factor = weights[proj]
            inp = inputs[(proj, transpose)]
            ms = None
            for stages in (3, 2):
                sizes = dict(block_m=block_m, block_n=bn, block_k=bk, num_warps=warps, num_stages=stages)

                def call(
                    sizes=sizes, codes=codes, scale=scale, factor=factor, inp=inp, transpose=transpose, table=table
                ):
                    nvfp4_grouped_gemm(inp, codes, scale, factor, lut, *table, transpose, **sizes)

                try:
                    ms = time_call(call)
                    break
                except Exception as exc:  # triton OutOfResources (shared memory) and the like
                    print(f"  skip {proj} T={int(transpose)} {sizes}: {type(exc).__name__}", flush=True)
            if ms is None:
                continue
            results[(proj, transpose)][(block_m, bn, bk, warps, stages)] = ms
            print(
                f"  {proj:7s} T={int(transpose)} bm{block_m:<3d} bn{bn:<3d} bk{bk:<2d} w{warps} s{stages}: {ms:.3f} ms",
                flush=True,
            )

    print("\nper-example cost (2 x forward + 1 x transposed, both projections) with the best sizes at each block_m:")
    best_cost, best_m = float("inf"), None
    for block_m in BLOCK_MS:
        cost = 0.0
        for key, n in CALLS_PER_EXAMPLE.items():
            cands = [ms for cfg, ms in results[key].items() if cfg[0] == block_m]
            cost += n * min(cands) if cands else float("inf")
        print(f"  block_m {block_m}: {cost:.2f} ms per layer per example")
        if cost < best_cost:
            best_cost, best_m = cost, block_m
    print(f"\nchosen block_m {best_m}; TUNED_BLOCK_SIZES = {{")
    for key in CALLS_PER_EXAMPLE:
        cfg, ms = min(((c, t) for c, t in results[key].items() if c[0] == best_m), key=lambda x: x[1])
        start = results[key].get((64, 64, 32, 4, 3))
        print(
            f'    {key!r}: {{"block_m": {cfg[0]}, "block_n": {cfg[1]}, "block_k": {cfg[2]}, "num_warps": {cfg[3]}, '
            f'"num_stages": {cfg[4]}}},  # {ms:.2f} ms (start values {start:.2f} ms)'
        )
    print("}")


if __name__ == "__main__":
    from gpu_checks.gpu_guard import gpu_session

    with gpu_session(min_free_mib=MIN_FREE_MIB):
        main()
