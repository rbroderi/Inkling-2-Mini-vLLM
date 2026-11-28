"""GPU acceptance check for the Triton NVFP4 grouped GEMM (ticket T3 of docs/inkling_fused_experts_plan.md).

Run in WSL (takes the GPU lock with min_free_mib=4096, exits 75 if the GPU is busy; peaks around 1-2 GiB with the
K1 case's 128 layer-0-shaped experts):
    cd /path/to/Inkling-2-Mini-vLLM && PYTHONPATH=. ~/inkling-venv/bin/python gpu_checks/check_kernel.py

CPU-only kernel check through Triton's interpreter (small shapes, float32 only, no GPU, no lock):
    cd /path/to/Inkling-2-Mini-vLLM && TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES="" PYTHONPATH=. ~/inkling-venv/bin/python \
        gpu_checks/check_kernel.py

Reference: dense weights from ``dequantize_nvfp4`` (rounded to bf16 unless stated) and a float64 matmul per expert.
The block table is built here by a local helper, independently of ``inkling_kernels``.

Ticket K1 (docs/inkling_speed_headroom.md section 7) adds ``check_k1_options`` (in-kernel gather, slot scatter,
row scale and ``round_out`` against ``TorchBackend``, exact structure, canaries around scattered rows, empty blocks)
and ``check_out_bit_identical`` (``out`` of the new forward ``torch.equal`` to a local copy of the pre-K1 forward).

Ticket K2 adds ``check_k2_act`` (the SwiGLU epilogue: ``a`` and ``dfac`` against ``TorchBackend``, float32 rel at most
1e-5, GPU bf16 ``a`` at most 2e-3 and ``dfac`` at most 5e-3; the gate/up column mapping with a weight whose gate rows
and up rows are distinct constants; nothing written through the dfac pointer without ``save_grads``) and a timing of
the gate_up forward with the epilogue. With K2 the forward's ``a`` comes from the kernel's tanh/sigmoid, so
``check_out_bit_identical`` accepts a rel difference (bf16 5e-3, the real-layer output bar; float32 1e-5) and reports
whether ``out`` is still bit-identical.
"""  # run as scripts, not a package

import os
import time
from typing import Any
from typing import cast

import torch

from inkling_qlora.inkling_nvfp4 import dequantize_nvfp4
from inkling_qlora.inkling_nvfp4 import e4m3_table
from inkling_qlora.inkling_nvfp4 import quantize_nvfp4
from inkling_qlora.inkling_triton import TritonBackend
from inkling_qlora.inkling_triton import nvfp4_grouped_gemm

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = torch.device("cpu" if INTERPRET else "cuda")
COUNTS = (0, 1, 63, 64, 65, 200, 0, 17)  # rows per expert, E = 8
MIN_FREE_MIB = 4096  # peaks around 0.6 GiB; must run beside a ~15 GB training job
INP_PAD = 3.0  # finite value in padding rows of inp
OUT_CANARY = -4096.0  # exact in bf16 and float32, far from any output value
TAIL = 8  # extra canary rows after the last row covered by the block table


def _gemm(*args: Any, **kwargs: Any) -> torch.Tensor:
    """Call ``nvfp4_grouped_gemm`` without ``save_grads``: it returns one tensor."""
    return cast("torch.Tensor", nvfp4_grouped_gemm(*args, **kwargs))


def _tensor(result: Any) -> torch.Tensor:
    """Narrow a backend ``gemm`` result to the tensor it is without ``save_grads``."""
    return cast("torch.Tensor", result)


def _pair(result: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Narrow a backend ``gemm`` result to ``(a, dfac)``, which is what ``save_grads=True`` returns."""
    return cast("tuple[torch.Tensor, torch.Tensor]", result)


def rel(x: torch.Tensor, ref: torch.Tensor) -> float:
    """Return the relative L2 error of ``x`` against ``ref``."""
    return ((x.double() - ref.double()).norm() / ref.double().norm()).item()


def block_table(counts, block_m: int, gap: int = 0):
    """(expert_of_block, block_row0, block_rows) int32, the first row of each expert, and the total row count.

    ``gap`` padding rows before every expert and at the end make stray writes visible as overwritten canaries.
    """
    eob, row0, nrows, starts = [], [], [], []
    r = 0
    for e, n in enumerate(counts):
        r += gap
        starts.append(r)
        for j in range(0, n, block_m):
            eob.append(e)
            row0.append(r + j)
            nrows.append(min(block_m, n - j))
        r += n
    r += gap

    def as_t(v):
        return torch.tensor(v, dtype=torch.int32, device=DEVICE)

    return as_t(eob), as_t(row0), as_t(nrows), starts, r


def make_weights(num_experts: int, rows: int, columns: int, seed: int):
    """Return random NVFP4 expert weights: codes, scales and per-tensor factors."""
    g = torch.Generator().manual_seed(seed)
    codes, scales, factors = [], [], []
    for _ in range(num_experts):
        w = torch.randn(rows, columns, generator=g) * (0.02 + 0.05 * torch.rand(rows, 1, generator=g))
        c, s, f = quantize_nvfp4(w.to(DEVICE))
        codes.append(c)
        scales.append(s)
        factors.append(f)
    return torch.stack(codes), torch.stack(scales), torch.stack(factors)


def reference(inp, codes, scale, factor, starts, transpose, round_bf16):  # noqa: PLR0913, PLR0917  # check helper, many inputs
    """float64 rows of every expert with rows, concatenated in expert order."""
    refs = []
    for e, n in enumerate(COUNTS):
        if n == 0:
            continue
        w = dequantize_nvfp4(codes[e], scale[e], factor[e])
        if round_bf16 and INTERPRET:
            # Triton 3.8's interpreter casts float32 -> bf16 by truncation (GPU: round to nearest even), so here the
            # reference truncates too; this still checks that the weight is rounded where the kernel should round it.
            w = (w.view(torch.int32) & -65536).view(torch.float32)
        elif round_bf16:
            w = w.to(torch.bfloat16)
        w = w.double()
        x = inp[starts[e] : starts[e] + n].double()
        refs.append(x @ w if transpose else x @ w.T)
    return torch.cat(refs)


def check_case(weights, transpose, in_dtype, out_dtype, round_bf16, sizes, label):  # noqa: PLR0913, PLR0917  # check helper, many inputs
    """Compare the Triton grouped GEMM with a float64 reference for one configuration."""
    codes, scale, factor = weights
    rows, columns = codes.shape[1], codes.shape[2] * 2
    k_dim, n_dim = (rows, columns) if transpose else (columns, rows)
    lut = e4m3_table(DEVICE)
    scale_u8 = scale.view(torch.uint8)
    g = torch.Generator(device=DEVICE).manual_seed(1)
    # round_out=False: the T3 cases compare the raw float32 accumulator (K1 made round_out=True the default, which
    # rounds bf16-input results to bf16 before widening to a float32 output)
    kw: dict[str, Any] = dict(out_dtype=out_dtype, round_weight_bf16=round_bf16, round_out=False, **sizes)

    # gapped layout with canaries: 3 padding rows before every expert, plus TAIL extra rows after the last one (past
    # row P of the table). Padding rows of inp hold a finite value, so a stray load produces a finite result that
    # would overwrite the distinct finite canary in out; canary rows are compared exactly.
    eob, row0, nrows, starts, total = block_table(COUNTS, sizes["block_m"], gap=3)
    total += TAIL
    inp = torch.full((total, k_dim), INP_PAD, device=DEVICE, dtype=in_dtype)
    valid = torch.zeros(total, dtype=torch.bool, device=DEVICE)
    for e, n in enumerate(COUNTS):
        inp[starts[e] : starts[e] + n] = torch.randn(n, k_dim, device=DEVICE, generator=g).to(in_dtype)
        valid[starts[e] : starts[e] + n] = True
    out = torch.full((total, n_dim), OUT_CANARY, device=DEVICE, dtype=out_dtype)
    _gemm(inp, codes, scale_u8, factor, lut, eob, row0, nrows, transpose, out=out, **kw)
    assert torch.equal(out[~valid], torch.full_like(out[~valid], OUT_CANARY)), f"{label}: padding rows were written"
    assert torch.equal(out[-TAIL:], torch.full_like(out[-TAIL:], OUT_CANARY)), f"{label}: rows past P were written"
    assert torch.isfinite(out[valid]).all(), f"{label}: valid rows not finite"
    assert not (out[valid] == OUT_CANARY).all(-1).any(), f"{label}: a valid row was not written"
    ref = reference(inp, codes, scale, factor, starts, transpose, round_bf16)
    got = out[valid]
    err = rel(got, ref)
    bar = 1e-5 if out_dtype == torch.float32 else 2e-3
    assert err <= bar, f"{label}: rel {err:.3e} > {bar}"

    # compact layout (out allocated by the function), run twice: bit-identical, and equal to the gapped run
    eob, row0, nrows, _, _ = block_table(COUNTS, sizes["block_m"], gap=0)
    inp_c = inp[valid].contiguous()
    o1 = _gemm(inp_c, codes, scale_u8, factor, lut, eob, row0, nrows, transpose, **kw)
    o2 = _gemm(inp_c, codes, scale_u8, factor, lut, eob, row0, nrows, transpose, **kw)
    assert torch.equal(o1, o2), f"{label}: two runs differ"
    assert torch.equal(o1, got), f"{label}: compact layout differs from the gapped layout"
    print(f"  {label}: rel {err:.2e} (bar {bar:.0e}), canaries intact, deterministic")


def check_zero_blocks(weights):
    """Check that the grouped GEMM on zero blocks returns an empty output without a launch."""
    codes, scale, factor = weights
    empty = torch.zeros(0, dtype=torch.int32, device=DEVICE)
    inp = torch.zeros(0, codes.shape[2] * 2, device=DEVICE)
    lut = e4m3_table(DEVICE)
    out = _gemm(inp, codes, scale.view(torch.uint8), factor, lut, empty, empty, empty, False)  # noqa: FBT003  # transpose is positional in the kernel API
    assert out.shape == (0, codes.shape[1])
    print("  zero blocks: empty output, no launch")


def check_backend_end_to_end(gu, dn, block_sizes=None):
    """TritonBackend vs TorchBackend through fused_experts: output, d(hidden), d(top_k_weights)."""
    from inkling_qlora.inkling_kernels import TorchBackend  # noqa: PLC0415  # heavy GPU dependency, lazy
    from inkling_qlora.inkling_kernels import fused_experts  # noqa: PLC0415  # heavy GPU dependency, lazy

    tri, ref = TritonBackend(*gu, *dn), TorchBackend(*gu, *dn)
    block_m = 64
    if block_sizes is not None:
        for key in tri.block_sizes:
            tri.block_sizes[key] = dict(block_sizes)
        block_m = block_sizes["block_m"]
    hidden_size = gu[0].shape[2] * 2
    tokens, k = (40, 4) if INTERPRET else (100, 4)
    g = torch.Generator(device=DEVICE).manual_seed(2)
    top_k_index = torch.randint(0, tri.num_experts + 1, (tokens, k), device=DEVICE, generator=g)  # incl. sentinel
    top_k_index[:, 0] = 1  # expert 1 gets more than block_m rows, so it spans several blocks
    top_k_index[top_k_index == 6] = 7  # expert 6 gets no rows
    for dtype in [torch.float32] if INTERPRET else [torch.float32, torch.bfloat16]:
        results = []
        for backend in (tri, ref):
            g = torch.Generator(device=DEVICE).manual_seed(3)
            hidden = torch.randn(tokens, hidden_size, device=DEVICE, generator=g).to(dtype).requires_grad_()
            weights = torch.rand(tokens, k, device=DEVICE, generator=g).requires_grad_()
            grad = torch.randn(tokens, hidden_size, device=DEVICE, generator=g).to(dtype)
            out = fused_experts(hidden, top_k_index, weights, backend, 4.0, 25.0, block_m=block_m)
            out.backward(grad)
            results.append((out.detach(), hidden.grad, weights.grad))
        bars = (1e-5, 1e-5, 1e-5) if dtype == torch.float32 else (5e-3, 1e-2, 1e-2)
        for name, a, b, bar in zip(("out", "d_hidden", "d_weights"), *results, bars, strict=True):
            err = rel(a, b)
            assert err <= bar, f"fused_experts {dtype} {name}: rel {err:.3e} > {bar}"
            print(f"  fused_experts {dtype} {name}: TritonBackend vs TorchBackend rel {err:.2e} (bar {bar:.0e})")


def _k1_routing(num_experts: int, tokens: int, k: int, seed: int):
    """Top-k indices with the sentinel, an expert without rows (6) and one spanning several blocks (1)."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    index = torch.randint(0, num_experts + 1, (tokens, k), device=DEVICE, generator=g)
    index[:, 0] = 1
    index[index == 6] = num_experts
    return index


def check_k1_options(gu, dn, block_sizes=None):  # noqa: PLR0915  # check script, one long flow
    """Ticket K1: each new ``gemm`` option on TritonBackend against TorchBackend, plus exact structure checks."""
    from inkling_qlora.inkling_kernels import TorchBackend  # noqa: PLC0415  # heavy GPU dependency, lazy
    from inkling_qlora.inkling_kernels import build_route_plan  # noqa: PLC0415  # heavy GPU dependency, lazy

    tri, ref = TritonBackend(*gu, *dn), TorchBackend(*gu, *dn)
    if block_sizes is not None:
        for key in tri.block_sizes:
            tri.block_sizes[key] = dict(block_sizes)
    block_m = tri.block_sizes[("gate_up", False)]["block_m"]
    tokens, k = (40, 4) if INTERPRET else (300, 4)
    plan = build_route_plan(_k1_routing(tri.num_experts, tokens, k, seed=4), tri.num_experts, block_m)
    tk, p = tokens * k, int(plan.counts.sum())
    num_real = int((plan.block_rows > 0).sum())
    assert num_real < plan.block_rows.numel(), "the table should end in empty blocks"
    print(
        f"  K1 plan: {tk} pairs ({tk - p} sentinel), "
        f"{num_real} real + {plan.block_rows.numel() - num_real} empty blocks"
    )
    dest = (2 * plan.slot + 1).to(torch.int32)  # scattered rows with a canary row between any two
    num_out = 2 * tk + 3
    written = torch.zeros(num_out, dtype=torch.bool, device=DEVICE)
    written[dest[:p].long()] = True
    i_size, h_size = dn[0].shape[2] * 2, gu[0].shape[2] * 2
    k_dims = {
        ("gate_up", False): h_size,
        ("gate_up", True): 2 * i_size,
        ("down", False): i_size,
        ("down", True): h_size,
    }
    # (input dtype, out_dtype, round_out); interpreter: float32 only (no bf16 dot, and it truncates float32 -> bf16)
    io = [(torch.float32, torch.float32, True)]
    if not INTERPRET:
        io += [
            (torch.bfloat16, torch.bfloat16, True),
            (torch.bfloat16, torch.float32, True),
            (torch.bfloat16, torch.float32, False),
        ]
    g = torch.Generator(device=DEVICE).manual_seed(5)
    scale = torch.empty(0, device=DEVICE)  # reassigned in the loop below (k_dims is never empty)
    for (proj, transpose), k_dim in k_dims.items():
        src = torch.randn(tokens, k_dim, device=DEVICE, generator=g)
        scale = torch.rand(tk, device=DEVICE, generator=g) + 0.5
        for in_dtype, out_dtype, round_out in io:
            x = src.to(in_dtype)
            gathered = x.index_select(0, plan.token_idx)
            kw: dict[str, Any] = {"out_dtype": out_dtype, "round_out": round_out}
            exact = out_dtype == torch.float32 and not (in_dtype == torch.bfloat16 and round_out)
            bar = (1e-6 if INTERPRET else 1e-5) if exact else 2e-3
            label = (
                f"{proj} T={int(transpose)} in={str(in_dtype)[6:]} out={str(out_dtype)[6:]} round_out={int(round_out)}"
            )
            cases: dict[str, tuple[torch.Tensor, dict[str, Any]]] = {
                "plain": (gathered, {}),
                "in_rows": (x, {"in_rows": plan.token_idx32}),
                "out_rows": (gathered, {"out_rows": dest, "num_out_rows": num_out}),
                "all": (x, {"in_rows": plan.token_idx32, "out_rows": dest, "num_out_rows": num_out}),
            }
            if out_dtype == torch.float32:
                cases["row_scale"] = (gathered, {"row_scale": scale})
                cases["all"][1]["row_scale"] = scale
            got, worst = {}, 0.0
            for name, (inp, opts) in cases.items():
                a = _tensor(tri.gemm(inp, proj, transpose, plan, **kw, **opts))
                b = _tensor(ref.gemm(inp, proj, transpose, plan, **kw, **opts))
                assert a.dtype == b.dtype == out_dtype, f"{label} {name}: dtype/shape"
                assert a.shape == b.shape, f"{label} {name}: dtype/shape"
                err = rel(a, b)
                assert err <= bar, f"{label} {name}: rel {err:.3e} > {bar}"
                got[name] = a
                worst = max(worst, err)
            plain = got["plain"]
            assert torch.equal(plain[p:], torch.zeros_like(plain[p:])), f"{label}: sentinel rows not zero"
            assert torch.equal(got["in_rows"], plain), f"{label}: in-kernel gather differs from a gathered copy"
            sc = got["out_rows"]
            assert torch.equal(sc[dest[:p].long()], plain[:p]), f"{label}: scattered rows differ"
            assert torch.equal(sc[~written], torch.zeros_like(sc[~written])), f"{label}: unwritten rows not zero"
            if "row_scale" in got:
                want = plain * scale[:, None]
                assert torch.equal(got["row_scale"], want), f"{label}: row_scale is not (rounded result) * scale"
                full = torch.zeros_like(sc)
                full[dest.long()] = want
                assert torch.equal(got["all"], full), f"{label}: gather + scale + scatter differs"
            # canaries around the scattered rows: a supplied ``out`` keeps every row no plan row writes
            codes, scale_u8, factor = tri.weights[proj]
            canary = torch.full((num_out, sc.shape[1]), OUT_CANARY, device=DEVICE, dtype=out_dtype)
            sizes = tri.block_sizes[(proj, transpose)]
            _gemm(
                x, codes, scale_u8, factor, tri.e4m3_lut, plan.expert_of_block, plan.block_row0, plan.block_rows,
                transpose, out=canary, in_rows=plan.token_idx32, out_rows=dest, num_out_rows=num_out,
                round_weight_bf16=in_dtype == torch.bfloat16, **kw, **sizes,
            )  # fmt: skip
            assert torch.equal(canary[~written], torch.full_like(canary[~written], OUT_CANARY)), f"{label}: canary hit"
            assert torch.equal(canary[written], sc[written]), f"{label}: canary run differs"
            print(
                f"  K1 {label}: {', '.join(got)} vs TorchBackend max rel {worst:.2e} (bar {bar:.0e}); "
                "exact, canaries intact"
            )

    # all-sentinel routing: every block of the static table is empty, nothing is decoded or stored
    empty = build_route_plan(torch.full((tokens, k), tri.num_experts, device=DEVICE), tri.num_experts, block_m)
    assert int(empty.block_rows.sum()) == 0
    codes, scale_u8, factor = tri.weights["down"]
    canary = torch.full((tk, h_size), OUT_CANARY, device=DEVICE)
    x = torch.randn(tk, i_size, device=DEVICE, generator=g)
    sizes = tri.block_sizes[("down", False)]
    _gemm(
        x, codes, scale_u8, factor, tri.e4m3_lut, empty.expert_of_block, empty.block_row0, empty.block_rows, False,  # noqa: FBT003  # transpose is positional in the kernel API
        out=canary, round_weight_bf16=False, **sizes,
    )  # fmt: skip
    assert torch.equal(canary, torch.full_like(canary, OUT_CANARY)), "an empty block stored"
    zeros = _tensor(tri.gemm(x, "down", False, empty, out_rows=empty.slot32, num_out_rows=tk, row_scale=scale))  # noqa: FBT003  # transpose is positional in the kernel API
    assert torch.equal(zeros, torch.zeros_like(zeros)), "all-sentinel output not zero"
    print("  K1 all-sentinel plan: no block stores, output exact zeros")


def _swiglu_ref(gu64: torch.Tensor, gate_cap: float = 4.0, up_cap: float = 25.0):
    """float64 (a, dfac interleaved) from float64 gu with gate = even, up = odd columns."""
    from inkling_qlora.inkling_kernels import SoftcappedSwiGLU  # noqa: PLC0415  # heavy GPU dependency, lazy
    from inkling_qlora.inkling_kernels import softcapped_swiglu_grads  # noqa: PLC0415  # heavy GPU dependency, lazy

    gate, up = gu64[:, 0::2], gu64[:, 1::2]
    a = SoftcappedSwiGLU(gate_cap, up_cap)(gate, up)
    dg, du = softcapped_swiglu_grads(gate, up, gate_cap, up_cap)
    return a, torch.stack((dg, du), -1).reshape(gu64.shape)


# Bar on the fraction of bf16 ``a`` elements more than one bf16 ulp from the reference. Kernel and TorchBackend sum
# gu in a different float32 order (relative difference ~1e-6), so a bf16 rounding of gu flips in roughly
# 2 * 1e-6 / 2**-8 ~ 5e-4 of the elements, and only some of those flips move ``a`` by more than one ulp: expected
# ~1e-4 (measured in the interpreter, with its less accurate tanh formula: 0.07% and 0%). Dropping the bf16 rounding
# of gu before the activation gives 20% and 32% (interpreter, see check_k2_bf16_rounding). The bar, 0.5%, is about 7x
# the measured value and 40x below the mutation.
ULP_FRAC_BAR = 5e-3


def frac_over_1ulp(x: torch.Tensor, ref: torch.Tensor) -> float:
    """Fraction of elements of ``x`` more than one bf16 ulp (of ``ref``'s magnitude) away from ``ref``."""
    r = ref.double()
    ulp = torch.exp2(torch.floor(torch.log2(r.abs().clamp_min(2.0**-126))) - 7)
    return float(((x.double() - r).abs() > ulp).double().mean())


def _round_bf16(t: torch.Tensor) -> torch.Tensor:
    """Round float32 to bf16 and back as the kernel casts.

    Round to nearest even on the GPU, truncation in Triton 3.8's interpreter.
    """
    t = t.float()
    if INTERPRET:
        return (t.contiguous().view(torch.int32) & -65536).view(torch.float32)
    return t.bfloat16().float()


def check_k2_bf16_rounding(gu, block_sizes=None):
    """Check ticket K2: the bf16 epilogue's rounding points, in units of bf16 ulps.

    The points are gu rounded to bf16 before the activation and ``a`` stored as bf16. Runs the kernel directly
    with float32 input holding bf16 values and bf16 rounding and output switched on, which the interpreter can
    execute (it has no bf16 dot); float32 products of bf16 values are exact, so this is the bf16 path up to the
    summation order. Reference: float64 GEMM with bf16 weights, then the kernel's casts (``_round_bf16``) around a
    float32 SoftcappedSwiGLU.
    """
    from inkling_qlora.inkling_kernels import SoftcappedSwiGLU  # noqa: PLC0415  # heavy GPU dependency, lazy
    from inkling_qlora.inkling_kernels import build_route_plan  # noqa: PLC0415  # heavy GPU dependency, lazy
    from inkling_qlora.inkling_triton import _nvfp4_grouped_gemm_kernel  # noqa: PLC0415  # heavy GPU dependency, lazy

    codes, scale, factor = gu
    num_experts, i2, h_size = codes.shape[0], codes.shape[1], codes.shape[2] * 2
    sizes = block_sizes or dict(TritonBackend(*gu, *gu).block_sizes[("gate_up", False)])
    tokens, k = (40, 4) if INTERPRET else (300, 4)
    plan = build_route_plan(_k1_routing(num_experts, tokens, k, seed=41), num_experts, sizes["block_m"])
    p = int(plan.counts.sum())
    g = torch.Generator(device=DEVICE).manual_seed(42)
    for mult in (1.0, 30.0):
        x = _round_bf16(torch.randn(tokens, h_size, device=DEVICE, generator=g) * mult)
        out = torch.zeros(tokens * k, i2 // 2, dtype=torch.bfloat16, device=DEVICE)
        dummy = plan.block_rows
        grid = (plan.block_rows.numel(), i2 // sizes["block_n"])
        _nvfp4_grouped_gemm_kernel[grid](  # pyright: ignore[reportIndexIssue]
            x, codes, scale.view(torch.uint8), factor, e4m3_table(DEVICE), out,
            plan.expert_of_block, plan.block_row0, plan.block_rows, plan.token_idx32, dummy, dummy, dummy, 4.0, 25.0,
            R=i2, C=h_size, BLOCK_M=sizes["block_m"], BLOCK_N=sizes["block_n"], BLOCK_K=sizes["block_k"],
            TRANSPOSE=False, OUT_BF16=True, IN_FP32=True, ROUND_W_BF16=True, GATHER_IN=True, SCATTER_OUT=False,
            HAS_ROW_SCALE=False, ROUND_OUT_BF16=True, ACT=True, SAVE_GRADS=False,
            num_warps=sizes["num_warps"], num_stages=sizes["num_stages"],
        )  # fmt: skip
        gu64 = torch.zeros(p, i2, dtype=torch.float64, device=DEVICE)
        start = 0
        for e, n in enumerate(plan.counts.tolist()):
            if n:
                w = _round_bf16(dequantize_nvfp4(codes[e], scale[e], factor[e])).double()
                gu64[start : start + n] = x[plan.token_idx[start : start + n]].double() @ w.T
                start += n
        gu_r = _round_bf16(gu64)
        a_r = _round_bf16(SoftcappedSwiGLU(4.0, 25.0)(gu_r[:, 0::2], gu_r[:, 1::2]))
        gu_u = gu64.float()  # the mutation: no rounding of gu
        a_u = _round_bf16(SoftcappedSwiGLU(4.0, 25.0)(gu_u[:, 0::2], gu_u[:, 1::2]))
        frac, frac_u = frac_over_1ulp(out[:p], a_r), frac_over_1ulp(a_u, a_r)
        assert torch.equal(out[p:], torch.zeros_like(out[p:])), "K2 bf16 rounding: sentinel rows of a not zero"
        assert frac <= ULP_FRAC_BAR, f"K2 bf16 rounding x{mult:g}: {frac:.3%} of a > 1 ulp off (bar {ULP_FRAC_BAR:.1%})"
        assert frac_u > ULP_FRAC_BAR, f"K2 bf16 rounding x{mult:g}: the check cannot see unrounded gu ({frac_u:.3%})"
        print(
            f"  K2 bf16 rounding points x{mult:g}: a > 1 bf16 ulp off in {frac:.3%} (bar {ULP_FRAC_BAR:.1%}); "
            f"reference without the gu rounding would give {frac_u:.2%}"
        )


def check_k2_act(gu, dn, block_sizes=None):  # noqa: C901, PLR0915  # check script, one long flow
    """Ticket K2: the gate_up gemm with the SwiGLU epilogue (``act``, ``save_grads``) against TorchBackend."""
    from inkling_qlora.inkling_kernels import TorchBackend  # noqa: PLC0415  # heavy GPU dependency, lazy
    from inkling_qlora.inkling_kernels import build_route_plan  # noqa: PLC0415  # heavy GPU dependency, lazy

    caps = (4.0, 25.0)
    tri, ref = TritonBackend(*gu, *dn), TorchBackend(*gu, *dn)
    if block_sizes is not None:
        for key in tri.block_sizes:
            tri.block_sizes[key] = dict(block_sizes)
    block_m = tri.block_sizes[("gate_up", False)]["block_m"]
    tokens, k = (40, 4) if INTERPRET else (300, 4)
    plan = build_route_plan(_k1_routing(tri.num_experts, tokens, k, seed=31), tri.num_experts, block_m)
    p = int(plan.counts.sum())
    h_size, i2 = gu[0].shape[2] * 2, gu[0].shape[1]
    g = torch.Generator(device=DEVICE).manual_seed(32)
    src = torch.randn(tokens, h_size, device=DEVICE, generator=g)
    # (dtype, input scale): the larger scale drives gu into both caps' saturation (|gate| >> 4, |up| near 25)
    cases = [(torch.float32, 1.0), (torch.float32, 30.0)]
    if not INTERPRET:
        cases += [(torch.bfloat16, 1.0), (torch.bfloat16, 30.0)]
    for dtype, mult in cases:
        x = (src * mult).to(dtype)
        table_before = plan.block_rows.clone()
        a_t, none_t = cast(
            "tuple[torch.Tensor, None]",
            tri.gemm(x, "gate_up", False, plan, in_rows=plan.token_idx32, act=caps),  # noqa: FBT003  # transpose is positional in the kernel API
        )
        assert none_t is None, "save_grads=False returned dfac"
        # the dfac pointer receives block_rows as a dummy when save_grads is False: it must stay untouched
        assert torch.equal(plan.block_rows, table_before), "something was written through the dfac pointer"
        a_t2, dfac_t = _pair(tri.gemm(x, "gate_up", False, plan, in_rows=plan.token_idx32, act=caps, save_grads=True))  # noqa: FBT003  # transpose is positional in the kernel API
        a_r, dfac_r = _pair(ref.gemm(x, "gate_up", False, plan, in_rows=plan.token_idx32, act=caps, save_grads=True))  # noqa: FBT003  # transpose is positional in the kernel API
        assert a_t.dtype == dtype
        assert a_t.shape == (tokens * k, i2 // 2)
        assert dfac_t.shape == (tokens * k, i2)
        assert torch.equal(a_t, a_t2), "a differs between save_grads=False and True"
        assert torch.equal(a_t[p:], torch.zeros_like(a_t[p:])), "sentinel rows of a not zero"
        assert torch.equal(dfac_t[p:], torch.zeros_like(dfac_t[p:])), "sentinel rows of dfac not zero"
        bar_a, bar_d = (1e-5, 1e-5) if dtype == torch.float32 else (2e-3, 5e-3)
        if dtype == torch.float32 and mult > 1:
            # saturated gate/up: 1 - tg^2 and 1 - s cancel, so one-ulp tanh/sigmoid differences between libdevice and
            # torch are magnified in dfac (GPU 2026-11-22: 1.27e-5); a is not affected
            bar_d = 1e-4
        err_a, err_d = rel(a_t, a_r), rel(dfac_t, dfac_r)
        err_g, err_u = rel(dfac_t[:, 0::2], dfac_r[:, 0::2]), rel(dfac_t[:, 1::2], dfac_r[:, 1::2])
        label = f"K2 act {str(dtype)[6:]} x{mult:g}"
        assert err_a <= bar_a, f"{label}: a rel {err_a:.3e} > {bar_a}"
        assert err_d <= bar_d, f"{label}: dfac rel {err_d:.3e} > {bar_d}"
        assert max(err_g, err_u) <= bar_d, f"{label}: dfac gate/up halves rel {err_g:.3e}/{err_u:.3e} > {bar_d}"
        gu_r = _tensor(ref.gemm(x, "gate_up", False, plan, in_rows=plan.token_idx32))  # noqa: FBT003  # transpose is positional in the kernel API
        sat = float((gu_r[:p, 0::2].float().abs() > 8).float().mean())
        ulp = ""
        if dtype == torch.bfloat16:  # the rel bars cannot see a dropped rounding point; one-ulp counts can
            frac_a, frac_d = frac_over_1ulp(a_t[:p], a_r[:p]), frac_over_1ulp(dfac_t[:p], dfac_r[:p])
            assert frac_a <= ULP_FRAC_BAR, f"{label}: {frac_a:.3%} of a > 1 bf16 ulp off (bar {ULP_FRAC_BAR:.1%})"
            ulp = f", > 1 ulp: a {frac_a:.3%} (bar {ULP_FRAC_BAR:.1%}), dfac {frac_d:.3%} (no bar)"
        print(
            f"  {label}: a rel {err_a:.2e} (bar {bar_a:.0e}), dfac rel {err_d:.2e} (gate {err_g:.2e}, up {err_u:.2e}; "
            f"bar {bar_d:.0e}), |gate|>8 in {sat:.0%}, sentinel rows zero, no dfac write without save_grads{ulp}"
        )

    # column mapping: every expert's gate rows hold one constant and its up rows another, so a, d/dgate and d/dup
    # each depend on which half is which. Reference from the dense dequantized weight, float64.
    from inkling_qlora.inkling_nvfp4 import quantize_nvfp4  # noqa: PLC0415  # heavy GPU dependency, lazy

    num_experts = tri.num_experts
    codes, scales, factors = [], [], []
    for e in range(num_experts):
        w = torch.empty(i2, h_size, device=DEVICE)
        w[0::2] = 0.010 * (e + 1)  # gate
        w[1::2] = -0.045 * (e + 2)  # up
        c, sc, f = quantize_nvfp4(w)
        codes.append(c)
        scales.append(sc)
        factors.append(f)
    const = (torch.stack(codes), torch.stack(scales), torch.stack(factors))
    tri_c = TritonBackend(*const, *dn)
    if block_sizes is not None:
        for key in tri_c.block_sizes:
            tri_c.block_sizes[key] = dict(block_sizes)
    rowsum = torch.linspace(-3.0, 3.0, tokens, device=DEVICE)
    x = (rowsum[:, None] / h_size).expand(tokens, h_size).contiguous() + 0.02 * src  # distinct per-row gate and up
    a_t, dfac_t = _pair(tri_c.gemm(x, "gate_up", False, plan, in_rows=plan.token_idx32, act=caps, save_grads=True))  # noqa: FBT003  # transpose is positional in the kernel API
    gu64 = torch.zeros(tokens * k, i2, dtype=torch.float64, device=DEVICE)
    expert_of_row = torch.repeat_interleave(torch.arange(num_experts, device=DEVICE), plan.counts)
    for e in range(num_experts):
        rows = (expert_of_row == e).nonzero().flatten()
        if rows.numel():
            w = dequantize_nvfp4(const[0][e], const[1][e], const[2][e]).double()
            gu64[rows] = x[plan.token_idx[rows]].double() @ w.T
    a_ref, dfac_ref = _swiglu_ref(gu64)
    a_swap, dfac_swap = _swiglu_ref(gu64.view(-1, i2 // 2, 2).flip(-1).reshape(gu64.shape))
    err_a, err_d = rel(a_t, a_ref), rel(dfac_t, dfac_ref)
    assert err_a <= 1e-5, f"K2 column mapping: a rel {err_a:.3e}, dfac rel {err_d:.3e} > 1e-5"
    assert err_d <= 1e-5, f"K2 column mapping: a rel {err_a:.3e}, dfac rel {err_d:.3e} > 1e-5"
    assert rel(a_t, a_swap) > 1e-2, "K2 column mapping test cannot see a swap"
    assert rel(dfac_t, dfac_swap) > 1e-2, "K2 column mapping test cannot see a swap"
    print(f"  K2 column mapping (distinct gate/up constants, float32): a rel {err_a:.2e}, dfac rel {err_d:.2e}")


def _pre_k1_gemm():
    """``nvfp4_grouped_gemm`` from the file named by ``K1_OLD_TRITON`` (the pre-K1 kernel), or None when unset."""
    path = os.environ.get("K1_OLD_TRITON")
    if not path:
        return None
    import importlib.util  # noqa: PLC0415  # heavy GPU dependency, lazy

    spec = importlib.util.spec_from_file_location("inkling_triton_pre_k1", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.nvfp4_grouped_gemm


def _old_forward_pre_k1(hidden, top_k_index, top_k_weights, tri, block_m):
    """Return a local copy of the pre-K1 ``_forward`` + ``_combine`` + plan.

    It uses bincount, repeat_interleave, a gathered copy, a rounded y, the weight multiply and the slot copy in torch.

    The GEMMs use the pre-K1 kernel when ``K1_OLD_TRITON`` names a saved copy of the pre-K1 ``inkling_triton.py``
    (loaded under another module name). Without it they use the new kernel in its plain mode (no gather, scatter or
    scale), so a regression shared by the plain mode of both sides would not show here.
    """
    old_gemm_fn = _pre_k1_gemm() or nvfp4_grouped_gemm
    num_experts, k = tri.num_experts, top_k_index.shape[1]
    d, f = hidden.dtype, torch.float32
    flat = top_k_index.reshape(-1).long()
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=num_experts + 1)[:num_experts]
    slot = order[: int(counts.sum())]
    token_idx = torch.div(slot, k, rounding_mode="floor")
    blocks_per_expert = torch.div(counts + block_m - 1, block_m, rounding_mode="floor")
    eob = torch.repeat_interleave(torch.arange(num_experts, device=hidden.device), blocks_per_expert)
    first_block = torch.cumsum(blocks_per_expert, 0) - blocks_per_expert
    first_row = torch.cumsum(counts, 0) - counts
    j = torch.arange(eob.numel(), device=hidden.device) - first_block[eob]
    table = (eob.int(), (first_row[eob] + j * block_m).int(), torch.clamp(counts[eob] - j * block_m, max=block_m).int())

    def gemm(x, proj):
        codes, scale_u8, factor = tri.weights[proj]
        sizes = tri.block_sizes[(proj, False)]
        bf16 = x.dtype == torch.bfloat16
        return cast(
            "torch.Tensor",
            old_gemm_fn(
                x.contiguous(),
                codes,
                scale_u8,
                factor,
                tri.e4m3_lut,
                *table,
                False,  # noqa: FBT003  # transpose is positional in the kernel API
                round_weight_bf16=bf16,
                **sizes,
            ),
        )

    xs = hidden.index_select(0, token_idx)
    gu = gemm(xs, "gate_up")
    g32, u32 = gu[:, 0::2].to(f), gu[:, 1::2].to(f)
    a = ((torch.tanh(g32 / 4.0) * 4.0) * torch.sigmoid(g32) * (torch.tanh(u32 / 25.0) * 25.0)).to(d)
    y = gemm(a, "down")
    ws = top_k_weights.reshape(-1).index_select(0, slot)
    contrib = (y.to(f) * ws[:, None]).to(f)
    buf = torch.zeros(hidden.shape[0] * k, hidden.shape[1], dtype=f, device=hidden.device)
    buf.index_copy_(0, slot, contrib)
    return buf.view(hidden.shape[0], k, hidden.shape[1]).sum(1).to(d)


def check_out_bit_identical(block_sizes=None):
    """Check ticket K1: the new forward's ``out`` equals the pre-K1 forward exactly.

    GPU: bf16, layer-0 shapes, 4,192 rows with sentinels; interpreter: float32, small shapes.
    K2 computes the SwiGLU in the kernel (a different tanh and sigmoid), so the bar is now rel <= 5e-3 (bf16) /
    1e-5 (float32); bit-identity is reported, not required.
    """
    from inkling_qlora.inkling_kernels import fused_experts  # noqa: PLC0415  # heavy GPU dependency, lazy

    if INTERPRET:
        num_experts, i_size, h_size, tokens, dtype = 8, 32, 64, 40, torch.float32
    else:
        num_experts, i_size, h_size, tokens, dtype = 128, 1152, 2304, 524, torch.bfloat16
    k = 8
    gu = make_weights(num_experts, 2 * i_size, h_size, seed=21)
    dn = make_weights(num_experts, h_size, i_size, seed=22)
    tri = TritonBackend(*gu, *dn)
    if block_sizes is not None:
        for key in tri.block_sizes:
            tri.block_sizes[key] = dict(block_sizes)
    block_m = tri.block_sizes[("gate_up", False)]["block_m"]
    g = torch.Generator(device=DEVICE).manual_seed(23)
    index = torch.rand(tokens, num_experts, device=DEVICE, generator=g).argsort(-1)[:, :k]  # distinct experts
    index[torch.rand(tokens, k, device=DEVICE, generator=g) < 0.05] = num_experts  # about 5% sentinel pairs
    hidden = torch.randn(tokens, h_size, device=DEVICE, generator=g).to(dtype)
    weights = torch.rand(tokens, k, device=DEVICE, generator=g)
    old = _old_forward_pre_k1(hidden, index, weights, tri, block_m)
    new = fused_experts(hidden, index, weights, tri, 4.0, 25.0, block_m=block_m)
    new_fn = fused_experts(hidden.clone().requires_grad_(), index, weights, tri, 4.0, 25.0, block_m=block_m)
    sentinels = int((index == num_experts).sum())
    bar = 1e-5 if dtype == torch.float32 else 5e-3
    err = rel(new, old)
    assert err <= bar, f"out vs the pre-K1 forward: rel {err:.3e} > {bar}"
    assert torch.equal(new_fn.detach(), new), "autograd-path out differs from the no-grad path"
    kernel = "pre-K1 kernel" if os.environ.get("K1_OLD_TRITON") else "new kernel, plain mode"
    same = "bit-identical" if torch.equal(new, old) else f"rel {err:.2e} (bar {bar:.0e}; K2's in-kernel SwiGLU)"
    print(f"  out vs the pre-K1 forward ({kernel}): {same}; {str(dtype)[6:]}, {tokens * k} rows ({sentinels} sentinel)")


def time_real_shapes():
    """Time per call at 6,000 rows over 128 experts, bf16 (record only, no gate). Random codes, valid scale bytes."""
    g = torch.Generator(device=DEVICE).manual_seed(6)
    counts = torch.bincount(torch.randint(0, 128, (6000,), device=DEVICE, generator=g), minlength=128).tolist()
    lut = e4m3_table(DEVICE)
    eob, row0, nrows, _, total = block_table(counts, 64)
    for name, rows, columns in (("gate_up", 2304, 2304), ("down", 2304, 1152)):
        codes = torch.randint(0, 256, (128, rows, columns // 2), device=DEVICE, dtype=torch.uint8, generator=g)
        scale = torch.randint(0x20, 0x40, (128, rows, columns // 16), device=DEVICE, dtype=torch.uint8, generator=g)
        factor = torch.full((128,), 1e-3, device=DEVICE)
        for transpose in (False, True):
            inp = torch.randn(total, rows if transpose else columns, device=DEVICE).to(torch.bfloat16)
            args = (inp, codes, scale, factor, lut, eob, row0, nrows, transpose)
            for _ in range(3):
                nvfp4_grouped_gemm(*args)
            torch.cuda.synchronize()
            reps = 20
            t0 = time.perf_counter()
            for _ in range(reps):
                nvfp4_grouped_gemm(*args)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / reps * 1e3
            tflops = 2 * total * rows * columns / ms / 1e9
            print(f"  {name} transpose={transpose}: {ms:.3f} ms per call ({tflops:.1f} TFLOP/s)")
            if name == "gate_up" and not transpose:
                # K2: the same call with the SwiGLU epilogue, without and with the derivative factors
                for save_grads in (False, True):
                    kw: dict[str, Any] = {"act": (4.0, 25.0), "save_grads": save_grads}
                    for _ in range(3):
                        nvfp4_grouped_gemm(*args, **kw)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(reps):
                        nvfp4_grouped_gemm(*args, **kw)
                    torch.cuda.synchronize()
                    ms = (time.perf_counter() - t0) / reps * 1e3
                    print(f"  {name} transpose=False act save_grads={int(save_grads)}: {ms:.3f} ms per call")
    print(f"  peak allocated {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")


def main() -> None:
    """Run every kernel check for the active mode (GPU or Triton interpreter)."""
    if not INTERPRET:
        torch.backends.cuda.matmul.allow_tf32 = False
    shapes = [(128, 64), (64, 128)] if INTERPRET else [(128, 64), (2304, 2304), (2304, 1152)]
    if INTERPRET:
        configs = [
            {"block_m": 16, "block_n": 64, "block_k": 32, "num_warps": 4, "num_stages": 1},
            {"block_m": 32, "block_n": 32, "block_k": 16, "num_warps": 4, "num_stages": 1},
        ]
    else:
        configs = [
            {"block_m": 64, "block_n": 64, "block_k": 32, "num_warps": 4, "num_stages": 3},
            {"block_m": 32, "block_n": 128, "block_k": 64, "num_warps": 8, "num_stages": 3},
        ]
    io = [(torch.float32, torch.float32, True), (torch.float32, torch.float32, False)]
    if not INTERPRET:  # bf16 comparisons are GPU-only (the interpreter has no bf16 dot)
        io += [(torch.bfloat16, torch.float32, True), (torch.bfloat16, torch.bfloat16, True)]
    print(f"device {DEVICE}{' (TRITON_INTERPRET=1)' if INTERPRET else ''}")
    for rows, columns in shapes:
        weights = make_weights(len(COUNTS), rows, columns, seed=rows * 7 + columns)
        for transpose in (False, True):
            for sizes in configs:
                if (columns if transpose else rows) % sizes["block_n"]:
                    continue
                for in_dtype, out_dtype, round_bf16 in io:
                    label = (
                        f"(R,C)=({rows},{columns}) T={int(transpose)} in={str(in_dtype)[6:]} out={str(out_dtype)[6:]} "
                        f"round_bf16={int(round_bf16)} bm{sizes['block_m']} bn{sizes['block_n']} bk{sizes['block_k']}"
                    )
                    check_case(weights, transpose, in_dtype, out_dtype, round_bf16, sizes, label)
        check_zero_blocks(weights)
        del weights
    # end to end: gate_up W (2I, H) and down W (H, I)
    i_size, h_size = (32, 64) if INTERPRET else (1152, 2304)
    gu = make_weights(len(COUNTS), 2 * i_size, h_size, seed=11)
    dn = make_weights(len(COUNTS), h_size, i_size, seed=12)
    small = {"block_m": 16, "block_n": 32, "block_k": 32, "num_warps": 4, "num_stages": 1} if INTERPRET else None
    check_backend_end_to_end(gu, dn, small)
    check_k1_options(gu, dn, small)
    check_k2_act(gu, dn, small)
    check_k2_bf16_rounding(gu, small)
    del gu, dn
    check_out_bit_identical(small)
    if not INTERPRET:
        time_real_shapes()
    print("all kernel checks passed")


if __name__ == "__main__":
    if INTERPRET:
        main()
    else:
        from gpu_checks.gpu_guard import gpu_session

        with gpu_session(min_free_mib=MIN_FREE_MIB):
            main()
