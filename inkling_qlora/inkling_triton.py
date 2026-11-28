"""Triton grouped GEMM over NVFP4 expert weights with in-register decode, and the ``TritonBackend``.

The backend serves ``inkling_kernels.fused_experts`` (docs/inkling_fused_experts_plan.md, sections 2.1 and 2.4).

Imported lazily, only on the CUDA fused path (it imports Triton). Under ``TRITON_INTERPRET=1`` it also accepts CPU
tensors, which is how the kernel is checked without a GPU.

For rows sorted by expert, with W[e] the expert's (R, C) weight in checkpoint row order:

- ``transpose=False``: ``out[p, r] = sum_c inp[p, c] * W[e(p), r, c]`` (``inp @ W.T``);
- ``transpose=True``: ``out[p, c] = sum_r inp[p, r] * W[e(p), r, c]`` (``inp @ W``).

Both read the same packed layout; no transposed or dense copy of the weights is made. Each output element is written
by exactly one program (no atomics, no stream-K), so results are deterministic.

Ticket K2 (docs/inkling_speed_headroom.md section 7): with ``act=(gate_cap, up_cap)`` the forward gate_up call ends in
the softcapped SwiGLU epilogue and stores ``a`` (and, with ``save_grads``, the derivative factors ``dfac``) instead
of ``gu``. ``tanh`` is libdevice's on the GPU; under ``TRITON_INTERPRET=1`` (chosen once at import) it is
``2 * sigmoid(2x) - 1``, which is less accurate for tiny arguments and only meant for CPU checks.
"""

import os
from typing import Any
from typing import no_type_check

import torch
import triton  # pyright: ignore[reportMissingImports]
import triton.language as tl  # pyright: ignore[reportMissingImports]
from triton.language.extra import libdevice  # pyright: ignore[reportMissingImports]

from inkling_qlora.inkling_kernels import check_gemm_options
from inkling_qlora.inkling_nvfp4 import NVFP4_BLOCK
from inkling_qlora.inkling_nvfp4 import e4m3_table

# Start values (section 2.4), the nvfp4_grouped_gemm defaults.
DEFAULT_BLOCK_SIZES = {"block_m": 64, "block_n": 64, "block_k": 32, "num_warps": 4, "num_stages": 3}
_PROJECTIONS = ("gate_up", "down")
CODES_NDIM = 3  # codes are (experts, rows, columns / 2)
INP_NDIM = 2  # inp is (rows, columns)
E4M3_TABLE_SIZE = 256  # one entry per e4m3 byte
# Tuned by gpu_checks/tune_blocks.py (T5, 2026-11-22, RTX 3090, layer 0, 6,000 rows over 128 experts, bf16). block_m
# stays 64, the routing plan's default block_m (BLOCK_M must be >= it; at 32 or 128 the per-example cost was higher).
TUNED_BLOCK_SIZES: dict[tuple[str, bool], dict[str, int]] = {
    ("gate_up", False): {"block_m": 64, "block_n": 128, "block_k": 64, "num_warps": 8, "num_stages": 3},  # 1.99 ms
    ("down", False): {"block_m": 64, "block_n": 128, "block_k": 64, "num_warps": 8, "num_stages": 3},  # 1.03 ms
    ("gate_up", True): {"block_m": 64, "block_n": 128, "block_k": 64, "num_warps": 8, "num_stages": 3},  # 1.93 ms
    ("down", True): {"block_m": 64, "block_n": 128, "block_k": 32, "num_warps": 4, "num_stages": 3},  # 0.96 ms
}


_INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"

if _INTERPRET:

    @triton.jit
    @no_type_check  # beartype must not wrap a Triton kernel (it compiles the function's source)
    def _tanh(x):
        return 2 * tl.sigmoid(2 * x) - 1

else:

    @triton.jit
    @no_type_check  # beartype must not wrap a Triton kernel (it compiles the function's source)
    def _tanh(x):
        return libdevice.tanh(x)


@triton.jit
@no_type_check  # beartype must not wrap a Triton kernel (it compiles the function's source)
def _decode_tile(b, sb, lut_ptr, factor, ROWS: tl.constexpr, COLS: tl.constexpr):  # noqa: N803, PLR0913, PLR0917  # Triton constexpr naming; kernel signature
    """Decode a [ROWS, COLS] weight tile in float32.

    ``b``: int32 [ROWS, COLS // 2] code bytes (even column in the low nibble); ``sb``: int32 [ROWS, COLS // 16] e4m3
    scale bytes. Computed as ``(e2m1 * scale) * factor``, the codec's order: ``e2m1 * scale`` is exact in float32, so
    the only rounding is the one by ``factor``, as in ``dequantize_nvfp4``.
    """
    nib = tl.interleave(b & 15, (b >> 4) & 15)  # [ROWS, COLS]: nib[:, 2j] = low nibble of byte j, nib[:, 2j+1] = high
    m = nib & 7
    ex = m >> 1
    fbit = m & 1
    mag2 = tl.where(ex == 0, fbit, (2 + fbit) << tl.maximum(ex - 1, 0))  # 2 * |value| in {0,1,2,3,4,6,8,12}
    sign = 1 - 2 * (nib >> 3)  # bit 3; negative zero gives -1 * 0 = 0
    val = (sign * mag2).to(tl.float32) * 0.5
    s = tl.load(lut_ptr + sb)  # [ROWS, COLS // 16] float32 scale values
    w = tl.reshape(val, (ROWS, COLS // 16, 16)) * s[:, :, None]
    return tl.reshape(w, (ROWS, COLS)) * factor


@triton.jit
@no_type_check  # beartype must not wrap a Triton kernel (it compiles the function's source)
def _nvfp4_grouped_gemm_kernel(  # noqa: C901, PLR0912, PLR0913, PLR0915, PLR0917  # Triton kernel: flat argument list and branches
    inp_ptr,
    codes_ptr,
    scale_ptr,
    factor_ptr,
    lut_ptr,
    out_ptr,
    expert_of_block_ptr,
    block_row0_ptr,
    block_rows_ptr,
    in_rows_ptr,
    out_rows_ptr,
    row_scale_ptr,
    dfac_ptr,
    gate_cap,
    up_cap,
    R: tl.constexpr,  # noqa: N803  # Triton constexpr naming
    C: tl.constexpr,  # noqa: N803  # Triton constexpr naming
    BLOCK_M: tl.constexpr,  # noqa: N803  # Triton constexpr naming
    BLOCK_N: tl.constexpr,  # noqa: N803  # Triton constexpr naming
    BLOCK_K: tl.constexpr,  # noqa: N803  # Triton constexpr naming
    TRANSPOSE: tl.constexpr,  # noqa: N803  # Triton constexpr naming
    OUT_BF16: tl.constexpr,  # noqa: N803  # Triton constexpr naming
    IN_FP32: tl.constexpr,  # noqa: N803  # Triton constexpr naming
    ROUND_W_BF16: tl.constexpr,  # noqa: N803  # Triton constexpr naming
    GATHER_IN: tl.constexpr,  # noqa: N803  # Triton constexpr naming
    SCATTER_OUT: tl.constexpr,  # noqa: N803  # Triton constexpr naming
    HAS_ROW_SCALE: tl.constexpr,  # noqa: N803  # Triton constexpr naming
    ROUND_OUT_BF16: tl.constexpr,  # noqa: N803  # Triton constexpr naming
    ACT: tl.constexpr,  # noqa: N803  # Triton constexpr naming
    SAVE_GRADS: tl.constexpr,  # noqa: N803  # Triton constexpr naming
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    nrows = tl.load(block_rows_ptr + pid_b)
    if nrows > 0:  # empty blocks of the static table do no decode and store nothing
        e = tl.load(expert_of_block_ptr + pid_b).to(tl.int64)
        row0 = tl.load(block_row0_ptr + pid_b).to(tl.int64)
        offs_m = tl.arange(0, BLOCK_M)
        mask_m = offs_m < nrows
        rows = row0 + offs_m  # plan rows, int64
        if GATHER_IN:  # plan row p reads input row in_rows[p]; padded rows of a partial block read row 0, masked  # noqa: SIM108  # Triton JIT: keep explicit if/else
            in_r = tl.load(in_rows_ptr + rows, mask=mask_m, other=0).to(tl.int64)
        else:
            in_r = rows
        if SCATTER_OUT:  # plan row p writes output row out_rows[p]  # noqa: SIM108  # Triton JIT: keep explicit if/else
            out_r = tl.load(out_rows_ptr + rows, mask=mask_m, other=0).to(tl.int64)
        else:
            out_r = rows
        factor = tl.load(factor_ptr + e)
        codes_base = codes_ptr + e * (R * (C // 2))
        scale_base = scale_ptr + e * (R * (C // 16))
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        if TRANSPOSE:
            # reduce over rows r; tile [BLOCK_K (r), BLOCK_N (c)], scale of (r, c // 16)
            cb = pid_n * (BLOCK_N // 2) + tl.arange(0, BLOCK_N // 2)
            cs = pid_n * (BLOCK_N // 16) + tl.arange(0, BLOCK_N // 16)
            for k0 in range(0, R, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)
                a = tl.load(inp_ptr + in_r[:, None] * R + offs_k[None, :], mask=mask_m[:, None], other=0.0)
                b = tl.load(codes_base + offs_k[:, None] * (C // 2) + cb[None, :]).to(tl.int32)
                sb = tl.load(scale_base + offs_k[:, None] * (C // 16) + cs[None, :]).to(tl.int32)
                w = _decode_tile(b, sb, lut_ptr, factor, BLOCK_K, BLOCK_N)
                if ROUND_W_BF16:
                    w = w.to(tl.bfloat16)
                if IN_FP32:
                    acc = tl.dot(a, w.to(tl.float32), acc, input_precision="ieee")
                else:
                    acc = tl.dot(a, w.to(tl.bfloat16), acc)
            out_offs = out_r[:, None] * C + offs_n[None, :]
        else:
            # reduce over columns c; tile [BLOCK_N (r), BLOCK_K (c)], transposed in registers before the dot
            for k0 in range(0, C, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)
                cb = k0 // 2 + tl.arange(0, BLOCK_K // 2)
                cs = k0 // 16 + tl.arange(0, BLOCK_K // 16)
                a = tl.load(inp_ptr + in_r[:, None] * C + offs_k[None, :], mask=mask_m[:, None], other=0.0)
                b = tl.load(codes_base + offs_n[:, None] * (C // 2) + cb[None, :]).to(tl.int32)
                sb = tl.load(scale_base + offs_n[:, None] * (C // 16) + cs[None, :]).to(tl.int32)
                w = _decode_tile(b, sb, lut_ptr, factor, BLOCK_N, BLOCK_K)
                if ROUND_W_BF16:
                    w = w.to(tl.bfloat16)
                if IN_FP32:
                    acc = tl.dot(a, tl.trans(w.to(tl.float32)), acc, input_precision="ieee")
                else:
                    acc = tl.dot(a, tl.trans(w.to(tl.bfloat16)), acc)
            out_offs = out_r[:, None] * R + offs_n[None, :]

        # epilogue, the old glue's order: round to bf16 -> widen -> times the row's scale (float32) -> output dtype
        res = acc
        if ROUND_OUT_BF16:
            res = res.to(tl.bfloat16).to(tl.float32)
        if ACT:
            # K2: res is gu (rounded to bf16 for bf16 input, the old rounding point) with gate/up interleaved along the
            # tile's BLOCK_N columns (even = gate). SwiGLU and its derivatives in float32, as softcapped_swiglu_grads.
            gate, up = tl.split(tl.reshape(res, (BLOCK_M, BLOCK_N // 2, 2)))
            tg = _tanh(gate / gate_cap)
            s = tl.sigmoid(gate)  # the raw gate; the caps apply to the tanh factors only
            tu = _tanh(up / up_cap)
            act = (tg * gate_cap) * s * (tu * up_cap)  # the order of _softcapped_swiglu
            offs_a = pid_n * (BLOCK_N // 2) + tl.arange(0, BLOCK_N // 2)
            a_offs = out_r[:, None] * (R // 2) + offs_a[None, :]
            if OUT_BF16:
                act = act.to(tl.bfloat16)
            tl.store(out_ptr + a_offs, act, mask=mask_m[:, None])
            if SAVE_GRADS:
                dg = up_cap * tu * ((1 - tg * tg) * s + gate_cap * tg * s * (1 - s))
                du = gate_cap * tg * s * (1 - tu * tu)
                dfac = tl.interleave(dg, du)  # [BLOCK_M, BLOCK_N], the tile's own gu columns
                if OUT_BF16:
                    dfac = dfac.to(tl.bfloat16)
                tl.store(dfac_ptr + out_offs, dfac, mask=mask_m[:, None])
        else:
            if HAS_ROW_SCALE:
                rs = tl.load(row_scale_ptr + rows, mask=mask_m, other=0.0)
                res = res * rs[:, None]
            if OUT_BF16:
                res = res.to(tl.bfloat16)
            tl.store(out_ptr + out_offs, res, mask=mask_m[:, None])


def _check_device(t: torch.Tensor, name: str, device: torch.device) -> None:
    if t.device != device:
        msg = f"{name} is on {t.device}, expected {device}"
        raise ValueError(msg)
    if not t.is_contiguous():
        msg = f"{name} must be contiguous"
        raise ValueError(msg)


def nvfp4_grouped_gemm(  # noqa: C901, PLR0912, PLR0913, PLR0915, PLR0917  # validation and launch of one kernel
    inp: torch.Tensor,
    codes: torch.Tensor,
    scale_u8: torch.Tensor,
    factor: torch.Tensor,
    e4m3_lut: torch.Tensor,
    expert_of_block: torch.Tensor,
    block_row0: torch.Tensor,
    block_rows: torch.Tensor,
    transpose: bool,  # noqa: FBT001  # callers pass it positionally
    out_dtype: torch.dtype | None = None,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 32,
    num_warps: int = 4,
    num_stages: int = 3,
    *,
    round_weight_bf16: bool = True,
    out: torch.Tensor | None = None,
    in_rows: torch.Tensor | None = None,
    out_rows: torch.Tensor | None = None,
    num_out_rows: int | None = None,
    row_scale: torch.Tensor | None = None,
    round_out: bool = True,
    act: tuple[float, float] | None = None,
    save_grads: bool = False,
    dfac: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
    """Multiply the plan rows by the NVFP4 weight of each row's expert (see the module docstring).

    ``inp`` (rows, K_dim) contiguous bf16 or float32; ``codes`` uint8 (E, R, C/2); ``scale_u8`` uint8 (E, R, C/16),
    the e4m3 bytes; ``factor`` float32 (E,); ``e4m3_lut`` float32 (256,) from ``e4m3_table``; the block table (int32)
    must have been built with blocks of at most ``block_m`` rows (blocks with 0 rows are skipped). Output rows no block
    writes are zeros when ``out`` is None (allocated with ``torch.zeros``) and left untouched in a supplied ``out``.

    The plan has P rows: ``in_rows.numel()`` when ``in_rows`` is given, else ``inp.shape[0]``.

    - ``in_rows`` int32 (P,): plan row p reads ``inp[in_rows[p]]`` (indices are not range-checked: no host sync).
    - ``out_rows`` int32 (P,) with ``num_out_rows``: plan row p is written to row ``out_rows[p]`` of a
      (``num_out_rows``, N_dim) output; the covered entries must be distinct (each output row has one writer).
    - ``row_scale`` float32 (P,): row p's result is multiplied by ``row_scale[p]`` in float32, after the rounding by
      ``round_out``. Requires ``out_dtype=torch.float32``.
    - ``round_out``: for bf16 input, the float32 accumulator is rounded to bf16 first, then widened or scaled. False
      keeps the float32 accumulator. Ignored for float32 input.

    The decoded weight is rounded to bf16 before the multiply when ``round_weight_bf16`` (the plan's kernel, matching
    the bf16 reference); with float32 input and ``round_weight_bf16=False`` it stays float32, matching
    ``TorchBackend`` for float32 input. Accumulation is float32; float32 dots use ``input_precision="ieee"``.
    ``out``: optional preallocated (P or ``num_out_rows``, N_dim) contiguous output of ``out_dtype``.

    ``act=(gate_cap, up_cap)`` (K2; ``transpose=False`` only, N_dim = 2I with gate/up rows interleaved, no
    ``row_scale``, ``out_dtype`` equal to the input dtype): the SwiGLU epilogue. The accumulator is rounded to bf16
    for bf16 input (the old rounding of ``gu``), then ``a = gate_cap*tanh(gate/gate_cap) * sigmoid(gate) *
    up_cap*tanh(up/up_cap)`` in float32 with gate = even, up = odd columns, stored to ``out`` (rows, I). With
    ``save_grads``, (d a / d gate, d a / d up) interleaved go to ``dfac`` (rows, 2I) in ``out_dtype`` (allocated with
    ``torch.zeros`` when not supplied); without it nothing is written through the dfac pointer. Returns
    ``(out, dfac or None)``.
    """
    if inp.dtype not in (torch.bfloat16, torch.float32):
        msg = f"inp must be bfloat16 or float32, got {inp.dtype}"
        raise TypeError(msg)
    out_dtype = inp.dtype if out_dtype is None else out_dtype
    if out_dtype not in (torch.bfloat16, torch.float32):
        msg = f"out_dtype must be bfloat16 or float32, got {out_dtype}"
        raise TypeError(msg)
    device = inp.device
    if device.type != "cuda" and os.environ.get("TRITON_INTERPRET") != "1":
        msg = f"inp must be on CUDA (or run with TRITON_INTERPRET=1), got {device}"
        raise ValueError(msg)
    if codes.dtype != torch.uint8 or codes.dim() != CODES_NDIM:
        msg = f"codes must be uint8 (E, R, C/2), got {codes.dtype} {tuple(codes.shape)}"
        raise TypeError(msg)
    num_experts, rows, columns = codes.shape[0], codes.shape[1], codes.shape[2] * 2
    if scale_u8.dtype != torch.uint8 or tuple(scale_u8.shape) != (num_experts, rows, columns // NVFP4_BLOCK):
        msg = (
            f"scale_u8 must be uint8 {(num_experts, rows, columns // NVFP4_BLOCK)}, got {scale_u8.dtype} "
            f"{tuple(scale_u8.shape)}"
        )
        raise TypeError(msg)
    if factor.dtype != torch.float32 or factor.numel() != num_experts:
        msg = f"factor must be float32 with {num_experts} entries, got {factor.dtype} {tuple(factor.shape)}"
        raise TypeError(msg)
    if e4m3_lut.dtype != torch.float32 or e4m3_lut.numel() != E4M3_TABLE_SIZE:
        msg = "e4m3_lut must be the float32 (256,) e4m3 table"
        raise TypeError(msg)
    for name, t in (("expert_of_block", expert_of_block), ("block_row0", block_row0), ("block_rows", block_rows)):
        if t.dtype != torch.int32:
            msg = f"{name} must be int32, got {t.dtype}"
            raise TypeError(msg)
    tensors = {
        "inp": inp,
        "codes": codes,
        "scale_u8": scale_u8,
        "factor": factor,
        "e4m3_lut": e4m3_lut,
        "expert_of_block": expert_of_block,
        "block_row0": block_row0,
        "block_rows": block_rows,
    }
    for name, t in tensors.items():
        _check_device(t, name, device)
    if inp.dim() != INP_NDIM:
        msg = f"inp must be 2-D, got {tuple(inp.shape)}"
        raise ValueError(msg)

    assert columns % NVFP4_BLOCK == 0, f"C ({columns}) must be divisible by {NVFP4_BLOCK}"  # noqa: S101  # invariant check on tile sizes
    k_dim, n_dim = (rows, columns) if transpose else (columns, rows)
    if inp.shape[1] != k_dim:
        msg = f"inp has {inp.shape[1]} columns, expected {k_dim} (transpose={transpose})"
        raise ValueError(msg)
    assert n_dim % block_n == 0, f"N_dim ({n_dim}) must be divisible by block_n ({block_n})"  # noqa: S101  # invariant check on tile sizes
    if transpose:
        assert block_n % NVFP4_BLOCK == 0, f"block_n ({block_n}) must be divisible by 16 for transpose=True"  # noqa: S101  # invariant check on tile sizes
        assert rows % block_k == 0, f"R ({rows}) must be divisible by block_k ({block_k}) for transpose=True"  # noqa: S101  # invariant check on tile sizes
    else:
        assert block_k % NVFP4_BLOCK == 0, f"block_k ({block_k}) must be divisible by 16 for transpose=False"  # noqa: S101  # invariant check on tile sizes
        assert columns % block_k == 0, f"C ({columns}) must be divisible by block_k ({block_k}) for transpose=False"  # noqa: S101  # invariant check on tile sizes

    if act is None:
        if save_grads or dfac is not None:
            msg = "save_grads and dfac need act"
            raise ValueError(msg)
    else:
        if transpose:
            msg = "act is only valid with transpose=False"
            raise ValueError(msg)
        if row_scale is not None:
            msg = "act does not combine with row_scale"
            raise ValueError(msg)
        if out_dtype != inp.dtype:
            msg = f"act needs out_dtype equal to the input dtype ({inp.dtype}), got {out_dtype}"
            raise TypeError(msg)
        if not round_out:
            msg = "act rounds the accumulator to the input dtype first; round_out=False is not supported"
            raise ValueError(msg)
        if block_n % 2:
            msg = f"act needs an even block_n (gate/up pairs), got {block_n}"
            raise ValueError(msg)
        if dfac is not None and not save_grads:
            msg = "dfac needs save_grads"
            raise ValueError(msg)
    out_cols = n_dim // 2 if act is not None else n_dim

    num_plan_rows = inp.shape[0] if in_rows is None else in_rows.numel()
    options = (("in_rows", in_rows, torch.int32), ("out_rows", out_rows, torch.int32))
    for name, t, dtype in (*options, ("row_scale", row_scale, torch.float32)):
        if t is None:
            continue
        if t.dtype != dtype or t.dim() != 1 or t.numel() != num_plan_rows:
            msg = f"{name} must be {dtype} ({num_plan_rows},), got {t.dtype} {tuple(t.shape)}"
            raise ValueError(msg)
        _check_device(t, name, device)
    if out_rows is None:
        if num_out_rows is not None:
            msg = "num_out_rows needs out_rows"
            raise ValueError(msg)
        num_rows = num_plan_rows
    else:
        if num_out_rows is None:
            msg = "out_rows needs num_out_rows"
            raise ValueError(msg)
        num_rows = num_out_rows
    if row_scale is not None and out_dtype != torch.float32:
        msg = "row_scale needs out_dtype=torch.float32"
        raise TypeError(msg)
    if out is None:
        out = torch.zeros(num_rows, out_cols, dtype=out_dtype, device=device)
    else:
        if out.dtype != out_dtype or tuple(out.shape) != (num_rows, out_cols):
            msg = f"out must be {out_dtype} {(num_rows, out_cols)}, got {out.dtype} {tuple(out.shape)}"
            raise ValueError(msg)
        _check_device(out, "out", device)
    if save_grads:
        if dfac is None:
            dfac = torch.zeros(num_rows, n_dim, dtype=out_dtype, device=device)
        else:
            if dfac.dtype != out_dtype or tuple(dfac.shape) != (num_rows, n_dim):
                msg = f"dfac must be {out_dtype} {(num_rows, n_dim)}, got {dfac.dtype} {tuple(dfac.shape)}"
                raise ValueError(msg)
            _check_device(dfac, "dfac", device)

    def result():
        return out if act is None else (out, dfac)

    num_blocks = expert_of_block.numel()
    if num_blocks == 0 or out.numel() == 0:
        return result()
    grid = (num_blocks, n_dim // block_n)
    # the act path rounds gu to bf16 before the activation (out_dtype is then bf16 too)
    widen = bool(round_out) and out_dtype == torch.float32
    round_acc_bf16 = inp.dtype == torch.bfloat16 and (act is not None or widen)
    dummy = block_rows  # stands in for unused pointer arguments; never read
    _nvfp4_grouped_gemm_kernel[grid](  # pyright: ignore[reportIndexIssue]  # triton's JIT launcher
        inp,
        codes,
        scale_u8,
        factor,
        e4m3_lut,
        out,
        expert_of_block,
        block_row0,
        block_rows,
        dummy if in_rows is None else in_rows,
        dummy if out_rows is None else out_rows,
        dummy if row_scale is None else row_scale,
        dummy if dfac is None else dfac,
        float(act[0]) if act is not None else 1.0,
        float(act[1]) if act is not None else 1.0,
        R=rows,
        C=columns,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        TRANSPOSE=bool(transpose),
        OUT_BF16=out_dtype == torch.bfloat16,
        IN_FP32=inp.dtype == torch.float32,
        ROUND_W_BF16=bool(round_weight_bf16) or inp.dtype == torch.bfloat16,
        GATHER_IN=in_rows is not None,
        SCATTER_OUT=out_rows is not None,
        HAS_ROW_SCALE=row_scale is not None,
        ROUND_OUT_BF16=round_acc_bf16,
        ACT=act is not None,
        SAVE_GRADS=bool(save_grads),
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return result()


class TritonBackend:
    """``inkling_kernels`` backend on the Triton kernel; same constructor and ``gemm`` as ``TorchBackend``.

    Holds references to the module's buffers (codes are never copied; the scales are viewed as uint8). Supports
    bfloat16 input (weight rounded to bf16, float32 accumulation, bf16 output) and float32 input (float32 weight,
    ieee float32 dots, float32 output), the same semantics as ``TorchBackend`` for those dtypes.
    """

    def __init__(  # noqa: PLR0913, PLR0917  # three tensors per projection, passed positionally
        self,
        gate_up_codes: torch.Tensor,
        gate_up_scale: torch.Tensor,
        gate_up_factor: torch.Tensor,
        down_codes: torch.Tensor,
        down_scale: torch.Tensor,
        down_factor: torch.Tensor,
    ) -> None:
        """Wrap the packed NVFP4 codes, block scales and factors of both projections (no copies)."""
        self.weights: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for proj, codes, scale, factor in (
            ("gate_up", gate_up_codes, gate_up_scale, gate_up_factor),
            ("down", down_codes, down_scale, down_factor),
        ):
            if not codes.is_contiguous() or not scale.is_contiguous():
                msg = f"{proj} codes and scale must be contiguous (they are not copied)"
                raise ValueError(msg)
            scale_u8 = scale if scale.dtype == torch.uint8 else scale.view(torch.uint8)
            factor_f32 = factor.to(device=codes.device, dtype=torch.float32).reshape(-1).contiguous()  # (E,), tiny
            self.weights[proj] = (codes, scale_u8, factor_f32)
        self.num_experts = gate_up_codes.shape[0]
        self.e4m3_lut = e4m3_table(device=gate_up_codes.device)
        self.block_sizes: dict[tuple[str, bool], dict[str, int]] = {
            (proj, transpose): dict(TUNED_BLOCK_SIZES[(proj, transpose)])
            for proj in _PROJECTIONS
            for transpose in (False, True)
        }

    def gemm(  # noqa: PLR0913  # keyword options of the backend protocol
        self,
        inp: torch.Tensor,
        proj: str,
        transpose: bool,  # noqa: FBT001  # callers pass it positionally
        plan,
        *,
        in_rows: torch.Tensor | None = None,
        out_rows: torch.Tensor | None = None,
        num_out_rows: int | None = None,
        row_scale: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
        round_out: bool = True,
        act: tuple[float, float] | None = None,
        save_grads: bool = False,
    ):
        """Implement the backend protocol of ``inkling_kernels`` (keyword options in its module docstring).

        No host sync.
        """
        if inp.dtype not in (torch.bfloat16, torch.float32):
            msg = f"TritonBackend supports bfloat16 and float32 input, got {inp.dtype}"
            raise TypeError(msg)
        out_dtype, _ = check_gemm_options(
            inp,
            plan,
            in_rows,
            out_rows,
            num_out_rows,
            row_scale,
            out_dtype,
            act,
            save_grads=save_grads,
            proj=proj,
            transpose=transpose,
            round_out=round_out,
        )
        codes, scale_u8, factor = self.weights[proj]
        sizes: dict[str, Any] = self.block_sizes[(proj, bool(transpose))]
        if plan.block_m > sizes["block_m"]:
            msg = (
                f"the plan was built with block_m {plan.block_m} but the kernel's block_m is {sizes['block_m']}; "
                "build the plan with block_m <= the backend's block_m"
            )
            raise ValueError(msg)
        return nvfp4_grouped_gemm(
            inp.contiguous(),
            codes,
            scale_u8,
            factor,
            self.e4m3_lut,
            plan.expert_of_block,
            plan.block_row0,
            plan.block_rows,
            transpose,
            out_dtype=out_dtype,
            round_weight_bf16=inp.dtype == torch.bfloat16,
            in_rows=in_rows,
            out_rows=out_rows,
            num_out_rows=num_out_rows,
            row_scale=row_scale,
            round_out=round_out,
            act=act,
            save_grads=save_grads,
            **sizes,
        )
