"""Fused frozen-expert MoE for Inkling 2 Mini: routing plan, softcapped SwiGLU, autograd Function, torch backend.

Imports only torch and the NVFP4 codec (no Triton, no transformers), so it runs on CPU for tests. The GEMMs go through
a *backend*; :class:`TorchBackend` is the slow reference, ``inkling_triton.TritonBackend`` the CUDA kernel.

Backend protocol (any object with):

- ``num_experts: int``;
- ``gemm(inp, proj, transpose, plan, *, in_rows=None, out_rows=None, num_out_rows=None, row_scale=None,
  out_dtype=None, round_out=True, act=None, save_grads=False) -> Tensor | (Tensor, Tensor | None)`` with
  ``proj in {"gate_up", "down"}``. The plan has R =
  ``plan.slot.numel()`` (= T*K) rows in plan order (sorted by expert, ``plan.counts[e]`` rows per expert, then the
  rows of dropped/sentinel pairs, which no block covers). With W the expert's weight in checkpoint row order,
  ``transpose=False`` computes ``x @ W.T`` and ``transpose=True`` computes ``x @ W`` for every covered plan row x.
  gate_up W is (2I, H) with gate/up rows interleaved (even = gate), down W is (H, I).

  - ``in_rows`` int32 (R,): plan row p reads ``inp[in_rows[p]]``; None: ``inp[p]`` (``inp`` then has R rows).
  - ``out_rows`` int32 (R,) with ``num_out_rows``: plan row p is written to row ``out_rows[p]`` of a
    (``num_out_rows``, N) output; None: row p of an (R, N) output.
  - ``row_scale`` (R,) in the output dtype (float32 for half input): row p's result is multiplied by ``row_scale[p]``
    after the ``round_out`` rounding. Requires a float32/float64 output.
  - ``out_dtype``: None means ``inp.dtype``; float32 is allowed for half input.
  - ``round_out``: for half input, the float32 accumulator is first rounded to the input dtype (the old rounding
    point), then widened or scaled; False keeps the float32 accumulator. Ignored for float32/float64 input.
  - The output is allocated with ``torch.zeros``: rows no covered plan row writes are exact zeros.
  - ``act=(gate_cap, up_cap)`` (only with ``proj="gate_up"``, ``transpose=False``, no ``out_rows``/``row_scale``, the
    input's output dtype and ``round_out=True``): the softcapped SwiGLU epilogue. The (R, 2I) ``gu`` is rounded to the
    input dtype as before, then ``a = SoftcappedSwiGLU(gate=gu[:, 0::2], up=gu[:, 1::2])`` (float32 for half input)
    and the call returns ``(a, dfac)``: ``a`` (R, I) in the input dtype and, when ``save_grads``, ``dfac`` (R, 2I) in
    the input dtype holding (d act / d gate, d act / d up) interleaved (even column = gate), else None. ``gu`` itself
    is not returned. ``save_grads`` without ``act`` is an error.

Notation as in docs/inkling_fused_experts_plan.md section 2.3: d = dtype of the hidden states, f = float32 when d is
bf16/fp16 (else d). Ticket K1 (docs/inkling_speed_headroom.md section 7) moved the gather, the routing-weight multiply
and the slot scatter into ``gemm`` and made the plan free of host syncs; ticket K2 moved the SwiGLU and its derivative
factors into the gate_up ``gemm`` (``act``, ``save_grads``).
"""

from dataclasses import dataclass
from typing import Any
from typing import Protocol
from typing import no_type_check
from typing import runtime_checkable

import torch
from torch import nn
from torch.autograd.function import once_differentiable

from inkling_qlora.inkling_nvfp4 import dequantize_nvfp4

_HALF = (torch.bfloat16, torch.float16)
ACT_CAPS = 2  # ``act`` is (gate_cap, up_cap)


def _compute_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float32 if dtype in _HALF else dtype


def _softcapped_swiglu(gate: torch.Tensor, up: torch.Tensor, gate_cap: float, up_cap: float) -> torch.Tensor:
    dtype = gate.dtype
    compute = _compute_dtype(dtype)
    gate, up = gate.to(compute), up.to(compute)
    act = (torch.tanh(gate / gate_cap) * gate_cap) * torch.sigmoid(gate) * (torch.tanh(up / up_cap) * up_cap)
    return act.to(dtype)


class SoftcappedSwiGLU(nn.Module):
    """``gate_cap*tanh(gate/gate_cap) * sigmoid(gate) * up_cap*tanh(up/up_cap)``, in float32 for half inputs."""

    def __init__(self, gate_cap: float, up_cap: float) -> None:
        """Create the activation with the two softcaps."""
        super().__init__()
        self.gate_cap = float(gate_cap)
        self.up_cap = float(up_cap)

    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """Apply the activation to ``gate`` and ``up``."""
        return _softcapped_swiglu(gate, up, self.gate_cap, self.up_cap)


def softcapped_swiglu_grads(
    gate: torch.Tensor, up: torch.Tensor, gate_cap: float, up_cap: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (d act / d gate, d act / d up), inputs already in the float compute dtype."""
    tg = torch.tanh(gate / gate_cap)
    s = torch.sigmoid(gate)
    tu = torch.tanh(up / up_cap)
    dgate = up_cap * tu * ((1 - tg * tg) * s + gate_cap * tg * s * (1 - s))
    dup = gate_cap * tg * s * (1 - tu * tu)
    return dgate, dup


@dataclass
class RoutePlan:
    """All T*K token-expert pairs sorted by expert (sentinel pairs last), and a static table of row blocks.

    The table has ``E + (T*K) // block_m`` entries, enough for any routing; the real blocks come first, the rest are
    empty (``block_rows == 0``, ``block_row0 == 0``) and do no work. Built without host syncs.
    """

    slot: torch.Tensor  # int64 (T*K,): index into the flattened (T*K) top-k arrays, a permutation
    token_idx: torch.Tensor  # int64 (T*K,): slot // K
    slot32: torch.Tensor  # int32 copy of slot, for the kernel
    token_idx32: torch.Tensor  # int32 copy of token_idx, for the kernel
    expert_of_block: torch.Tensor  # int32 (NB,)
    block_row0: torch.Tensor  # int32 (NB,): first row of the block, in plan order
    block_rows: torch.Tensor  # int32 (NB,): rows in the block, 0..block_m (0 = empty block)
    counts: torch.Tensor  # int64 (E,): rows per expert
    block_m: int


def build_route_plan(top_k_index: torch.Tensor, num_experts: int, block_m: int) -> RoutePlan:
    """Return the routing plan for ``top_k_index`` (T, K); the id ``num_experts`` means "no expert".

    No block covers a "no expert" pair.

    Everything stays on the device: no ``int()``, ``.item()``, ``.tolist()``, ``bincount``, ``nonzero`` or
    ``repeat_interleave`` with tensor repeats (each would force a host sync).
    """
    if block_m < 1:
        msg = f"block_m must be >= 1, got {block_m}"
        raise ValueError(msg)
    if num_experts < 1:
        msg = f"num_experts must be >= 1, got {num_experts}"
        raise ValueError(msg)
    k = top_k_index.shape[-1]
    num_slots = top_k_index.numel()
    device = top_k_index.device
    flat = top_k_index.reshape(-1).long()
    slot = torch.argsort(flat, stable=True)  # sentinel pairs (id num_experts) sort last
    token_idx = torch.div(slot, k, rounding_mode="floor")
    ones = torch.ones_like(flat)
    counts = torch.zeros(num_experts + 1, dtype=torch.int64, device=device).scatter_add_(0, flat, ones)[:num_experts]
    first_row = torch.cumsum(counts, 0) - counts

    blocks_per_expert = torch.div(counts + block_m - 1, block_m, rounding_mode="floor")
    cum = torch.cumsum(blocks_per_expert, 0)
    num_blocks = num_experts + num_slots // block_m  # >= sum(ceil(counts / block_m)) for any routing
    b = torch.arange(num_blocks, device=device)
    e = torch.searchsorted(cum, b, right=True).clamp(max=num_experts - 1)
    valid = b < cum[-1]
    j = b - (cum[e] - blocks_per_expert[e])  # block within its expert
    zero = torch.zeros((), dtype=torch.int64, device=device)
    block_row0 = torch.where(valid, first_row[e] + j * block_m, zero)
    block_rows = torch.where(valid, torch.clamp(counts[e] - j * block_m, 0, block_m), zero)
    return RoutePlan(
        slot=slot,
        token_idx=token_idx,
        slot32=slot.to(torch.int32),
        token_idx32=token_idx.to(torch.int32),
        expert_of_block=e.to(torch.int32),
        block_row0=block_row0.to(torch.int32),
        block_rows=block_rows.to(torch.int32),
        counts=counts,
        block_m=block_m,
    )


@runtime_checkable  # beartype checks parameters typed with it
class ExpertsBackend(Protocol):
    """What the fused experts need from a GEMM backend (see the module docstring)."""

    num_experts: int

    def gemm(  # noqa: PLR0913  # keyword options of the backend protocol
        self,
        inp: torch.Tensor,
        proj: str,
        transpose: bool,  # noqa: FBT001  # callers pass it positionally
        plan: RoutePlan,
        *,
        in_rows: torch.Tensor | None = None,
        out_rows: torch.Tensor | None = None,
        num_out_rows: int | None = None,
        row_scale: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
        round_out: bool = True,
        act: tuple[float, float] | None = None,
        save_grads: bool = False,
    ) -> Any:
        """Run the grouped GEMM for ``proj`` over the plan's rows."""
        ...


def check_gemm_options(  # noqa: C901, PLR0912, PLR0913, PLR0915, PLR0917  # one flat list of option checks
    inp: torch.Tensor,
    plan: RoutePlan,
    in_rows: torch.Tensor | None,
    out_rows: torch.Tensor | None,
    num_out_rows: int | None,
    row_scale: torch.Tensor | None,
    out_dtype: torch.dtype | None,
    act: tuple[float, float] | None = None,
    *,
    save_grads: bool = False,
    proj: str = "gate_up",
    transpose: bool = False,
    round_out: bool = True,
) -> tuple[torch.dtype, int]:
    """Validate the ``gemm`` keyword options (shapes and dtypes only, no host sync).

    Return (out_dtype, out rows).
    """
    num_rows = plan.slot.numel()
    if act is None:
        if save_grads:
            msg = "save_grads needs act"
            raise ValueError(msg)
    else:
        if proj != "gate_up" or transpose:
            msg = f"act is only valid with proj='gate_up', transpose=False (got {proj!r}, {transpose})"
            raise ValueError(msg)
        if len(act) != ACT_CAPS:
            msg = f"act must be (gate_cap, up_cap), got {act!r}"
            raise ValueError(msg)
        if out_rows is not None or num_out_rows is not None or row_scale is not None:
            msg = "act does not combine with out_rows, num_out_rows or row_scale"
            raise ValueError(msg)
        if out_dtype is not None and out_dtype != inp.dtype:
            msg = f"act returns the input dtype ({inp.dtype}); out_dtype {out_dtype} is not allowed"
            raise TypeError(msg)
        if not round_out:
            msg = "act rounds gu to the input dtype first; round_out=False is not supported"
            raise ValueError(msg)
    out_dtype = inp.dtype if out_dtype is None else out_dtype
    if out_dtype != inp.dtype and not (inp.dtype in _HALF and out_dtype == torch.float32):
        msg = f"out_dtype {out_dtype} is not allowed for {inp.dtype} input"
        raise TypeError(msg)
    if in_rows is None:
        if inp.shape[0] != num_rows:
            msg = f"inp has {inp.shape[0]} rows, the plan has {num_rows} (pass in_rows to gather)"
            raise ValueError(msg)
    elif in_rows.dtype != torch.int32 or in_rows.shape != (num_rows,):
        msg = f"in_rows must be int32 ({num_rows},), got {in_rows.dtype} {tuple(in_rows.shape)}"
        raise ValueError(msg)
    if out_rows is None:
        if num_out_rows is not None:
            msg = "num_out_rows needs out_rows"
            raise ValueError(msg)
        total = num_rows
    else:
        if out_rows.dtype != torch.int32 or out_rows.shape != (num_rows,):
            msg = f"out_rows must be int32 ({num_rows},), got {out_rows.dtype} {tuple(out_rows.shape)}"
            raise ValueError(msg)
        if num_out_rows is None:
            msg = "out_rows needs num_out_rows"
            raise ValueError(msg)
        total = num_out_rows
    if row_scale is not None:
        if out_dtype in _HALF:
            msg = "row_scale needs a float32 or float64 output (out_dtype=torch.float32 for half input)"
            raise TypeError(msg)
        if row_scale.dtype != out_dtype or row_scale.shape != (num_rows,):
            got = f"{row_scale.dtype} {tuple(row_scale.shape)}"
            msg = f"row_scale must be {out_dtype} ({num_rows},), got {got}"
            raise ValueError(msg)
    return out_dtype, total


class TorchBackend:
    """Reference backend: loops over the experts with rows and dequantizes each one's weight with the codec."""

    def __init__(  # noqa: PLR0913, PLR0917  # three tensors per projection, passed positionally
        self,
        gate_up_codes: torch.Tensor,
        gate_up_scale: torch.Tensor,
        gate_up_factor: torch.Tensor,
        down_codes: torch.Tensor,
        down_scale: torch.Tensor,
        down_factor: torch.Tensor,
    ) -> None:
        """Wrap the packed NVFP4 codes, block scales and factors of both projections."""
        self.weights = {
            "gate_up": (gate_up_codes, gate_up_scale, gate_up_factor),
            "down": (down_codes, down_scale, down_factor),
        }
        self.num_experts = gate_up_codes.shape[0]

    def gemm(  # noqa: PLR0913  # keyword options of the backend protocol
        self,
        inp: torch.Tensor,
        proj: str,
        transpose: bool,  # noqa: FBT001  # callers pass it positionally
        plan: RoutePlan,
        *,
        in_rows: torch.Tensor | None = None,
        out_rows: torch.Tensor | None = None,
        num_out_rows: int | None = None,
        row_scale: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
        round_out: bool = True,
        act: tuple[float, float] | None = None,
        save_grads: bool = False,
    ) -> Any:
        """Run the grouped GEMM expert by expert on the CPU (reference; see the module docstring)."""
        out_dtype, total = check_gemm_options(
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
        if act is not None:
            gate_cap, up_cap = act
            gu = self.gemm(inp, proj, transpose, plan, in_rows=in_rows)  # (R, 2I) in the input dtype
            gate, up = gu[:, 0::2], gu[:, 1::2]
            a = _softcapped_swiglu(gate, up, gate_cap, up_cap)  # rows no block covers: act(0, 0) = 0 exactly
            if not save_grads:
                return a, None
            f = _compute_dtype(inp.dtype)
            dgate, dup = softcapped_swiglu_grads(gate.to(f), up.to(f), gate_cap, up_cap)  # 0 at (0, 0)
            dfac = torch.stack((dgate, dup), -1).reshape(gu.shape).to(inp.dtype)  # even column = gate
            return a, dfac
        codes, scale, factor = self.weights[proj]
        rows, columns = codes.shape[1], codes.shape[2] * 2
        dtype = inp.dtype
        out = torch.zeros(total, rows if not transpose else columns, dtype=out_dtype, device=inp.device)
        half = dtype in _HALF
        if half:
            # weight rounded to the input dtype as the reference does, then multiplied and accumulated in float32
            def weight(e: int) -> torch.Tensor:
                return dequantize_nvfp4(codes[e], scale[e], factor[e]).to(dtype).float()

            compute = torch.float32
        else:

            def weight(e: int) -> torch.Tensor:
                return dequantize_nvfp4(codes[e], scale[e], factor[e], dtype=dtype)

            compute = dtype
        start = 0
        for e, n in enumerate(plan.counts.tolist()):  # CPU reference: the sync here does not matter
            if n == 0:
                continue
            w = weight(e)
            src = in_rows[start : start + n].long() if in_rows is not None else slice(start, start + n)
            x = inp[src].to(compute)
            res = x @ w if transpose else x @ w.T
            if half and round_out:
                res = res.to(dtype).to(compute)
            if row_scale is not None:
                res = res.to(out_dtype) * row_scale[start : start + n, None]
            dst = out_rows[start : start + n].long() if out_rows is not None else slice(start, start + n)
            out[dst] = res.to(out_dtype)
            start += n
        return out


def _forward(  # noqa: PLR0913, PLR0917  # shared by the Function and the no-grad path
    hidden: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    backend: ExpertsBackend,
    gate_cap: float,
    up_cap: float,
    block_m: int,
    *,
    save_grads: bool = False,
) -> tuple[torch.Tensor, RoutePlan, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Return (out, plan, dfac, a, ws), all rows in plan order (T*K rows; rows of sentinel pairs are zero).

    ``dfac`` (TK, 2I) in d: the SwiGLU's (d a / d gate, d a / d up) interleaved, when ``save_grads``; else None.
    """
    d = hidden.dtype
    f = _compute_dtype(d)
    tokens, k = top_k_index.shape
    plan = build_route_plan(top_k_index, backend.num_experts, block_m)
    # gate_up GEMM with the SwiGLU epilogue: gu (rounded to d) never leaves the kernel
    a, dfac = backend.gemm(
        hidden,
        "gate_up",
        transpose=False,
        plan=plan,
        in_rows=plan.token_idx32,
        act=(gate_cap, up_cap),
        save_grads=save_grads,
    )  # a (TK, I) d; dfac (TK, 2I) d or None
    ws = top_k_weights.reshape(-1).index_select(0, plan.slot).to(f)  # (TK,) f
    # each pair's row: rounded to d, widened, times its routing weight, into its own slot (slots are unique)
    buf = backend.gemm(
        a,
        "down",
        transpose=False,
        plan=plan,
        out_rows=plan.slot32,
        num_out_rows=tokens * k,
        row_scale=ws,
        out_dtype=f,
        round_out=True,
    )  # (TK, H) f
    out = buf.view(tokens, k, buf.shape[1]).sum(1).to(d)  # row t*K + k; fixed summation order, no atomics
    return out, plan, dfac, a, ws


@no_type_check  # autograd calls forward/backward with its own context objects; keep it unwrapped
class FusedExpertsFunction(torch.autograd.Function):
    """Experts forward and backward; the frozen expert weights get no gradient."""

    @staticmethod
    def forward(  # type: ignore[override]  # noqa: PLR0913, PLR0917  # autograd Function inputs
        ctx: Any,
        hidden: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
        backend: ExpertsBackend,
        gate_cap: float,
        up_cap: float,
        block_m: int,
    ) -> torch.Tensor:
        """Compute the experts output; keep what the backward needs on ``ctx``."""
        # the derivative factors are needed only for d(hidden)
        out, plan, dfac, a, ws = _forward(
            hidden, top_k_index, top_k_weights, backend, gate_cap, up_cap, block_m, save_grads=ctx.needs_input_grad[0]
        )
        ctx.plan, ctx.backend = plan, backend
        ctx.hidden_meta = (hidden.shape, hidden.dtype)
        ctx.weights_meta = (top_k_weights.shape, top_k_weights.dtype)
        ctx.save_for_backward(dfac, a, ws)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[torch.Tensor | None, ...]:  # type: ignore[override]  # autograd Function: one grad_out
        """Return the gradients for the hidden states and the routing weights; the experts are frozen."""
        plan: RoutePlan = ctx.plan
        backend = ctx.backend
        shape, d = ctx.hidden_meta
        w_shape, w_dtype = ctx.weights_meta
        f = _compute_dtype(d)
        tokens, k = w_shape
        need_hidden, need_weights = ctx.needs_input_grad[0], ctx.needs_input_grad[2]
        d_hidden = d_weights = None
        dfac, a, ws = ctx.saved_tensors
        go = grad_out.to(d).contiguous()  # (T, H)
        # W_down^T go for every pair, unscaled and unrounded; zero on sentinel rows
        da_u = backend.gemm(
            go, "down", transpose=True, plan=plan, in_rows=plan.token_idx32, out_dtype=f, round_out=False
        )  # (TK, I)
        if need_weights:
            # sum_h go[h] (W a)[h] = sum_i (W^T go)[i] a[i], with the bf16 a the down projection saw
            dws = (da_u * a.to(f)).sum(-1)
            flat = torch.zeros(tokens * k, dtype=f, device=dws.device).index_copy_(0, plan.slot, dws)
            d_weights = flat.view(w_shape).to(w_dtype)
        if need_hidden:
            da = da_u * ws[:, None]  # (TK, I) f
            num_rows, inter = a.shape
            dfac2 = dfac.view(num_rows, inter, 2).to(f)  # [..., 0] = d a / d gate, [..., 1] = d a / d up
            dgu = (da[:, :, None] * dfac2).reshape(num_rows, 2 * inter).to(d)  # even = gate
            buf = backend.gemm(
                dgu,
                "gate_up",
                transpose=True,
                plan=plan,
                out_rows=plan.slot32,
                num_out_rows=tokens * k,
                out_dtype=f,
                round_out=True,
            )  # (TK, H) f
            d_hidden = buf.view(tokens, k, buf.shape[1]).sum(1).to(d).view(shape)
        return d_hidden, None, d_weights, None, None, None, None


def fused_experts(  # noqa: PLR0913, PLR0917  # public API; callers pass the caps positionally
    hidden: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    backend: ExpertsBackend,
    gate_cap: float,
    up_cap: float,
    block_m: int = 64,
) -> torch.Tensor:
    """Return the experts output (T, H) in ``hidden.dtype``; use the autograd Function only if a gradient is needed."""
    if not torch.is_grad_enabled() or not (hidden.requires_grad or top_k_weights.requires_grad):
        with torch.no_grad():
            return _forward(hidden, top_k_index, top_k_weights, backend, gate_cap, up_cap, block_m)[0]
    return FusedExpertsFunction.apply(hidden, top_k_index, top_k_weights, backend, gate_cap, up_cap, block_m)
