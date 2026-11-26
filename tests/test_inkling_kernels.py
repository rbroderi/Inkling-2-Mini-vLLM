"""CPU tests for the fused-experts routing plan, SwiGLU, autograd Function and torch backend."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from inkling_qlora import inkling_kernels
from inkling_qlora.inkling_kernels import (
    FusedExpertsFunction,
    SoftcappedSwiGLU,
    TorchBackend,
    build_route_plan,
    fused_experts,
    softcapped_swiglu_grads,
)
from inkling_qlora.inkling_nvfp4 import dequantize_nvfp4, quantize_nvfp4

E, H, I, T, K = 4, 32, 16, 5, 2  # noqa: E741  (I = moe intermediate size, as in the plan)
GATE_CAP, UP_CAP = 4.0, 25.0


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm())


def _packed(rows: int, columns: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    parts = [quantize_nvfp4(torch.randn(rows, columns, generator=g) * 0.3) for _ in range(E)]
    codes, scale, factor = zip(*parts, strict=True)
    return torch.stack(codes), torch.stack(scale), torch.stack(factor)


@pytest.fixture(scope="module")
def weights():
    return _packed(2 * I, H, seed=1) + _packed(H, I, seed=2)


@pytest.fixture(scope="module")
def backend(weights):
    return TorchBackend(*weights)


def _routing(dtype: torch.dtype, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    # experts 0-2 only (expert 3 receives no token), one sentinel (id E)
    index = torch.tensor([[0, 2], [1, E], [2, 0], [1, 2], [0, 1]])
    weights = torch.rand(T, K, generator=g, dtype=torch.float64).to(dtype)
    hidden = torch.randn(T, H, generator=g, dtype=torch.float64).to(dtype)
    return hidden, index, weights


def _eager_loop(hidden, top_k_index, top_k_weights, weights):
    """The original algorithm (Inkling2LoopExperts.forward / Quant4bitExperts) with gate/up de-interleaved."""
    gu_codes, gu_scale, gu_factor, dn_codes, dn_scale, dn_factor = weights
    d = hidden.dtype
    f = torch.float32 if d in (torch.bfloat16, torch.float16) else d
    wdt = torch.float64 if d == torch.float64 else torch.float32
    final = torch.zeros_like(hidden, dtype=f)
    mask = F.one_hot(top_k_index, num_classes=E + 1).permute(2, 1, 0)
    for e in range(E):
        top_k_pos, token_idx = torch.where(mask[e])
        if token_idx.numel() == 0:
            continue
        w13 = dequantize_nvfp4(gu_codes[e], gu_scale[e], gu_factor[e], dtype=wdt)
        w13 = w13.view(I, 2, H).transpose(0, 1).reshape(2 * I, H).to(d)  # checkpoint interleave -> [gate; up]
        w2 = dequantize_nvfp4(dn_codes[e], dn_scale[e], dn_factor[e], dtype=wdt).to(d)
        x = hidden[token_idx]
        gate, up = F.linear(x, w13).chunk(2, dim=-1)
        g32, u32 = gate.to(f), up.to(f)
        act = (GATE_CAP * torch.tanh(g32 / GATE_CAP) * torch.sigmoid(g32) * UP_CAP * torch.tanh(u32 / UP_CAP)).to(d)
        y = F.linear(act, w2) * top_k_weights[token_idx, top_k_pos, None]
        final.index_add_(0, token_idx, y.to(f))
    return final.to(d)


def _run(fn, hidden, index, weights):
    h = hidden.clone().requires_grad_()
    w = weights.clone().requires_grad_()
    out = fn(h, index, w)
    g = torch.randn(out.shape, generator=torch.Generator().manual_seed(7), dtype=torch.float64).to(out.dtype)
    (out * g).sum().backward()
    return out.detach(), h.grad, w.grad


def test_float64_matches_eager_loop(weights, backend):
    hidden, index, tkw = _routing(torch.float64)
    got = _run(lambda h, i, w: fused_experts(h, i, w, backend, GATE_CAP, UP_CAP, block_m=2), hidden, index, tkw)
    want = _run(lambda h, i, w: _eager_loop(h, i, w, weights), hidden, index, tkw)
    for a, b in zip(got, want, strict=True):
        assert a.dtype == torch.float64
        assert _rel(a, b) <= 1e-12
    # the sentinel pair gets a zero weight gradient
    assert got[2][1, 1] == 0


def test_float32_matches_eager_loop(weights, backend):
    hidden, index, tkw = _routing(torch.float32)
    got = _run(lambda h, i, w: fused_experts(h, i, w, backend, GATE_CAP, UP_CAP), hidden, index, tkw)
    want = _run(lambda h, i, w: _eager_loop(h, i, w, weights), hidden, index, tkw)
    for a, b in zip(got, want, strict=True):
        assert a.dtype == torch.float32
        assert _rel(a, b) <= 1e-5


def test_bf16_dtypes_and_rough_agreement(weights, backend):
    hidden, index, _ = _routing(torch.bfloat16)
    tkw = _routing(torch.float32)[2]  # the router runs in float32
    got = _run(lambda h, i, w: fused_experts(h, i, w, backend, GATE_CAP, UP_CAP), hidden, index, tkw)
    assert got[0].dtype == torch.bfloat16 and got[1].dtype == torch.bfloat16 and got[2].dtype == torch.float32
    want = _run(lambda h, i, w: _eager_loop(h, i, w, weights), hidden.float(), index, tkw)
    # CPU bf16 is no exact reference (section 2.5); only check it is in the right place
    for a, b in zip(got, want, strict=True):
        assert _rel(a, b) <= 3e-2


def test_gradcheck(backend):
    hidden, index, tkw = _routing(torch.float64)
    h = hidden.clone().requires_grad_()
    w = tkw.clone().requires_grad_()
    assert torch.autograd.gradcheck(
        lambda a, b: fused_experts(a, index, b, backend, GATE_CAP, UP_CAP, block_m=1), (h, w), eps=1e-6, atol=1e-5
    )


def test_swiglu_grads_match_autograd():
    grid = torch.linspace(-40, 40, 161, dtype=torch.float64)
    gate = grid.repeat_interleave(grid.numel()).requires_grad_()
    up = grid.repeat(grid.numel()).requires_grad_()
    out = SoftcappedSwiGLU(GATE_CAP, UP_CAP)(gate, up)
    assert out.dtype == torch.float64
    dg_auto, du_auto = torch.autograd.grad(out.sum(), (gate, up))
    dg, du = softcapped_swiglu_grads(gate.detach(), up.detach(), GATE_CAP, UP_CAP)
    assert _rel(dg, dg_auto) <= 1e-12
    assert _rel(du, du_auto) <= 1e-12


def test_swiglu_half_computes_in_float32():
    gate = torch.randn(64, dtype=torch.float64) * 5
    up = torch.randn(64, dtype=torch.float64) * 30
    out = SoftcappedSwiGLU(GATE_CAP, UP_CAP)(gate.bfloat16(), up.bfloat16())
    g, u = gate.bfloat16().float(), up.bfloat16().float()
    want = (GATE_CAP * torch.tanh(g / GATE_CAP) * torch.sigmoid(g) * UP_CAP * torch.tanh(u / UP_CAP)).bfloat16()
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, want)


@pytest.mark.parametrize("block_m", [1, 2, 64])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_route_plan(block_m, seed):
    g = torch.Generator().manual_seed(seed)
    num_experts, tokens, k = 6, 37, 3
    index = torch.stack([torch.randperm(num_experts + 1, generator=g)[:k] for _ in range(tokens)])
    index[0, 0] = 5  # expert 5 always present; drop expert 2 to have an empty one
    index[index == 2] = num_experts
    plan = build_route_plan(index, num_experts, block_m)
    flat = index.reshape(-1)
    p = int((flat < num_experts).sum())
    assert int(plan.counts.sum()) == p
    for t in (plan.expert_of_block, plan.block_row0, plan.block_rows, plan.slot32, plan.token_idx32):
        assert t.dtype == torch.int32
    assert plan.slot.dtype == torch.int64 and plan.token_idx.dtype == torch.int64 and plan.counts.dtype == torch.int64
    assert plan.counts[2] == 0 and plan.counts.sum() == p
    assert plan.block_m == block_m
    # all T*K pairs, a permutation; the sentinel pairs come last and no block covers them
    assert torch.equal(torch.sort(plan.slot).values, torch.arange(tokens * k))
    assert (flat[plan.slot[p:]] == num_experts).all()
    assert torch.equal(plan.slot32.long(), plan.slot) and torch.equal(plan.token_idx32.long(), plan.token_idx)

    covered = torch.zeros(tokens * k, dtype=torch.long)
    expert_of_row = flat[plan.slot]
    for e, r0, n in zip(plan.expert_of_block.tolist(), plan.block_row0.tolist(), plan.block_rows.tolist(), strict=True):
        assert 0 <= n <= block_m
        if n == 0:
            continue
        covered[r0 : r0 + n] += 1
        assert (expert_of_row[r0 : r0 + n] == e).all()
    assert (covered[:p] == 1).all() and (covered[p:] == 0).all()
    assert torch.equal(plan.token_idx, plan.slot // k)
    expert_of_row = expert_of_row[:p]

    mask = F.one_hot(index, num_classes=num_experts + 1).permute(2, 1, 0)
    loop_pairs = set()
    for e in range(num_experts):
        top_k_pos, token_idx = torch.where(mask[e])
        loop_pairs |= {(e, t, kk) for t, kk in zip(token_idx.tolist(), top_k_pos.tolist(), strict=True)}
    plan_pairs = set(
        zip(expert_of_row.tolist(), plan.token_idx[:p].tolist(), (plan.slot[:p] % k).tolist(), strict=True)
    )
    assert plan_pairs == loop_pairs and len(plan_pairs) == p


def test_no_grad_path_skips_function(backend, monkeypatch):
    calls = []
    original = FusedExpertsFunction.forward

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(FusedExpertsFunction, "forward", staticmethod(counting))
    hidden, index, tkw = _routing(torch.float64)
    h = hidden.clone().requires_grad_()
    w = tkw.clone().requires_grad_()
    with torch.no_grad():
        out_nograd = fused_experts(h, index, w, backend, GATE_CAP, UP_CAP)
    assert not calls and out_nograd.grad_fn is None
    out_plain = fused_experts(hidden, index, tkw, backend, GATE_CAP, UP_CAP)  # nothing requires grad
    assert not calls and out_plain.grad_fn is None
    out_grad = fused_experts(h, index, w, backend, GATE_CAP, UP_CAP)
    assert len(calls) == 1 and out_grad.grad_fn is not None
    assert torch.equal(out_nograd, out_grad.detach())
    assert torch.equal(out_plain, out_grad.detach())


def test_only_weights_require_grad_runs_one_gemm_backward(backend, monkeypatch):
    # K1: d(top_k_weights) comes from W_down^T go (one transposed down GEMM); the gate_up GEMM is skipped
    hidden, index, tkw = _routing(torch.float64)
    w = tkw.clone().requires_grad_()
    out = fused_experts(hidden, index, w, backend, GATE_CAP, UP_CAP)
    seen = []
    original = backend.gemm

    def recording(inp, proj, transpose, plan, **kw):
        seen.append((proj, transpose))
        return original(inp, proj, transpose, plan, **kw)

    monkeypatch.setattr(backend, "gemm", recording)
    out.sum().backward()
    assert seen == [("down", True)] and w.grad is not None


def test_all_sentinel_gives_zeros(backend):
    hidden, _, tkw = _routing(torch.float64)
    index = torch.full((T, K), E)
    h = hidden.clone().requires_grad_()
    w = tkw.clone().requires_grad_()
    out = fused_experts(h, index, w, backend, GATE_CAP, UP_CAP)
    assert torch.equal(out, torch.zeros_like(hidden))
    out.sum().backward()
    assert torch.equal(h.grad, torch.zeros_like(hidden))
    assert torch.equal(w.grad, torch.zeros_like(tkw))


def test_module_imports_only_torch_and_codec():
    tree = ast.parse(Path(inkling_kernels.__file__).read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert roots <= {"__future__", "dataclasses", "typing", "torch", "inkling_qlora"}, roots


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_experts_two_runs_bit_identical(backend, dtype):
    hidden0, index, weights0 = _routing(dtype)
    results = []
    for _ in range(2):
        hidden = hidden0.clone().requires_grad_()
        top_w = weights0.clone().requires_grad_()
        out = fused_experts(hidden, index, top_w, backend, GATE_CAP, UP_CAP, block_m=2)
        out.backward(torch.ones_like(out))
        results.append((out.detach(), hidden.grad, top_w.grad))
    for a, b in zip(*results, strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize("block_m", [1, 3, 64])
def test_float64_matches_eager_loop_block_sizes(weights, backend, block_m):
    # the static table has trailing empty blocks at every block_m; expert 3 gets no token; one sentinel pair
    hidden, index, tkw = _routing(torch.float64)
    plan = build_route_plan(index, E, block_m)
    assert (plan.block_rows == 0).any()
    got = _run(lambda h, i, w: fused_experts(h, i, w, backend, GATE_CAP, UP_CAP, block_m=block_m), hidden, index, tkw)
    want = _run(lambda h, i, w: _eager_loop(h, i, w, weights), hidden, index, tkw)
    for a, b in zip(got, want, strict=True):
        assert _rel(a, b) <= 1e-12
    assert got[2][1, 1] == 0


# --- K1: the static, sync-free route plan ---


def _old_route_blocks(top_k_index, num_experts, block_m):
    """The pre-K1 block construction (bincount + repeat_interleave), kept as the reference."""
    flat = top_k_index.reshape(-1).long()
    counts = torch.bincount(flat, minlength=num_experts + 1)[:num_experts]
    blocks_per_expert = torch.div(counts + block_m - 1, block_m, rounding_mode="floor")
    expert_of_block = torch.repeat_interleave(torch.arange(num_experts), blocks_per_expert)
    first_block = torch.cumsum(blocks_per_expert, 0) - blocks_per_expert
    first_row = torch.cumsum(counts, 0) - counts
    j = torch.arange(expert_of_block.numel()) - first_block[expert_of_block]
    block_row0 = first_row[expert_of_block] + j * block_m
    block_rows = torch.clamp(counts[expert_of_block] - j * block_m, max=block_m)
    return expert_of_block.int(), block_row0.int(), block_rows.int()


def _random_index(seed, num_experts, tokens, k):
    g = torch.Generator().manual_seed(seed)
    index = torch.randint(0, num_experts + 1, (tokens, k), generator=g)  # includes the sentinel
    index[index == 4] = num_experts  # expert 4 gets no token
    return index


@pytest.mark.parametrize(
    "index,num_experts,block_m",
    [(_random_index(s, 7, 41, 3), 7, bm) for s in range(3) for bm in (1, 2, 3, 64)]
    + [
        (torch.tensor([[0], [1], [2]]), 3, 4),  # 3 real blocks in a table of exactly E + 3 // 4 = 3 entries
        (torch.full((5, 2), 3), 3, 2),  # all sentinel: every block empty
        (torch.zeros(9, 2, dtype=torch.long), 3, 3),  # one expert takes all 18 pairs: 6 full blocks
        (torch.zeros(0, 4, dtype=torch.long), 3, 2),  # no tokens
    ],
)
def test_route_plan_blocks_match_old_construction(index, num_experts, block_m):
    plan = build_route_plan(index, num_experts, block_m)
    old = _old_route_blocks(index, num_experts, block_m)
    n_old = old[0].numel()
    assert plan.block_rows.numel() == num_experts + index.numel() // block_m >= n_old
    for new_t, old_t in zip((plan.expert_of_block, plan.block_row0, plan.block_rows), old, strict=True):
        assert torch.equal(new_t[:n_old], old_t)
    assert (plan.block_rows[:n_old] >= 1).all()
    assert (plan.block_rows[n_old:] == 0).all() and (plan.block_row0[n_old:] == 0).all()
    assert ((plan.expert_of_block >= 0) & (plan.expert_of_block < num_experts)).all()
    assert plan.slot.numel() == index.numel()


def _call_names(node: ast.AST) -> set[str]:
    names = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            func = sub.func
            names.add(func.id if isinstance(func, ast.Name) else getattr(func, "attr", ""))
    return names


def test_plan_forward_and_backward_have_no_host_syncs():
    tree = ast.parse(Path(inkling_kernels.__file__).read_text(encoding="utf-8"))
    funcs = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    banned_plan = {"int", "item", "tolist", "bincount", "nonzero", "repeat_interleave", "max", "cpu", "numpy"}
    assert not (_call_names(funcs["build_route_plan"]) & banned_plan)
    for name in ("_forward", "backward"):
        assert not (_call_names(funcs[name]) & {"int", "item", "tolist"}), name
    # the check works: it sees the sync the CPU reference backend is allowed to have
    assert "tolist" in _call_names(funcs["gemm"])


# --- K1: the gemm keyword options on TorchBackend ---

_K_DIM = {("gate_up", False): H, ("gate_up", True): 2 * I, ("down", False): I, ("down", True): H}


@pytest.mark.parametrize("proj,transpose", list(_K_DIM))
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
def test_torch_backend_gemm_options(backend, proj, transpose, dtype):
    _, index, _ = _routing(dtype)
    plan = build_route_plan(index, E, 2)
    tk, p = T * K, int(plan.counts.sum())
    g = torch.Generator().manual_seed(5)
    src = torch.randn(T, _K_DIM[(proj, transpose)], generator=g, dtype=torch.float64).to(dtype)
    gathered = src.index_select(0, plan.token_idx)  # (TK, K_dim) in plan order
    plain = backend.gemm(gathered, proj, transpose, plan)
    assert plain.dtype == dtype and plain.shape[0] == tk
    assert (plain[p:] == 0).all() and (plain[:p] != 0).any(-1).all()  # sentinel rows are exact zeros

    # in_rows: gather through the index instead of a copy
    assert torch.equal(backend.gemm(src, proj, transpose, plan, in_rows=plan.token_idx32), plain)

    # out_rows: scatter into a larger buffer; unwritten rows stay exact zeros
    dest = (2 * plan.slot + 1).to(torch.int32)
    num_out = 2 * tk + 3
    scattered = backend.gemm(gathered, proj, transpose, plan, out_rows=dest, num_out_rows=num_out)
    want = torch.zeros(num_out, plain.shape[1], dtype=dtype)
    want[dest[:p].long()] = plain[:p]
    assert torch.equal(scattered, want)
    untouched = torch.ones(num_out, dtype=torch.bool)
    untouched[dest[:p].long()] = False
    assert (scattered[untouched] == 0).all()

    # row_scale: after the rounding to the input dtype, in the float compute dtype
    f = torch.float32 if dtype == torch.bfloat16 else dtype
    scale = torch.rand(tk, generator=g, dtype=torch.float64).to(f) + 0.5
    scaled = backend.gemm(gathered, proj, transpose, plan, row_scale=scale, out_dtype=f)
    assert scaled.dtype == f and torch.equal(scaled, plain.to(f) * scale[:, None])

    # round_out both ways
    if dtype == torch.bfloat16:
        wide = backend.gemm(gathered, proj, transpose, plan, out_dtype=torch.float32, round_out=True)
        raw = backend.gemm(gathered, proj, transpose, plan, out_dtype=torch.float32, round_out=False)
        assert torch.equal(wide, plain.float())
        assert torch.equal(raw.bfloat16(), plain)  # the unrounded accumulator rounds to the default output
        assert not torch.equal(raw, wide)
    else:
        assert torch.equal(backend.gemm(gathered, proj, transpose, plan, round_out=False), plain)


def test_torch_backend_gemm_option_errors(backend):
    _, index, _ = _routing(torch.bfloat16)
    plan = build_route_plan(index, E, 2)
    x = torch.randn(T * K, H).bfloat16()
    scale = torch.ones(T * K)
    with pytest.raises(TypeError):
        backend.gemm(x, "gate_up", False, plan, row_scale=scale)  # bf16 output
    with pytest.raises(TypeError):
        backend.gemm(x, "gate_up", False, plan, out_dtype=torch.float64)
    with pytest.raises(ValueError):
        backend.gemm(x, "gate_up", False, plan, in_rows=plan.token_idx)  # int64
    with pytest.raises(ValueError):
        backend.gemm(x[:-1], "gate_up", False, plan)  # rows != T*K without in_rows
    with pytest.raises(ValueError):
        backend.gemm(x, "gate_up", False, plan, out_rows=plan.slot32)  # no num_out_rows
    with pytest.raises(ValueError):
        backend.gemm(x, "gate_up", False, plan, num_out_rows=T * K)  # no out_rows


# --- K1: the new forward equals the pre-change glue exactly (CPU mirror of check_kernel.py's GPU case) ---


def _old_gemm(weights, inp, proj, transpose, counts):
    """The pre-K1 TorchBackend.gemm: rows in plan order, output in the input dtype."""
    gu_codes, gu_scale, gu_factor, dn_codes, dn_scale, dn_factor = weights
    codes, scale, factor = (gu_codes, gu_scale, gu_factor) if proj == "gate_up" else (dn_codes, dn_scale, dn_factor)
    dtype = inp.dtype
    out = inp.new_empty(inp.shape[0], codes.shape[1] if not transpose else codes.shape[2] * 2)
    start = 0
    for e, n in enumerate(counts.tolist()):
        if n == 0:
            continue
        if dtype in (torch.bfloat16, torch.float16):
            w, x = dequantize_nvfp4(codes[e], scale[e], factor[e]).to(dtype).float(), inp[start : start + n].float()
        else:
            w, x = dequantize_nvfp4(codes[e], scale[e], factor[e], dtype=dtype), inp[start : start + n]
        out[start : start + n] = (x @ w if transpose else x @ w.T).to(dtype)
        start += n
    return out


def _old_forward(hidden, top_k_index, top_k_weights, weights):
    """The pre-K1 ``_forward`` + ``_combine`` (gather copy, bf16 y, weight multiply and slot copy in torch)."""
    d = hidden.dtype
    f = torch.float32 if d in (torch.bfloat16, torch.float16) else d
    k = top_k_index.shape[1]
    flat = top_k_index.reshape(-1).long()
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=E + 1)[:E]
    slot = order[: int(counts.sum())]
    xs = hidden.index_select(0, slot // k)
    gu = _old_gemm(weights, xs, "gate_up", False, counts)
    g32, u32 = gu[:, 0::2].to(f), gu[:, 1::2].to(f)
    a = ((torch.tanh(g32 / GATE_CAP) * GATE_CAP) * torch.sigmoid(g32) * (torch.tanh(u32 / UP_CAP) * UP_CAP)).to(d)
    y = _old_gemm(weights, a, "down", False, counts)
    ws = top_k_weights.reshape(-1).index_select(0, slot)
    contrib = (y.to(f) * ws[:, None]).to(f)
    buf = torch.zeros(hidden.shape[0] * k, hidden.shape[1], dtype=f)
    buf.index_copy_(0, slot, contrib)
    return buf.view(hidden.shape[0], k, hidden.shape[1]).sum(1).to(d)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_forward_bit_identical_to_pre_k1_glue(weights, backend, dtype):
    hidden, index, _ = _routing(dtype)
    tkw = _routing(torch.float32)[2]
    want = _old_forward(hidden, index, tkw, weights)
    got = fused_experts(hidden, index, tkw, backend, GATE_CAP, UP_CAP, block_m=2)
    assert torch.equal(got, want)
    got_fn = fused_experts(hidden.clone().requires_grad_(), index, tkw, backend, GATE_CAP, UP_CAP, block_m=2)
    assert torch.equal(got_fn.detach(), want)


@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16])
def test_zero_tokens(backend, dtype):
    hidden = torch.zeros(0, H, dtype=dtype)
    index = torch.zeros(0, K, dtype=torch.long)
    tkw = torch.zeros(0, K, dtype=torch.float32 if dtype == torch.bfloat16 else dtype)
    out = fused_experts(hidden, index, tkw, backend, GATE_CAP, UP_CAP)
    assert out.shape == (0, H) and out.dtype == dtype
    h, w = hidden.clone().requires_grad_(), tkw.clone().requires_grad_()
    out = fused_experts(h, index, w, backend, GATE_CAP, UP_CAP)
    assert out.shape == (0, H) and out.grad_fn is not None
    out.sum().backward()
    assert h.grad.shape == (0, H) and w.grad.shape == (0, K)


# --- K2: the SwiGLU epilogue of the gate_up gemm (act, save_grads) ---


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
def test_act_epilogue_matches_swiglu_on_gu(backend, dtype):
    _, index, _ = _routing(dtype)
    plan = build_route_plan(index, E, 2)
    p = int(plan.counts.sum())
    src = torch.randn(T, H, generator=torch.Generator().manual_seed(8), dtype=torch.float64).to(dtype)
    gu = backend.gemm(src, "gate_up", False, plan, in_rows=plan.token_idx32)
    a, dfac = backend.gemm(src, "gate_up", False, plan, in_rows=plan.token_idx32, act=(GATE_CAP, UP_CAP))
    assert dfac is None and a.dtype == dtype and a.shape == (T * K, I)
    assert torch.equal(a, SoftcappedSwiGLU(GATE_CAP, UP_CAP)(gu[:, 0::2], gu[:, 1::2]))  # same a as K1's glue
    a2, dfac = backend.gemm(
        src, "gate_up", False, plan, in_rows=plan.token_idx32, act=(GATE_CAP, UP_CAP), save_grads=True
    )
    assert torch.equal(a2, a) and dfac.dtype == dtype and dfac.shape == (T * K, 2 * I)
    f = torch.float32 if dtype == torch.bfloat16 else dtype
    dg, du = softcapped_swiglu_grads(gu[:, 0::2].to(f), gu[:, 1::2].to(f), GATE_CAP, UP_CAP)
    assert torch.equal(dfac[:, 0::2], dg.to(dtype)) and torch.equal(dfac[:, 1::2], du.to(dtype))  # even = gate
    assert (a[p:] == 0).all() and (dfac[p:] == 0).all()  # sentinel rows are exact zeros


def test_act_dfac_equals_autograd_of_softcapped_swiglu(backend):
    # gu spread over -40..40 (both caps saturate); dfac against autograd of the module, float64
    _, index, _ = _routing(torch.float64)
    plan = build_route_plan(index, E, 2)
    p = int(plan.counts.sum())
    src = torch.randn(T, H, generator=torch.Generator().manual_seed(9), dtype=torch.float64)
    gu = backend.gemm(src, "gate_up", False, plan, in_rows=plan.token_idx32)
    src = src * (40.0 / gu.abs().max())
    gu = backend.gemm(src, "gate_up", False, plan, in_rows=plan.token_idx32)
    assert gu.abs().max() > 39 and (gu[:p] < -20).any() and (gu[:p] > 20).any()
    _, dfac = backend.gemm(
        src, "gate_up", False, plan, in_rows=plan.token_idx32, act=(GATE_CAP, UP_CAP), save_grads=True
    )
    gate = gu[:p, 0::2].clone().requires_grad_()
    up = gu[:p, 1::2].clone().requires_grad_()
    dg_auto, du_auto = torch.autograd.grad(SoftcappedSwiGLU(GATE_CAP, UP_CAP)(gate, up).sum(), (gate, up))
    assert _rel(dfac[:p, 0::2], dg_auto) <= 1e-12
    assert _rel(dfac[:p, 1::2], du_auto) <= 1e-12


def test_act_option_errors(backend):
    _, index, _ = _routing(torch.bfloat16)
    plan = build_route_plan(index, E, 2)
    x = torch.randn(T * K, H).bfloat16()
    caps = (GATE_CAP, UP_CAP)
    with pytest.raises(ValueError):
        backend.gemm(torch.randn(T * K, I).bfloat16(), "down", False, plan, act=caps)
    with pytest.raises(ValueError):
        backend.gemm(torch.randn(T * K, 2 * I).bfloat16(), "gate_up", True, plan, act=caps)
    with pytest.raises(ValueError):
        backend.gemm(x, "gate_up", False, plan, save_grads=True)  # save_grads without act
    with pytest.raises(ValueError):
        backend.gemm(x, "gate_up", False, plan, act=caps, out_rows=plan.slot32, num_out_rows=T * K)
    with pytest.raises(ValueError):
        backend.gemm(x, "gate_up", False, plan, act=caps, round_out=False)
    with pytest.raises(TypeError):
        backend.gemm(x, "gate_up", False, plan, act=caps, out_dtype=torch.float32)


def test_save_grads_only_when_hidden_needs_a_gradient(backend, monkeypatch):
    hidden, index, tkw = _routing(torch.float64)
    seen = []
    original = backend.gemm

    def recording(inp, proj, transpose, plan, **kw):
        result = original(inp, proj, transpose, plan, **kw)
        if kw.get("act") is not None:
            seen.append((kw.get("save_grads", False), result[1] is None))
        return result

    monkeypatch.setattr(backend, "gemm", recording)
    with torch.no_grad():
        fused_experts(hidden.clone().requires_grad_(), index, tkw, backend, GATE_CAP, UP_CAP)
    fused_experts(hidden, index, tkw.clone().requires_grad_(), backend, GATE_CAP, UP_CAP)  # only the weights
    fused_experts(hidden.clone().requires_grad_(), index, tkw, backend, GATE_CAP, UP_CAP)
    assert seen == [(False, True), (False, True), (True, False)]
