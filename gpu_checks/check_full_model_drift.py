"""Measure full-model drift of the fused experts against the loop and against bf16-sized noise.

Criteria (b) and (c) of docs/inkling_fused_experts_plan.md section 9 (ticket T7): full-model agreement of the
fused experts with the loop, judged against the loop's own sensitivity to bf16-rounding-sized noise.

The noise: every ``Quant4bitExperts`` output is multiplied by (1 + 1e-3 N(0, 1)) in fp32 and rounded back to bf16.
The generator is reseeded from the call's content (a hash of its bf16 input and its expert indices, plus the module's
index), so the gradient-checkpointing recompute of a call draws exactly the same noise as the original forward; the
script asserts this (every recomputed noisy output equals the original bit for bit).

(b) Forced routing (eval, no grad): the loop's top-k indices are recorded per router call and replayed (the routing
weights are still computed from the live scores). For each of the 48 decoder-layer calls (24 layers, the stack runs
twice) and each of the drift examples, rel(layer output, the loop's) is measured for fused and for loop+noise.
D_X(c) = mean over the examples of that rel. Pass iff D_fused(c) <= 1.5 x D_noise(c) at every call c.
Sanity: the loop with replayed routing reproduces the loop exactly at every call.

(c) Free routing, train mode as in training (LoRA r 16 / alpha 32 with B = 0, gradient checkpointing, loss.backward()):
mean over the examples of |L_fused - L_loop| <= 1.5 x mean of |L_loop+noise - L_loop|. LoRA-gradient cosines are
printed for information only (that bar was dropped).

Run in WSL (takes the GPU lock, needs 21 GiB free):
    cd /path/to/Inkling-2-Mini-vLLM && HF_HUB_OFFLINE=1 PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        ~/inkling-venv/bin/python gpu_checks/check_full_model_drift.py --drift-examples 8 --loss-examples 32
"""  # run as scripts, not a package

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any
from typing import cast

import torch
import torch.nn.functional as F  # noqa: N812  # torch convention

MODEL = "thinkingmachines/Inkling-2-Mini"
DATA = Path(os.environ.get("INKLING_CHECK_DATA", "data/train.jsonl"))
MAX_TOKENS = 1536
EPS = 1e-3
RATIO = 1.5

ROUTE = {"mode": "off", "rec": {}, "ctr": {}}


def router_forward(self, hidden_states):
    """Run ``Inkling2LoopTopKRouter.forward`` with optional record/replay of the top-k indices.

    Ported from the T5 review harness, rv_route.py. "off" is the original forward.
    """
    hidden_states = hidden_states.float()
    flat = hidden_states.reshape(-1, self.hidden_dim)
    logits = F.linear(flat, self.weight)
    scores = self.score_fn(logits)
    indices = torch.topk(scores + self.e_score_correction_bias, self.top_k, dim=-1, sorted=False).indices
    if ROUTE["mode"] == "record":
        ROUTE["rec"].setdefault(id(self), []).append(indices.detach().clone())
    elif ROUTE["mode"] == "replay":
        c = ROUTE["ctr"].get(id(self), 0)
        ROUTE["ctr"][id(self)] = c + 1
        indices = ROUTE["rec"][id(self)][c]
    weights = scores.gather(1, indices)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return logits, weights * self.routed_scaling_factor, indices


def set_route(mode: str) -> None:
    """Select the routing mode and reset its per-call counters."""
    ROUTE["mode"] = mode
    ROUTE["ctr"] = {}
    if mode == "record":
        ROUTE["rec"] = {}


def fingerprint(t: torch.Tensor) -> int:
    """Return a position-weighted integer hash of a tensor's bits (bf16 viewed as int16, integer tensors as is)."""
    bits = t.detach().reshape(-1)
    bits = bits.view(torch.int16) if bits.dtype == torch.bfloat16 else bits
    bits = bits.to(torch.int64)
    w = torch.arange(bits.numel(), device=bits.device, dtype=torch.int64) % 65521 + 1
    return int((bits * w).sum().item())


class Noise:
    """Forward hook on each experts module: content-seeded multiplicative noise, and a recompute-equality record."""

    def __init__(self) -> None:
        """Start with noise off and nothing recorded."""
        self.eps = 0.0
        self.seen: dict[tuple[int, int], int] = {}
        self.recomputed = 0
        self.mismatched = 0

    def reset(self, eps: float) -> None:
        """Set the noise size ``eps`` and clear the recompute record."""
        self.eps = eps
        self.seen = {}
        self.recomputed = self.mismatched = 0

    def hook(self, index: int):
        """Return a forward hook that adds content-seeded noise to the experts module at ``index``."""

        def fn(mod, args, out):
            if not self.eps:
                return out
            key = (index, fingerprint(args[0]) * 1_000_003 + fingerprint(args[1]))
            gen = torch.Generator(device=out.device).manual_seed((key[1] * 7919 + index) % (2**62))
            n = torch.randn(out.shape, device=out.device, generator=gen)
            out = (out.float() * (1 + self.eps * n)).to(out.dtype)
            fp = fingerprint(out)
            if key in self.seen:
                self.recomputed += 1
                self.mismatched += int(self.seen[key] != fp)
            else:
                self.seen[key] = fp
            return out

        return fn


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return the relative L2 error of ``a`` against ``b``."""
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return the cosine similarity of two flat tensors."""
    return (a @ b / (a.norm() * b.norm())).item()


def main() -> None:  # noqa: C901, PLR0915  # check script, one long flow
    """Measure the fused experts' drift against the loop and against the noise baseline."""
    import transformers.models.inkling2_loop.modeling_inkling2_loop as modeling  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
    from peft import LoraConfig  # noqa: PLC0415  # heavy GPU dependency, lazy
    from peft import get_peft_model  # noqa: PLC0415  # heavy GPU dependency, lazy
    from transformers import AutoTokenizer  # noqa: PLC0415  # heavy GPU dependency, lazy

    from inkling_qlora import inkling as ik  # noqa: PLC0415  # heavy GPU dependency, lazy
    from inkling_qlora.data import read_records  # noqa: PLC0415  # heavy GPU dependency, lazy

    parser = argparse.ArgumentParser()
    parser.add_argument("--drift-examples", type=int, default=8)
    parser.add_argument("--loss-examples", type=int, default=32)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    from inkling_qlora import inkling_kernels  # noqa: PLC0415  # heavy GPU dependency, lazy
    from inkling_qlora import inkling_triton  # noqa: PLC0415  # heavy GPU dependency, lazy

    print(f"kernels from {inkling_kernels.__file__}, {inkling_triton.__file__}", flush=True)
    modeling.Inkling2LoopTopKRouter.forward = router_forward
    torch.manual_seed(0)
    model: Any = ik.load_inkling_4bit(MODEL, device="cuda", experts_impl="loop", mxfp8_storage="nvfp4")
    experts = [m for m in model.modules() if isinstance(m, ik.Quant4bitExperts)]
    assert {m.storage for m in experts} == {"nvfp4"}
    noise = Noise()
    for i, m in enumerate(experts):
        m.register_forward_hook(noise.hook(i))
    tok: Any = AutoTokenizer.from_pretrained(MODEL)
    examples = read_records(DATA)
    random.Random(0).shuffle(examples)  # noqa: S311  # seeded sampling, not security
    batches = []
    for ex in examples:
        rec = ex
        prompt = tok.apply_chat_template(
            rec["prompt"], add_generation_prompt=True, tokenize=False, **ik.INKLING_CHAT_TEMPLATE_KWARGS
        )
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tok(rec["completion"][0]["content"] + (tok.eos_token or ""), add_special_tokens=False)["input_ids"]
        if len(p_ids) + len(c_ids) <= MAX_TOKENS:
            batches.append(
                (
                    torch.tensor([p_ids + c_ids], device="cuda"),
                    torch.tensor([[-100] * len(p_ids) + c_ids], device="cuda"),
                )
            )
        if len(batches) == max(args.drift_examples, args.loss_examples):
            break

    def set_impl(name: str) -> None:
        for m in experts:
            m.impl = name

    report: dict = {}

    # (b) forced routing, eval, no grad
    layers = [m for n, m in model.named_modules() if n.split(".")[-2:-1] == ["layers"] and "language_model" in n]
    assert len(layers) == 24, len(layers)
    cap: dict = {"ref": None, "out": [], "rels": []}

    def layer_hook(mod, a, out):
        o = (out[0] if isinstance(out, tuple) else out).detach()
        if cap["ref"] is None:
            cap["out"].append(o.cpu())
        else:
            cap["rels"].append(rel(o, cap["ref"][len(cap["rels"])].cuda()))

    handles = [m.register_forward_hook(layer_hook) for m in layers]
    model.eval()
    drift = {"fused": [], "noise": [], "loop_replay": []}
    for bi, (ids, labels) in enumerate(batches[: args.drift_examples]):
        with torch.no_grad():
            set_impl("loop")
            set_route("record")
            noise.reset(0.0)
            cap["ref"], cap["out"] = None, []
            loss_ref = model(input_ids=ids, labels=labels).loss.item()
            cap["ref"] = cap["out"]
            losses = {}
            for name, impl, eps in (("loop_replay", "loop", 0.0), ("fused", "fused", 0.0), ("noise", "loop", EPS)):
                set_impl(impl)
                set_route("replay")
                noise.reset(eps)
                cap["rels"] = []
                losses[name] = model(input_ids=ids, labels=labels).loss.item()
                assert len(cap["rels"]) == 48, len(cap["rels"])
                drift[name].append(cap["rels"])
        max_ratio = max(f / n for f, n in zip(drift["fused"][-1], drift["noise"][-1], strict=True))
        print(
            f"(b) example {bi}, {ids.shape[1]} tokens: loss loop {loss_ref:.5f}, "
            + ", ".join(f"{k} {v:.5f}" for k, v in losses.items())
            + f"; max per-call ratio fused/noise {max_ratio:.2f}",
            flush=True,
        )
    for h in handles:
        h.remove()
    cap["ref"] = cap["out"] = None
    set_route("off")
    noise.reset(0.0)
    assert all(v == 0.0 for r in drift["loop_replay"] for v in r), "loop with replayed routing is not exact"
    d_f = [sum(r[c] for r in drift["fused"]) / len(drift["fused"]) for c in range(48)]
    d_n = [sum(r[c] for r in drift["noise"]) / len(drift["noise"]) for c in range(48)]
    ratios = [f / n for f, n in zip(d_f, d_n, strict=True)]
    print("(b) mean layer-output rel per call, fused: " + " ".join(f"{v:.1e}" for v in d_f))
    print("(b) mean layer-output rel per call, noise: " + " ".join(f"{v:.1e}" for v in d_n))
    print("(b) ratio fused/noise per call:            " + " ".join(f"{v:.2f}" for v in ratios))
    failing = [c + 1 for c, r in enumerate(ratios) if r > RATIO]
    pass_b = not failing
    print(
        f"(b) max ratio {max(ratios):.2f} at call {ratios.index(max(ratios)) + 1}; calls over {RATIO}: {failing}; "
        f"criterion (b): {'PASS' if pass_b else 'FAIL'}",
        flush=True,
    )
    report["b"] = {"fused": d_f, "noise": d_n, "ratios": ratios, "per_example": drift, "pass": pass_b}

    # (c) free routing, train mode with gradient checkpointing and LoRA
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(
        model, LoraConfig(r=16, lora_alpha=32, target_modules=ik.INKLING_LORA_TARGETS, task_type="CAUSAL_LM")
    )
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    torch.cuda.reset_peak_memory_stats()

    def run(ids, labels, impl: str, eps: float):
        set_impl(impl)
        noise.reset(eps)
        loss = model(input_ids=ids, labels=labels).loss
        loss.backward()
        g = torch.cat([cast("torch.Tensor", p.grad).float().flatten() for p in params]).double().cpu()
        for p in params:
            p.grad = None
        return loss.item(), g

    rows = []
    recomputed = mismatched = 0
    for bi, (ids, labels) in enumerate(batches[: args.loss_examples]):
        l_loop, g_loop = run(ids, labels, "loop", 0.0)
        l_noise, g_noise = run(ids, labels, "loop", EPS)
        rc, mm = noise.recomputed, noise.mismatched
        recomputed, mismatched = recomputed + rc, mismatched + mm
        l_fused, g_fused = run(ids, labels, "fused", 0.0)
        l_fused2, g_fused2 = run(ids, labels, "fused", 0.0)
        row = {
            "tokens": ids.shape[1],
            "loop": l_loop,
            "noise": l_noise,
            "fused": l_fused,
            "fused_again": l_fused2,
            "cos_noise": cos(g_noise, g_loop),
            "cos_fused": cos(g_fused, g_loop),
            "cos_fused_again": cos(g_fused2, g_fused),
        }
        rows.append(row)
        print(
            f"(c) example {bi}, {row['tokens']} tokens: loss loop {l_loop:.5f}, noise {l_noise:.5f} "
            f"(|d| {abs(l_noise - l_loop):.2e}), fused {l_fused:.5f} (|d| {abs(l_fused - l_loop):.2e}), fused again "
            f"{'identical' if l_fused2 == l_fused else f'{l_fused2:.5f}'}; grad cos noise {row['cos_noise']:.3f}, "
            f"fused {row['cos_fused']:.3f}, fused again {row['cos_fused_again']:.6f}; noise recomputes {rc} "
            f"(mismatched {mm})",
            flush=True,
        )
    assert recomputed > 0, (recomputed, mismatched)
    assert mismatched == 0, (recomputed, mismatched)
    m_f = sum(abs(r["fused"] - r["loop"]) for r in rows) / len(rows)
    m_n = sum(abs(r["noise"] - r["loop"]) for r in rows) / len(rows)
    pass_c = m_f <= RATIO * m_n
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(
        f"(c) {len(rows)} examples: mean |dloss| fused vs loop {m_f:.4e}, loop+noise vs loop {m_n:.4e}, ratio "
        f"{m_f / m_n:.2f} (bar {RATIO}); criterion (c): {'PASS' if pass_c else 'FAIL'}; fused runs bit-identical "
        f"{sum(r['fused'] == r['fused_again'] for r in rows)}/{len(rows)}; noise recomputes {recomputed}, all "
        f"identical; peak {peak:.2f} GiB",
        flush=True,
    )
    report["c"] = {"rows": rows, "mean_fused": m_f, "mean_noise": m_n, "pass": pass_c, "peak_gib": peak}
    if args.out:
        args.out.write_text(json.dumps(report))


if __name__ == "__main__":
    from gpu_checks.gpu_guard import gpu_session
    from gpu_checks.inkling_snapshot import pin_full_snapshot

    pin_full_snapshot()
    with gpu_session(min_free_mib=21000):
        main()
