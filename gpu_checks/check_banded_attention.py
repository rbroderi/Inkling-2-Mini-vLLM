"""Ticket A1 (docs/inkling_general_speed.md section 10): the banded sliding attention against the original forward on
real inputs of every sliding attention call of the real model (fused experts).

One no-grad forward of a 2,048-token prompt (validation examples concatenated to length) and one of a right-padded
batch of two (1,500 and 2,048 tokens) are run with the patch off; a forward pre-hook copies the inputs of every
sliding attention call to the CPU (40 calls per forward: 20 sliding modules, 10 of them in the loop layers that
run three times). Then, with the patch on, each captured call is run through the class's original forward and the
patched forward on the same inputs:
  - bf16: output rel <= 2e-3 and cosine >= 0.9999;
  - the module deep-copied to fp32 with fp32 inputs: output rel <= 1e-5;
  - layers 0, 9 and 22: the gradient of ``output.float().sum()`` with respect to ``hidden_states`` (and to the key
    and value inputs where the loop layers pass them): rel <= 1e-2 in bf16, <= 1e-4 in fp32.
Every captured call must take the banded path (no fallback), and the full-attention modules must be unpatched.
Whole-model logits are not compared (the model is chaotic at bf16 rounding).

Run in WSL (takes the GPU lock, needs 21 GiB free):
    cd /path/to/Inkling-2-Mini-vLLM && HF_HUB_OFFLINE=1 PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        ~/inkling-venv/bin/python gpu_checks/check_banded_attention.py
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
from pathlib import Path
from typing import Any

import torch

MODEL = "thinkingmachines/Inkling-2-Mini"
DATA = Path(os.environ.get("INKLING_CHECK_DATA", "data/validation.jsonl"))
GRAD_LAYERS = (0, 9, 22)
BARS = {"bf16_rel": 2e-3, "bf16_cos": 0.9999, "fp32_rel": 1e-5, "bf16_grad_rel": 1e-2, "fp32_grad_rel": 1e-4}


def long_token_ids(tok: Any, length: int) -> list[int]:
    """Validation examples (prompt + completion, chat template) concatenated until ``length`` tokens, truncated."""
    from inkling_qlora import inkling as ik
    from inkling_qlora.data import iter_records

    ids: list[int] = []
    for rec in iter_records(DATA):
        prompt = tok.apply_chat_template(
            rec["prompt"], add_generation_prompt=True, tokenize=False, **ik.INKLING_CHAT_TEMPLATE_KWARGS
        )
        text = prompt + rec["completion"][0]["content"] + (tok.eos_token or "")
        ids += tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) >= length:
            return ids[:length]
    raise ValueError(f"{DATA} has fewer than {length} tokens")


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    return (a @ b / (a.norm() * b.norm()).clamp_min(1e-30)).item()


def to_device(obj: Any, device: str, memo: dict[int, Any], dtype: torch.dtype | None = None) -> Any:
    """Copy tensors (in nested dicts) to ``device``; floating tensors to ``dtype`` if given. Shared tensors (the
    mask, the distance matrix) are copied once."""
    if isinstance(obj, dict):
        return {k: to_device(v, device, memo, dtype) for k, v in obj.items()}
    if not isinstance(obj, torch.Tensor):
        return obj
    if id(obj) not in memo:
        t = obj.detach().to(device)
        if dtype is not None and t.is_floating_point():
            t = t.to(dtype)
        memo[id(obj)] = t
    return memo[id(obj)]


def capture(model: torch.nn.Module, modeling: Any, ids: torch.Tensor, mask: torch.Tensor | None) -> list[tuple]:
    calls: list[tuple] = []
    memo: dict[int, Any] = {}
    handles = []
    for module in model.modules():
        if isinstance(module, modeling.Inkling2LoopSlidingAttention) and module.is_sliding:

            def hook(mod, args, kwargs):
                assert not args, "the decoder layers pass every attention input by keyword"
                calls.append((mod, to_device(dict(kwargs), "cpu", memo)))

            handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))
    try:
        with torch.no_grad():
            model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=1)
    finally:
        for h in handles:
            h.remove()
    return calls


def grad_inputs(kwargs: dict[str, Any]) -> list[str]:
    return [k for k in ("hidden_states", "key_hidden_states", "value_hidden_states") if kwargs.get(k) is not None]


def run_pair(module, modeling, ib, kwargs: dict[str, Any], want_grad: bool) -> dict[str, Any]:
    """Original vs patched forward on the same inputs; output (and input gradients if ``want_grad``)."""
    original = modeling.Inkling2LoopSlidingAttention.forward
    assert not ib._use_original(
        module,
        kwargs["hidden_states"],
        kwargs["attention_mask"],
        kwargs["relative_distance"],
        kwargs.get("cache"),
        {k: v for k, v in kwargs.items() if k not in ("hidden_states", "attention_mask", "relative_distance", "cache")},
    ), "this call would take the fallback"
    out: dict[str, Any] = {}
    for name, fn in (("ref", lambda **kw: original(module, **kw)), ("band", module.forward)):
        kw = dict(kwargs)
        names = grad_inputs(kw) if want_grad else []
        for k in names:
            kw[k] = kw[k].detach().clone().requires_grad_()
        with torch.set_grad_enabled(want_grad):
            y = fn(**kw)[0]
            grads = torch.autograd.grad(y.float().sum(), [kw[k] for k in names]) if want_grad else ()
        out[name] = (y.detach(), dict(zip(names, (g.detach() for g in grads), strict=True)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, help="write the per-call numbers as JSON")
    args = parser.parse_args()

    import transformers.models.inkling2_loop.modeling_inkling2_loop as modeling
    from transformers import AutoTokenizer

    from inkling_qlora import inkling as ik
    from inkling_qlora import inkling_banded as ib

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = ik.load_inkling_4bit(MODEL, device="cuda", experts_impl="fused", mxfp8_storage="nvfp4")
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = long_token_ids(tok, 2048)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    single = torch.tensor([ids], device="cuda")
    batch = torch.tensor([ids[:1500] + [pad_id] * 548, ids], device="cuda")  # right padding, as in training
    batch_mask = torch.ones_like(batch)
    batch_mask[0, 1500:] = 0

    cases = {
        "2048": capture(model, modeling, single, None),
        "batch 1500+2048": capture(model, modeling, batch, batch_mask),
    }
    for name, calls in cases.items():
        print(f"captured {len(calls)} sliding calls for {name}", flush=True)
        assert len(calls) == 40, len(calls)

    n = ib.enable_banded_attention(model)
    assert n == 20, n
    full_attn = [m for m in model.modules() if isinstance(m, modeling.Inkling2LoopFullAttention)]
    full_attn += [
        m for m in model.modules() if isinstance(m, modeling.Inkling2LoopSlidingAttention) and not m.is_sliding
    ]
    assert full_attn and all(not hasattr(m, "_banded_original_forward") for m in full_attn), "full attention patched"
    print(f"patched {n} sliding modules; {len(full_attn)} full-attention modules unpatched", flush=True)

    rows, failures = [], []
    worst = dict.fromkeys(("bf16_rel", "fp32_rel", "bf16_grad_rel", "fp32_grad_rel"), 0.0) | {"bf16_cos": 1.0}
    for case, calls in cases.items():
        seen: dict[int, int] = {}
        for module, cpu_kwargs in calls:
            layer = module.layer_idx
            call = seen[layer] = seen.get(layer, -1) + 1
            want_grad = layer in GRAD_LAYERS
            row: dict[str, Any] = {"case": case, "layer": layer, "call": call}

            kw16 = to_device(cpu_kwargs, "cuda", {})
            r = run_pair(module, modeling, ib, kw16, want_grad)
            row["bf16_rel"] = rel(r["band"][0], r["ref"][0])
            row["bf16_cos"] = cos(r["band"][0], r["ref"][0])
            if want_grad:
                row["bf16_grad_rel"] = max(rel(r["band"][1][k], r["ref"][1][k]) for k in r["ref"][1])
            del r, kw16

            module32 = copy.deepcopy(module).float()
            kw32 = to_device(cpu_kwargs, "cuda", {}, dtype=torch.float32)
            r = run_pair(module32, modeling, ib, kw32, want_grad)
            row["fp32_rel"] = rel(r["band"][0], r["ref"][0])
            if want_grad:
                row["fp32_grad_rel"] = max(rel(r["band"][1][k], r["ref"][1][k]) for k in r["ref"][1])
            del r, kw32, module32
            torch.cuda.empty_cache()

            for key, value in row.items():
                if key in BARS:
                    bad = value < BARS[key] if key == "bf16_cos" else value > BARS[key]
                    worst[key] = min(worst[key], value) if key == "bf16_cos" else max(worst[key], value)
                    if bad:
                        failures.append(f"{case} layer {layer} call {call}: {key} {value:.3g} (bar {BARS[key]})")
            rows.append(row)
            print(
                "  " + ", ".join(f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()),
                flush=True,
            )
        gc.collect()

    print("worst: " + ", ".join(f"{k} {v:.4g} (bar {BARS[k]})" for k, v in worst.items()))
    if args.out:
        args.out.write_text(json.dumps({"rows": rows, "worst": worst, "failures": failures}, indent=1))
    assert not failures, f"bars failed: {failures}"
    print("banded attention check: PASS")


if __name__ == "__main__":
    from gpu_checks.gpu_guard import gpu_session
    from gpu_checks.inkling_snapshot import pin_full_snapshot

    pin_full_snapshot()
    with gpu_session(min_free_mib=21000):
        main()
