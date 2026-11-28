"""Criterion (d) of docs/inkling_fused_experts_plan.md section 9 (ticket T7): LoRA training curves, fused vs loop.

For each seed, both implementations train a fresh LoRA from the same initialization on the same data order:
batch 1, AdamW lr 1e-4 (no weight decay, no schedule), gradient clipping 0.3, r 16 / alpha 32 / dropout 0 on
``INKLING_LORA_TARGETS``, gradient checkpointing, examples of at most 1536 tokens from the factcheck training set
shuffled with ``random.Random(seed)``. The model is loaded once (``mxfp8_storage="nvfp4"``); ``impl`` is switched on
every ``Quant4bitExperts`` and the LoRA weights are re-initialized under ``torch.manual_seed(seed)`` before each run.

Bar (fixed before running): M(impl, s) = mean loss of steps (N-19)..N of seed s. The loop's seed-to-seed spread is
[min_s M(loop, s), max_s M(loop, s)]. Pass iff the fused mean, mean_s M(fused, s) (the mean of all 3 x 20 last-step
losses), lies inside that interval. Per-seed values are printed as well.

Run in WSL (takes the GPU lock, needs 21 GiB free; the loop arm takes about 13 s per step):
    cd /path/to/Inkling-2-Mini-vLLM && HF_HUB_OFFLINE=1 PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        ~/inkling-venv/bin/python gpu_checks/check_lora_seeds.py --seeds 0,1,2 --steps 100 --out results.json
"""  # run as scripts, not a package

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import torch

MODEL = "thinkingmachines/Inkling-2-Mini"
DATA = Path(os.environ.get("INKLING_CHECK_DATA", "data/train.jsonl"))
MAX_TOKENS = 1536
LAST = 20


def batches_for_seed(tok, examples, seed: int, steps: int) -> list[tuple[list[int], list[int]]]:
    """Return ``steps`` training batches in the shuffled order for ``seed``."""
    from inkling_qlora import inkling as ik  # noqa: PLC0415  # heavy GPU dependency, lazy

    order = list(examples)
    random.Random(seed).shuffle(order)  # noqa: S311  # seeded sampling, not security
    out = []
    for ex in order:
        rec = ex
        prompt = tok.apply_chat_template(
            rec["prompt"], add_generation_prompt=True, tokenize=False, **ik.INKLING_CHAT_TEMPLATE_KWARGS
        )
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tok(rec["completion"][0]["content"] + (tok.eos_token or ""), add_special_tokens=False)["input_ids"]
        if len(p_ids) + len(c_ids) > MAX_TOKENS:
            continue
        out.append((p_ids + c_ids, [-100] * len(p_ids) + c_ids))
        if len(out) == steps:
            return out
    msg = f"only {len(out)} examples of at most {MAX_TOKENS} tokens"
    raise ValueError(msg)


def main() -> None:  # noqa: PLR0915  # check script, one long flow
    """Train a LoRA with each implementation and seed and compare their final losses."""
    from peft import LoraConfig  # noqa: PLC0415  # heavy GPU dependency, lazy
    from peft import get_peft_model  # noqa: PLC0415  # heavy GPU dependency, lazy
    from peft.tuners.lora.layer import LoraLayer  # noqa: PLC0415  # heavy GPU dependency, lazy
    from transformers import AutoTokenizer  # noqa: PLC0415  # heavy GPU dependency, lazy

    from gpu_checks.inkling_snapshot import pin_full_snapshot  # noqa: PLC0415  # heavy GPU dependency, lazy
    from inkling_qlora import inkling as ik  # noqa: PLC0415  # heavy GPU dependency, lazy
    from inkling_qlora.data import read_records  # noqa: PLC0415  # heavy GPU dependency, lazy

    pin_full_snapshot()
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--impls", default="fused,loop")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    impls = args.impls.split(",")
    assert args.steps >= LAST
    assert set(impls) <= {"fused", "loop"}

    model: Any = ik.load_inkling_4bit(MODEL, device="cuda", experts_impl="loop", mxfp8_storage="nvfp4")
    experts = [m for m in model.modules() if isinstance(m, ik.Quant4bitExperts)]
    print(f"storage {sorted({m.storage for m in experts})}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    examples = read_records(DATA)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.0, target_modules=ik.INKLING_LORA_TARGETS, task_type="CAUSAL_LM"
        ),
    )
    model.train()
    lora_layers = [m for m in model.modules() if isinstance(m, LoraLayer)]
    params = [p for p in model.parameters() if p.requires_grad]
    results: dict[str, dict[str, list[float]]] = {impl: {} for impl in impls}
    peak = 0.0
    for seed in seeds:
        batches = batches_for_seed(tok, examples, seed, args.steps)
        print(f"seed {seed}: {len(batches)} examples, tokens {sum(len(b[0]) for b in batches)}", flush=True)
        for impl in impls:
            for m in experts:
                m.impl = impl
            torch.manual_seed(seed)
            for layer in lora_layers:
                layer.reset_lora_parameters("default", True)  # noqa: FBT003  # PEFT API, second argument is positional
            opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.0)
            torch.cuda.reset_peak_memory_stats()
            losses, t0 = [], time.perf_counter()
            for step, (ids, labels) in enumerate(batches, 1):
                loss = model(
                    input_ids=torch.tensor([ids], device="cuda"), labels=torch.tensor([labels], device="cuda")
                ).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 0.3)
                opt.step()
                opt.zero_grad(set_to_none=True)
                losses.append(loss.item())
                if step % 10 == 0:
                    print(
                        f"  seed {seed} {impl} step {step}: loss {losses[-1]:.4f}, mean of last 10 "
                        f"{sum(losses[-10:]) / 10:.4f}, {(time.perf_counter() - t0) / step:.2f} s/step",
                        flush=True,
                    )
            run_peak = torch.cuda.max_memory_allocated() / 2**30
            peak = max(peak, run_peak)
            results[impl][str(seed)] = losses
            print(
                f"seed {seed} {impl}: mean loss of the last {LAST} steps {sum(losses[-LAST:]) / LAST:.4f}, peak "
                f"{run_peak:.2f} GiB",
                flush=True,
            )
            del opt
            args.out.write_text(json.dumps({"results": results, "peak_gib": peak}))

    def last_mean(impl: str, s: int) -> float:
        return sum(results[impl][str(s)][-LAST:]) / LAST

    if set(impls) == {"fused", "loop"}:
        loop_means = [last_mean("loop", s) for s in seeds]
        fused_means = [last_mean("fused", s) for s in seeds]
        lo, hi = min(loop_means), max(loop_means)
        fused = sum(fused_means) / len(fused_means)
        print(f"loop per-seed last-{LAST} means {[f'{v:.4f}' for v in loop_means]} -> spread [{lo:.4f}, {hi:.4f}]")
        print(f"fused per-seed last-{LAST} means {[f'{v:.4f}' for v in fused_means]} -> mean {fused:.4f}")
        print(f"per-seed fused - loop: {[f'{f - lp:+.4f}' for f, lp in zip(fused_means, loop_means, strict=True)]}")
        print(f"criterion (d): {'PASS' if lo <= fused <= hi else 'FAIL'}; peak {peak:.2f} GiB")


if __name__ == "__main__":
    from gpu_checks.gpu_guard import gpu_session

    with gpu_session(min_free_mib=21000):
        main()
