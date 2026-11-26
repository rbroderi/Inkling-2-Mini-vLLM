"""Ticket A1 (docs/inkling_general_speed.md section 10): time and peak memory of the real model with the banded
sliding attention off and on, at T = 1,024, 2,048, 3,072 and 4,096 tokens (fused experts by default).

Per length and setting: (a) one no-grad forward; (b) one training step with LoRA (r 16, ``INKLING_LORA_TARGETS``),
gradient checkpointing and the loss on the last 64 positions only. In both, the script runs the decoder itself and
applies the output layer (and the logit softcap) to the last 64 positions, so the logits do not set the peak. No
optimizer step: the weights stay the same across cases. 2 warm-up runs, median of 5, ``torch.cuda.synchronize()``
around each run, ``max_memory_allocated`` reset per case. A case that runs out of memory is recorded as "OOM" and not
retried smaller. Prints the table and the gate for A2's default (T = 1,024 within 2% in both modes; the T = 2,048
training step at least 5% faster with the patch). The card should otherwise be idle. No torch.profiler.

Run in WSL (takes the GPU lock, needs 21 GiB free):
    cd /path/to/Inkling-2-Mini-vLLM && HF_HUB_OFFLINE=1 PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        ~/inkling-venv/bin/python gpu_checks/bench_banded.py --out /tmp/bench_banded.json
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

MODEL = "thinkingmachines/Inkling-2-Mini"
LOSS_POSITIONS = 64
WARMUP, RUNS = 2, 5


def last_positions_loss(base: Any, ids: torch.Tensor) -> torch.Tensor:
    """Decoder on all positions, output layer and softcap on the last 64 (+1 shifted), next-token loss."""
    hidden = base.model(input_ids=ids, use_cache=False).last_hidden_state
    logits = base.logit_softcap(base.lm_head(hidden[:, -LOSS_POSITIONS - 1 : -1]).float())
    return F.cross_entropy(logits.flatten(0, 1), ids[:, -LOSS_POSITIONS:].flatten())


def time_case(fn) -> dict[str, Any]:
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    times = []
    try:
        for i in range(WARMUP + RUNS):
            torch.cuda.synchronize()
            t = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            if i >= WARMUP:
                times.append(time.perf_counter() - t)
    except (torch.OutOfMemoryError, RuntimeError) as exc:
        # With expandable segments an OOM can surface as "CUDA driver error: out of memory" (a plain RuntimeError).
        if not isinstance(exc, torch.OutOfMemoryError) and "out of memory" not in str(exc):
            raise
        return {"result": "OOM"}
    return {
        "median_s": statistics.median(times),
        "times_s": times,
        "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="1024,2048,3072,4096")
    parser.add_argument("--modes", default="forward,train_step", help="comma-separated subset of forward,train_step")
    parser.add_argument(
        "--banded",
        choices=("both", "off", "on"),
        default="both",
        help="run one setting only (one case per process after a CUDA driver OOM, which leaves the allocator unusable)",
    )
    parser.add_argument("--experts-impl", choices=("fused", "loop"), default="fused")
    parser.add_argument("--mxfp8-storage", choices=("nf4", "nvfp4"), default="nvfp4")
    parser.add_argument("--out", type=Path, help="write the results as JSON")
    args = parser.parse_args()
    lengths = [int(x) for x in args.lengths.split(",")]

    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer

    from gpu_checks.check_banded_attention import long_token_ids
    from inkling_qlora import inkling as ik
    from inkling_qlora import inkling_banded as ib

    model = ik.load_inkling_4bit(MODEL, device="cuda", experts_impl=args.experts_impl, mxfp8_storage=args.mxfp8_storage)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(
        model, LoraConfig(r=16, lora_alpha=32, target_modules=ik.INKLING_LORA_TARGETS, task_type="CAUSAL_LM")
    )
    base = model.get_base_model()
    tok = AutoTokenizer.from_pretrained(MODEL)
    all_ids = long_token_ids(tok, max(lengths))
    print(f"experts {args.experts_impl}, MXFP8 layers as {args.mxfp8_storage}; lengths {lengths}", flush=True)

    def forward_no_grad(ids: torch.Tensor) -> None:
        model.eval()
        with torch.no_grad():
            last_positions_loss(base, ids)

    def train_step(ids: torch.Tensor) -> None:
        model.train()
        last_positions_loss(base, ids).backward()
        model.zero_grad(set_to_none=True)

    results: list[dict[str, Any]] = []
    for length in lengths:
        ids = torch.tensor([all_ids[:length]], device="cuda")
        for mode, fn in (("forward", forward_no_grad), ("train_step", train_step)):
            if mode not in args.modes.split(","):
                continue
            for banded in {"both": (False, True), "off": (False,), "on": (True,)}[args.banded]:
                if banded:
                    ib.enable_banded_attention(model)
                try:
                    r = time_case(lambda fn=fn, ids=ids: fn(ids))
                finally:
                    ib.disable_banded_attention(model)
                model.zero_grad(set_to_none=True)
                r |= {"length": length, "mode": mode, "banded": banded}
                results.append(r)
                shown = r.get("result") or f"{r['median_s']:.3f}s median, peak {r['peak_gib']:.2f} GiB"
                print(f"T={length:5d} {mode:10s} banded={banded!s:5s} {shown}", flush=True)

    def get(length: int, mode: str, banded: bool) -> dict[str, Any] | None:
        return next((r for r in results if (r["length"], r["mode"], r["banded"]) == (length, mode, banded)), None)

    print("\nlength  mode        off (s)   on (s)   speed-up   peak off / on (GiB)")
    speedups: dict[tuple[int, str], float | None] = {}
    for length in lengths:
        for mode in ("forward", "train_step"):
            off, on = get(length, mode, False), get(length, mode, True)
            if off is None or on is None or "median_s" not in off or "median_s" not in on:
                speedups[(length, mode)] = None
                print(
                    f"{length:6d}  {mode:10s}  {off.get('result', '') if off else ''} / {on.get('result', '') if on else ''}"
                )
                continue
            s = 1 - on["median_s"] / off["median_s"]  # time saved by the patch
            speedups[(length, mode)] = s
            print(
                f"{length:6d}  {mode:10s}  {off['median_s']:7.3f}  {on['median_s']:7.3f}  {s:+10.1%}   "
                f"{off['peak_gib']:.2f} / {on['peak_gib']:.2f}"
            )

    gate = {}
    if 1024 in lengths:
        gate["T=1024 within 2%"] = all(
            (s := speedups.get((1024, m))) is not None and abs(s) <= 0.02 for m in ("forward", "train_step")
        )
    if 2048 in lengths:
        gate["T=2048 train step >= 5% faster"] = (s := speedups.get((2048, "train_step"))) is not None and s >= 0.05
    print("\ngate for A2's default (the correctness bars of check_banded_attention.py also count):")
    for k, v in gate.items():
        print(f"  {k}: {'pass' if v else 'FAIL'}")
    if args.out:
        args.out.write_text(
            json.dumps({"results": results, "gate": gate, "args": vars(args) | {"out": str(args.out)}}, indent=1)
        )


if __name__ == "__main__":
    from gpu_checks.gpu_guard import gpu_session
    from gpu_checks.inkling_snapshot import pin_full_snapshot

    pin_full_snapshot()
    with gpu_session(min_free_mib=21000):
        main()
