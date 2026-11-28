"""Compare fused and loop experts on real expert inputs captured from a full-model forward.

Criterion (a) of docs/inkling_fused_experts_plan.md section 9 (ticket T7): fused vs loop vs the fp32 golden path
on real expert inputs captured from a full-model forward, in all 24 layers.

Phase 1 loads the full model (``mxfp8_storage="nvfp4"``, loop), runs one real factcheck training example forward
(eval, no grad) and copies every experts call's (hidden, top_k_index, top_k_weights) to the CPU: 24 layers x 2 calls
(the model runs its layer stack twice). Phase 2 frees the model and, for each layer, loads that layer's experts alone
and runs ``check_real_layer.compare`` on both captured calls with a random bf16 upstream gradient, against the
section 2.5 bars (fused vs loop rel <= 5e-3 output, <= 1e-2 gradients, cosine >= 0.9999; fused error vs the golden
path <= 1.25 x loop error + 1e-4).

Run in WSL (takes the GPU lock, needs 21 GiB free for phase 1):
    cd /path/to/Inkling-2-Mini-vLLM && HF_HUB_OFFLINE=1 PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        ~/inkling-venv/bin/python gpu_checks/check_captured_layers.py
"""  # run as scripts, not a package

import argparse
import gc
import os
import random
from pathlib import Path
from typing import Any

import torch

from gpu_checks import check_real_layer as crl

DATA = Path(os.environ.get("INKLING_CHECK_DATA", "data/train.jsonl"))


def capture(example_index: int) -> dict[int, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    """Run one training example through the full model and return each layer's experts call inputs."""
    from transformers import AutoTokenizer  # noqa: PLC0415  # heavy GPU dependency, lazy

    from inkling_qlora import inkling as ik  # noqa: PLC0415  # heavy GPU dependency, lazy
    from inkling_qlora.data import read_records  # noqa: PLC0415  # heavy GPU dependency, lazy

    model = ik.load_inkling_4bit(crl.MODEL, device="cuda", experts_impl="loop", mxfp8_storage="nvfp4")
    tok: Any = AutoTokenizer.from_pretrained(crl.MODEL)
    examples = read_records(DATA)
    random.Random(0).shuffle(examples)  # noqa: S311  # seeded sampling, not security
    rec = examples[example_index]
    prompt = tok.apply_chat_template(
        rec["prompt"], add_generation_prompt=True, tokenize=False, **ik.INKLING_CHAT_TEMPLATE_KWARGS
    )
    ids = tok(prompt + rec["completion"][0]["content"] + (tok.eos_token or ""), add_special_tokens=False)["input_ids"]
    captured: dict[int, list] = {}
    handles = []
    for name, mod in model.named_modules():
        if isinstance(mod, ik.Quant4bitExperts):
            layer = int(name.split("layers.")[1].split(".")[0])

            def hook(m, args, layer=layer):
                hidden, top_k_index, top_k_weights = args
                captured.setdefault(layer, []).append(
                    (hidden.detach().cpu(), top_k_index.detach().cpu(), top_k_weights.detach().cpu())
                )

            handles.append(mod.register_forward_pre_hook(hook))
    model.eval()
    with torch.no_grad():
        model(input_ids=torch.tensor([ids], device="cuda"))
    for h in handles:
        h.remove()
    print(f"captured example {example_index}: {len(ids)} tokens, {sum(map(len, captured.values()))} experts calls")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return captured


def main() -> None:
    """Compare fused and loop experts on the captured inputs of every layer."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--example", type=int, default=2, help="index into the training set shuffled with seed 0")
    args = parser.parse_args()
    captured = capture(args.example)
    assert sorted(captured) == list(range(24)), sorted(captured)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    failures, worst = [], {}
    for layer in range(24):
        mod = crl.load_layer(layer, "nvfp4")
        for call, (hidden, top_k_index, weights) in enumerate(captured[layer]):
            g = torch.Generator(device="cuda").manual_seed(1000 * layer + call)
            grad = torch.randn(hidden.shape, device="cuda", generator=g).to(torch.bfloat16)
            inputs = (hidden.cuda(), top_k_index.cuda(), weights.cuda(), grad)
            label = f"layer {layer} call {call} (storage {mod.storage}), {hidden.shape[0]} tokens, captured input"
            results, failed = crl.compare(mod, inputs, label)
            failures += [(layer, call, f) for f in failed]
            for key in ("out", "d_hidden", "d_weights"):
                worst[key] = max(worst.get(key, 0.0), results[f"{key}_rel"])
                ratio = results[f"{key}_fused_gold"] / results[f"{key}_loop_gold"]
                worst[key + "_gold_ratio"] = max(worst.get(key + "_gold_ratio", 0.0), ratio)
                worst[key + "_cos_min"] = min(worst.get(key + "_cos_min", 1.0), results[f"{key}_cos"])
        mod = None
        torch.cuda.empty_cache()
    print("worst over all layers and calls: " + ", ".join(f"{k} {v:.7g}" for k, v in worst.items()))
    assert not failures, f"bars failed: {failures}"
    print("criterion (a): PASS (all 24 layers, both calls)")


if __name__ == "__main__":
    from gpu_checks.gpu_guard import gpu_session
    from gpu_checks.inkling_snapshot import pin_full_snapshot

    pin_full_snapshot()
    with gpu_session(min_free_mib=21000):
        main()
