"""Merge PEFT LoRA adapters into one by concatenating ranks ("cat"), each scaled by a weight.

Usage: merge_adapters_cat.py OUT SRC [SRC ...] [--weights 0.5 0.5]

The merged adapter adds sum_i w_i * (alpha_i / r_i) * B_i A_i to each target module, exactly what PEFT's
``add_weighted_adapter(combination_type="cat")`` (``adapter_eval.py --stack --weights --combine cat``) computes, but as
one ordinary adapter, so it can be served by vLLM in one pass (``serve.sh`` needs ``MAX_LORA_RANK`` >= the summed
rank). A is stacked along the rank axis, B is scaled by w_i and stacked along the rank axis. All sources must share
the target modules and the scaling alpha / r, which the merge keeps (alpha = scaling x summed rank). Other files
(agent config, tokenizer, chat template) are copied from the first source.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file
from safetensors.torch import save_file

COPY = ("agent_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "README.md")


def merge(sources: list[dict[str, torch.Tensor]], weights: list[float]) -> dict[str, torch.Tensor]:
    keys = set(sources[0])
    for s in sources[1:]:
        if set(s) != keys:
            raise SystemExit(f"adapters target different modules: {sorted(keys ^ set(s))[:4]}")
    out: dict[str, torch.Tensor] = {}
    for k in sorted(keys):
        if ".lora_A." in k:
            out[k] = torch.cat([s[k] for s in sources], dim=0)
        elif ".lora_B." in k:
            out[k] = torch.cat([w * s[k] for s, w in zip(sources, weights, strict=True)], dim=1)
        else:
            raise SystemExit(f"unexpected tensor {k} (only lora_A / lora_B are merged)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("sources", type=Path, nargs="+")
    ap.add_argument("--weights", type=float, nargs="+", help="default: equal, summing to 1")
    args = ap.parse_args()
    weights = args.weights or [1 / len(args.sources)] * len(args.sources)
    if len(weights) != len(args.sources):
        raise SystemExit("one weight per source")

    cfgs = [json.loads((s / "adapter_config.json").read_text()) for s in args.sources]
    scales = {c["lora_alpha"] / c["r"] for c in cfgs}
    if len(scales) != 1 or any(c.get("rank_pattern") or c.get("alpha_pattern") for c in cfgs):
        raise SystemExit(f"sources need one alpha / r and no rank/alpha patterns (got {scales})")
    targets = {json.dumps(c["target_modules"], sort_keys=True) for c in cfgs}
    if len(targets) != 1:
        raise SystemExit("sources have different target_modules")

    tensors = merge([load_file(str(s / "adapter_model.safetensors")) for s in args.sources], weights)
    rank = sum(c["r"] for c in cfgs)
    cfg = {**cfgs[0], "r": rank, "lora_alpha": scales.pop() * rank}
    args.out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out / "adapter_model.safetensors"), metadata={"format": "pt"})
    (args.out / "adapter_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    for name in COPY:
        if (args.sources[0] / name).exists():
            shutil.copy2(args.sources[0] / name, args.out / name)
    (args.out / "merge.json").write_text(
        json.dumps({"sources": [str(s) for s in args.sources], "weights": weights, "r": rank}, indent=2) + "\n"
    )
    n = len(args.sources)
    print(f"merged {n} adapters -> {args.out} (r={rank}, alpha={cfg['lora_alpha']}, {len(tensors)} tensors)")


if __name__ == "__main__":
    main()
