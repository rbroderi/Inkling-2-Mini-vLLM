"""Write a random, non-zero PEFT adapter for Inkling-2-Mini (PEFT key names) to test the vLLM LoRA path.

Every module's update (alpha / r) * B @ A has norm rel_norm * ||W||_F, W being the base weight it adapts.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inkling_qlora.inkling_vllm_adapter import BASE_MODEL
from inkling_qlora.inkling_vllm_adapter import MODULE_MAP
from inkling_qlora.inkling_vllm_adapter import NUM_LAYERS

DEFAULT_MODULES = "q_proj.linear,k_proj.linear,v_proj.linear,o_proj.linear,o_proj.gate_proj"


def scaled_pair(W: torch.Tensor, r: int, alpha: float, rel_norm: float, gen: torch.Generator):
    """A (r, in) and B (out, r), float32, with ||(alpha / r) B A||_F = rel_norm * ||W||_F."""
    out, inp = W.shape
    A = torch.randn(r, inp, generator=gen, dtype=torch.float32)
    B = torch.randn(out, r, generator=gen, dtype=torch.float32)
    delta = (alpha / r) * (B.double() @ A.double())
    B = B * float(rel_norm * W.double().norm() / delta.norm())
    return A, B


def find_snapshot() -> Path:
    from huggingface_hub.constants import HF_HUB_CACHE

    repo = Path(HF_HUB_CACHE) / f"models--{BASE_MODEL.replace('/', '--')}" / "snapshots"
    found = sorted(p for p in repo.glob("*") if (p / "model.safetensors.index.json").exists()) if repo.exists() else []
    if not found:
        raise FileNotFoundError(f"no cached snapshot of {BASE_MODEL} with model.safetensors.index.json under {repo}")
    return found[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rel-norm", type=float, default=0.05)
    ap.add_argument("--modules", default=DEFAULT_MODULES)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--snapshot", type=Path, default=None)
    args = ap.parse_args()
    modules = args.modules.split(",")
    unknown = [m for m in modules if m not in MODULE_MAP]
    if unknown:
        raise SystemExit(f"unknown modules: {unknown}")
    snap = args.snapshot or find_snapshot()
    wmap = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
    names = list(MODULE_MAP)
    tensors: dict[str, torch.Tensor] = {}
    for layer in range(NUM_LAYERS):
        for hf in modules:
            wname = f"model.llm.layers.{layer}.attn.{MODULE_MAP[hf]}.weight"
            with safe_open(str(snap / wmap[wname]), framework="pt", device="cpu") as f:
                W = f.get_tensor(wname)
            gen = torch.Generator().manual_seed(args.seed * 1000003 + layer * 101 + names.index(hf))
            A, B = scaled_pair(W, args.r, args.alpha, args.rel_norm, gen)
            base = f"base_model.model.model.language_model.layers.{layer}.self_attn.{hf}"
            tensors[f"{base}.lora_A.weight"] = A
            tensors[f"{base}.lora_B.weight"] = B
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out_dir / "adapter_model.safetensors"), metadata={"format": "pt"})
    regex = r".*language_model\..*\.(?:" + "|".join(re.escape(m) for m in modules) + ")"
    cfg = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": BASE_MODEL,
        "r": args.r,
        "lora_alpha": args.alpha,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": regex,
        "use_rslora": False,
        "use_dora": False,
        "inference_mode": True,
        "init_lora_weights": True,
    }
    (args.out_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"wrote {len(tensors)} tensors to {args.out_dir}")


if __name__ == "__main__":
    main()
