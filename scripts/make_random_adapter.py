"""Write a random, non-zero PEFT adapter for Inkling-2-Mini (PEFT key names) to test the vLLM LoRA path.

Every module's update (alpha / r) * B @ A has norm rel_norm * ||W||_F, W being the base weight it adapts.
"""

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


def scaled_pair(weight: torch.Tensor, r: int, alpha: float, rel_norm: float, gen: torch.Generator):
    """Return A (r, in) and B (out, r), float32, with ||(alpha / r) B A||_F = rel_norm * ||weight||_F."""
    out, inp = weight.shape
    a_mat = torch.randn(r, inp, generator=gen, dtype=torch.float32)
    b_mat = torch.randn(out, r, generator=gen, dtype=torch.float32)
    delta = (alpha / r) * (b_mat.double() @ a_mat.double())
    b_mat = b_mat * float(rel_norm * weight.double().norm() / delta.norm())
    return a_mat, b_mat


def find_snapshot() -> Path:
    """Return the newest cached snapshot of the base model that has a weight index."""
    from huggingface_hub.constants import HF_HUB_CACHE  # noqa: PLC0415  # optional heavy dependency

    repo = Path(HF_HUB_CACHE) / f"models--{BASE_MODEL.replace('/', '--')}" / "snapshots"
    found = sorted(p for p in repo.glob("*") if (p / "model.safetensors.index.json").exists()) if repo.exists() else []
    if not found:
        msg = f"no cached snapshot of {BASE_MODEL} with model.safetensors.index.json under {repo}"
        raise FileNotFoundError(msg)
    return found[-1]


def main() -> None:
    """Write the random adapter to the output folder."""
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
        msg = f"unknown modules: {unknown}"
        raise SystemExit(msg)
    snap = args.snapshot or find_snapshot()
    wmap = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
    names = list(MODULE_MAP)
    tensors: dict[str, torch.Tensor] = {}
    for layer in range(NUM_LAYERS):
        for hf in modules:
            wname = f"model.llm.layers.{layer}.attn.{MODULE_MAP[hf]}.weight"
            with safe_open(str(snap / wmap[wname]), framework="pt", device="cpu") as f:
                weight = f.get_tensor(wname)
            gen = torch.Generator().manual_seed(args.seed * 1000003 + layer * 101 + names.index(hf))
            a_mat, b_mat = scaled_pair(weight, args.r, args.alpha, args.rel_norm, gen)
            base = f"base_model.model.model.language_model.layers.{layer}.self_attn.{hf}"
            tensors[f"{base}.lora_A.weight"] = a_mat
            tensors[f"{base}.lora_B.weight"] = b_mat
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
