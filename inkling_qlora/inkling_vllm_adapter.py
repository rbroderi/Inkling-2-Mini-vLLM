"""Convert our PEFT Inkling adapters to the names vLLM's Inkling-2 LoRA loader expects.

The conversion is a key rename only. No tensor is scaled, transposed or cast: vLLM applies ``lora_alpha / r`` itself.

``MODULE_MAP`` follows transformers' ``conversion_mapping.py`` (checkpoint ``attn.X`` to HF ``self_attn.Y``) in
reverse. ``wq_du`` and ``gate_attn`` share a shape, as do ``wk_dv`` and ``wv_dv``, so a swapped pair would pass every
shape check; the tests pin the pairs with literals.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

BASE_MODEL = "thinkingmachines/Inkling-2-Mini"
NUM_LAYERS = 24

# HF module name (under ...self_attn.) -> checkpoint module name (under ...attn.)
MODULE_MAP: dict[str, str] = {
    "q_proj.linear": "wq_du",
    "k_proj.linear": "wk_dv",
    "v_proj.linear": "wv_dv",
    "o_proj.linear": "wo_ud",
    "o_proj.gate_proj": "gate_attn",
    "relative_position_bias.linear": "wr_du",
    "sink_proj": "attn_sink_proj",
}

# checkpoint module name -> (out, in)
MODULE_SHAPES: dict[str, tuple[int, int]] = {
    "wq_du": (3072, 2304),
    "wk_dv": (1024, 2304),
    "wv_dv": (1024, 2304),
    "wo_ud": (2304, 3072),
    "gate_attn": (3072, 2304),
    "wr_du": (384, 2304),
    "attn_sink_proj": (24, 2304),
}

_ALTERNATIVES = "|".join(re.escape(k) for k in sorted(MODULE_MAP, key=len, reverse=True))
PEFT_KEY = re.compile(
    r"^base_model\.model\.model\.language_model\.layers\.(\d+)\.self_attn\.("
    + _ALTERNATIVES
    + r")\.lora_([AB])\.weight$"
)
_CKPT_ALTERNATIVES = "|".join(re.escape(v) for v in sorted(MODULE_MAP.values(), key=len, reverse=True))
CKPT_KEY = re.compile(
    r"^base_model\.model\.model\.llm\.layers\.(\d+)\.attn\.(" + _CKPT_ALTERNATIVES + r")\.lora_([AB])\.weight$"
)
_INVERSE = {v: k for k, v in MODULE_MAP.items()}


def convert_key(key: str) -> str:
    m = PEFT_KEY.match(key)
    if not m or int(m.group(1)) >= NUM_LAYERS:
        raise ValueError(f"cannot convert adapter key: {key}")
    n, hf, ab = m.groups()
    return f"base_model.model.model.llm.layers.{int(n)}.attn.{MODULE_MAP[hf]}.lora_{ab}.weight"


def invert_key(key: str) -> str:
    m = CKPT_KEY.match(key)
    if not m or int(m.group(1)) >= NUM_LAYERS:
        raise ValueError(f"cannot invert adapter key: {key}")
    n, ckpt, ab = m.groups()
    return f"base_model.model.model.language_model.layers.{int(n)}.self_attn.{_INVERSE[ckpt]}.lora_{ab}.weight"


def convert_state_dict(tensors: dict[str, torch.Tensor], r: int) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    bad: list[str] = []
    for key in tensors:
        try:
            new = convert_key(key)
        except ValueError:
            bad.append(key)
            continue
        if new in out:
            raise ValueError(f"two input keys convert to {new} (second: {key})")
        out[new] = tensors[key]
    if bad:
        raise ValueError("unconvertible adapter keys: " + ", ".join(bad))
    if len(out) != len(tensors):
        raise ValueError(f"converted {len(out)} tensors from {len(tensors)}")
    for key, t in out.items():
        m = CKPT_KEY.match(key)
        assert m is not None
        _, ckpt, ab = m.groups()
        partner = key[: -len(f"lora_{ab}.weight")] + f"lora_{'B' if ab == 'A' else 'A'}.weight"
        if partner not in out:
            raise ValueError(f"missing partner of {invert_key(key)}: {invert_key(partner)}")
        o, i = MODULE_SHAPES[ckpt]
        want = (r, i) if ab == "A" else (o, r)
        if tuple(t.shape) != want:
            raise ValueError(f"{invert_key(key)}: shape {tuple(t.shape)}, expected {want}")
    return out


def convert_config(cfg: dict, present: set[str]) -> dict:
    problems = []
    if cfg.get("peft_type") != "LORA":
        problems.append(f"peft_type={cfg.get('peft_type')!r}")
    if cfg.get("use_dora"):
        problems.append("use_dora")
    if cfg.get("lora_bias"):
        problems.append("lora_bias")
    if cfg.get("bias", "none") != "none":
        problems.append(f"bias={cfg.get('bias')!r}")
    if cfg.get("modules_to_save"):
        problems.append("modules_to_save")
    if cfg.get("rank_pattern"):
        problems.append("rank_pattern")
    if cfg.get("alpha_pattern"):
        problems.append("alpha_pattern")
    if cfg.get("init_lora_weights", True) not in (True, "gaussian"):
        problems.append(f"init_lora_weights={cfg.get('init_lora_weights')!r}")
    if cfg.get("layer_replication"):
        problems.append("layer_replication")
    if cfg.get("alora_invocation_tokens"):
        problems.append("alora_invocation_tokens")
    if problems:
        raise ValueError("adapter config not supported by vLLM: " + ", ".join(problems))
    return {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": BASE_MODEL,
        "r": cfg["r"],
        "lora_alpha": cfg["lora_alpha"],
        "use_rslora": bool(cfg.get("use_rslora", False)),
        "target_modules": sorted(present),
        "bias": "none",
        "modules_to_save": None,
        "use_dora": False,
        "lora_bias": False,
        "init_lora_weights": True,
        "rank_pattern": {},
        "alpha_pattern": {},
    }


def convert_adapter(src: Path, dst: Path) -> dict:
    src, dst = Path(src), Path(dst)
    if dst.resolve() == src.resolve():
        raise ValueError("destination must differ from the source")
    if dst.exists() and any(dst.iterdir()):
        raise ValueError(f"destination {dst} exists and is not empty")
    weights = src / "adapter_model.safetensors"
    cfg = json.loads((src / "adapter_config.json").read_text(encoding="utf-8"))
    tensors = load_file(str(weights))
    converted = convert_state_dict(tensors, cfg["r"])
    present: set[str] = set()
    layers: set[int] = set()
    for key in converted:
        m = CKPT_KEY.match(key)
        assert m is not None
        layers.add(int(m.group(1)))
        present.add(m.group(2))
    new_cfg = convert_config(cfg, present)
    agent = src / "agent_config.json"
    summary = {
        "source": str(src),
        "source_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "num_tensors": len(converted),
        "modules": sorted(present),
        "layers": sorted(layers),
        "r": cfg["r"],
        "lora_alpha": cfg["lora_alpha"],
        "agent_config": json.loads(agent.read_text(encoding="utf-8")) if agent.exists() else None,
    }
    dst.mkdir(parents=True, exist_ok=True)
    save_file(
        {k: v.contiguous() for k, v in converted.items()},
        str(dst / "adapter_model.safetensors"),
        metadata={"format": "pt"},
    )
    (dst / "adapter_config.json").write_text(json.dumps(new_cfg, indent=2), encoding="utf-8")
    (dst / "conversion.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    written = load_file(str(dst / "adapter_model.safetensors"))
    if len(written) != len(tensors):
        raise ValueError(f"wrote {len(written)} tensors, expected {len(tensors)}")
    for k, t in tensors.items():
        if not torch.equal(t, written[convert_key(k)]):
            raise ValueError(f"tensor {k} changed in conversion")
    return summary
