"""Tests for inkling vllm adapter."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from safetensors.torch import save_file

from inkling_qlora.inkling_vllm_adapter import MODULE_MAP
from inkling_qlora.inkling_vllm_adapter import MODULE_SHAPES
from inkling_qlora.inkling_vllm_adapter import convert_adapter
from inkling_qlora.inkling_vllm_adapter import convert_config
from inkling_qlora.inkling_vllm_adapter import convert_key
from inkling_qlora.inkling_vllm_adapter import convert_state_dict
from inkling_qlora.inkling_vllm_adapter import invert_key

# Literal pairs (HF name, checkpoint name), typed independently of MODULE_MAP; see
# transformers/conversion_mapping.py lines 379-396 (inkling2 text/loop renames).
EXPECTED = {
    "q_proj.linear": "wq_du",
    "k_proj.linear": "wk_dv",
    "v_proj.linear": "wv_dv",
    "o_proj.linear": "wo_ud",
    "o_proj.gate_proj": "gate_attn",
    "relative_position_bias.linear": "wr_du",
    "sink_proj": "attn_sink_proj",
}
FIVE = ["q_proj.linear", "k_proj.linear", "v_proj.linear", "o_proj.linear", "o_proj.gate_proj"]
SHAPES = {
    "q_proj.linear": (3072, 2304),
    "k_proj.linear": (1024, 2304),
    "v_proj.linear": (1024, 2304),
    "o_proj.linear": (2304, 3072),
    "o_proj.gate_proj": (3072, 2304),
}
R = 4


def peft_key(n, hf, ab):
    """Return the PEFT state-dict key of layer ``n``, projection ``hf`` and LoRA half ``ab``."""
    return f"base_model.model.model.language_model.layers.{n}.self_attn.{hf}.lora_{ab}.weight"


def synthetic(r=R, modules=FIVE):
    """Return a random PEFT state dict of rank ``r`` for the projections in ``modules``."""
    t = {}
    for n in range(24):
        for hf in modules:
            out, inp = SHAPES[hf]
            t[peft_key(n, hf, "A")] = torch.randn(r, inp)
            t[peft_key(n, hf, "B")] = torch.randn(out, r)
    return t


def test_module_map_literal():
    """Check module map literal."""
    assert MODULE_MAP == EXPECTED
    assert MODULE_SHAPES["wq_du"] == MODULE_SHAPES["gate_attn"] == (3072, 2304)


def test_full_adapter_converts():
    """Check full adapter converts."""
    src = synthetic()
    out = convert_state_dict(src, R)
    expected = {
        f"base_model.model.model.llm.layers.{n}.attn.{ck}.lora_{ab}.weight"
        for n in range(24)
        for ck in ["wq_du", "wk_dv", "wv_dv", "wo_ud", "gate_attn"]
        for ab in "AB"
    }
    assert len(out) == 240
    assert set(out) == expected
    for k, v in src.items():
        assert out[convert_key(k)] is v
    # same-shape swaps: q must land on wq_du, gate_proj on gate_attn, etc.
    for n in (0, 23):
        assert (
            out[f"base_model.model.model.llm.layers.{n}.attn.wq_du.lora_A.weight"]
            is src[peft_key(n, "q_proj.linear", "A")]
        )
        assert (
            out[f"base_model.model.model.llm.layers.{n}.attn.gate_attn.lora_B.weight"]
            is src[peft_key(n, "o_proj.gate_proj", "B")]
        )
        assert (
            out[f"base_model.model.model.llm.layers.{n}.attn.wk_dv.lora_A.weight"]
            is src[peft_key(n, "k_proj.linear", "A")]
        )
        assert (
            out[f"base_model.model.model.llm.layers.{n}.attn.wv_dv.lora_A.weight"]
            is src[peft_key(n, "v_proj.linear", "A")]
        )


def test_invert_and_injective():
    """Check invert and injective."""
    keys = [peft_key(n, hf, ab) for n in range(24) for hf in EXPECTED for ab in "AB"]
    converted = [convert_key(k) for k in keys]
    assert len(set(converted)) == len(keys) == 336
    assert all(invert_key(convert_key(k)) == k for k in keys)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda t: t.__setitem__(
            "base_model.model.model.language_model.layers.3.mlp.gate.lora_A.weight", torch.zeros(1, 1)
        ),
        lambda t: t.__setitem__(peft_key(24, "q_proj.linear", "A"), torch.zeros(R, 2304)),
        lambda t: t.__setitem__(
            "base_model.model.model.vision_tower.layers.0.self_attn.q_proj.lora_A.weight", torch.zeros(R, 2304)
        ),
    ],
)
def test_bad_key_raises(mutate):
    """Check bad key raises."""
    t = synthetic()
    mutate(t)
    with pytest.raises(ValueError, match=r"layers\.3\.mlp\.gate|layers\.24\.|vision_tower"):
        convert_state_dict(t, R)


def test_missing_b_wrong_rank_transposed():
    """Check missing b wrong rank transposed."""
    t = synthetic()
    del t[peft_key(5, "k_proj.linear", "B")]
    with pytest.raises(ValueError, match=r"layers\.5\.self_attn\.k_proj\.linear"):
        convert_state_dict(t, R)
    t = synthetic()
    t[peft_key(2, "q_proj.linear", "A")] = torch.zeros(R + 1, 2304)
    with pytest.raises(ValueError, match=r"layers\.2\.self_attn\.q_proj\.linear"):
        convert_state_dict(t, R)
    t = synthetic()
    t[peft_key(7, "o_proj.linear", "B")] = torch.zeros(R, 2304)
    with pytest.raises(ValueError, match=r"layers\.7\.self_attn\.o_proj\.linear"):
        convert_state_dict(t, R)


PEFT_CFG = {
    "peft_type": "LORA",
    "task_type": "CAUSAL_LM",
    "base_model_name_or_path": "somewhere/else",
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "target_modules": r".*language_model\..*\.(?:(?:q|k|v|o)_proj\.linear|o_proj\.gate_proj)",
    "use_rslora": False,
    "use_dora": False,
    "inference_mode": True,
    "init_lora_weights": True,
    "modules_to_save": None,
    "rank_pattern": {},
    "alpha_pattern": {},
    "lora_bias": False,
    "layer_replication": None,
}


def test_convert_config_literal():
    """Check convert config literal."""
    present = {"wq_du", "wk_dv", "wv_dv", "wo_ud", "gate_attn"}
    assert convert_config(PEFT_CFG, present) == {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": "thinkingmachines/Inkling-2-Mini",
        "r": 16,
        "lora_alpha": 32,
        "use_rslora": False,
        "target_modules": ["gate_attn", "wk_dv", "wo_ud", "wq_du", "wv_dv"],
        "bias": "none",
        "modules_to_save": None,
        "use_dora": False,
        "lora_bias": False,
        "init_lora_weights": True,
        "rank_pattern": {},
        "alpha_pattern": {},
    }


@pytest.mark.parametrize(
    "change",
    [
        {"peft_type": "IA3"},
        {"use_dora": True},
        {"lora_bias": True},
        {"bias": "all"},
        {"modules_to_save": ["lm_head"]},
        {"rank_pattern": {"x": 8}},
        {"alpha_pattern": {"x": 8}},
        {"init_lora_weights": "pissa"},
        {"layer_replication": [[0, 2]]},
        {"alora_invocation_tokens": [1]},
    ],
)
def test_convert_config_rejects(change):
    """Check convert config rejects."""
    with pytest.raises(ValueError, match="adapter config not supported by vLLM"):
        convert_config({**PEFT_CFG, **change}, {"wq_du"})


def test_convert_adapter_roundtrip(tmp_path: Path):
    """Check convert adapter roundtrip."""
    src = tmp_path / "src"
    src.mkdir()
    t = synthetic()
    save_file(t, str(src / "adapter_model.safetensors"))
    (src / "adapter_config.json").write_text(json.dumps({**PEFT_CFG, "r": R}))
    dst = tmp_path / "dst"
    summary = convert_adapter(src, dst)
    assert {p.name for p in dst.iterdir()} == {"adapter_model.safetensors", "adapter_config.json", "conversion.json"}
    assert summary == json.loads((dst / "conversion.json").read_text())
    assert summary["num_tensors"] == 240
    assert summary["source_sha256"] == hashlib.sha256((src / "adapter_model.safetensors").read_bytes()).hexdigest()
    assert summary["layers"] == list(range(24))
    assert summary["agent_config"] is None
    assert len(load_file(str(dst / "adapter_model.safetensors"))) == 240
    with pytest.raises(ValueError, match="exists and is not empty"):
        convert_adapter(src, dst)
    with pytest.raises(ValueError, match="destination must differ"):
        convert_adapter(src, src)


def test_scaled_pair(monkeypatch):
    """Check scaled pair."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "make_random_adapter.py"
    spec = importlib.util.spec_from_file_location("make_random_adapter", path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    w = torch.randn(64, 32).to(torch.bfloat16)
    a, b = mod.scaled_pair(w, 4, 8, 0.05, torch.Generator().manual_seed(1))
    ratio = ((8 / 4) * (b.double() @ a.double())).norm() / w.double().norm()
    assert abs(float(ratio) - 0.05) < 1e-5
    a2, b2 = mod.scaled_pair(w, 4, 8, 0.05, torch.Generator().manual_seed(1))
    assert torch.equal(a, a2)
    assert torch.equal(b, b2)
    assert a.dtype == torch.float32
    assert b.abs().sum() > 0
