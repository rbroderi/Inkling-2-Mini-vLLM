import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from safetensors.torch import save_file

_spec = importlib.util.spec_from_file_location(
    "merge_adapters_cat", Path(__file__).resolve().parents[1] / "scripts" / "merge_adapters_cat.py"
)
mac = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mac)

K = "base_model.model.layers.0.q_proj.linear"


def _adapter(path: Path, seed: int, r: int = 4, alpha: int = 8) -> dict:
    g = torch.Generator().manual_seed(seed)
    t = {f"{K}.lora_A.weight": torch.randn(r, 6, generator=g), f"{K}.lora_B.weight": torch.randn(5, r, generator=g)}
    path.mkdir()
    save_file(t, str(path / "adapter_model.safetensors"))
    (path / "adapter_config.json").write_text(json.dumps({"r": r, "lora_alpha": alpha, "target_modules": "x"}))
    return t


def test_cat_merge_equals_weighted_sum_of_deltas(tmp_path, monkeypatch):
    a, b = _adapter(tmp_path / "a", 0), _adapter(tmp_path / "b", 1)
    out = tmp_path / "m"
    monkeypatch.setattr(
        sys, "argv", ["x", str(out), str(tmp_path / "a"), str(tmp_path / "b"), "--weights", "0.5", "0.5"]
    )
    mac.main()
    m = load_file(str(out / "adapter_model.safetensors"))
    cfg = json.loads((out / "adapter_config.json").read_text())
    assert cfg["r"] == 8 and cfg["lora_alpha"] == 16  # scaling alpha / r stays 2

    def delta(t, scale):
        return scale * t[f"{K}.lora_B.weight"] @ t[f"{K}.lora_A.weight"]

    expected = 0.5 * delta(a, 2.0) + 0.5 * delta(b, 2.0)
    assert torch.allclose(delta(m, cfg["lora_alpha"] / cfg["r"]), expected, atol=1e-5)


def test_mismatched_scaling_is_refused(tmp_path, monkeypatch):
    _adapter(tmp_path / "a", 0, r=4, alpha=8)
    _adapter(tmp_path / "b", 1, r=4, alpha=4)
    monkeypatch.setattr(sys, "argv", ["x", str(tmp_path / "m"), str(tmp_path / "a"), str(tmp_path / "b")])
    with pytest.raises(SystemExit, match="alpha / r"):
        mac.main()
