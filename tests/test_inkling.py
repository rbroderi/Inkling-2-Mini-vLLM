import pytest

torch = pytest.importorskip("torch")

from inkling_qlora.inkling import _embeddings_to_cpu  # noqa: E402


class NormedEmbedding(torch.nn.Embedding):
    """Shaped like Inkling2LoopNormedEmbedding: a lookup, then a norm with its own weight."""

    def __init__(self) -> None:
        super().__init__(50, 8, padding_idx=0)
        self.embed_norm = torch.nn.RMSNorm(8)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_norm(super().forward(input_ids))


def test_cpu_embeddings_match_and_leave_the_parameters() -> None:
    torch.manual_seed(0)
    model = torch.nn.Sequential(NormedEmbedding(), torch.nn.Linear(8, 3))
    ids = torch.tensor([[0, 5, 49, 7]])
    expected = model(ids)

    _embeddings_to_cpu(model[0])

    names = [n for n, _ in model.named_parameters()]
    assert "0.weight" not in names and "0.embed_norm.weight" in names
    assert next(model.parameters()).device == model[0].embed_norm.weight.device
    assert model[0].cpu_table.device.type == "cpu"
    emb = model[0]
    assert emb.weight is emb.cpu_table and not emb.weight.requires_grad
    assert not isinstance(emb.weight, torch.nn.Parameter)
    assert all(p is not emb.weight for p in model.parameters())
    assert all(b is not emb.weight for b in model.buffers())
    assert not any(k.startswith("0.weight") for k in model.state_dict()) and "0.cpu_table" not in model.state_dict()
    torch.testing.assert_close(model(ids), expected)


def test_cpu_embedding_weight_survives_the_bnb_override_lookup() -> None:
    bnb = pytest.importorskip("bitsandbytes")
    emb = NormedEmbedding()
    _embeddings_to_cpu(emb)
    manager = bnb.optim.GlobalOptimManager.get_instance()
    manager.register_module_override(emb, "weight", {"optim_bits": 32})
    try:
        # what Optimizer8bit.check_overrides does for each registered (module, attr, config)
        for module, attr, _config in manager.module_weight_config_triple:
            if module is emb:
                pmodule = getattr(module, attr)
                assert isinstance(pmodule, torch.Tensor)
                assert all(id(p) != id(pmodule) for p in emb.parameters())
    finally:
        manager.module_weight_config_triple = [t for t in manager.module_weight_config_triple if t[0] is not emb]


def test_cpu_embeddings_work_with_input_require_grads() -> None:
    # what enable_input_require_grads does for LoRA with gradient checkpointing: mark the embedding output as needing
    # gradients so they reach the adapters
    model = torch.nn.Sequential(NormedEmbedding(), torch.nn.Linear(8, 3))
    _embeddings_to_cpu(model[0])
    model[0].register_forward_hook(lambda _m, _i, out: out.requires_grad_(True))
    for p in model[1].parameters():
        p.requires_grad_(False)
    out = model(torch.tensor([[1, 2, 3]]))
    out.sum().backward()
    assert model[0].embed_norm.weight.grad is not None


# ---- fused experts integration (T4) ----

from types import SimpleNamespace  # noqa: E402

from inkling_qlora import inkling  # noqa: E402
from inkling_qlora.inkling import Quant4bitExperts, requantize_mxfp8_to_nvfp4  # noqa: E402
from inkling_qlora.inkling_nvfp4 import dequantize_mxfp8, dequantize_nvfp4  # noqa: E402

E, H, I = 4, 32, 16  # noqa: E741


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        n_routed_experts=E, hidden_size=H, moe_intermediate_size=I, mlp_gate_softcap=4.0, mlp_up_softcap=25.0
    )


def _random_packed(seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    packed = {}
    for name, rows, cols in (("gate_up", 2 * I, H), ("down", H, I)):
        codes = torch.randint(0, 256, (E, rows, cols // 2), generator=g, dtype=torch.uint8)
        scale = (torch.rand(E, rows, cols // 16, generator=g) * 3 + 0.5).to(torch.float8_e4m3fn)
        packed[name] = (codes, scale, torch.full((E,), 2.0**-6))
    return packed


def _experts(impl: str, packed: dict) -> Quant4bitExperts:
    module = Quant4bitExperts(_config())
    module.set_nvfp4_packed(packed, "cpu")
    module.impl = impl
    return module


def _route(tokens: int = 24, k: int = 3, seed: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    index = torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(tokens)])
    index[0, 0] = E  # a dropped slot ("no expert")
    return index, torch.softmax(torch.randn(tokens, k, generator=g), -1)


def _run(module: Quant4bitExperts, x: torch.Tensor, index: torch.Tensor, w: torch.Tensor):
    x = x.clone().requires_grad_(True)
    w = w.clone().requires_grad_(True)
    out = module(x, index, w)
    out.pow(2).sum().backward()
    return out.detach(), x.grad, w.grad


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm())


def test_fused_matches_loop_forward_and_backward() -> None:
    packed = _random_packed()
    x = torch.randn(24, H, generator=torch.Generator().manual_seed(2))
    index, weights = _route()
    ref = _run(_experts("loop", packed), x, index, weights)
    fused = _run(_experts("fused", packed), x, index, weights)
    assert _rel(fused[0], ref[0]) <= 1e-5
    assert _rel(fused[1], ref[1]) <= 1e-5
    assert _rel(fused[2], ref[2]) <= 1e-5


def test_set_nvfp4_keeps_buffers() -> None:
    g = torch.Generator().manual_seed(3)
    parts = {}
    for proj, rows, cols in (("w13", 2 * I, H), ("w2", H, I)):
        parts[f"{proj}_weight"] = torch.randint(0, 256, (E, rows, cols // 2), generator=g, dtype=torch.uint8)
        parts[f"{proj}_weight_scale"] = torch.rand(E, rows, cols // 16, generator=g).to(torch.float8_e4m3fn)
        parts[f"{proj}_weight_amax"] = torch.rand(E, generator=g) + 1
        parts[f"{proj}_weight_outer_scale_denominator"] = torch.full((E,), 2688.0)
    module = Quant4bitExperts(_config())
    module.set_nvfp4(parts, "cpu")
    assert module.storage == "nvfp4" and module.loaded
    for proj, name in (("w13", "gate_up"), ("w2", "down")):
        assert torch.equal(getattr(module, f"{name}_codes"), parts[f"{proj}_weight"])
        assert torch.equal(
            getattr(module, f"{name}_scale").view(torch.uint8), parts[f"{proj}_weight_scale"].view(torch.uint8)
        )
        expected = parts[f"{proj}_weight_amax"].float() / parts[f"{proj}_weight_outer_scale_denominator"].float()
        assert torch.equal(getattr(module, f"{name}_factor"), expected)
    assert module.byte_to_values.shape == (256, 2)


def test_nf4_storage_falls_back_to_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    module = Quant4bitExperts(_config())
    module.storage, module.impl = "nf4", "fused"
    called = []
    monkeypatch.setattr(module, "_forward_loop", lambda *a: called.append(a) or "loop")
    assert module(torch.zeros(2, H), torch.zeros(2, 1, dtype=torch.long), torch.ones(2, 1)) == "loop"
    assert len(called) == 1


def test_backend_is_cached_and_follows_buffer_replacement() -> None:
    module = _experts("fused", _random_packed())
    first = module._backend(torch.device("cpu"))
    assert module._backend(torch.device("cpu")) is first
    assert not any(n.startswith("_backend") for n, _ in module.named_buffers())
    module.gate_up_codes = module.gate_up_codes.clone()  # what .to() does: a new tensor replaces the buffer
    assert module._backend(torch.device("cpu")) is not first


def test_experts_impl_environment_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INKLING_EXPERTS_IMPL", raising=False)
    assert inkling.resolve_experts_impl(None) == "fused"  # the default since T7
    assert inkling.resolve_experts_impl("loop") == "loop"
    monkeypatch.setenv("INKLING_EXPERTS_IMPL", "loop")  # the rollback switch
    assert inkling.resolve_experts_impl(None) == "loop"
    assert inkling.resolve_experts_impl("fused") == "fused"  # an explicit argument wins in the loader itself
    monkeypatch.setattr(inkling.os, "environ", {"INKLING_EXPERTS_IMPL": "bogus"})
    with pytest.raises(ValueError):
        inkling.load_inkling_4bit()
    with pytest.raises(ValueError):
        inkling.load_inkling_4bit(experts_impl="fast")
    with pytest.raises(ValueError):
        inkling.load_inkling_4bit(experts_impl="loop", mxfp8_storage="int4")
    # a valid value passes the validation (then needs the Inkling transformers build, which is absent or slow here)
    monkeypatch.setattr(inkling.os, "environ", {"INKLING_EXPERTS_IMPL": "fused"})
    monkeypatch.setitem(__import__("sys").modules, "transformers.models.inkling2_loop.modeling_inkling2_loop", None)
    with pytest.raises(ImportError):
        inkling.load_inkling_4bit()


def test_banded_attention_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INKLING_BANDED_ATTENTION", raising=False)
    assert inkling.BANDED_ATTENTION_DEFAULT is True  # A1's gate passed (section 10.4)
    assert inkling.resolve_banded_attention(None) is True  # unset -> the default
    monkeypatch.setenv("INKLING_BANDED_ATTENTION", "0")  # the rollback switch
    assert inkling.resolve_banded_attention(None) is False
    monkeypatch.setenv("INKLING_BANDED_ATTENTION", "1")
    assert inkling.resolve_banded_attention(None) is True
    assert inkling.resolve_banded_attention(False) is False  # an explicit argument wins over the variable
    monkeypatch.delenv("INKLING_BANDED_ATTENTION")
    assert inkling.resolve_banded_attention(False) is False
    assert inkling.resolve_banded_attention(True) is True
    monkeypatch.setattr(inkling, "BANDED_ATTENTION_DEFAULT", False)  # the default is read at call time
    assert inkling.resolve_banded_attention(None) is False
    monkeypatch.setenv("INKLING_BANDED_ATTENTION", "1")
    assert inkling.resolve_banded_attention(None) is True
    for bad in ("", "true", "on", "2"):
        monkeypatch.setenv("INKLING_BANDED_ATTENTION", bad)
        with pytest.raises(ValueError):
            inkling.resolve_banded_attention(None)
        with pytest.raises(ValueError):  # the loader rejects it before it touches the model
            inkling.load_inkling_4bit(device="cpu")


def test_mxfp8_layer_requantized_to_nvfp4_keeps_checkpoint_row_order() -> None:
    g = torch.Generator().manual_seed(4)
    raw = {}
    for proj, rows, cols in (("w13", 2 * I, 64), ("w2", 64, I * 2)):
        raw[f"{proj}_weight"] = (torch.randn(E, rows, cols, generator=g) * 3).to(torch.float8_e4m3fn)
        raw[f"{proj}_weight_scale"] = torch.randint(120, 127, (E, rows, cols // 32), generator=g, dtype=torch.uint8)
    packed = requantize_mxfp8_to_nvfp4(raw, "cpu")
    for proj, name in (("w13", "gate_up"), ("w2", "down")):
        codes, scale, factor = packed[name]
        original = dequantize_mxfp8(raw[f"{proj}_weight"], raw[f"{proj}_weight_scale"])
        assert factor.shape == (E,)
        for e in range(E):
            assert _rel(dequantize_nvfp4(codes[e], scale[e], factor[e]), original[e]) < 0.12

    # the module's loop weight() de-interleaves it: gate rows (even checkpoint rows) come first
    module = Quant4bitExperts(
        SimpleNamespace(
            n_routed_experts=E, hidden_size=64, moe_intermediate_size=I, mlp_gate_softcap=4.0, mlp_up_softcap=25.0
        )
    )
    module.set_nvfp4_packed(packed, "cpu")
    w = module.weight("gate_up", 1).float()
    original = dequantize_mxfp8(raw["w13_weight"], raw["w13_weight_scale"])[1]
    assert _rel(w[:I], original[0::2]) < 0.12 and _rel(w[I:], original[1::2]) < 0.12




def test_to_drops_backend_cache_and_keeps_packed_dtypes() -> None:
    module = _experts("fused", _random_packed())
    old = module._backend(torch.device("cpu"))
    assert module._backend_cache is not None
    module.to(torch.device("cpu"))
    assert module._backend_cache is None
    assert module._backend(torch.device("cpu")) is not old

    factor = module.gate_up_factor.clone()
    module.to(torch.bfloat16)  # a dtype cast must not touch the packed buffers
    module.float()
    assert module._backend_cache is None
    for name in ("gate_up", "down"):
        assert getattr(module, f"{name}_codes").dtype == torch.uint8
        assert getattr(module, f"{name}_scale").dtype == torch.float8_e4m3fn
        assert getattr(module, f"{name}_factor").dtype == torch.float32
    assert torch.equal(module.gate_up_factor, factor)
    assert module.byte_to_values.dtype == torch.float32  # the loop's lookup table follows .float()/.to(dtype)
    x = torch.randn(8, H)
    index, weights = _route(8)
    assert torch.isfinite(module(x, index, weights)).all()


def test_qlora_env_overrides_config_impl(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    from inkling_qlora.train import inkling_experts_impl

    cfg = "fused"
    monkeypatch.delenv("INKLING_EXPERTS_IMPL", raising=False)
    assert inkling_experts_impl(cfg) == "fused"
    monkeypatch.setenv("INKLING_EXPERTS_IMPL", "loop")
    with caplog.at_level("INFO"):
        assert inkling_experts_impl(cfg) == "loop"
    assert "INKLING_EXPERTS_IMPL=loop overrides" in caplog.text






def test_loader_pins_the_inkling_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    import huggingface_hub
    import transformers

    calls: dict[str, list] = {"snapshot": [], "from_pretrained": []}
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda mid, revision=None: calls["snapshot"].append((mid, revision)) or ".",
    )
    monkeypatch.delenv("INKLING_REVISION", raising=False)
    model_id = inkling.INKLING_MODEL_ID
    inkling._snapshot(model_id)
    inkling._snapshot("someone/else")
    monkeypatch.setenv("INKLING_REVISION", "abc123")
    inkling._snapshot(model_id)
    assert calls["snapshot"] == [(model_id, inkling.INKLING_REVISION), ("someone/else", None), (model_id, "abc123")]

    class Stop(Exception):
        pass

    class FakeAuto:
        @staticmethod
        def from_pretrained(mid, **kwargs):
            calls["from_pretrained"].append((mid, kwargs.get("revision")))
            raise Stop

    fake = types.ModuleType("modeling_inkling2_loop")
    fake.Inkling2LoopExperts = object
    fake.AutoModel = object
    fake.Inkling2LoopPreTrainedModel = type("Base", (), {"_init_weights": lambda self, module: None})
    package = types.ModuleType("inkling2_loop")
    package.modeling_inkling2_loop = fake
    monkeypatch.setitem(sys.modules, "transformers.models.inkling2_loop", package)
    monkeypatch.setitem(sys.modules, "transformers.models.inkling2_loop.modeling_inkling2_loop", fake)
    monkeypatch.setattr(transformers, "AutoModelForImageTextToText", FakeAuto)
    for kwargs, expected in (({}, "abc123"), ({"revision": "explicit"}, "explicit")):
        with pytest.raises(Stop):
            inkling.load_inkling_4bit(model_id, device="cpu", experts_impl="loop", **kwargs)
        assert calls["from_pretrained"][-1] == (model_id, expected)
    monkeypatch.delenv("INKLING_REVISION")
    with pytest.raises(Stop):
        inkling.load_inkling_4bit(model_id, device="cpu", experts_impl="loop")
    assert calls["from_pretrained"][-1] == (model_id, inkling.INKLING_REVISION)
    assert fake.Inkling2LoopExperts is object  # the loader restored the patched names
