"""TRL's labelled-positions LM head (add_fused_lm_head) against the model's own loss, and the qlora guard."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
trl_utils = pytest.importorskip("trl.trainer.utils")

from inkling_qlora.train import assert_fused_head  # noqa: E402

VOCAB, SEQ = 50, 9


def make_gpt2():
    cfg = transformers.GPT2Config(
        n_layer=1, n_embd=16, n_head=2, vocab_size=VOCAB, n_positions=32, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0
    )
    return transformers.GPT2LMHeadModel(cfg)


def make_gemma2():
    cfg = transformers.Gemma2Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=VOCAB,
        final_logit_softcapping=30.0,
        attn_logit_softcapping=None,
        sliding_window=8,
        max_position_embeddings=32,
        layer_types=["full_attention"],
    )
    model = transformers.Gemma2ForCausalLM(cfg)
    model.logit_softcap = SimpleNamespace(softcap=30.0)  # Inkling's model carries this module; Gemma2 does not
    return model


MAKERS = {"gpt2": make_gpt2, "gemma2": make_gemma2}


def make(name):
    torch.manual_seed(0)
    return MAKERS[name]().float().eval()


def batch(model):
    ids = torch.randint(0, VOCAB, (2, SEQ))
    labels = ids.clone()
    labels[:, :5] = -100  # prompt
    labels[1, 6:] = -100  # padded row: only positions 5 and 6 carry a label... then padding
    mask = torch.ones(2, SEQ, dtype=torch.long)
    mask[1, 7:] = 0
    # Gemma2's embedding is scaled inside the model; feed embeddings so the gradient has a common leaf.
    embeds = model.get_input_embeddings()(ids).detach().requires_grad_(True)
    return embeds, labels, mask


def run(model, fused, embeds, labels, mask):
    embeds = embeds.detach().clone().requires_grad_(True)
    out = model(inputs_embeds=embeds, attention_mask=mask, labels=labels, fused_lm_head=fused)
    out.loss.backward()
    return out.loss.detach(), embeds.grad


@pytest.mark.parametrize("name", sorted(MAKERS))
def test_fused_loss_and_gradient_match_model(name):
    model = make(name)
    trl_utils.add_fused_lm_head(model)
    embeds, labels, mask = batch(model)
    loss_full, grad_full = run(model, False, embeds, labels, mask)
    loss_fused, grad_fused = run(model, True, embeds, labels, mask)
    assert abs(loss_fused - loss_full) / abs(loss_full) <= 1e-5
    rel = (grad_fused - grad_full).norm() / grad_full.norm()
    assert rel <= 1e-4


def test_softcap_is_applied_by_fused_head():
    """Mutation guard: a fused head that dropped the softcap would give a different loss from the model's own."""
    model = make("gemma2")
    model.lm_head.weight.data.mul_(40.0)  # large logits, so tanh capping visibly changes the loss
    trl_utils.add_fused_lm_head(model)
    embeds, labels, mask = batch(model)
    loss_full, _ = run(model, False, embeds, labels, mask)
    loss_fused, _ = run(model, True, embeds, labels, mask)
    assert abs(loss_fused - loss_full) / abs(loss_full) <= 1e-5
    model.config.get_text_config().final_logit_softcapping = None
    ids = embeds.detach()
    uncapped = model.lm_head(model.model(inputs_embeds=ids, attention_mask=mask).last_hidden_state).float()
    capped = torch.tanh(uncapped / 30.0) * 30.0
    assert (uncapped - capped).abs().max() > 1.0  # the two heads really differ at this scale


@pytest.mark.parametrize("name", sorted(MAKERS))
def test_mean_is_over_labelled_tokens_only(name):
    model = make(name)
    trl_utils.add_fused_lm_head(model)
    embeds, labels, mask = batch(model)
    out = model(inputs_embeds=embeds, attention_mask=mask, labels=labels, fused_lm_head=True)
    n_labelled = int((labels[:, 1:] != -100).sum())
    assert out.label_mask.sum() == n_labelled
    assert torch.allclose(out.loss, -out.log_probs.sum() / n_labelled)


@pytest.mark.parametrize("name", sorted(MAKERS))
def test_no_labelled_position_gives_zero_loss(name):
    model = make(name)
    trl_utils.add_fused_lm_head(model)
    embeds, labels, mask = batch(model)
    labels = torch.full_like(labels, -100)
    loss, grad = run(model, True, embeds, labels, mask)
    assert torch.isfinite(loss) and loss.item() == 0.0
    assert torch.isfinite(grad).all()


def test_assert_fused_head_passes_on_patched_model():
    model = make("gemma2")
    trl_utils.add_fused_lm_head(model)
    assert_fused_head(model)


def test_assert_fused_head_raises_on_unpatched_model():
    with pytest.raises(RuntimeError, match="not TRL's fused LM head"):
        assert_fused_head(make("gemma2"))


def test_assert_fused_head_raises_when_patched_with_no_softcap():
    """Mutation: patched while the config said None, config restored to 30 afterwards. The closure holds None."""
    model = make("gemma2")
    text_config = model.config.get_text_config()
    text_config.final_logit_softcapping = None
    trl_utils.add_fused_lm_head(model)
    text_config.final_logit_softcapping = 30.0
    with pytest.raises(RuntimeError, match="softcap mismatch"):
        assert_fused_head(model)


def test_assert_fused_head_ignores_config_changes_after_patching():
    """The closure value is what counts: a correct patch passes whatever the config says later."""
    model = make("gemma2")
    trl_utils.add_fused_lm_head(model)
    model.config.get_text_config().final_logit_softcapping = 10.0
    assert_fused_head(model)


def test_assert_fused_head_raises_when_closure_cell_is_missing():
    model = make("gemma2")
    trl_utils.add_fused_lm_head(model)

    def other_forward(self, *args, **kwargs):
        return None

    other_forward.__code__ = other_forward.__code__.replace(co_name="_fused_forward")
    model.forward = other_forward.__get__(model)
    with pytest.raises(RuntimeError, match="cannot find final_logit_softcapping"):
        assert_fused_head(model)


def test_assert_fused_head_unwraps_peft_style_models():
    model = make("gemma2")
    trl_utils.add_fused_lm_head(model)
    wrapper = SimpleNamespace(get_base_model=lambda: model)
    assert_fused_head(wrapper)
