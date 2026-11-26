"""GPU check of TRL's labelled-positions head on the real Inkling model, under the GPU lock (ticket H1).

Part 1 (eval, no_grad): for the 4 examples of ab_step.py, capture last_hidden_state ONCE with a forward hook, then
compute the loss three ways from that one tensor:
  full   - the model's own head: softcap(lm_head(h).float()) and the model's loss_function (bf16 logits rounding)
  fused  - ChunkedLogProbFunction exactly as TRL's _fused_forward calls it (labelled positions only)
  golden - softcap(h.float() @ W.float().T) with a float32 cross-entropy on the labelled positions
Bars per example: (a) |fused - full| <= 2e-3 (the same loss computed two ways) and (b) |fused - golden| <=
1.25 * |full - golden| + 5e-4 (fused no worse than the model's own head).
Changed after the first GPU run: the absolute bar |fused - golden| <= 2e-3 was missed by the reference full head
too (|full - golden| up to 3.1e-3), because both heads start from the bf16 hidden state while golden is float32.
Part 2 (train mode, LoRA, gradient checkpointing as in ab_step.py): max_memory_allocated of one step with each head
at the example nearest 1002 tokens and the one nearest 495 tokens (experiment E2). Printed, not asserted.

Run: cd /path/to/Inkling-2-Mini-vLLM && HF_HUB_OFFLINE=1 PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
     ~/inkling-venv/bin/python gpu_checks/check_fused_head.py
"""

import os
import random

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer
from trl.kernels.chunked_logprob import ChunkedLogProbFunction
from trl.trainer.utils import _CHUNKED_LOGPROB_CHUNK_SIZE, add_fused_lm_head

from gpu_checks.gpu_guard import gpu_session
from inkling_qlora import inkling as ik
from inkling_qlora.data import read_records

FUSED_FULL_TOL, REL, ABS = 2e-3, 1.25, 5e-4
FULL_TOL = 3e-2  # Part 2: train-mode loss agreement
GiB = 2**30

with gpu_session(min_free_mib=21000):
    MODEL = "thinkingmachines/Inkling-2-Mini"
    model = ik.load_inkling_4bit(MODEL, device="cuda", experts_impl="fused", mxfp8_storage="nvfp4")
    add_fused_lm_head(
        model, outputs=("log_probs", "entropy", "is_top1")
    )  # SFTTrainer's outputs  # before get_peft_model, as SFTTrainer does
    tok = AutoTokenizer.from_pretrained(MODEL)
    examples = read_records(os.environ.get("INKLING_CHECK_DATA", "data/train.jsonl"))
    random.Random(0).shuffle(examples)
    records = examples[:40]

    def encode(rec):
        prompt = tok.apply_chat_template(
            rec["prompt"], add_generation_prompt=True, tokenize=False, **ik.INKLING_CHAT_TEMPLATE_KWARGS
        )
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tok(rec["completion"][0]["content"] + (tok.eos_token or ""), add_special_tokens=False)["input_ids"]
        return (
            torch.tensor([p_ids + c_ids], device="cuda"),
            torch.tensor([[-100] * len(p_ids) + c_ids], device="cuda"),
        )

    encoded = [encode(r) for r in records]
    ab_examples = encoded[2:6]  # the 4 timed examples of ab_step.py

    # ---- Part 1: the three heads from one captured hidden state
    captured: dict[str, torch.Tensor] = {}
    hook = model.model.register_forward_hook(lambda mod, args, out: captured.__setitem__("h", out.last_hidden_state))
    softcap = model.config.get_text_config().final_logit_softcapping
    assert softcap == model.logit_softcap.softcap == 30.0, softcap
    weight = model.lm_head.weight
    model.eval()
    failures = []
    worst_fused = worst_full = 0.0
    for i, (ids, labels) in enumerate(ab_examples):
        with torch.no_grad():
            model(input_ids=ids)  # one backbone forward; the hook keeps its hidden state
            h = captured["h"][:, :-1]  # the heads below all consume this same tensor
            shifted = labels[:, 1:]  # the harness passes unshifted labels; shifted exactly once, here
            mask = shifted != -100
            n = int(mask.sum())
            logits = model.logit_softcap(model.lm_head(captured["h"]).float())
            full = model.loss_function(logits=logits, labels=labels, vocab_size=logits.shape[-1]).item()
            per_token = ChunkedLogProbFunction.apply(
                h[mask], weight, None, shifted[mask], 1.0, _CHUNKED_LOGPROB_CHUNK_SIZE, softcap, 1.0, ("log_probs",),
                None, None,
            )  # fmt: skip
            fused = (-per_token[0].sum() / n).item()
            z = h[mask].float() @ weight.float().T
            z = torch.tanh(z / softcap) * softcap
            golden = F.cross_entropy(z, shifted[mask], reduction="mean").item()
        d_fused, d_full = abs(fused - golden), abs(full - golden)
        worst_fused, worst_full = max(worst_fused, d_fused), max(worst_full, d_full)
        d_ff = abs(fused - full)
        if d_ff > FUSED_FULL_TOL:
            failures.append(f"example {i}: |fused - full| {d_ff:.2e} > {FUSED_FULL_TOL:g}")
        if d_fused > REL * d_full + ABS:
            failures.append(f"example {i}: |fused - golden| {d_fused:.2e} > {REL} * {d_full:.2e} + {ABS:g}")
        print(
            f"example {i}: {ids.shape[1]} tokens, {n} labelled; golden {golden:.5f} fused {fused:.5f} "
            f"full {full:.5f}; |fused-golden| {d_fused:.2e}, |full-golden| {d_full:.2e}, |fused-full| {d_ff:.2e}",
            flush=True,
        )
    hook.remove()
    assert not failures, failures
    print(f"PART 1 PASS: worst |fused - golden| {worst_fused:.2e}, worst |full - golden| {worst_full:.2e}", flush=True)

    # ---- Part 2: training peak per head (E2)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(
        model, LoraConfig(r=16, lora_alpha=32, target_modules=ik.INKLING_LORA_TARGETS, task_type="CAUSAL_LM")
    )
    model.train()

    def pick(target):
        return min(encoded, key=lambda e: abs(e[0].shape[1] - target))

    def step(ids, labels, fused):
        kw = {"fused_lm_head": True} if fused else {}
        loss = model(input_ids=ids, labels=labels, **kw).loss
        loss.backward()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        return loss.item()

    short, long_ = pick(495), pick(1002)
    for fused in (False, True):  # warm-up in each mode (kernel compilation, allocator)
        step(*short, fused)
    losses: dict[tuple[int, bool], float] = {}
    for ids, labels in (long_, short):
        for fused in (False, True):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            loss = step(ids, labels, fused)
            losses[(ids.shape[1], fused)] = loss
            print(
                f"{ids.shape[1]:5d} tokens, head {'fused' if fused else 'full ':5s}: loss {loss:.4f}, "
                f"max_memory_allocated {torch.cuda.max_memory_allocated() / GiB:.2f} GiB",
                flush=True,
            )
    for n_tokens in {long_[0].shape[1], short[0].shape[1]}:
        # LoRA B starts at zero, so both heads see the same model: the trainer's own fused wiring must agree
        diff = abs(losses[(n_tokens, True)] - losses[(n_tokens, False)])
        assert diff <= FULL_TOL, f"{n_tokens} tokens: train-mode fused vs full loss differ by {diff:.3e}"
    print("PART 2 PASS: train-mode fused and full losses agree", flush=True)
    print("PASS", flush=True)
