"""QLoRA fine-tuning of Inkling 2 Mini on one 24 GB Ampere GPU (RTX 3090 / A5000 / A6000 / A100).

    python -m inkling_qlora.train --train data/train.jsonl --validation data/validation.jsonl --output out/my-adapter

Data: JSONL in TRL's conversational prompt/completion format (see :mod:`inkling_qlora.data`); the loss is on the
completion only, and TRL's fused labelled-positions head is used so the full-vocabulary logits are never built.

What is specific to Inkling here, compared with a plain TRL SFT script:

* the model comes from :func:`inkling_qlora.inkling.load_inkling_4bit`: the NVFP4/MXFP8 experts stay packed and
  frozen (fused Triton kernels), attention/router/embeddings are bf16, the vision tower is dropped;
* LoRA goes on the language model's attention projections (:data:`INKLING_LORA_TARGETS`), not the experts;
* the chat template runs with ``reasoning_effort="none"``;
* optional length bucketing (similar lengths share a batch, so batch 2 wastes little padding);
* wall-clock checkpoints and automatic resume from the latest ``checkpoint-N``, for long runs on machines that hang
  or get preempted.

The defaults are the settings the 40k-example judges were trained with (~17 s per optimizer step of 16 examples at
up to 1024 tokens on a 3090, ~12 h for 40k examples).
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

from inkling_qlora.data import read_records

log = logging.getLogger(__name__)

INKLING_MODEL_ID = "thinkingmachines/Inkling-2-Mini"


def inkling_experts_impl(configured: str) -> str:
    """``INKLING_EXPERTS_IMPL`` when set (the rollback switch), else the configured experts implementation."""
    env = os.environ.get("INKLING_EXPERTS_IMPL")
    if env:
        log.info("INKLING_EXPERTS_IMPL=%s overrides experts_impl=%s", env, configured)
        return env
    return configured


def assert_fused_head(model: Any) -> None:
    """Raise RuntimeError unless TRL's labelled-positions head is active on a (PEFT-wrapped) model.

    SFTTrainer patches the model with ``trl.trainer.utils.add_fused_lm_head``; this guards that the patch is there
    and that the softcap it captured is the model's own (a silent ``None`` would change the loss).
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    forward = getattr(base.forward, "__func__", base.forward)
    # functools.wraps copies __name__ from the original forward, so the patched function is recognised by its code.
    code_name = getattr(getattr(forward, "__code__", None), "co_name", None)
    if code_name != "_fused_forward":
        raise RuntimeError(
            f"the model's forward is not TRL's fused LM head (code name {code_name!r}): the loss would run the "
            "full-vocabulary head"
        )
    names = forward.__code__.co_freevars
    if "final_logit_softcapping" not in names or forward.__closure__ is None:
        raise RuntimeError("cannot find final_logit_softcapping in the fused forward's closure (TRL version change?)")
    patched_cap = forward.__closure__[names.index("final_logit_softcapping")].cell_contents
    model_cap = getattr(getattr(base, "logit_softcap", None), "softcap", None)
    if patched_cap != model_cap:
        raise RuntimeError(
            f"fused head softcap mismatch: the patched forward uses {patched_cap!r}, the model's logit_softcap is "
            f"{model_cap!r}"
        )


def resume_checkpoint(output_dir: Path, enabled: bool) -> str | None:
    """The latest ``checkpoint-N`` directory in ``output_dir`` to resume from, or None (off, or none yet)."""
    if not enabled or not Path(output_dir).is_dir():
        return None
    from transformers.trainer_utils import get_last_checkpoint

    return get_last_checkpoint(str(output_dir))


def timed_save_callback(minutes: float, clock: Any = None) -> Any:
    """A ``TrainerCallback`` that asks for a checkpoint whenever ``minutes`` of wall-clock time pass since the last."""
    import time

    from transformers import TrainerCallback

    now = clock or time.monotonic
    interval = minutes * 60.0

    class TimedSave(TrainerCallback):
        def __init__(self) -> None:
            self.last = now()

        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            self.last = now()

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            if now() - self.last >= interval:
                control.should_save = True
                self.last = now()
            return control

    return TimedSave()


def bucketed_trainer_class(base: type, *, window_batches: int, seed: int) -> type:
    """A subclass of ``base`` (``SFTTrainer``) whose training sampler is a :class:`BucketedSampler`.

    Lengths come from the ``input_ids`` column of the tokenized training set; the Trainer calls ``set_epoch`` on the
    sampler each epoch, so a resume replays the right order. Evaluation batches are unaffected.
    """
    from inkling_qlora.bucketing import BucketedSampler

    class BucketedSFTTrainer(base):  # type: ignore[misc, valid-type]
        def _get_train_sampler(self, train_dataset: Any = None) -> Any:
            ds = train_dataset if train_dataset is not None else self.train_dataset
            if "input_ids" not in getattr(ds, "column_names", ()):
                raise ValueError("length bucketing needs a tokenized training set with an input_ids column")
            batch_size = getattr(self, "_train_batch_size", None) or self.args.train_batch_size
            return BucketedSampler([len(x) for x in ds["input_ids"]], batch_size, window_batches, seed)

    return BucketedSFTTrainer


def _warmup_kwargs(cls: Any, ratio: float) -> dict[str, Any]:
    """Warmup as a fraction of steps: transformers 5 dropped ``warmup_ratio`` (``warmup_steps`` < 1 is a fraction)."""
    if "warmup_ratio" in inspect.signature(cls.__init__).parameters:
        return {"warmup_ratio": ratio}
    if not 0 <= ratio < 1:
        raise ValueError(f"warmup_ratio must be in [0, 1), got {ratio}")
    return {"warmup_steps": ratio}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="QLoRA fine-tuning of Inkling 2 Mini on one 24 GB Ampere GPU")
    ap.add_argument("--train", type=Path, required=True, help="training JSONL (prompt/completion message lists)")
    ap.add_argument("--validation", type=Path, help="validation JSONL (optional)")
    ap.add_argument("--output", type=Path, required=True, help="output dir (checkpoints, final adapter)")
    ap.add_argument("--model", default=INKLING_MODEL_ID)
    ap.add_argument("--experts-impl", choices=["fused", "loop"], default="fused")
    ap.add_argument("--mxfp8-storage", choices=["nvfp4", "nf4"], default="nvfp4")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1, help="> 0 stops early (smoke tests)")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-seq-length", type=int, default=1024)
    ap.add_argument("--no-bucketing", action="store_true", help="turn length bucketing off")
    ap.add_argument("--bucket-window-batches", type=int, default=32)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--scheduler", default="cosine")
    ap.add_argument("--max-grad-norm", type=float, default=0.3)
    ap.add_argument("--optim", default="paged_adamw_8bit")
    ap.add_argument("--eval-steps", type=int, default=250)
    ap.add_argument("--max-eval-examples", type=int, default=200)
    ap.add_argument("--save-steps", type=int, default=250)
    ap.add_argument("--save-total-limit", type=int, default=3)
    ap.add_argument("--save-every-minutes", type=float, default=30.0, help="also checkpoint on wall-clock time; 0 off")
    ap.add_argument("--no-resume", action="store_true", help="do not resume from the latest checkpoint in --output")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:  # pragma: no cover - needs the GPU stack
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from inkling_qlora.inkling import INKLING_CHAT_TEMPLATE_KWARGS, INKLING_LORA_TARGETS, load_inkling_4bit

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    a = parse_args(argv)
    a.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)

    tokenizer = AutoTokenizer.from_pretrained(a.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    experts_impl = inkling_experts_impl(a.experts_impl)
    model = load_inkling_4bit(a.model, device="cuda", experts_impl=experts_impl, mxfp8_storage=a.mxfp8_storage)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    peft_config = LoraConfig(
        r=a.lora_r,
        lora_alpha=a.lora_alpha,
        lora_dropout=a.lora_dropout,
        target_modules=INKLING_LORA_TARGETS,
        bias="none",
        task_type="CAUSAL_LM",
    )

    def rows(path: Path) -> list[dict[str, Any]]:
        return [
            {"prompt": r["prompt"], "completion": r["completion"], "chat_template_kwargs": INKLING_CHAT_TEMPLATE_KWARGS}
            for r in read_records(path)
        ]

    train = rows(a.train)
    validation = rows(a.validation) if a.validation else []
    if a.max_eval_examples and len(validation) > a.max_eval_examples:
        validation = random.Random(a.seed).sample(validation, a.max_eval_examples)
    eval_ds = Dataset.from_list(validation) if validation else None
    args = SFTConfig(
        output_dir=str(a.output),
        learning_rate=a.lr,
        num_train_epochs=a.epochs,
        max_steps=a.max_steps,
        per_device_train_batch_size=a.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=a.grad_accum,
        **_warmup_kwargs(SFTConfig, a.warmup_ratio),
        lr_scheduler_type=a.scheduler,
        max_grad_norm=a.max_grad_norm,
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=a.save_steps,
        save_total_limit=a.save_total_limit,
        eval_steps=a.eval_steps,
        eval_strategy="steps" if eval_ds is not None else "no",
        optim=a.optim,
        bf16=True,
        seed=a.seed,
        report_to=["none"],
        completion_only_loss=True,
        max_length=a.max_seq_length,
    )
    trainer_cls: type = SFTTrainer
    if not a.no_bucketing:
        trainer_cls = bucketed_trainer_class(SFTTrainer, window_batches=a.bucket_window_batches, seed=a.seed)
    trainer = trainer_cls(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(train),
        eval_dataset=eval_ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    if a.save_every_minutes:
        trainer.add_callback(timed_save_callback(a.save_every_minutes))
    assert_fused_head(trainer.model)
    checkpoint = resume_checkpoint(a.output, not a.no_resume)
    if checkpoint:
        log.info("resuming from %s", checkpoint)
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(str(a.output))
    tokenizer.save_pretrained(str(a.output))
    run = {**vars(a), "experts_impl": experts_impl, "resumed_from": checkpoint}
    (a.output / "train_config.json").write_text(json.dumps(run, indent=2, default=str), encoding="utf-8")
    return a.output


if __name__ == "__main__":
    main()
