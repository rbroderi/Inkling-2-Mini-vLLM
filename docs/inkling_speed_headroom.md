# Inkling-2-Mini QLoRA on one RTX 3090: speed headroom after the fused experts

Written 2026-11-22. Analysis only: nothing here was run on the GPU. Inputs are the code, the plan
(`docs/inkling_fused_experts_plan.md`), the existing logs in the scratchpad (`ab_nvfp4.log`, `tune_blocks.log`,
`rv_*.log`, `smoke_*.log`) and two CPU-only checks (checkpoint tensor sizes, token lengths of 400 training examples).
Every number marked "est." is an estimate from operation counts and two-point fits, not a measurement. Section 6
lists the experiments that would replace the estimates.

## 1. Answer

No, 1.72 s per example is not the floor. There is roughly a factor of 2 left.

- About 1.05-1.1 s per example is reachable with low-risk changes that touch no kernel (loss only at the labelled
  positions, batch 2 with length bucketing).
- About 0.8-0.9 s is a realistic floor with one more kernel ticket and batches sized by token count.
- Below about 0.6 s needs work that isn't worth it on this card (a custom attention kernel, caching expert
  activations, compiling the layers). The arithmetic-only limit is about 0.35-0.4 s.

Most of the remaining time is no longer the experts. It is per-operation overhead that does not depend on the
sequence length, and batching divides it.

## 2. Facts found while reading (measured, from existing logs and CPU checks)

| Fact | Source |
|---|---|
| The 4 timed examples have 1002, 488, 507 and 495 tokens; fused times 2.5, 1.6, 1.4, 1.4 s | `ab_nvfp4.log` |
| Training set (`factcheck_full`, 40,000 examples, sample of 400): total tokens mean 524, p50 489, p90 664, p99 1120, max 1825 | CPU tokenizer |
| The completion is 50-57 tokens (mean 53), 10% of the sequence | CPU tokenizer |
| Allocated after load 14.99 GiB; peak 18.26-18.28 GiB at 488-507 tokens and 20.09 GiB at 1002 tokens | `rv_*.log`, `smoke_*.log` |
| Device memory in use during training 22.1-22.5 GiB (the card has 24) | `smoke_*.log` |
| Loop and fused peaks are identical (20.08 GiB), although the loop keeps about 2 GiB more per layer | plan section 8 |
| Kernel cost per layer call per example at 6,000 rows: block_m 32: 12.28 ms, 64: 8.93 ms, 128: 14.05 ms | `tune_blocks.log` |
| Padded tokens over real tokens: batch 2 random 1.13, sorted by length 1.005; batch 4 random 1.28, sorted 1.016 | CPU |
| `liger_kernel` is installed in `~/inkling-venv`; `causal_conv1d`, `kernels`, `flash_attn` are not | CPU |
| The model class sets `_supports_sdpa = False` and `_supports_flash_attn = False`; attention is the eager path | modeling file |
| The MTP head and vision tower are not on the GPU (static non-expert weights: 1.27 GiB layers + 0.86 GiB output layer) | checkpoint headers |

Three consequences:

1. **1.72 s is the mean of an unusual mix.** One of the four examples is at the dataset's 98th percentile. On the
   real length distribution today's code runs at about 1.5 s per example (est.), about 16.7 h per 40k epoch.
2. **Time is roughly 0.46 s + 2.0 ms per token** (two-point fit through 497 tokens / 1.47 s and 1002 tokens / 2.5 s;
   the intercept is uncertain by about 0.2 s because the times are rounded to 0.1 s and the long example ran first).
   A third to a half of an average example's time does not depend on its length.
3. **The memory peak is in the loss, not in the layers.** The loop path holds about 2 GiB more per layer than the
   fused path, yet both peak at the same value, so the peak is somewhere they share. The peak grows by 3.6 MB per
   token, which matches full-vocabulary logits: 200,058 columns in bf16, then fp32, then the softcap and the
   log-softmax and their gradients. The model computes them for all positions although only 10% carry a label.

## 3. Where the time goes (est.)

For one example of 524 tokens (4,192 token-expert pairs per layer call, 48 layer calls, each run twice forward
because of gradient checkpointing and once backward):

| Part | s per example | Scales with | Basis |
|---|---|---|---|
| Expert kernels (6 calls per layer call) | 0.37 | row blocks, not rows | 8.93 ms x 48 at 6,000 rows = 0.43 s; fewer blocks at 4,192 rows |
| Expert glue in torch (sort, gather, SwiGLU, weights, combine) | 0.15-0.2 | tokens, plus fixed | about 25 operations per call on 20-40 MB tensors |
| Fixed per-operation overhead outside the experts | 0.5-0.6 | nothing (per step) | about 35,000 small operations at 15-18 microseconds each |
| Attention arithmetic on (24, T, T) tensors (bias gather, fp32 softmax with sinks) | 0.15-0.25 | T squared | about 12 passes over 26-52 MB per call |
| Dense GEMMs (q, k, v, gate, output projections, LoRA, router, output layer) | 0.1 | tokens | about 7 TFLOP |
| Full-vocabulary softcap and cross-entropy | 0.04-0.06 | tokens | about 10 passes over 420 MB |
| Short convolutions | 0.05 | tokens | 0.07 s measured on the longer mix |
| Total | about 1.4-1.6 | | measured 1.4-1.6 s at 488-507 tokens |

So the experts are now about 0.55 s of 1.5 s (35%), and the rest about 0.95 s. The plan's "0.98 s outside the
experts" slightly understates the rest: in `no_experts` mode the experts return zeros without a gradient, so the
backward pass through the router and the MLP input is skipped.

How the kernel time splits (est., from the block-size tuning): going from 32-row to 64-row blocks left the cost per
block unchanged (about 53-55 microseconds), so at 64 rows a block's cost is mostly decoding the weights, not the
multiply. An expert with 33 rows (the average at 524 tokens) and one with 64 cost the same. At 128 rows the cost per
block doubles, which the multiply alone doesn't explain; the kernel seems to fall off a code-generation cliff there.

## 4. Speed-ups, ranked by gain per effort

Gains are per example at 524 tokens against 1.5 s. They overlap; section 5 stacks them.

| # | Change | Gain (est.) | Cost | Risk |
|---|---|---|---|---|
| 1 | Loss only at labelled positions | 0.05 s, and 1.2-1.7 GiB of peak (2.5-3.5 GiB at 1,000 tokens) | Small: a forward wrapper for training | Low; identical loss |
| 2 | Batch 2 with length bucketing | 0.3-0.4 s | Small: config plus a sampler; needs #1 for memory | Low |
| 3 | Batches by token budget (up to 4 short examples) | another 0.15-0.2 s | Medium: custom batch sampler in the trainer | Low-medium (memory at long lengths) |
| 4 | Gather, SwiGLU, routing weight and slot scatter inside the kernel; drop the two host syncs per call | 0.08-0.12 s | Medium: one Opus kernel ticket, rerun all bars | Medium-low |
| 5 | A kernel whose one decode serves 128-256 rows | 0.05-0.1 s, only with #3 | Medium-high: kernel work and tuning | Medium |
| 6 | Cap `max_seq_length` at 1024; trim prompt boilerplate | 2 ms per token removed | Small | Changes the data; needs an eval |
| 7 | Cache `gu` to skip the experts' forward recompute | 0.1-0.17 s | Medium-high; about 0.9 GiB per 524 tokens in the batch | Medium-high (silent wrong gradients if the cache mismatches) |
| 8 | Attention: trim the bias table to T columns, or a fused kernel | 0.03 s (trim), up to 0.15 s (kernel) | Small / high | Low / medium |
| 9 | `torch.compile` on the layers | 0.1-0.3 s at batch 1, under 0.1 s after #3 | High | High |
| 10 | Fewer loop passes (3 to 2) | 0.35 s (25%) | Small | It is a different model; quality unknown |
| - | Selective or no gradient checkpointing | 0.01 s per layer call freed, each costing about 0.6 GiB | - | Not worth the memory |
| - | Lower precision in the transposed GEMMs | none available on sm86 | - | - |
| - | Liger, sdpa, flash attention, CUDA graphs, sequence packing | none or not applicable | - | - |

### 4.1 Loss only at the labelled positions (do first)

> **Correction (2026-11-22, found while writing the tickets in section 7).** TRL already does this in the real
> training path. `SFTTrainer` (1.18.2 in `~/inkling-venv`, 1.16.0 in the Qwen environment) patches the model with
> `add_fused_lm_head` and calls `model(..., fused_lm_head=True)`: only positions whose next-token label is not -100
> go through `lm_head`, in tiles, with the softcap read from `final_logit_softcapping` (30.0 in Inkling's config).
> The full-vocabulary peak and the 0.05 s are therefore properties of the harnesses (`ab_step.py`,
> `lora_smoke.py`, `check_lora_seeds.py`), which call `model(input_ids, labels)` directly. No Inkling run has gone
> through `SFTTrainer` yet, so nothing shows that the patched head works on Inkling. Item 1 becomes: prove it, guard
> it, and make the harnesses measure what training runs (ticket H1). The text below describes the model's own
> forward, not the trainer.

`Inkling2LoopForConditionalGeneration.forward` runs `lm_head` on every position, casts to fp32, applies the softcap
and hands everything to the cross-entropy. With completion-only loss, 471 of 524 positions are ignored. Selecting
the hidden states whose next-token label is not -100 before `lm_head` gives the same loss (the mean is over labelled
tokens either way) and the same gradients.

- Time: about 0.05 s per example.
- Memory: this is the peak today, so the peak falls to whatever the layers need, which is not known yet (it is at
  least 1.8 GiB lower than the loop's layer peak). This is what makes room for batch 2.
- Detail to handle: TRL's `SFTTrainer.compute_loss` computes token accuracy from `outputs.logits` unless
  `use_liger_kernel` is set, so the wrapper has to return something that code accepts, or the trainer's method has
  to be overridden. Evaluation must use the same wrapper.
- Liger's fused linear cross-entropy (it has a softcap argument) would also remove the peak by chunking, but it
  still multiplies all 524 positions, and Liger has no patch for `inkling2_loop`, so `use_liger_kernel=True` would
  do nothing or fail. Slicing is simpler and 10 times less work.

### 4.2 Batching (the largest gain)

The MoE block flattens (batch, tokens) before routing, so a batch of 2 already gives each expert twice the rows; no
change to the fused path is needed. What batching buys:

- **Fixed overhead halves.** About 0.5-0.6 s per step becomes 0.25-0.3 s per example at batch 2 and about 0.15 s at
  batch 4. This is the main gain and needs no kernel change.
- **Fuller blocks, modestly.** At 524 tokens an expert gets 33 rows on average in a 64-row block. Two examples give
  65 rows, so about half the experts need a second block: about 25% fewer blocks per example (est.; with uniform
  routing, less or more with the real skew). Four examples give 131 rows in 3 blocks: about 25% fewer again. The
  decode is not truly shared until the kernel handles more rows per decode (#5).
- **Bucketing is required.** Random batches of 2 waste 13% in padding and batches of 4 waste 28%; sorted by length
  the waste is 0.5% and 1.6%. Shuffle the order of the buckets, not their contents.

Memory (est.): with #1 done, a batch-1 step of 524 tokens should peak near 16.5-17.5 GiB. Each extra 524-token
example adds about 0.6-0.9 GiB (one layer's graph during its backward and recompute, plus the saved layer inputs).
Batch 2 fits. Batch 4 fits for short examples but not for long ones, because the attention tensors grow with T
squared: 4 examples of 1,100 tokens need about 4.4 times the attention memory of 4 examples of 524. Hence a token
budget (for instance batch x T at most 2,200 and batch x T squared at most 1.3 M) rather than a fixed batch size.
Only 22.1-22.5 of the 24 GiB are usable in practice, as the device figure shows.

Keep the effective batch at 16 examples: batch 2 x accumulation 8, or accumulate until 16 examples have been seen
when the batch size varies.

Sequence packing (several examples in one sequence) is not recommended: attention would need a block-diagonal mask
on top of the relative-position bias and the convolutions a `seq_idx`, and its attention memory grows with the
square of the packed length, where batching grows linearly.

### 4.3 Glue into the kernel

Four changes to the existing kernel, none of which changes its inner loop:

- read input rows through `token_idx` (one indirect load) instead of `index_select` into a (P, H) copy;
- a gate_up epilogue that applies the softcapped SwiGLU in registers and stores `a` (and `gu` when a gradient is
  needed); gate and up are adjacent rows of the same tile, so nothing needs to be regrouped. Rounding the
  accumulator to bf16 and back before the activation keeps it bit-compatible with today's path;
- a down epilogue that multiplies by the routing weight and writes each row to its `slot` row of the (T x K, H)
  buffer, which removes `index_copy_` and two casts (slots are unique, so it stays deterministic);
- remove the two host synchronisations per layer call (`int(counts.sum())` in `build_route_plan` and
  `int(plan.block_rows.max())` in `TritonBackend.gemm`): 192 forced syncs per example that stop the CPU from
  queueing work ahead of the GPU. The first can be replaced by using T x K rows always (sentinel rows are sorted
  last and no block covers them); the second is a guard that the plan's own `block_m` already guarantees.

Also possible in the same ticket: keep the per-expert `factor` out of the weight tile (multiply the accumulator
once at the end). Code value times block scale has at most 6 significant bits, so it is exact in bf16 and the
per-element fp32 multiply and rounding disappear. This changes the result slightly (towards the fp32 golden path),
so it needs the section 9 criteria rerun. Gain unknown, perhaps 5-10% of the kernel.

### 4.4 A kernel that shares one decode across more rows

At batch 4 the multiply would cost about 1.1-1.4 ms per layer call per example at a good tensor-core rate, against
3.0 ms for the forward pair today; the rest is decode that could be shared. The existing 128-row configuration is
slower per row than the 64-row one, so this needs a different structure, for instance two 64-row accumulators fed
by one decoded tile inside the same loop. Worth doing only after #3, and only if experiment E4 confirms the
per-block cost model.

The plan's contingency (decode 16 experts to a bf16 scratch buffer, then `torch.bmm`) does not beat the current
kernel: writing 2 GB of bf16 per layer call per pass costs about as much as the in-register decode does now.

### 4.5 Gradient checkpointing

Checkpointing costs one extra forward, about 0.5 s of the 1.5 s. Turning it off needs about 0.6 GiB per layer call
at 524 tokens (the attention tensors, the fp32 residual stream, the experts' saved rows): about 29 GiB for 48
calls. It does not fit. Turning it off for the few layer calls that fit (5 calls in 3 GiB) saves about 0.05 s. The
same 3 GiB spent on batching saves about 0.4 s, so checkpointing stays on everywhere.

Skipping only the experts' recompute (#7) is the useful part of this idea. The second forward needs the experts'
output (the layer continues after it) and the backward needs `gu`. `y` is not needed: the routing-weight gradient
`(go * y).sum(-1)` equals `(down-transposed(go) * a).sum(-1)`, and the transposed product is computed anyway.
Caching `out` (2.4 MB per call) and `gu` (19 MB per call at 524 tokens) costs 0.9 GiB per example in the batch and
removes 2 of the 6 kernel calls plus half the glue. The cache must be matched to the right layer call (the loop
layers are called three times per forward) and cleared every step; a mismatch would give wrong gradients without
an error, so it needs a check of `top_k_index` on reuse. It competes with batching for memory. Do it last, if at
all.

### 4.6 Attention

- Flash attention cannot express the per-head additive position bias or the sink logit. SDPA can (bias and causal
  mask as a float mask, the sink as one extra zero key and value), but with a float mask on sm86 it runs the same
  arithmetic in C++, and its softmax would be bf16 where the model uses fp32. Gain about 0.05 s, with a numerical
  change in a model that section 8 showed is chaotic at the bf16 rounding level. Not recommended.
- Exact and small: `rel_logits` is (tokens, 24, 1024) whatever the sequence length; for sequences under 1,024 tokens
  only the first T columns of `proj` are ever gathered. Slicing `proj[:, :T]` halves that tensor and its
  scatter-add backward at 524 tokens. About 0.03 s (est.).
- A fused Triton attention kernel (bias gather, sink, fp32 softmax and the value product in one pass) would remove
  most of the T-squared cost, about 0.15 s at 524 tokens and 0.5 s at 1,000. High effort; the long examples that
  would benefit are 2% of the data.

### 4.7 `torch.compile` and CUDA graphs

The overhead that compiling would remove is the same overhead that batching divides. At batch 1 the gain could be
0.1-0.3 s; after batch 4, under 0.1 s. Against that: sequence lengths change every step (recompiles, or dynamic
shapes), the layers run under non-reentrant checkpointing with a custom autograd Function that launches a Triton
kernel (a graph break in every layer), the transformers build is a pull-request snapshot, and compile times are in
minutes. CUDA graphs (`reduce-overhead`) need static shapes and static memory. Not recommended before #1-#4, and
probably not after.

### 4.8 Precision of the transposed GEMMs

Already bf16 inputs and weights with fp32 accumulation, which is the fastest mode Ampere tensor cores have. There is
no fp8 on sm86, Triton's `tl.dot` has no int8 path, and quantising gradients would add noise to a chaotic model.
No headroom here.

### 4.9 Data

- Each token costs about 2 ms at batch 1 (about 1.5 ms after batching). The prompt averages 472 tokens; cutting 100
  tokens of fixed instructions saves about 13%. Nothing can be cached across examples, because LoRA changes every
  layer's activations.
- `max_seq_length` is 1536; about 1.5% of examples exceed 1,024 tokens. A cap of 1,024 costs those examples their
  tail (the answer is at the end, so truncation would have to drop passages, not the tail) and bounds the T-squared
  memory, which makes the token budget in #3 simpler.
- Fewer passages per example is a quality decision; the time saved is linear in tokens.

### 4.10 The three loop passes

Layers 6-17 run 3 times, so they are 36 of the 48 layer calls. Running 2 passes removes 12 calls, 25% of the layer
time (about 0.35 s); 1 pass removes 50%. `loop_num_passes` is a config value, so it is a one-line experiment, and
inference gets the same speed-up. But the checkpoint was trained with 3 fixed passes (`is_varloop` is false), so
this is a weaker base model, and the adapter would have to be trained and served with the same count. Only worth
considering if a zero-shot and a short-LoRA comparison show no loss on the judge sets (E6).

### 4.11 Not worth pursuing

- Embeddings on the CPU: one lookup of 524 rows and a 2.4 MB copy per step. Negligible.
- The optimizer (`paged_adamw_8bit` on LoRA weights, once per 16 examples). Negligible.
- Native Linux instead of WSL2 might reduce the per-launch overhead; the size of that is unknown, and batching
  removes most of what it would save.

## 5. Stacked estimate and floor

At the dataset's mean of 524 tokens:

| Stage | s per example | Hours per 40k epoch |
|---|---|---|
| Measured harness mix (one 1,002-token example in four) | 1.72 | 19.1 |
| Today on the real length distribution (est.) | 1.5 | 16.7 |
| + loss at labelled positions (#1) | 1.45 | 16.1 |
| + batch 2 with bucketing (#2) | 1.05-1.1 | 11.7-12.2 |
| + token-budget batches up to 4 (#3) | 0.9-0.95 | 10-10.5 |
| + glue in the kernel, syncs removed (#4) | 0.8-0.85 | 9-9.5 |
| + shared-decode kernel (#5) | 0.72-0.8 | 8-9 |
| + `gu` cache, attention kernel (#7, #8) | 0.55-0.65 | 6-7 |
| Arithmetic only (GEMMs at 100 TFLOP/s, decode shared, no overhead) | 0.35-0.4 | 4-4.5 |

**Realistic floor: 0.8-0.9 s per example, 9-10 h per 40k-example epoch**, reached by #1-#4. That is about half of
today's time and about half of Qwen3.6-27B's 1.9 s. The first two rows of work (#1, #2) get most of it for little
effort and no kernel change.

The uncertainty is mainly the size of the fixed overhead (0.46 s from a two-point fit, 0.5-0.6 s from operation
counts). If it is 0.3 s instead, batching gains about half as much and the floor is nearer 1.0 s.

## 6. Experiments that would confirm the top items

All under `tests/gpu/gpu_guard.gpu_session(min_free_mib=21000)`, GPU otherwise idle, `torch.cuda.synchronize()`
before and after each timed region, `time.perf_counter()`, 2 warm-up steps per shape, no `torch.profiler`. Fused
path, `--mxfp8-storage nvfp4`, no optimizer step, as in `ab_step.py`.

**E1. Time against length (settles the fixed overhead).** One example truncated to 16, 64, 256, 512, 768 and 1,024
tokens (keep the completion, cut the prompt), 5 timed steps each after warm-up at that length, median. Fit
a + b T + c T squared. Predicts: a = 0.45-0.6 s. Also record `max_memory_allocated` per length. About 3 minutes.

**E2. Where the peak is.** For 500 and 1,000 tokens: reset the peak counter, run the model body only
(`model.model(...)` and a backward from `last_hidden_state.float().sum()`), record the peak; then the full loss.
Predicts: the body's peak is 1.2-1.7 GiB (500 tokens) and 2.5-3.5 GiB (1,000) below the full peak. This gives the
memory that #1 frees and the layer peak that batching starts from.

**E3. Batch size (the main result).** With the loss sliced to labelled positions in the harness (a 10-line local
function, no repo change needed for the experiment): 16 examples sorted by length, run as 16 x 1, 8 x 2, 4 x 4 with
right padding and an attention mask; report s per example and peak memory for each. Repeat 4 x 4 with random
(unsorted) batches. Predicts: 1.45, 1.05-1.1 and 0.9-0.95 s per example; batch 4 peak under 20 GiB at about 500
tokens. Also run one batch of 4 at 1,000 tokens to find where it stops fitting.

**E4. Kernel cost per block.** Extend `tune_blocks.py`: capture `top_k_index` from a real forward at batch 1, 2 and
4 (layers 0, 12 and 23), and time the four kernels on those plans for `block_m` 64 and 128. Report the number of
blocks, microseconds per block and ms per example. Predicts: cost follows the block count (about 55 microseconds
per block over the six calls at 64 rows), about 25% fewer blocks per example at batch 2. If the cost follows rows
instead, #5 is pointless and the batching gain in the kernel is nil.

**E5. Glue and syncs.** Two runs of E1's 512-token case: (a) with `backend.gemm` wrapped in synchronising timers
(kernel total), and the autograd Function's forward and backward wrapped likewise in a separate run (experts
total); glue is the difference. (b) with the two host syncs removed by a local patch. Predicts: kernels 0.35-0.4 s,
glue 0.15-0.2 s, syncs worth 0-0.05 s. Timers inside the step perturb it, so use one wrapped region per run.

**E6 (only if #10 is of interest).** Zero-shot judge accuracy on 200 validation examples with `loop_num_passes` 3,
2 and 1 (inference only, fused path), then 200 LoRA steps at 2 passes against 3 with the same seed.

After any change to the fused path or the kernel (#4, #5, #7), rerun criteria (a)-(d) of the plan's section 9.

## 7. Tickets (decided 2026-11-22)

The user's decision: implement item 1 (loss at labelled positions) and item 4 (glue into the kernel, host syncs
removed); test item 2 (batch 2 with length bucketing) for quality before adopting it.

### 7.0 Common constraints, order and parallelism

- T7 is running on the GPU and edits `inkling.py` and `configs.py`. Until it finishes: no GPU work, and no edits to
  those two files. Code and CPU tests in other files can be written now.
- GPU work only inside `tests.gpu.gpu_guard.gpu_session` (`min_free_mib=21000` for full-model scripts, 4096 for
  kernel checks). Exit code 75 means wait; no retry loops. Timers use `torch.cuda.synchronize()` and
  `time.perf_counter()`. No `torch.profiler`. No installs, downloads or commits.
- CPU tests: `uv run pytest tests/<file>` on Windows (no Triton, no Inkling transformers there). GPU scripts:
  `cd /mnt/e/code/aarr && HF_HUB_OFFLINE=1 PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  ~/inkling-venv/bin/python <script>`. Kernel tests without a GPU: the same command with `TRITON_INTERPRET=1
  CUDA_VISIBLE_DEVICES=""` in WSL.
- Every ticket's diff is reviewed by a fresh Opus reviewer with the ticket's checklist.

| ID | Tier | Summary | Depends on | Files |
|---|---|---|---|---|
| H1 | sonnet | Prove and guard TRL's labelled-positions head on Inkling; harnesses use it | - (GPU part after T7) | `tests/test_fused_head.py`, `tests/gpu/check_fused_head.py`, `scripts/inkling/ab_step.py`, `scripts/inkling/lora_smoke.py`, `training/qlora.py` |
| K1 | opus | Gather, routing weight and slot scatter in the kernel; sync-free plan | - (GPU part after T7 and H1) | `training/inkling_kernels.py`, `training/inkling_triton.py`, `tests/test_inkling_kernels.py`, `tests/gpu/check_kernel.py` |
| K2 | opus | SwiGLU epilogue in the gate_up kernel | K1 | same files as K1 |
| B1 | sonnet | Bucketed sampler module with CPU tests | - | `training/bucketing.py`, `tests/test_bucketing.py` |
| B2 | sonnet | Quality-test script and its pre-gate | B1, H1 (GPU part after T7) | `scripts/inkling/bucketing_quality.py`, `tests/test_bucketing_quality.py` |
| B3 | sonnet | Run the quality test, apply the decision rule, record the result | B2, GPU free for about 8 h | this document (results section) |
| B4 | sonnet | Adopt batch 2 in the trainer, only if B3 passes | B3 pass, T7 finished | `training/configs.py`, `training/qlora.py`, `tests/test_training.py` |

Parallel now (disjoint files): H1, K1 and B1. B2 can start when B1's module exists. K2 follows K1 in the same
files. H1 and B4 both edit `qlora.py`, but B4 comes much later. GPU order after T7: H1 check (10 min), K1 checks
(40 min), K2 checks (40 min), B2 pre-gate (20 min), B3 (about 8 h, overnight). B3 does not depend on K1 or K2: both
arms run the same code, whichever is merged; the script records `git status --short` and the experts settings.

GPU hours in total: about 10 (H1 0.2, K1 0.7, K2 0.7, B2 0.3, B3 about 8).

---

### H1. The labelled-positions loss on Inkling: prove, guard, and use in the harnesses (sonnet)

> **Done on the CPU 2026-11-22; GPU check pending (after T7).** 21 CPU tests pass. The Opus review confirmed that TRL's SFTTrainer patches Inkling via add_fused_lm_head in both TRL versions, with the softcap from text_config, the unpadded vocab, a single shift, and the mean over labelled tokens; token accuracy reads `is_top1`, not logits (the worry in section 4.1 is out of date). Fixed and rechecked:
> - `assert_fused_head` now checks the softcap TRL captured in the patched forward's closure; a mutation with the closure set to None and the config at 30 now raises.
> - check_fused_head.py Part 2 asserts full vs fused within 3e-2.
> - The harnesses patch with SFTTrainer's outputs ("log_probs", "entropy", "is_top1").
>
> GPU next: `tests/gpu/check_fused_head.py`.
>
> **GPU check passed 2026-11-22** (WSL ~/gpu_h1b.log). Part 1's absolute bar (|fused−golden| ≤ 2e-3) was recalibrated by the lead after the first run: the model's own full head missed it too (up to 3.08e-3), since both differ from the fp32 golden by the bf16 rounding of the hidden state. New bars, passed:
> - |fused−full| ≤ 2e-3 (worst 1.65e-3);
> - |fused−golden| ≤ 1.25·|full−golden| + 5e-4.
>
> Part 2 (train mode, LoRA, checkpointing, fused experts): the losses agree within 3e-2. Peak memory: 20.50 → 18.39 GiB at 1,002 tokens, and 17.94 → 16.87 GiB at 495.

Tier reason: no new numerics; the work is tests, a guard and harness flags.

Reference behaviour: the model's own forward: `logits = softcap(lm_head(hidden).float())`, then
`ForCausalLMLoss` (labels shifted by one, mean over positions whose label is not -100). TRL's
`trl.trainer.utils.add_fused_lm_head` must give the same loss from the same hidden states.

Decisions:

1. No new loss code. The implementation is TRL's `add_fused_lm_head`, which `SFTTrainer.__init__` already applies
   for both loaders. `qlora.py`'s HF/Qwen path is not changed at all.
2. Guard in `qlora.py`, after `SFTTrainer(...)` is built and only when `cfg.loader == "inkling"`: a new
   module-level function `assert_fused_head(model)` that raises `RuntimeError` unless (a) the forward of
   `model.get_base_model()` (or `model` when it is not a PEFT model) is the patched one
   (`getattr(base.forward, "__func__", base.forward).__name__ == "_fused_forward"`), and (b)
   `base.config.get_text_config().final_logit_softcapping == base.logit_softcap.softcap`. It is called with
   `trainer.model`.
3. `scripts/inkling/ab_step.py` and `scripts/inkling/lora_smoke.py` get `--head full|fused`, default `full` (so
   T7's commands keep their meaning). With `fused`: call `add_fused_lm_head(model, outputs=("log_probs",))` right
   after `load_inkling_4bit` and before `get_peft_model`, and compute the loss as
   `model(input_ids=ids, labels=labels, fused_lm_head=True).loss`. The printed header states the head. All later
   tickets time with `--head fused`.
4. Tolerances. CPU, float32, tiny models: relative loss difference at most 1e-5, gradient of the hidden states
   relative difference at most 1e-4. GPU, real model, bf16, both heads fed the *same* captured
   `last_hidden_state`: per-example |fused - golden| at most 2e-3 and |full - golden| at most 3e-2, where golden is
   `softcap(hidden.float() @ lm_head.weight.float().T)` with float32 cross-entropy on the labelled positions. The
   model's own head rounds every logit to bf16 (a logit near 20 has a rounding step of 0.125), so the fused head is
   expected to be the closer one; "identical" means identical to the golden value within 2e-3.

Files and work:

- `tests/test_fused_head.py` (new, CPU, runs in the Windows environment with TRL 1.16.0):
  - a tiny `GPT2LMHeadModel` (`n_layer=1, n_embd=16, n_head=2, vocab_size=50`, built from a config, no download)
    and a tiny `Gemma2ForCausalLM` (hidden 16, 1 layer, `final_logit_softcapping=30.0`, vocabulary 50), each with
    labels whose first 5 of 9 positions are -100 and a second row padded with -100: fused loss equals the model's
    own `loss` within 1e-5; gradients with respect to the input embeddings agree within 1e-4;
  - a row with no labelled position gives a finite zero loss;
  - `assert_fused_head` passes on the patched model, raises on an unpatched one, and raises when
    `final_logit_softcapping` is changed after patching.
- `tests/gpu/check_fused_head.py` (new, `gpu_session(min_free_mib=21000)`): load with
  `experts_impl="fused", mxfp8_storage="nvfp4"`, `model.eval()`, `torch.no_grad()`; for 4 fixed examples (the
  `ab_step.py` ones) capture `last_hidden_state` once with a forward hook on `model.model`; compute full, fused
  (by calling `trl.kernels.chunked_logprob.ChunkedLogProbFunction` exactly as `_fused_forward` does) and golden
  from that one tensor; apply decision 4. Then, in train mode with LoRA and gradient checkpointing as in
  `ab_step.py`, report `max_memory_allocated` for one step with each head at the 1002-token and a 495-token
  example (this is experiment E2: it gives the real training peak).
- `qlora.py`: `assert_fused_head` and its one call.

Acceptance:

- CPU: `uv run pytest tests/test_fused_head.py tests/test_training.py` passes.
- GPU (after T7): `check_fused_head.py` passes decision 4 and prints the two peaks per length;
  `ab_step.py --modes fused --mxfp8-storage nvfp4 --head fused` runs and prints s per example and the peak.

Review checklist (silent failure modes):

- The GPU comparison uses one captured hidden-state tensor for all three heads. Two separate backbone forwards
  would compare chaos, not heads.
- The labels are shifted exactly once (the fused head shifts internally; the harness passes unshifted labels).
- The softcap value used by the fused head is the model's (30.0), not `None`; the Gemma2 test fails if the softcap
  is dropped (check by mutation).
- `add_fused_lm_head` is applied before `get_peft_model` in the harnesses.
- The loss is the mean over labelled tokens, not over all positions; the padded-row test covers this.
- The HF/Qwen path in `qlora.py` is byte-identical (diff it).
- `--head` defaults to `full` in both harnesses.

---

### K1. Gather, routing weight and slot scatter in the kernel; a plan without host syncs (opus)

> **Done on the CPU 2026-11-22; GPU checks and the A/B pending (after T7).** The Opus review found no bugs affecting training: block-table bounds and empty blocks are safe, the indexing and epilogue order match, the saved-`y`-free backward is correct, no host syncs remain, and the tests catch real regressions. Fixed and rechecked: zero-token input crashed in the reshapes (now explicit sizes, plus `test_zero_tokens`); the check_kernel bit-identity case can use the real pre-K1 kernel (`K1_OLD_TRITON=/home/ubuntu/k1_old/inkling_triton.py`). The pre-K1 files are kept in WSL ~/k1_old. Divergences accepted: row_scale takes the output dtype; one transposed GEMM runs when only the weights need a gradient; a supplied `out` with zero blocks isn't zeroed. Note: `scatter_add_` in the plan is an integer atomic, which is still deterministic. **GPU next:** check_kernel.py (with K1_OLD_TRITON), check_determinism.py, check_real_layer.py, check_captured_layers.py, then ab_step.py 3× old vs 3× new.
>
> **GPU 2026-11-22, run on the K2 code (K2 was kept, and it contains every K1 path).** Every K1 case in check_kernel.py passes (gather/scatter/row_scale/round_out, bf16 and float32, canaries, empty blocks; max rel 1.1e-4 against the 2e-3 bf16 bar, 2.4e-6 against the 1e-5 float32 bar). check_determinism.py: bit-identical over 4 runs. check_real_layer.py and check_captured_layers.py: criterion (a) passes (see K2). A test bug from K1 was fixed: T3's `check_case` bf16→float32 cases expected the raw accumulator, but K1 made `round_out=True` the default (rel 1.65e-3 against the 1e-5 bar), so the case now passes `round_out=False`. Not run: K1 on its own against pre-K1 (bit-identity with `K1_OLD_TRITON`, and the ab_step A/B against the pre-K1 files). With K2's in-kernel SwiGLU, `out` is no longer bit-identical to pre-K1 (rel 3.1e-5 in bf16; the bar is now 5e-3).

Tier reason: kernel indexing and a rewritten backward; mistakes give plausible numbers.

Reference behaviour: today's `inkling_kernels._forward`, `FusedExpertsFunction.backward` and `_combine` (T6):
every pair's row lands in its own slot of a zeroed (T x K, H) float32 buffer and the K rows of a token are summed in
fixed order. Bars: plan section 9 criterion (a) (`tests/gpu/check_captured_layers.py`), and bit-identical repeat
runs (`tests/gpu/check_determinism.py`).

Decisions:

1. **Backend protocol.** `gemm` gains keyword-only arguments, all optional, in `TorchBackend` and `TritonBackend`:
   `gemm(inp, proj, transpose, plan, *, in_rows=None, out_rows=None, num_out_rows=None, row_scale=None,
   out_dtype=None, round_out=True)`.
   - `in_rows` int32 (R,): plan row p reads `inp[in_rows[p]]`. None: `inp[p]`.
   - `out_rows` int32 (R,) with `num_out_rows`: plan row p is written to row `out_rows[p]` of a
     (`num_out_rows`, N) output. None: row p of an (R, N) output.
   - `row_scale` float32 (R,): row p's result is multiplied by `row_scale[p]` in float32, after `round_out`.
     Requires `out_dtype=torch.float32` when the input is bf16.
   - `out_dtype`: None means `inp.dtype`; float32 is allowed for bf16 input.
   - `round_out`: for bf16 input, the float32 accumulator is rounded to bf16 first (today's rounding point) and
     then widened or scaled. False keeps the float32 accumulator. Ignored for float32/float64 input.
   - R is `plan.slot.numel()` (= T x K, see decision 3). **Every output tensor is allocated with `torch.zeros`**,
     so rows no block covers are exact zeros.
2. **Kernel** (`_nvfp4_grouped_gemm_kernel`): new pointer arguments `in_rows_ptr`, `out_rows_ptr`,
   `row_scale_ptr` and constexprs `GATHER_IN`, `SCATTER_OUT`, `HAS_ROW_SCALE`, `ROUND_OUT_BF16`,
   `OUT_BF16` (existing). `rows = row0 + offs_m` stays the plan row. Input row index:
   `tl.load(in_rows_ptr + rows, mask=mask_m, other=0).to(tl.int64)` when `GATHER_IN`, else `rows`. Output row
   index likewise from `out_rows_ptr`. Epilogue order: `acc` -> (`ROUND_OUT_BF16`: `.to(bf16).to(float32)`) ->
   (`HAS_ROW_SCALE`: `* row_scale[rows][:, None]`) -> cast to the output dtype -> masked store. The whole body
   after loading `nrows` is inside `if nrows > 0:` so empty blocks do no decode. If Triton 3.8 rejects a runtime
   `if` there, the fixed fallback is: keep the static table but pass `num_blocks = int(valid.sum())` to the grid
   (one host sync per plan) and say so in the ticket's Done note. Unused pointer arguments receive the block
   table's `block_rows` tensor as a dummy. `nvfp4_grouped_gemm` gets the same keyword arguments and validates
   dtypes, devices and lengths.
3. **Sync-free plan.** `build_route_plan` must not call `int()`, `.item()`, `.tolist()`, `bincount`, `nonzero` or
   `repeat_interleave` with tensor repeats (each forces a host sync). New construction, all on the device:
   - `flat = top_k_index.reshape(-1).long()`; `order = argsort(flat, stable=True)`; `slot = order` (all T x K
     entries; sentinel pairs sort last and no block covers them); `token_idx = slot // K`;
   - `counts = zeros(E + 1, int64).scatter_add_(0, flat, ones)[:E]`; `first_row = cumsum(counts) - counts`;
   - `blocks_per_expert = (counts + block_m - 1) // block_m`; `cum = cumsum(blocks_per_expert)`;
   - static table size `NB = E + (T * K) // block_m`; `b = arange(NB)`;
     `e = searchsorted(cum, b, right=True).clamp(max=E - 1)`; `valid = b < cum[-1]`;
     `j = b - (cum[e] - blocks_per_expert[e])`; `block_row0 = first_row[e] + j * block_m`;
     `block_rows = where(valid, clamp(counts[e] - j * block_m, 0, block_m), 0)`; `block_row0` is 0 where not
     valid.
   - `RoutePlan` fields: `slot`, `token_idx` (int64), `slot32`, `token_idx32` (int32, for the kernel),
     `expert_of_block`, `block_row0`, `block_rows` (int32, length NB), `counts`, `block_m: int`. `num_pairs` is
     removed. `TritonBackend.gemm` replaces the `_max_block_rows` guard by
     `if plan.block_m > sizes["block_m"]: raise ValueError`.
   - `TorchBackend` iterates `plan.counts.tolist()` as now (CPU reference; a sync there does not matter).
4. **Forward** (`_forward`), d = hidden dtype, f = float32 for bf16:
   ```
   gu  = gemm(hidden, "gate_up", False, plan, in_rows=plan.token_idx32)            # (TK, 2I) d
   a   = _softcapped_swiglu(gu[:, 0::2], gu[:, 1::2], gate_cap, up_cap)            # (TK, I)  d
   ws  = top_k_weights.reshape(-1).index_select(0, plan.slot).to(f)                # (TK,)
   buf = gemm(a, "down", False, plan, out_rows=plan.slot32, num_out_rows=T*K,
              row_scale=ws, out_dtype=f, round_out=True)                           # (TK, H) f
   out = buf.view(T, K, H).sum(1).to(d)
   ```
   For bf16 this is the same arithmetic as today (round to bf16, widen, multiply by the weight, slot buffer, sum
   over K), so **`out` must be bit-identical to the pre-change code** on the GPU. `_combine` is deleted. The
   `num_pairs == 0` special cases are deleted (an all-sentinel input gives all-empty blocks and zero buffers).
5. **Backward.** `y` is no longer saved; `ctx.save_for_backward(gu, a, ws)`. With `go = grad_out` (T, H) in d:
   ```
   da_u = gemm(go.contiguous(), "down", True, plan, in_rows=plan.token_idx32,
               out_dtype=f, round_out=False)                                       # (TK, I) f, zero on sentinel rows
   dws  = (da_u * a.to(f)).sum(-1)                                                 # equals (go * y).sum(-1)
   d_weights = zeros(T*K, f).index_copy_(0, plan.slot, dws).view(w_shape).to(w_dtype)
   # only when hidden needs a gradient:
   da   = da_u * ws[:, None]
   dgate, dup = softcapped_swiglu_grads(gu[:, 0::2].to(f), gu[:, 1::2].to(f), gate_cap, up_cap)
   dgu  = stack((da * dgate, da * dup), -1).reshape(T*K, -1).to(d)
   buf  = gemm(dgu, "gate_up", True, plan, out_rows=plan.slot32, num_out_rows=T*K,
               out_dtype=f, round_out=True)
   d_hidden = buf.view(T, K, H).sum(1).to(d)
   ```
   The identity `sum_h go[h] * (W a)[h] = sum_i (W^T go)[i] * a[i]` replaces the saved `y`. `da_u` is always
   computed (one transposed GEMM even when only `top_k_weights` needs a gradient). Gradients are therefore not
   bit-identical to the old code; they are held to criterion (a).
6. `fused_experts`' signature and `inkling.py` are not touched. `softcapped_swiglu_grads` and `SoftcappedSwiGLU`
   keep their signatures.

Acceptance:

- CPU (`uv run pytest tests/test_inkling_kernels.py tests/test_inkling.py`): all existing tests pass after
  replacing `plan.num_pairs` by `int(plan.counts.sum())` in the tests; plus new tests:
  - the float64 comparison with the independent eager loop (output, d(hidden), d(top_k_weights), rel at most
    1e-12) and `gradcheck` still pass, with sentinel indices and an expert without tokens;
  - `build_route_plan`: the non-empty blocks equal the old construction's blocks exactly (keep the old
    construction as a test helper); empty blocks have `block_rows == 0`; table length is `E + T*K // block_m`;
  - each new `gemm` option on `TorchBackend` against plain indexing: `in_rows`, `out_rows` with zero untouched
    rows, `row_scale`, `round_out` both ways;
  - `build_route_plan`'s source contains none of the banned calls (an AST test), and `_forward` and `backward`
    contain no `int(`/`.item(`/`.tolist(`;
  - two runs give `torch.equal` results (existing T6 test).
- CPU in WSL (`TRITON_INTERPRET=1`): `tests/gpu/check_kernel.py`'s interpreter cases extended with every new
  option against `TorchBackend`, rel at most 1e-6 in float32, including a table with trailing empty blocks.
- GPU (after T7 and H1), in this order:
  1. `tests/gpu/check_kernel.py` (extended: new options, bf16 and float32, canaries around scattered rows, empty
     blocks) passes the T3 bars;
  2. bit-identity of `out`: a new case in `check_kernel.py` runs the pre-change forward (kept in the script as a
     local copy of the old `_forward` + `_combine`, about 25 lines) and the new one on layer-0-shaped synthetic
     weights, bf16, 4,192 rows with sentinels: `torch.equal`;
  3. `tests/gpu/check_determinism.py`: bit-identical over 4 runs;
  4. `tests/gpu/check_real_layer.py` and `tests/gpu/check_captured_layers.py`: all section 2.5 per-layer bars in
     all 24 layers (criterion (a));
  5. `scripts/inkling/ab_step.py --modes fused --mxfp8-storage nvfp4 --head fused`, three times with the old
     kernel files and three times with the new ones (keep copies of the two old files in the scratchpad and swap
     them in; do not commit): report the median s per example of each. Expected gain 0.05-0.1 s per example. A
     gain below 0.02 s is reported as "no gain"; the change is still kept if the bars pass, because K2 builds on
     it.

Review checklist (silent failure modes):

- `in_rows` and `out_rows` are indexed by the plan row (`row0 + offs_m`), not by the offset within the block.
- Scattered stores go to distinct rows (slots are a permutation); nothing is written for masked rows; padded
  rows of a partial block do not read `in_rows` out of bounds (masked load with `other=0`).
- `row_scale` is applied after the bf16 rounding and in float32 (the old order); `ws` is in plan order.
- Every output is `torch.zeros`, including `gu` and `da_u`; sentinel pairs get exactly zero weight gradient and
  contribute zero to `out` and `d_hidden`.
- `dws` uses the saved bf16 `a` (what the down projection saw) and the unscaled `da_u`; `da` is scaled by `ws`
  exactly once.
- The sum over K is over the buffer viewed as (T, K, H): row `t*K + k`. A view as (K, T, H) would still pass
  shape checks.
- The static table: `searchsorted(..., right=True)`, clamped expert ids, and zero rows for every block at or
  beyond the real count. A block with `block_rows == 0` must not store.
- No host sync was reintroduced (`.max()` in a Python `if`, `int()`, printing a tensor).
- The old-forward copy in `check_kernel.py` is the pre-change code, not a call into the new code.

---

### K2. SwiGLU epilogue in the gate_up kernel (opus)

> **Done and kept 2026-11-22: every bar passes, 0.11 s per example faster than K1.** CPU: the float64 loop comparison (1e-12) and gradcheck pass through `act`, `dfac` equals autograd of `SoftcappedSwiGLU` within 1e-12 for gate/up over -40..40, and the no-grad path gives `dfac is None`. GPU (RTX 3090), check_kernel.py against `TorchBackend`:
> - bf16 `a` rel 1.4e-4 / 6.6e-5 at input scale 1 / 30 (bar 2e-3), and `dfac` 1.2e-4 / 2.6e-4 (bar 5e-3);
> - more than one bf16 ulp off: `a` 0.035% / 0.038% (bar 0.5%, added after the review; dropping the gu rounding gives 20-33%, shown by mutation in the interpreter);
> - float32 `a` 1.1e-6 / 1.7e-6 (bar 1e-5), and `dfac` 9.5e-7 at scale 1 (bar 1e-5) but 1.27e-5 at scale 30. That bar was raised to 1e-4 for saturated inputs, where `1 - tg²` and `1 - s` cancel and one-ulp tanh/sigmoid differences get magnified; this case is not in the ticket;
> - the column-mapping case passes (rel 9e-7), and nothing is written through the `dfac` pointer without `save_grads`;
> - fused_experts bf16 against TorchBackend: out 4.8e-4, d_hidden 6.8e-4, d_weights 1.2e-4.
>
> check_determinism.py: bit-identical over 4 runs. check_real_layer.py, layer 0, fused against loop: out 1.86e-3, d_hidden 6.18e-3, d_weights 2.10e-3. check_captured_layers.py, all 24 layers and both calls: worst out 4.06e-3, d_hidden 6.52e-3, d_weights 3.40e-3, cosine at least 0.99998; criterion (a) passes. Kernel timing (default block sizes, 6,000 rows): gate_up forward 6.38 ms plain, 5.23 ms with the epilogue, 5.35 ms with `save_grads`. ab_step.py `--modes fused --mxfp8-storage nvfp4 --head fused`, 3 K1 runs alternating with 3 K2 runs: K1 1.84 / 1.62 / 1.62 s per example (median 1.62), K2 1.51 / 1.51 / 1.51 (median 1.51), a gain of 0.11 s; peak 18.13 GiB for K1, 17.98 GiB for K2.
>
> Divergences accepted: `dfac` is computed only when hidden needs a gradient; `act` refuses `out_rows`, `row_scale`, another output dtype and `round_out=False`; the K1 bit-identity case is now a rel bar. The K1 files are kept in WSL ~/k1_new, the K2 files in ~/k2_new.

Tier reason: in-kernel activation and its derivative; a wrong derivative trains without an error.

Reference behaviour: `_softcapped_swiglu` and `softcapped_swiglu_grads` in `inkling_kernels.py`.

Decisions:

1. Forward `gate_up` calls get an activation epilogue, selected by a new keyword `act=(gate_cap, up_cap)` on
   `gemm` (only valid with `proj="gate_up"`, `transpose=False`), plus `save_grads: bool`. Return value: a pair
   `(a, dfac)`: `a` (R, I) in the input dtype, and, when `save_grads`, `dfac` (R, 2I) in the input dtype holding
   (d act / d gate, d act / d up) interleaved (even column = gate), else None. `gu` is no longer produced on this
   path.
2. Kernel: constexprs `ACT`, `SAVE_GRADS`; arguments `gate_cap`, `up_cap`, `dfac_ptr`. The N tile is `BLOCK_N`
   interleaved rows, so after the K loop: `g2 = acc.to(bf16).to(float32)` (today's rounding of `gu`),
   `gate, up = tl.split(tl.reshape(g2, (BLOCK_M, BLOCK_N // 2, 2)))`, then in float32
   `tg = tanh(gate / gate_cap)`, `s = tl.sigmoid(gate)`, `tu = tanh(up / up_cap)`,
   `act = gate_cap * tg * s * up_cap * tu`, `dg = up_cap * tu * ((1 - tg*tg) * s + gate_cap * tg * s * (1 - s))`,
   `du = gate_cap * tg * s * (1 - tu*tu)`. `tanh` is `triton.language.extra.libdevice.tanh` (present in 3.8.0;
   `tl.split` is too). Under `TRITON_INTERPRET=1` the module uses `2 * tl.sigmoid(2 * x) - 1` instead, chosen
   once at import from the environment variable. `act` is stored to `out` at columns
   `pid_n * (BLOCK_N // 2) + arange(BLOCK_N // 2)`; `tl.interleave(dg, du)` to `dfac` at the tile's own columns.
   `BLOCK_N` must be even (it is 128).
3. `TorchBackend.gemm` implements `act` and `save_grads` with `_softcapped_swiglu` and
   `softcapped_swiglu_grads` on its own `gu`, rounding `dfac` to the input dtype.
4. `_forward` calls `a, dfac = gemm(hidden, "gate_up", False, plan, in_rows=..., act=(gate_cap, up_cap),
   save_grads=need_grad)`; the Function saves `(dfac, a, ws)`. Backward:
   `dgu = (da[:, :, None] * dfac.view(T*K, I, 2).to(f)).reshape(T*K, -1).to(d)`; everything else as K1.
5. Rounding `dfac` to bf16 adds rounding-sized noise to `d_hidden`. It is accepted only inside criterion (a).
6. **Keep rule, fixed now:** K2 is kept only if the per-layer bars pass and the `ab_step.py` median improves on
   K1's by at least 0.02 s per example; otherwise it is reverted and the Done note records the numbers.

Acceptance:

- CPU: `tests/test_inkling_kernels.py`: float64 loop comparison (1e-12) and `gradcheck` still pass through the
  `act` path of `TorchBackend`; `dfac` equals autograd's derivatives of `SoftcappedSwiGLU` (1e-12, inputs over
  -40..40); the no-grad path returns `dfac is None`.
- CPU in WSL (`TRITON_INTERPRET=1`): kernel `a` and `dfac` against `TorchBackend` in float32, rel at most 1e-5
  (the tanh formula differs); column mapping checked with a weight whose gate rows and up rows are distinct
  constants.
- GPU (after K1's GPU checks): `check_kernel.py` (bf16 `a` rel at most 2e-3 and `dfac` rel at most 5e-3 against
  `TorchBackend`), `check_determinism.py`, `check_real_layer.py`, `check_captured_layers.py` (criterion (a)),
  then `ab_step.py` as in K1 step 5 against the K1 code.

Review checklist (silent failure modes):

- Even columns are gate, odd are up, in `a`'s source and in `dfac`; a swap passes shape checks and the forward
  comparison would catch only half of it, so the distinct-constants test must cover `dfac` too.
- The sigmoid sees the raw gate; the caps apply to the tanh factors only (as in `Inkling2LoopSoftcappedSwiGLU`).
- The accumulator is rounded to bf16 before the activation (same point as today).
- `a`'s column offset is half the tile's; every column of `a` is written exactly once.
- With `save_grads=False` nothing is written through `dfac_ptr`.
- Small arguments: `tanh` of values near 1e-4 is accurate (libdevice on the GPU; the interpreter formula is only
  for CPU tests and its tolerance is the looser 1e-5).

---

### B1. Bucketed sampler (sonnet)

> **Done 2026-11-22.** 12 CPU tests pass. The Opus review found a HIGH bug, now fixed: the implicit epoch counter made a resume after epoch 0 replay the wrong epoch's order (Trainer only calls `set_epoch` if the sampler has one). Divergence approved by the lead: `set_epoch(epoch)` was added and `__iter__` no longer advances the epoch; `batches(epoch)` is public. The recheck simulated a mid-epoch resume through accelerate (epoch 1 → exactly `batches(1)[5:]`, then `batches(2)`), and the shuffle check now fails when the shuffle is removed. Known and accepted: the short final batch is the last window's longest examples, placed last.

Tier reason: plain Python with exact CPU tests.

Why not the TRL or transformers options: `train_sampling_strategy="group_by_length"` builds megabatches of
`batch x accumulation x 50` examples, sorts each by length and yields them longest first, so every optimizer
update would see 16 examples of nearly one length and each megabatch would run from long to short. That couples
the update to the length. TRL's `padding_free` and packing put several examples in one sequence, which is not
wanted here (section 4.2). Hence a small sampler of our own.

Decisions (new file `domain_verifier_factory/training/bucketing.py`, imports only the standard library and torch):

- `class BucketedSampler(torch.utils.data.Sampler[int])`,
  `__init__(self, lengths: Sequence[int], batch_size: int, window_batches: int = 32, seed: int = 0)`.
- `__iter__`, with an epoch counter starting at 0 and incremented at each call, and
  `rng = random.Random(seed * 1_000_003 + epoch)`: (1) a random permutation of all indices; (2) cut into windows
  of `batch_size * window_batches` indices (the last window may be shorter); (3) sort each window by length (ties
  by index); (4) cut each window into consecutive batches of `batch_size`; (5) shuffle the list of full batches
  from all windows; the one batch that may be short (the end of the last window) is appended last, because a
  short batch in the middle would shift every later `DataLoader` batch; (6) yield the indices batch by batch. A
  `DataLoader` with the same `batch_size` and `drop_last=False` then reproduces the batches.
- `__len__` is `len(lengths)`. `batch_size == 1` yields a plain seeded permutation.
- `def padding_ratio(lengths, order, batch_size) -> float`: padded tokens over real tokens for an index order.

Acceptance (CPU, `uv run pytest tests/test_bucketing.py`):

- every index appears exactly once per epoch; two epochs differ; the same seed and epoch reproduce the order;
- through a real `DataLoader`, the batches equal the sampler's batches, also when the number of examples is not
  divisible by the batch size or the window;
- on 4,000 lognormal lengths (mean 524, fixed seed): `padding_ratio` at batch 2 is at most 1.03 and a random
  order's is at least 1.08;
- the correlation between consecutive batches' mean lengths is below 0.2 in absolute value, and the means of
  runs of 8 consecutive batches have a standard deviation below half that of single batches (no length-sorted
  updates).

Review checklist (silent failure modes): the batches are shuffled after bucketing; the short batch is last; the
epoch counter changes the order; no dependence on the global `random` state; the sort is by length, not by index.

---

### B2. Quality-test script and pre-gate (sonnet)

Tier reason: a training loop assembled from existing harness code; the judgement is fixed in advance.

What is being decided: whether training with batch 2 x accumulation 8, length-bucketed, gives the same quality
as batch 1 x accumulation 16 in random order. The model is chaotic (two runs of the same code differed by up to
2e-2 in loss, and gradient directions are not reproducible), so single-run or per-step comparisons mean nothing.
The comparison is between end-of-run held-out metrics, against the baseline's own seed-to-seed spread.

Design (file `scripts/inkling/bucketing_quality.py`, one process, model loaded once, under
`gpu_session(min_free_mib=21000)`):

- **Model:** `load_inkling_4bit(experts_impl="fused", mxfp8_storage="nvfp4")`, `add_fused_lm_head` (H1), gradient
  checkpointing, LoRA r 64 / alpha 128 / dropout 0.05 on `INKLING_LORA_TARGETS` (the `LoraSettings` defaults).
- **Data:** from `datasets/factcheck_full/factcheck_agent_dataset/train.jsonl`, the examples of at most 1,024
  tokens (about 98.5%), a fixed subset of 3,200 drawn with `random.Random(1234)`. All six runs train on these
  same 3,200 examples, one pass: 200 optimizer updates of 16 examples.
- **Arms:** A (baseline): batch 1, accumulation 16, order = `random.Random(seed)` permutation. B (candidate):
  batch 2, accumulation 8, `BucketedSampler(lengths, 2, 32, seed)`, right padding with the tokenizer's pad
  token, `attention_mask`, labels -100 on prompt and padding. Seeds 0, 1, 2 for each arm. Run order A0, B0, A1,
  B1, A2, B2, so an interrupted run leaves balanced arms.
- **Per run:** LoRA re-initialised under `torch.manual_seed(seed)` (arms A and B of one seed start from the same
  weights); `torch.optim.AdamW` lr 2e-4, weight decay 0; linear warm-up over the first 6 updates, then cosine to
  0 at update 200; gradient clipping 0.3; loss per micro-batch = mean over its labelled tokens, divided by the
  accumulation count (what `Trainer` does for this model, whose `accepts_loss_kwargs` is False).
- **Held-out set:** `validation.jsonl`, the first 500 after a `random.Random(99)` shuffle, never trained on.
  Evaluated teacher-forced, batch 1, no padding, `model.eval()`, `torch.no_grad()`, at updates 100 and 200 (about
  3.5 minutes each). No generation.
- **Metrics per evaluation:** (1) `val_loss`: mean over examples of the mean negative log-likelihood of the
  completion tokens, computed from the hidden states at the labelled positions (`lm_head`, float32, softcap,
  cross-entropy). (2) `verdict_acc`: at the completion position that holds the first token of the verdict value,
  the argmax over the four labels' first tokens equals the gold label's. The four ids are obtained at start-up
  as `tok('"' + label + '"', add_special_tokens=False)["input_ids"][1]` (86995 supports, 31891 contradicts, 256
  inconclusive, 340 irrelevant on the current tokenizer) and must be distinct; the position is the first
  completion token that is one of the four ids, must lie within the first 8 completion tokens and must equal the
  gold id, else the script raises. Also stored: per-example NLL, s per example, peak memory.
- **Output:** one JSON per finished run in `--out-dir` (default `results/bucketing_quality/`), written by
  atomic rename; finished runs are skipped on restart. A final `summary.json` with the decision.
- **Pre-gate** (`--pregate`, about 20 minutes, run first): (i) the longest pair of the subset: one training
  step at batch 2, peak allocated at most 21.0 GiB; (ii) 16 examples sorted by length, as 16 x 1 and 8 x 2: s
  per example and peak for each (experiment E3); (iii) padding invariance: for those 8 pairs, in eval mode
  without grad, each example's loss at batch 2 against its loss at batch 1: mean absolute difference at most
  0.03 (reported next to T7's loop-with-noise spread for criterion (c)). The pre-gate fails, and B3 does not
  start, if (i) or (iii) fails, or if batch 2 is not at least 15% faster per example than batch 1.

**Decision rule, fixed before running** (final evaluation, update 200; three per-seed values per arm):

- Spread: `R_loss = max(A val_loss) - min(A val_loss)`, `R_acc = max(A verdict_acc) - min(A verdict_acc)`.
- PASS iff all three hold:
  1. `mean(B val_loss) <= mean(A val_loss) + R_loss`;
  2. `max(B val_loss) <= max(A val_loss) + R_loss`;
  3. `mean(B verdict_acc) >= mean(A verdict_acc) - max(R_acc, 0.015)` (0.015 is about one standard error of an
     accuracy on 500 examples).
- INCONCLUSIVE iff `R_loss > 0.10 * mean(A val_loss)` (the baseline itself is too noisy at this length): rerun
  both arms once with 6,400 examples (400 updates) under the same rule. Still inconclusive counts as FAIL.
- FAIL otherwise: batch 1 stays, B4 is cancelled, and the bucketed arm's deficit is recorded.
- Reported but not part of the rule: the update-100 values, the paired per-example difference of seed-averaged
  NLL with a bootstrap 95% interval over the 500 examples, and the speed-up.

GPU time: arm A about 3,200 x 1.45 s = 1.3 h per run; arm B about 1.0-1.1 h per run; evaluations 7 minutes per
run; load 5 minutes. About 8 h in all (up to 9 h if batch 2 gains little).

Files: `scripts/inkling/bucketing_quality.py`; `tests/test_bucketing_quality.py` (CPU) for the pure parts,
which live in functions that need neither CUDA nor transformers at import: the decision rule
(`decide(a_runs, b_runs) -> dict`), the verdict-position finder, the schedule, the collate function.

Acceptance:

- CPU: `decide` on hand-made numbers gives PASS, FAIL and INCONCLUSIVE as specified, including the boundary
  cases (equality passes); the verdict finder raises on a completion without a verdict token in the first 8;
  the collate pads right and masks padding in the labels; the schedule is 0 at update 0, 2e-4 at update 6 and 0
  at update 200; `--dry-run` (lengths only, no model) prints the six runs' counts (3,200 micro-batches for A,
  1,600 for B, 200 updates each).
- GPU (after T7): `--pregate` passes, or stops the ticket chain with its numbers.

Review checklist (silent failure modes):

- Both arms see the same 3,200 examples and the same 500 validation examples; validation is a different file
  from training and is evaluated identically (batch 1) for both arms.
- The effective batch is 16 examples per update in both arms; the schedule counts updates, not micro-batches.
- The loss scaling is the per-micro-batch mean divided by the accumulation count in both arms.
- Padding: right side, attention mask passed, padded labels -100.
- LoRA is really re-initialised per run (adapter weights equal across arms at update 0 for the same seed, and
  different across seeds); the optimizer and the sampler are rebuilt per run.
- The verdict position is found in the completion, not the prompt (the prompt also names the labels).
- The rule is applied to update-200 values only; the thresholds are module constants echoed in `summary.json`.
- Resume: a half-written run file is never read as finished.

---

### B3. Run the quality test (sonnet)

Run `bucketing_quality.py --pregate`, then the full run (about 8 h, GPU otherwise idle, overnight). Append a
results section to this document: the pre-gate numbers, the per-seed table for both arms at updates 100 and 200,
`R_loss`, `R_acc`, the three conditions, the verdict (PASS, FAIL, or INCONCLUSIVE and its rerun), speed and peak
memory. No threshold is changed after seeing results. If the run is interrupted, restart the same command; it
skips finished runs.

Review checklist: the verdict follows the rule literally; an INCONCLUSIVE was rerun once at 400 updates, not
reinterpreted; failures are reported as failures.

---

### B4. Adopt batch 2 with bucketing in the trainer (sonnet; only if B3 passes, and after T7)

Decisions:

- `configs.py`, `TrainingHyperparameters`: `length_bucketing: bool = False` and `bucket_window_batches: int = 32`.
  The defaults of `per_device_train_batch_size` (1) and `gradient_accumulation_steps` (16) are not changed;
  Inkling run configs set batch 2, accumulation 8, `length_bucketing=True`, `max_seq_length=1024`.
- `qlora.py`: when `hp.length_bucketing`, the trainer is a local subclass of `SFTTrainer` overriding
  `_get_train_sampler(self, train_dataset=None)` to return
  `BucketedSampler([len(x) for x in ds["input_ids"]], self.args.train_batch_size, hp.bucket_window_batches,
  hp.seed)`, where `ds` is `train_dataset` or, when that is None, `self.train_dataset`; it raises `ValueError`
  if the dataset has no `input_ids` column. With `length_bucketing=False` the plain `SFTTrainer` is used and
  nothing changes for the Qwen path. Both settings are written to the saved run metadata.

Acceptance: CPU (`tests/test_training.py`): with a fake dataset of known lengths, the override returns a
`BucketedSampler` over the right lengths; with the flag off, `SFTTrainer` itself is the class used. GPU: a
20-update Inkling run through `qlora.py` with the new settings finishes, reports `mean_token_accuracy` and a
peak under 21 GiB.

Review checklist (silent failure modes): the lengths come from the tokenized training set the trainer actually
uses (after truncation to `max_seq_length`); the flag off leaves the HF path byte-identical; the sampler's seed
is the run seed; evaluation batches are unaffected; the short final batch does not break accumulation.

**B2 pre-gate decision (the user, 2026-11-23):** the pre-gate failed (i), with a 21.39 GiB peak on the longest pair against a 21.0 bar, and (iii) padding invariance, 0.148 against 0.03. Speed passed (22.1% faster). (iii) is invalid: all 16 pre-gate examples were 438 tokens, so there was no padding, and the difference is the known batch-composition chaos (plan sections 8–9). (i) fits on the 24 GiB card, and only the rare longest pairs come near it. Decision: run B3 as written, accepting the 21.4 GiB peak and dropping (iii); B3's seed-spread rule decides.

**B3 done 2026-11-23: PASS.** 3 seeds × {batch 1 × accum 16 random, batch 2 × accum 8 bucketed}, 200 updates on 3,200 mixture examples. Mean val loss A 0.15118 / B 0.15109 (R_loss 0.0018), max A 0.15230 / B 0.15206, mean verdict accuracy A 92.13% / B 92.07% (R_acc 1.8 points); all three conditions pass. Speed: about 1.25 vs 0.98 s/example. Peak memory: B 21.48 GiB. Two driver-621.24 hangs during the run were handled by a fresh WSL plus resume (results/bucketing_quality/*.json, summary.json).

**B4 done 2026-11-23.** `length_bucketing` and `bucket_window_batches` are in configs (default off). `trainer_class(base, hp)` picks the bucketed SFTTrainer subclass, which overrides `_get_train_sampler` with `BucketedSampler` over the final tokenized dataset's lengths, using the DataLoader's `_train_batch_size`. Inkling runs set batch 2 × accum 8, length_bucketing true and max_seq_length 1024. GPU: 5 steps at 16.99 s/step (1.06 s/example vs 1.27 at batch 1). The Opus review found no bugs; its three test gaps were fixed (flag-off branch, batches received equal the sampler's, `_train_batch_size`).
