# Inkling-2-Mini on one RTX 3090: is the fused expert kernel the only general speed fix?

Written 2026-11-22. Analysis only: nothing here was run on the GPU. Sources are the code, the two existing documents
(`docs/inkling_fused_experts_plan.md`, `docs/inkling_speed_headroom.md`), the vLLM fork in WSL
(`~/src/vllm-ampere`, `INKLING2-RTX3090.md` and `vllm/models/inkling2/nvidia/`), the model code
(`modeling_inkling2_loop.py`) and the checkpoint's `config.json`. Numbers marked "est." come from operation counts
and from fits to existing timings; they are not measurements. Section 7 lists the experiments that would replace
them. Training speed for our short judge examples is covered in `inkling_speed_headroom.md` and is only summarised
here.

## 1. Answer

No. The fused NVFP4 kernel is the general fix for one thing: anything that runs through our own Hugging Face (HF)
loader, which is the only stack that can train. There it is required (7.5x per training step, 2x in HF decoding).
It is not the fix for generation, long contexts or serving, and it is not the last fix for training.

There are three general fixes, not one:

1. **The fused expert kernel** (done; the defaults flip when T7 passes). It serves QLoRA training and any
   forward-only scoring done inside the training process.
2. **The vLLM fork for every inference task** (generation, long context, batch evaluation, serving adapters). It
   decodes at 93 tok/s where our HF path reaches about 4 tok/s, and no kernel work on our side closes that gap,
   because the gap is no longer in the experts. What is missing is small: an adapter key converter, a predictor
   that talks to the server with the right options, and one parity check.
3. **Dividing or bounding what the kernel does not touch in training**: batches sized by tokens with the loss
   computed only at labelled positions (no kernel), and, only for sequences above about 1,500 tokens, an attention
   path that does not build full T x T tensors. The second is the only other place where a new kernel is justified.

Nothing else in the model (router, short convolutions, message gates, norms) deserves a kernel: each is small
arithmetic, and their cost is per-operation overhead that batching (training) or CUDA graphs (vLLM) already removes.

## 2. Facts this rests on

| Fact | Source |
|---|---|
| 20 of 24 layers use sliding-window attention (window 1,024); only layers 5, 11, 17, 23 attend to the full context. With the loop that is 40 sliding and 8 full layer calls of 48 | `config.json` (`local_layer_ids`, `loop_config`) |
| 24 query heads, 8 key/value heads, head size 128; relative-position bias over 1,024 distances; one sink logit per head | `config.json`, model code |
| The HF attention is the eager path for both layer types: it builds (24, T, T) logits, bias and an fp32 softmax even in the sliding layers | `eager_attention_forward`; `_supports_sdpa = False` |
| HF greedy generation, short prompt, 65 new tokens: loop 2.2 tok/s, fused 4.2 tok/s | plan section 8 (`lora_smoke.py`) |
| `lora_smoke.py` loads with the default `mxfp8_storage="nf4"`, so in that run layers 20-23 probably still used the loop | `scripts/inkling/lora_smoke.py` line 29 |
| vLLM fork in WSL: 93 tok/s decode, 17.64 GiB of weights, 2.5 GiB key/value cache at 8k context, utilization 0.94; CPU offload of the embeddings fails (pinned memory) | the brief (2026-11-21) |
| vLLM fork, the fork's own document: decode 91 tok/s, 112-147 tok/s with DSpark; prefill 2,500-2,700 tok/s up to 33k tokens; 131k context with an FP8 cache and embeddings on the CPU | `INKLING2-RTX3090.md` (not re-measured here; that setup used the CPU offload that fails in our WSL) |
| vLLM's experts on Ampere use the `humming` kernels on repacked weights, forward only | `nvidia/moe.py` |
| vLLM's LoRA covers `wq_du`, `wk_dv`, `wv_dv`, `wo_ud`, `gate_attn` (plus `wr_du`, `attn_sink_proj`); expert LoRA needs bf16 experts and is not available on this checkpoint | `nvidia/lora.py` |
| HF names map to checkpoint names by renaming only (no transpose, no scaling): `q/k/v/o_proj.linear` and `o_proj.gate_proj` | `transformers/conversion_mapping.py` lines 379-396 |
| Our evaluation code has no Inkling path: `AdapterPredictor` loads with `AutoModelForCausalLM` and bitsandbytes and generates one example at a time; `EndpointPredictor` exists but sends no chat-template options, no log-probabilities and one request at a time | `domain_verifier_factory/evaluation/predictors.py` |
| A rank-16 adapter on our targets is about 8.7 M parameters, 17 MB in bf16 | counted from the layer shapes |

## 3. Why the fused kernel does not fix HF generation

Decoding one token at batch 1 runs 48 layer calls with 8 token-expert pairs each.

- **Arithmetic is negligible.** The experts need about 6 GFLOP per token. The real cost is reading weights: 8
  experts x 4.3 MB of packed codes and scales x 48 calls is 1.65 GB per token, plus 0.93 GB for the output layer.
  At the card's memory bandwidth that is a floor near 3-4 ms per token, about 250-300 tok/s (est.). vLLM's
  10.8 ms per token is within about 3x of that floor.
- **Our kernel pays for 64-row blocks holding 1 row.** Each of the 8 experts gets one block; each block decodes its
  whole weight. The work is small but each kernel launch has a latency of roughly 0.3-1 ms (est. from the tuned
  timings: 2,300 programs finish in 2 ms, in 2-4 waves). Two launches per layer call give 30-70 ms per token.
- **The rest of the token time is per-operation overhead.** A forward pass is about 10,000-12,000 small torch
  operations (a third of the 35,000 counted for a training step), more with the cache updates, at 15-18
  microseconds each: 150-200 ms per token (est.). Add 15-20 ms for the experts' torch glue and its two host
  synchronisations per call.

That adds up to the measured 238 ms per token (4.2 tok/s). The loop cost about 217 ms more.

| HF decode variant | tok/s | Basis |
|---|---|---|
| Loop (old default) | 2.1-2.2 | measured |
| Fused, layers 20-23 probably looped | 4.2 | measured |
| Fused with `mxfp8_storage="nvfp4"` | 4.5-5 | est. |
| Plus a small-block decode kernel and the glue folded in | 5-6.5 | est.; experts near 15 ms, overhead unchanged |
| Experts free (the ceiling of eager HF at batch 1) | 5-7 | est. |
| HF batch of 8 sequences, total | 25-35 | est.; the overhead is per step, not per sequence |
| vLLM, one stream | 93 | measured |

So the fused kernel doubles HF decoding, and a decode-specific kernel on top would add at most 30%. The remaining
factor of 15-20 against vLLM is Python and dispatch overhead, which vLLM removes by replaying the decode step as a
CUDA graph. Getting that in HF would mean rebuilding what the fork already has (static shapes, a compiled layer
stack, a cache that works inside a graph). Not recommended. **HF generation is for smoke tests only.**

## 4. Task by task

Example sizes: a judge example is about 470 prompt tokens and 53 reply tokens.

| Task | Fastest path today | Speed | Main limit |
|---|---|---|---|
| QLoRA training, short examples (about 500 tokens) | HF loader, fused kernel, batch 1 | 1.5 s per example on the real mix (1.72 s on the harness mix), measured and est. | Per-operation overhead, then the experts' decode; batching not yet used |
| QLoRA training, long examples | Same; `max_seq_length` 1,536 | About 2 ms per token plus a T-squared term; about 3.5-4 s at 1,536 tokens (est.) | Memory: full-vocabulary logits (3.6 MB per token) today, then eager T x T attention |
| Chat, long outputs | vLLM fork | 93 tok/s (112-147 with DSpark per the fork's document) | Weight reads per token; 17.6 GiB of weights leaves little room |
| Long contexts | vLLM fork | Prefill 2,500-2,700 tok/s; 8k context verified in WSL, 131k claimed with an FP8 cache and CPU offload | Cache memory. HF cannot do this at all: about 340 T-squared bytes of attention temporaries caps one pass near 4,500 tokens (est.) |
| Batch evaluation of judges, generating the reply | vLLM fork with the adapter, once the converter exists. Today in our code: nothing for Inkling (HF would be about 13 s per example) | About 0.8 s per example serial, 0.3-0.4 s with 8 concurrent requests (est.) | Missing converter and predictor; batched decode throughput not measured |
| Batch scoring, verdict only | One forward pass with the reply prefix forced, reading the label probabilities: vLLM (1 output token with log-probabilities) or HF fused under no-grad | vLLM about 0.2 s per example; HF about 0.35-0.4 s per example in batches (est.) | Prefill speed. Works for verdict-first judges only |
| Serving adapters | vLLM fork, adapters applied at run time, chosen per request | As chat; one base model for all adapters (17 MB each) | Converter; LoRA with CUDA graphs and DSpark untested on this fork |
| Evaluation during training | HF fused in the training process | As the scoring row (HF) | vLLM cannot share the card with a training run (17.6 + 20 GiB) |

### 4.1 Training with any data

For sequences like ours, see `inkling_speed_headroom.md`: about 0.8-0.9 s per example is reachable; the largest
steps are the loss at labelled positions only and batches sized by tokens. Both are general, not judge-specific.
The only judge-specific number is the 10% of positions that carry a label; data with labels on every token (plain
language modelling) gets no saving from slicing, and must chunk the output layer instead (compute the logits and
the loss for 256 positions at a time, keeping only the gradient with respect to the hidden states).

Long sequences are a different regime, with two walls:

- **Memory.** After load 15.0 GiB is allocated and about 22.1-22.5 GiB is usable. Today the peak grows by 3.6 MB
  per token (logits), so about 1,500-1,800 tokens is the limit, which is why `max_seq_length` is 1,536. With the
  loss sliced or chunked, the next wall is the attention graph of one layer during its backward and recompute:
  roughly 400-500 T-squared bytes (bf16 logits, bias, their sum, the fp32 concatenation with the sink, the fp32
  softmax and its bf16 copy). That is 1.7-2.1 GiB at 2,048 tokens and 6.7-8.4 GiB at 4,096, plus 0.22 MB per token
  of saved layer inputs. **Limit about 3,000-3,500 tokens at batch 1 (est.).** Gradient checkpointing already keeps
  this to one layer at a time.
- **Time.** The linear part is about 2 ms per token. The T-squared part was estimated at 0.15-0.25 s at 524 tokens;
  if that holds it is 2-3.5 s at 2,048 and 9-15 s at 4,096, where it would exceed the linear part. This coefficient
  is the least certain number in this document (experiment E1 of the headroom document settles it).

What would lift the walls, in order of effort:

1. Exact slicing of the bias table to T columns below 1,024 tokens (headroom item 8). Small.
2. **Banded sliding attention.** 40 of 48 layer calls only attend 1,024 tokens back, yet build T x T tensors. For
   T above 1,024, processing the queries in chunks of 1,024 against the previous 2,047 keys gives the same result
   with those calls' memory and time linear in T. This is plain torch and exact. Corrected in section 10: the
   saving is 50% of those calls' attention work at 2,048 tokens and 73% at 4,096 (five sixths of the total only
   in the limit), and the **peak** memory does not fall, because the 8 full-attention calls still set it.
   Medium effort; the model's chaos at bf16 rounding level (plan section 8) means it needs the section 9 criteria,
   not a bit-equality test.
3. PyTorch's memory-efficient scaled-dot-product kernel for the 8 full-attention calls: it accepts an additive
   float mask (bias plus causal mask) and the sink can be one extra key with a zero value. It avoids most
   temporaries. Open points: whether its backward returns the gradient of the mask (the bias depends on the hidden
   states, so it must), and that its softmax is not the model's fp32 softmax. The headroom document judged this not
   worth it at 524 tokens; that stays true. It becomes worth a test above about 2,000 tokens.
4. A fused Triton attention kernel with a hand-written backward (bias gather, sink, fp32 softmax, value product).
   High effort and the same risk class as the expert kernel. Only if long-sequence training becomes a real need
   and items 2-3 are not enough.

Flash attention cannot express the per-head bias or the sink; that is unchanged.

### 4.2 Generation, long outputs, long contexts

vLLM is the path, for three reasons that our code cannot match cheaply: CUDA graphs for the decode step, kernels for
every block (experts, attention with bias and sinks, short convolutions, the loop gates), and a paged cache that
keeps only 1,024 tokens for the 40 sliding layer calls.

Cache size (est. from the head counts, assuming one cache per layer call as in the HF code): 4 KB per token per
layer call in bf16, so 197 KB per token up to 1,024 tokens and 33 KB per token after that. An 8k context is about
0.43 GiB; a 600-token judge request about 0.12 GiB. With 2.5 GiB of cache that is roughly 20 concurrent judge
requests or about 70k tokens in one sequence in bf16, and twice that with the fork's emulated FP8 cache (a
precision change whose quality has not been re-benchmarked, per the fork's own document).

Limits specific to our machine: Windows holds about 1.3 GiB, and the embedding offload that the fork's document
relies on (0.86 GiB) fails in WSL. Together that is about 2 GiB less than the fork's tested setup, which is the
difference between 8k and 131k contexts.

### 4.3 Batch evaluation and scoring

Two different jobs:

- **Reply generation** (needed for reason-first judges, and for rationales): vLLM with concurrent requests. Prefill
  is 0.19 s for 470 tokens and the reply 0.57 s at 93 tok/s, so about 0.8 s per example serial. With 8 or more
  requests in flight the decode steps are shared; if total decode throughput reaches 300 tok/s (not measured) an
  example costs 0.3-0.4 s. Our HF path would take about 13 s.
- **Verdict scoring** (verdict-first judges): the verdict is the first value in the reply, so the answer is the
  next-token distribution after the forced prefix `{"verdict": "`. That is one forward pass and no decoding: about
  0.2 s per example on vLLM (prefill-bound), and about 0.35-0.4 s on HF fused in no-grad batches (est.: a forward
  pass costs about 0.67 ms per token plus a fixed 0.15-0.2 s per batch). It also yields the label probabilities and
  margin that `record_margin` computes today. On vLLM the probabilities come from `logprobs` with the top 20
  alternatives, so `label_mass` is a lower bound when a label's first token is outside the top 20. The HF variant
  is the only one that can run inside a training process.

Prefix caching: vLLM caches prompt prefixes in whole cache blocks, per adapter. Judges share only the system
prompt, perhaps a quarter to a third of the 470 tokens, so the gain is about 0.05 s per example: free, but small.
It would be large only if many claims shared one evidence block placed before the claim, which is a prompt-format
change that trained adapters would not survive. Not recommended as a reason to change the template.

### 4.4 Serving adapters

What exists: vLLM applies LoRA at run time on the attention projections under the checkpoint's names, with several
adapters loaded and one chosen per request. That fits the ensemble goal directly: one 17.6 GiB base and five 17 MB
adapters in one server, instead of one model load per judge.

What is missing:

1. **A converter** from our PEFT adapter to the checkpoint's names. It is a key rename
   (`...language_model.layers.N.self_attn.q_proj.linear.lora_A.weight` to
   `model.llm.layers.N.attn.wq_du.lora_A.weight`, and likewise `wk_dv`, `wv_dv`, `wo_ud`, `gate_attn`) plus an
   `adapter_config.json` with matching target names. No tensor is transformed. The exact prefix PEFT and vLLM each
   expect must be checked against a real adapter.
2. **A predictor.** `EndpointPredictor` is nearly it. It needs the chat-template options
   (`reasoning_effort: none`; without it the template asks for thinking), the reason-first prompt, concurrent
   requests, and optionally log-probabilities for the margin.
3. **A parity check.** Training and serving differ in three ways that cannot be argued away in a model that is
   chaotic at bf16 rounding level: (a) we train with layers 20-23's experts re-quantized to 4 bits (relative error
   about 0.09) while vLLM serves the checkpoint's original MXFP8 experts; (b) vLLM's kernels round differently; (c)
   an FP8 cache or DSpark, if enabled. vLLM's weights are the more faithful ones, but the adapter was fitted to
   ours. The check is verdict agreement and label-probability differences between HF fused and vLLM on a few
   hundred validation examples, compared with the HF path's own run-to-run spread.
4. Unknowns to test once: adapters together with CUDA graphs, and with DSpark.

Merging the adapter into the base weights is not worth pursuing: the attention weights are bf16 in the checkpoint,
so a merge is possible, but each adapter would then need its own 17 GiB checkpoint copy and its own server.

### 4.5 Speculative decoding (DSpark)

The fork's document reports 112-147 tok/s against 91, a factor of 1.2-1.6, with exactly 7 draft tokens. Caveats for
us: the draft head was trained for the base model, so the acceptance rate under an adapter that changes the output
style (short JSON) is unknown and could be lower or higher; the draft needs memory that is already short in WSL; the
fork's FP16 cache refuses to run with it, and FP8 is the tested combination; verification keeps the greedy result
only up to rounding. It helps long outputs. It does nothing for verdict scoring and little for 53-token replies,
where prefill is a quarter of the time. Low effort to try (flags only), so try it after the adapter path works.

## 5. Other candidate bottlenecks: does anything else deserve a kernel?

| Candidate | Cost | Verdict |
|---|---|---|
| Python and dispatch overhead per operation | The largest non-expert cost in HF: 0.5-0.6 s per training step, 150-200 ms per decoded token (est.) | Not a kernel problem. Training: batching divides it. Inference: vLLM's CUDA graphs remove it. `torch.compile` on the HF layers stays not recommended (headroom 4.7) |
| Expert glue and host synchronisations | 0.08-0.12 s per training example; 15-20 ms per decoded token | Fold into the existing kernel (headroom item 4). Helps every HF use. Worth one ticket |
| Attention (bias gather, fp32 softmax with sinks) | Small at 500 tokens; dominant in time and memory above about 2,000 | The only second kernel worth considering, and only for long-sequence training (section 4.1) |
| Router (fp32 linear to 128 scores, top-8) | About 6 small operations per layer call | No |
| Short convolutions (4 per layer, 4 taps, depthwise) | 0.05-0.07 s per training example | No. The optional `causal_conv1d` kernels would save a few hundredths of a second and need an install |
| Message gates and block-attention residuals (loop layers) | Rank-64 projections and a few fp32 operations on (T, 2,304) | No. Their cost is operation count |
| Output layer and loss | The memory peak in training | Slice or chunk (no kernel) |
| Embeddings on the CPU, optimizer | Negligible | No |

## 6. General improvements, ranked by gain per effort

| # | Improvement | Helps | Gain (est.) | Effort | Confidence |
|---|---|---|---|---|---|
| 1 | Finish T7 and make the fused path the default | Training, HF scoring | 7.5x training (measured), 2x HF decode (measured) | Already in progress | High |
| 2 | Adapter converter + endpoint predictor + parity check, then use vLLM for all generation, evaluation and serving | Generation, evaluation, serving | About 20x per stream against HF fused (93 vs 4.2 tok/s); 15-40x per judge example | Small-medium: two Sonnet tickets and one GPU session | High for speed; medium for parity until checked |
| 3 | Loss at labelled positions (or a chunked output layer), then batches sized by tokens with length bucketing | Training on any data | 1.5 s to 0.9-0.95 s per example; lifts the length limit from about 1,500 to about 3,000 tokens | Small-medium, no kernel | Medium-high (depends on the fixed overhead, E1/E3 of the headroom document) |
| 4 | Verdict scoring by one forward pass | Evaluation of verdict-first judges, in-training evaluation | 2-4x over generating the reply on vLLM; about 30x over HF generation | Small | High for speed; needs an agreement check against generated replies |
| 5 | Glue into the kernel, remove the host synchronisations | Every HF use | 0.08-0.12 s per training example; 5-10% of HF decode | Medium: one Opus ticket, rerun the section 9 criteria | Medium |
| 6 | Banded sliding attention (exact, torch) | Training and HF scoring above 1,024 tokens | Time only: about 40% of the T-squared cost at 2,048 tokens and 60% at 4,096; peak memory and the trainable length unchanged (section 10) | Medium | Medium; gain unmeasured |
| 7 | DSpark in vLLM | Long outputs | 1.2-1.6x decode per the fork's document | Small (flags), if it fits in memory | Low-medium with adapters |
| 8 | FP8 cache in vLLM | Long contexts, more concurrent requests | 2x cache capacity, no speed | Small (flag) | Quality not re-benchmarked |
| 9 | Native Linux instead of WSL2 | Everything | Up to about 1-2 GiB of memory back (CPU offload works; a headless desktop holds less), and less overhead per kernel launch (size unknown) | A dual-boot setup | Low; nobody has measured the WSL share |
| 10 | Shared-decode expert kernel; a decode-specific small-block kernel | Training at batch 4; HF decode | 0.05-0.1 s per training example; at most 30% of HF decode | Medium-high | Low-medium |
| 11 | Memory-efficient or fused attention kernel for the full-attention calls | Training above about 2,000 tokens | Unknown until E-L1 | Medium (SDPA test) to high (Triton with backward) | Low |

Quality-affecting options: flag, do not adopt without an evaluation.

- **Fewer loop passes (3 to 2).** Removes 12 of 48 layer calls: 25% of the time everywhere, vLLM included, and a
  quarter of the cache. The checkpoint was trained with 3 fixed passes, so this is a different, weaker base model,
  and adapters must be trained and served with the same count. One config value to try (headroom E6).
- **Fewer experts per token (top-8 to top-6 or 4).** Helps much less than it looks. In training and prefill nearly
  all 128 experts are hit whatever K is, and the kernel's cost follows the number of blocks decoded, not the rows,
  so the saving is small. It helps only batch-1 decoding, where the cost is the 8 experts' weights read per layer
  call, and there vLLM is already within 3x of the memory floor. It changes every layer's output. Not recommended.
- **Expert pruning or merging (fewer than 128 experts).** Would save memory (10.7 GiB of experts is what makes
  everything tight) more than time. It needs routing statistics on our data, re-quantization, and a router that was
  never trained for the smaller set. A research project, not a speed fix.
- **Re-quantizing layers 20-23 to 4 bits** is already such an option and is in use for training; section 4.4
  explains why serving on vLLM puts it back in question.

## 7. Follow-up experiments

Rules for all of them: run under `tests/gpu/gpu_guard.gpu_session(...)` (HF) or with the lock held by a wrapper
script (vLLM), GPU otherwise idle; `torch.cuda.synchronize()` before and after every timed HF region and
`time.perf_counter()`; for vLLM, wall-clock per request from the client plus the server's own token counts; no
`torch.profiler` with CUDA activity (it leaked about 60 GB of Windows kernel memory in WSL); warm up before timing.

| ID | Question | Method | Prediction |
|---|---|---|---|
| G1 | HF decode with all layers fused | `generate` with `experts_impl="fused"`, `mxfp8_storage="nvfp4"`: 200 new tokens after a 500-token prompt; time prefill and decode separately | 4.5-5 tok/s; prefill 0.35-0.45 s |
| G2 | Where an HF decode step goes | One run with `backend.gemm` wrapped in synchronising timers, one with the experts module wrapped, one unwrapped | Kernels 30-70 ms, glue 15-20 ms, rest 150-200 ms per token |
| V1 | vLLM batched throughput | 1, 4, 8, 16 concurrent judge requests (470 in, 53 out), base model; examples per second and total tok/s | 1.2, then 2.5-3.5 examples per second at 8 or more |
| V2 | Adapter parity | Convert one adapter; 300 validation examples on HF fused and on vLLM; verdict agreement, label-probability difference; repeat HF twice for its own spread | Agreement within the HF run-to-run spread; if not, test vLLM without FP8 cache and DSpark first |
| V3 | Verdict scoring against generation | Same 300 examples: forced prefix plus one token with log-probabilities, against the generated verdict | The same verdicts except near-ties; 0.2 s per example |
| V4 | DSpark with an adapter | V1 at 1 and 8 requests with DSpark on; then a 1,000-token free-text reply | 1.2-1.6x on long replies, under 1.2x on judge replies; may not fit in memory |
| V5 | Prefix caching | V1 at 8 requests with and without a shared system prompt warm | About 0.05 s per example |
| L1 | Long-sequence training cost | Headroom E1 extended to 1,536, 2,048, 3,072 tokens with the loss sliced; fit a + bT + cT squared; record peak memory | c gives 2-3.5 s at 2,048; the peak follows 400-500 T-squared bytes |
| L2 | Banded sliding attention | A local patch in the harness; compare the per-layer outputs with the eager path on captured inputs, then L1 again | The same outputs within bf16 rounding; about 40% of the T-squared term gone at 2,048 tokens, 60% at 4,096 (ticket A1) |

## 8. Tickets: Inkling adapters on vLLM (approved 2026-11-22)

Goal: the vLLM fork handles generation, evaluation and serving of our Inkling adapters. Four tickets.

### 8.1 Facts checked for these tickets (CPU only)

- PEFT saves our adapter keys as
  `base_model.model.model.language_model.layers.{N}.self_attn.{hf}.lora_{A|B}.weight` (the model is
  `Inkling2LoopForConditionalGeneration.model.language_model.layers[N].self_attn`).
- The checkpoint names are `model.llm.layers.{N}.attn.{ckpt}.weight` (safetensors headers), all bf16, with these
  shapes (out, in):

  | HF module (`{hf}`) | Checkpoint (`{ckpt}`) | Shape | In our targets today |
  |---|---|---|---|
  | `q_proj.linear` | `wq_du` | (3072, 2304) | yes |
  | `k_proj.linear` | `wk_dv` | (1024, 2304) | yes |
  | `v_proj.linear` | `wv_dv` | (1024, 2304) | yes |
  | `o_proj.linear` | `wo_ud` | (2304, 3072) | yes |
  | `o_proj.gate_proj` | `gate_attn` | (3072, 2304) | yes |
  | `relative_position_bias.linear` | `wr_du` | (384, 2304) | no |
  | `sink_proj` | `attn_sink_proj` | (24, 2304) | no |

  **`wq_du` and `gate_attn` have the same shape, and so do `wk_dv` and `wv_dv`. A swapped mapping passes every
  shape check.** Only the behavioural test in V3 can catch it.
- vLLM reads `adapter_model.safetensors` and `adapter_config.json` itself (no PEFT needed in `~/vllm-venv`). It
  strips `base_model.model.`, maps the prefix `model.llm.` to `model.`, and takes the module name up to
  `.lora_A.weight`. So the converted key must be
  `base_model.model.model.llm.layers.{N}.attn.{ckpt}.lora_{A|B}.weight`. It rejects an adapter whose last name
  component is not a supported module, and rejects DoRA, `lora_bias`, `modules_to_save` and unusual inits. It
  applies `lora_alpha / r` itself (`alpha / sqrt(r)` with `use_rslora`).
- vLLM has `/tokenize` and `/detokenize` at the server root, `chat_template_kwargs` and `logprobs`/`top_logprobs`
  on chat completions, token-id prompts and `logprobs` on completions, and at most 20 alternatives
  (`max_logprobs` default).
- `domain_verifier_factory/llm.py` talks to endpoints with `urllib` only. Neither WSL environment has `openai` or
  `httpx`, and none is needed.
- The Windows `uv` environment has `safetensors` 0.8.0 and torch, so the converter and its tests run there.

### 8.2 Constraints for every ticket

- **Files.** Only the files a ticket lists. Never touch `training/inkling.py`, `inkling_kernels.py`,
  `inkling_triton.py`, `inkling_nvfp4.py`, `qlora.py`, `configs.py`, `training/bucketing.py`, `scripts/inkling/*`
  or the existing files in `tests/gpu/` (importing from them is fine). Do not edit
  `evaluation/predictors.py` either: the new predictor is its own module.
- **GPU.** GPU steps wait for the coordinator's word that the card is free. HF scripts run inside
  `tests.gpu.gpu_guard.gpu_session(min_free_mib=21000)`. The vLLM server takes the same lock file with `flock`
  (V0). A client script that talks to a running server takes no lock (the server holds it). Exit code 75 means
  busy: stop and report, do not retry in a loop.
- No `torch.profiler` with CUDA activity. No installs, downloads or commits. Nothing in these tickets needs an
  install or a download: the model snapshot and both environments exist, and the server runs with
  `HF_HUB_OFFLINE=1`.
- CPU tests: `uv run pytest tests/<file>` on Windows. They must not import Triton, vLLM or the Inkling
  transformers build.
- Do not print or log adapter tensors; do not write outputs outside the directory given on the command line.

### 8.3 Ticket list

| ID | Tier | Summary | Depends on | GPU |
|---|---|---|---|---|
| V0 | haiku | `serve.sh`: the working launch flags as a repo script, with LoRA and the GPU lock | - | 10 min smoke (run inside V3) |
| V1 | sonnet | Adapter converter (PEFT names to checkpoint names), random-adapter generator, CPU tests | - | none |
| V2 | sonnet | `VLLMJudgePredictor` and an `--endpoint` path in `scripts/adapter_eval.py`, CPU tests against a fake server | - | none (exercised in V3) |
| V3 | opus | Parity check: HF side, vLLM side, report with the fixed rule; runs V0-V2 on the GPU | V0, V1, V2 | about 1.5 h |

V0, V1 and V2 touch disjoint files and run in parallel. V3's three scripts can be written while V0-V2 are in
review (they depend only on the interfaces fixed below), but V3 is reviewed and run last.

---

### V0. vLLM launch script (haiku)

> **Done 2026-11-22.** Byte-for-byte the spec; `bash -n` passes; with the GPU lock held it prints "held; exiting 75" and doesn't start vLLM. Accepted without an Opus review (haiku ticket, verify passed). Gap noted for V3: unlike gpu_guard.py, it doesn't refresh the lock file's mtime (the WSL 30-day /tmp clean); it matters only for a lock file left untouched for 30 days.

Tier reason: a mechanical copy of a working command with fixed additions.

File (new): `scripts/inkling_vllm/serve.sh`.

Reference: scratchpad `tools/vllm_serve.sh` (ran on 2026-11-21: 8k context, 93 tok/s).

Spec, exactly:

```bash
#!/usr/bin/env bash
# Serve Inkling-2-Mini with the vLLM Ampere fork (WSL): text only, bf16 KV cache, no speculative decoding, LoRA on.
# Usage: scripts/inkling_vllm/serve.sh [NAME=ADAPTER_DIR ...]   (converted adapters, see convert_adapter.py)
# No --cpu-offload: on WSL it needs pinned memory for CUDA graphs, which WSL lacks (failed 2026-11-21).
set -euo pipefail
exec 9>/tmp/inkling_gpu.lock
flock -n 9 || { echo "GPU lock /tmp/inkling_gpu.lock is held; exiting 75" >&2; exit 75; }
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0) || exit 75
[ "$free" -ge 21000 ] || { echo "only $free MiB free on GPU 0, need 21000; exiting 75" >&2; exit 75; }
M=$(ls -d ~/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/*/ | head -n 1)
lora=()
[ "$#" -gt 0 ] && lora=(--lora-modules "$@")
HF_HUB_OFFLINE=1 exec ~/vllm-venv/bin/vllm serve "$M" --served-model-name inkling-2-mini --dtype bfloat16 \
  --max-model-len 8192 --max-num-seqs "${MAX_NUM_SEQS:-8}" --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.94 --kv-cache-memory=2684354560 \
  --limit-mm-per-prompt.image=0 \
  --compilation-config.cudagraph_mode=FULL_DECODE_ONLY \
  --enable-lora --max-lora-rank 16 --max-loras "${MAX_LORAS:-1}" --max-cpu-loras 8 \
  --port "${PORT:-8011}" "${lora[@]}"
```

Decisions: `exec` keeps file descriptor 9, so the lock is held for the server's lifetime and released when it
exits. `--max-num-seqs` rises from 1 to 8 (the cache pool is a fixed 2.5 GiB, so this changes concurrency, not the
pool); if the server fails to start at 8, V3 retries once with `MAX_NUM_SEQS=4`, then `1`, and records which
worked. `--max-loras 1` (one adapter active per batch, up to 8 kept on the CPU and swapped in). No FP8 cache, no
DSpark, no CPU offload. The scratchpad's background `nvidia-smi` logger is dropped. Base model requests use the
name `inkling-2-mini`; an adapter is addressed by its `NAME`.

Acceptance:

- CPU: `bash -n scripts/inkling_vllm/serve.sh` passes in WSL; the file has LF line endings and is executable.
- CPU (WSL): with the lock held by `flock /tmp/inkling_gpu.lock sleep 5 &`, the script exits 75 without starting
  vLLM.
- GPU (run by V3): the server answers `GET /v1/models` with `inkling-2-mini` and every `NAME`.

Review checklist:

- The lock is taken before the memory check, on the same path `gpu_guard.py` uses, and survives the `exec`.
- Every flag of the scratchpad script is present unchanged except `--max-num-seqs`; no `--cpu-offload`,
  `--kv-cache-dtype` or `--speculative-config`.
- With no arguments, no empty `--lora-modules` is passed (`set -u` and an empty array must not abort on the bash
  in WSL: test both cases).
- CRLF line endings would break the script in WSL: check with `file`.

---

### V1. Adapter converter and random-adapter generator (sonnet)

> **Done 2026-11-22.** 20 CPU tests pass. The Opus review found no blocking issues: all 7 module pairs match transformers' conversion_mapping (no swap in the same-shape pairs); vLLM's parse_fine_tuned_lora_name gives the right target for all 336 converted keys; PEFTHelper accepts the config (scaling 2.0, applied once by vLLM); the looped layers 6–17 share one adapter weight in both stacks. Accepted minor notes: a zero-padded layer index wouldn't round-trip through invert_key; the bad-key test matches loosely; `lora_alpha` is written as a float. The real swap check is V3's single-module adapters.

Tier reason: a pure rename with exact CPU tests; the one silent failure (same-shape swaps) is fixed by the table
above and caught in V3.

Files (all new): `domain_verifier_factory/export/inkling_vllm_adapter.py`,
`scripts/inkling_vllm/convert_adapter.py`, `scripts/inkling_vllm/make_random_adapter.py`,
`tests/test_inkling_vllm_adapter.py`.

Spec for `inkling_vllm_adapter.py` (imports: `json`, `re`, `hashlib`, `pathlib`, `torch`, `safetensors.torch`):

- `MODULE_MAP: dict[str, str]`: the seven rows of the table in 8.1, HF name to checkpoint name.
- `MODULE_SHAPES: dict[str, tuple[int, int]]`: checkpoint name to (out, in), the seven rows.
- `NUM_LAYERS = 24`.
- `PEFT_KEY = re.compile(r"^base_model\.model\.model\.language_model\.layers\.(\d+)\.self_attn\.(" + alternatives + r")\.lora_([AB])\.weight$")`,
  with the alternatives being the escaped keys of `MODULE_MAP`, longest first.
- `convert_key(key: str) -> str`: returns
  `f"base_model.model.model.llm.layers.{n}.attn.{MODULE_MAP[hf]}.lora_{ab}.weight"`; raises `ValueError` naming the
  key if it does not match or `n >= NUM_LAYERS`.
- `invert_key(key: str) -> str`: the exact inverse; `invert_key(convert_key(k)) == k`.
- `convert_state_dict(tensors: dict[str, Tensor], r: int) -> dict[str, Tensor]`:
  1. convert every key; collect all failures and raise one `ValueError` listing every unconvertible key (never
     skip a key);
  2. raise if two input keys give the same output key, or if the output has a different number of entries;
  3. every module must have both `lora_A` and `lora_B`; `lora_A` must have shape `(r, in)` and `lora_B`
     `(out, r)` from `MODULE_SHAPES`; raise otherwise, naming the key and both shapes;
  4. tensors are passed through unchanged: same object, same dtype, no scaling, no transpose.
- `convert_config(cfg: dict, present: set[str]) -> dict`: raises `ValueError` if any of these holds:
  `peft_type != "LORA"`, `use_dora`, `lora_bias`, `bias != "none"`, non-empty `modules_to_save`, non-empty
  `rank_pattern` or `alpha_pattern`, `init_lora_weights` not in `(True, "gaussian")`, `layer_replication`,
  `alora_invocation_tokens`. Returns exactly these keys:
  `{"peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": "thinkingmachines/Inkling-2-Mini",
  "r": cfg["r"], "lora_alpha": cfg["lora_alpha"], "use_rslora": bool(cfg.get("use_rslora", False)),
  "target_modules": sorted(present), "bias": "none", "modules_to_save": None, "use_dora": False,
  "lora_bias": False, "init_lora_weights": True, "rank_pattern": {}, "alpha_pattern": {}}`.
  `present` is the set of checkpoint module names found in the tensors (for our adapters:
  `["gate_attn", "wk_dv", "wo_ud", "wq_du", "wv_dv"]`). `lora_dropout` is dropped (inference only). The source's
  regex `target_modules` is not copied.
- `convert_adapter(src: Path, dst: Path) -> dict`: reads `src/adapter_model.safetensors` and
  `src/adapter_config.json`; refuses if `dst` exists and is not empty, or if `dst == src`; writes
  `dst/adapter_model.safetensors` (with safetensors metadata `{"format": "pt"}`), `dst/adapter_config.json`, and
  `dst/conversion.json` = `{"source": str(src), "source_sha256": <hex of the source safetensors file>,
  "num_tensors": n, "modules": sorted(present), "layers": sorted layer numbers, "r", "lora_alpha",
  "agent_config": <the source's agent_config.json content, or null>}`. Then it reloads the written file and
  checks: the same number of tensors, and for every source key `torch.equal(src[k], written[convert_key(k)])`.
  Returns the `conversion.json` content. It copies no tokenizer files (the server uses the base model's).
- `scripts/inkling_vllm/convert_adapter.py SRC DST`: calls `convert_adapter` and prints the summary as JSON.

Spec for `make_random_adapter.py` (a test adapter with known non-zero weights, since no trained Inkling adapter
exists; PEFT's own init leaves `lora_B` at zero, which would test nothing):

- Arguments: `OUT_DIR`, `--seed 0`, `--rel-norm 0.05`, `--modules` (comma-separated HF names, default the five
  targets), `--r 16`, `--alpha 32`, `--snapshot` (default: the local Inkling snapshot, found as
  `tests/gpu/inkling_snapshot.py` does, or by globbing the Hugging Face cache; no download, error if missing).
- For each layer 0-23 and each requested module, read the base weight `W` (bf16, CPU, from the safetensors file
  that holds `model.llm.layers.{N}.attn.{ckpt}.weight`); draw `A` (r, in) and `B` (out, r) from
  `torch.Generator().manual_seed(seed * 1000003 + layer * 101 + index of the module in MODULE_MAP)` as standard
  normals in float32; scale `B` by `rel_norm * ||W||_F / ||(alpha / r) * B @ A||_F` (norms in float64). So every
  module's update has the same relative size.
- Write `OUT_DIR/adapter_model.safetensors` with **PEFT key names** (the format of 8.1) in float32, and a PEFT
  `adapter_config.json`: `{"peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path":
  "thinkingmachines/Inkling-2-Mini", "r": 16, "lora_alpha": 32, "lora_dropout": 0.0, "bias": "none",
  "target_modules": <the regex of the requested modules, built like INKLING_LORA_TARGETS>, "use_rslora": false,
  "use_dora": false, "inference_mode": true, "init_lora_weights": true}`.
- Runs on the CPU in WSL with `~/inkling-venv/bin/python` (it reads about 1.3 GiB of weights in total, one tensor
  at a time).

Acceptance (CPU, `uv run pytest tests/test_inkling_vllm_adapter.py`; synthetic tensors, no checkpoint):

- A full synthetic adapter (24 layers x 5 modules x A/B = 240 tensors, r = 4, the real shapes) converts to 240
  keys; the key set equals a set written out by a separate literal comprehension in the test; every tensor is
  `torch.equal` to its source.
- The expected mapping in the test is a literal dict of the seven pairs, typed independently of `MODULE_MAP`,
  with a comment pointing to `transformers/conversion_mapping.py` lines 379-396.
- `invert_key(convert_key(k)) == k` for every key; `convert_key` is injective over all 24 x 7 x 2 keys.
- One extra key (`...layers.3.mlp.gate.lora_A.weight`), a layer 24, a vision-tower key
  (`base_model.model.model.vision_tower...q_proj.lora_A.weight`), a missing `lora_B`, a wrong rank and a
  transposed tensor each raise `ValueError` naming the key.
- `convert_config`: the output equals a literal expected dict for a config like PEFT writes for our adapters
  (regex `target_modules`, `lora_dropout`, `inference_mode` and other extra fields present); each rejected feature
  raises.
- `convert_adapter` on a temporary directory writes the three files; a second run into the same directory
  raises; `conversion.json` has the right hash and count.
- The scale rule of `make_random_adapter` (factored into a function `scaled_pair(W, r, alpha, rel_norm, gen)`)
  gives `||(alpha/r) B A||_F / ||W||_F == rel_norm` within 1e-5 on a random 64 x 32 `W`, and the same seed gives
  identical tensors.

Acceptance (CPU, WSL, after the above; not pytest): run `make_random_adapter.py` into a scratchpad directory,
convert it, then run this check with `~/vllm-venv/bin/python` (CPU, no model load): for every converted key,
`vllm.lora.utils.parse_fine_tuned_lora_name(key, mapper)` with
`mapper = WeightsMapper(orig_to_new_prefix={"model.llm.": "model."})` must return
`(f"model.layers.{n}.attn.{ckpt}", ab == "A")`, and `PEFTHelper.from_dict` (or the class constructor) must accept
the converted config. If importing those two vLLM modules needs a GPU or fails, report it; do not work around it.

Review checklist (silent failure modes):

- `MODULE_MAP` against the table in 8.1 and against `conversion_mapping.py`, pair by pair: `o_proj.gate_proj`
  goes to `gate_attn`, not to `wo_ud`; `q_proj.linear` to `wq_du`; `k` and `v` not swapped.
- The regex alternatives are escaped and anchored, so `o_proj.linear` cannot match inside another name and no
  key outside `language_model...self_attn` converts.
- No key is ever dropped: an unknown key is an error, not a warning. The count check and the reload check exist.
- The output path segment is `.attn.`, not `.self_attn.`, and the prefix is `base_model.model.model.llm.`.
- No scaling is applied anywhere in the converter (vLLM applies alpha / r; scaling twice would look plausible).
- `r` and `lora_alpha` are copied, not recomputed; `use_rslora` is carried over.
- The random adapter's `lora_B` is non-zero and the generator does not depend on dict ordering.
- The test's expected names are literals, not built from the module under test.

---

### V2. vLLM judge predictor and the `--endpoint` path (sonnet)

> **Done 2026-11-22.** 20 CPU tests (fake HTTP server). The Opus review found no correctness bugs in chunking, ordering, margins or the CLI guards. Fixed over 2 rounds:
> - connection drops are retried;
> - the chunk/resume tests have index-dependent preds, plus a resume at a different batch size;
> - a one-time /v1/models check that the model exists and its parent is the base;
> - label tokens are matched by id (`return_tokens_as_token_ids`), falling back to text only when a 400 names that option;
> - a cut-off partial line is truncated on resume;
> - `--concurrency < 1` is rejected, and `--endpoint --stock` works without `--served-model`.
>
> 4xx errors raise `HTTPFailure`.

Tier reason: clear specification, testable on the CPU against a fake HTTP server.

Files: `domain_verifier_factory/evaluation/vllm_predictor.py` (new), `scripts/adapter_eval.py` (edit),
`tests/test_vllm_predictor.py` (new), `tests/test_adapter_eval_resume.py` (extend only).

Reference behaviour: `AdapterPredictor` in `evaluation/predictors.py` (prompts, token budget, parsing,
`verdict_probs`). The new predictor must return the same dictionary: the keys of `parse_prediction` and, with
`record_margin`, `label_probs`, `label_mass`, `margin`.

Spec for `VLLMJudgePredictor`:

- `__init__(self, model: str, base_url: str = "http://localhost:8011/v1", *, domain: str,
  scheme: str = "four_way", display_name: str | None = None, max_passages: int = 10,
  rationale_first: bool = False, chat_template_kwargs: dict | None = None, concurrency: int = 8,
  timeout: float = 300.0, api_key: str | None = None)`.
  `name = f"vllm:{model}"`; `batch_size = 4 * concurrency`; `record_margin = False` (an attribute, as on
  `AdapterPredictor`). `chat_template_kwargs=None` means `{"reasoning_effort": "none"}` (the literal, with a
  comment naming `INKLING_CHAT_TEMPLATE_KWARGS`; do not import `training.inkling`). System and user prompts come
  from `templates.system_prompt(domain, display_name, scheme, rationale_first=rationale_first)` and
  `templates.user_prompt(claim, evidence, max_passages)`.
- HTTP with `urllib.request` only, JSON bodies, optional bearer token. A private `_post(path_or_url, body)`
  retries on `URLError`, a timeout or HTTP 5xx after 2, 4 and 8 s, then raises; HTTP 4xx raises at once with the
  response body in the message. A failed request never becomes an "invalid" prediction.
- One request per example to `{base_url}/chat/completions`:
  `{"model", "messages": [system, user], "temperature": 0.0, "max_tokens": M, "chat_template_kwargs": {...}}`,
  where `M = 200`, or `24` when `record_margin` and not `rationale_first` (the same rule as `AdapterPredictor`).
  With `record_margin` add `"logprobs": true, "top_logprobs": 20`.
- The reply text is `strip_thinking(content or "")` (from `llm.py`), then `parse_prediction(reply, scheme)`.
- `predict(examples)`: `ThreadPoolExecutor(max_workers=concurrency)` with `executor.map`, so results are in input
  order. An exception in any request propagates.
- Margins. At construction (lazily, on the first `record_margin` request, once, under a lock): for each label of
  `label_vocabulary(scheme)`, `POST {root}/tokenize` with
  `{"model": model, "prompt": label, "add_special_tokens": false}` and take the first id, then
  `POST {root}/detokenize` with `{"model": model, "tokens": [id]}` to get that token's text; `root` is `base_url`
  without a trailing `/v1`. Raise if two labels share a first token or a text is empty. This is the same token
  rule as `AdapterPredictor.verdict_probs` (the label's first token, encoded alone).
  Per reply: walk `choices[0].logprobs.content`; with `text` the concatenation of the `token` fields before
  position j, the verdict step is the first j where `re.search(r'"verdict"\s*:\s*"$', text)`. At that step,
  `p[label] = exp(logprob)` of the entry in `top_logprobs` whose `token` equals the label's token text, else
  `0.0`. `label_mass = sum(p)`; `label_probs = p / label_mass`; `margin` = the top minus the runner-up of the
  renormalised values. If the step is not found or `label_mass == 0`, add nothing (as the reference returns `{}`).
  Known difference from the reference, to state in the docstring: only the 20 most likely tokens are visible, so
  a label outside them counts as 0 and `label_mass` is a lower bound.
- No stock/adapter switch inside the predictor: the stock model is the served name `inkling-2-mini`.

Spec for `scripts/adapter_eval.py`:

- New options: `--endpoint URL` (default none), `--served-model NAME` (required with `--endpoint`),
  `--base-served-model NAME` (default `inkling-2-mini`), `--concurrency N` (default 8).
- With `--endpoint`: build `VLLMJudgePredictor(base name if --stock else served name, URL, domain=...,
  display_name=..., max_passages=..., rationale_first=<as AdapterPredictor derives it from
  AGENT_DIR/agent_config.json: rationale_first and include_rationale>, chat_template_kwargs=<agent_config's
  chat_template_kwargs or None>, concurrency=...)`; set `record_margin`; do not load any model; skip the
  `disable_adapter` context. `--stack`, `--weights` or a non-default `--combine` with `--endpoint` exit with a
  clear message (weighted stacks are a PEFT feature; not supported on the endpoint). The report label is
  `f"vllm:{name} ({agent_dir})"`.
- Without `--endpoint`: behaviour unchanged.
- `evaluate_test_set`: predict the examples that are not done yet in chunks of
  `getattr(pred, "batch_size", 1)`, in order, and append each chunk's rows to the partial file (one line per
  example, same format) before starting the next chunk. With `batch_size` 1 the file contents and the progress
  output are identical to today's. Type hints change from `AdapterPredictor` to the `Predictor` protocol.
- Update the module docstring with one `--endpoint` example.

Acceptance (CPU, `uv run pytest tests/test_vllm_predictor.py tests/test_adapter_eval_resume.py`): a fake server
(`http.server.ThreadingHTTPServer` on port 0 in a thread) that records request bodies and returns canned
replies.

- The request body has the model name, both messages equal to the `templates` functions' output,
  `temperature` 0, `max_tokens` 200, and `chat_template_kwargs == {"reasoning_effort": "none"}`; with
  `record_margin`, `max_tokens` 24 and the two logprob fields; with `rationale_first`, 200 and the reason-first
  system prompt.
- 20 examples with `concurrency=4` against a server that delays replies by a random 0-30 ms and echoes the
  claim: the predictions come back in input order.
- A canned logprobs reply whose tokens spell `{"verdict": "supports", ...` with known alternatives gives the
  expected `label_probs`, `label_mass` and `margin` (computed by hand in the test); a reply where one label is
  missing from the alternatives gives 0 for it; a reply without the verdict key adds no margin keys.
- Two failures (HTTP 503) then success returns the prediction (sleep monkeypatched); four failures raise; HTTP
  400 raises at once.
- Two labels with the same first token make the predictor raise.
- `adapter_eval.evaluate_test_set` with a fake predictor of `batch_size` 4 over 10 examples writes the same final
  files as with `batch_size` 1, and a run interrupted after the first chunk (the predictor raises on its second
  call) resumes without predicting the first 4 again. The existing resume tests pass unchanged.
- `--endpoint` with `--stack` exits with an error; `--endpoint` without `--served-model` exits with an error.

GPU (run inside V3): `/tokenize` and `/detokenize` requests succeed with these bodies, and the chat request's
prompt token ids equal the HF tokenizer's (V3 step 5).

Review checklist (silent failure modes):

- `chat_template_kwargs` is sent on every request. Without `reasoning_effort: none` the model thinks first and
  the 24-token budget cuts the reply: plausible-looking but wrong verdicts.
- Results are in input order under concurrency; no shared mutable state between threads except the guarded
  label-token cache.
- The verdict step uses the text **before** token j, and the alternatives of token j (an off-by-one gives the
  probabilities of the token after the label).
- A request error is never turned into `{"verdict": None}`; it raises, and the partial file lets the run resume.
- Label tokens come from the server's tokenizer for the same model name, and the match is on exact token text.
- `evaluate_test_set` with `batch_size` 1 is unchanged (compare the partial-file lines), and chunks are written
  before the next chunk starts.
- `AdapterPredictor` and `predictors.py` are untouched; no new dependency is imported.

---

### V3. Parity check between HF and vLLM (opus)

> **Done 2026-11-22: FAIL by the fixed rule** (P1, P3, P4, P5 fail; P2, P6 pass). P1 misses by one prompt: agree(V, H1; base) 0.945 (189 of 200) against a bar of 0.950; its tv part passes (0.049 <= 0.070). By the reading order, P1 failing stops the check, so the P3 regeneration of `full` at rel-norm 0.10/0.20 was not run (both adapters are made and converted, ready). Inputs were token-identical: the server's chat tokenization equals the HF ids for 20 prompts, the assistant header (200001, 200004), prefix and label ids match, and the engine's prompt_logprobs cover exactly the ids sent (no added BOS). Every HF adapter passed the tensor-equality check (24 x targets modules); vLLM applies every adapter (P6). A caution for reading P4/P5: between our own two HF paths, shift_cos(full) is only 0.34, so at these adapter sizes the shift direction is mostly stack noise, and the low V cosines don't by themselves show a wrong mapping. Results and numbers in 8.5.
>
> **Bigger-adapter rerun, 2026-11-22 evening (approved): HF side done, vLLM side blocked, no verdict.** `full` at rel-norm 0.10 and 0.20 ran on H1 (now the K2 kernel; H1 base is bit-identical to the morning's) and H2. P3 passes at 0.20 (tv 0.133, n_shift 84) but not at 0.10 (tv 0.079). The P4 floor, shift_cos(full) between H2 and H1, is 0.56 at 0.10 and 0.67 at 0.20, so even at 0.20 our own two HF paths don't reach the 0.80 bar. The shift toward `inconclusive` seen on vLLM at 0.05 also appears on HF: H2 at 0.10, and H1 and H2 at 0.20. So it is the adapter's own effect, not a vLLM artifact. The vLLM server no longer starts: in 4 attempts after the WSL restart at 22:18, it fails while repacking the MoE weights with `CUDA driver error: device not ready` / `Failed to create GPU mapping`. The HF runs on the same card work. V at 0.10/0.20 and P5 are not measured. A retry after a clean WSL restart at 23:04 failed the same way on the first start. Details in 8.5.1.
>
> **Final, 2026-11-23: strict rule FAIL; judged against the HF floor, vLLM applies adapters equivalently. Use it for evaluation, and gate on the trained-adapter accuracy rerun.** The serve failure was `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` inherited by `serve.sh` (now unset there). At 0.20: P3 pass; P4 shift_cos 0.735, above the HF floor (0.67) but under the 0.80 bar; P6 pass for all six adapters (V/H1 tv ratio 0.80-1.17). P5 cosines are 0.51-0.75 against an expected floor of about 0.5-0.6 for shifts this size (no H2 singles measured). The swap checks are clean: cross-cosine H1 q vs V gate is 0.06, and H1 k vs V v is 0.14. One residual: on V every adapter tilts toward `inconclusive` more than on H1. Details in 8.5.2.

Tier reason: it judges numerics in a model that is chaotic at rounding level, its acceptance rule decides whether
adapters can be served on vLLM, and a wrong mapping can look plausible.

Files (all new): `scripts/inkling_vllm/parity_common.py`, `scripts/inkling_vllm/parity_hf.py`,
`scripts/inkling_vllm/parity_vllm.py`, `scripts/inkling_vllm/parity_report.py`,
`tests/test_inkling_vllm_parity.py`, and a results section appended to this document (8.5).

What is compared. Three differences exist between training and serving: (1) the adapter's names (V1); (2) layers
20-23's experts, which we train re-quantized to 4 bits and vLLM serves as the checkpoint's MXFP8; (3) different
kernels and rounding everywhere. The plan's section 8 showed that rounding alone moves this model's loss by 1-10%,
so equality is not available. The test therefore measures the distance between the two stacks against the
distance between our own two HF paths, and separately checks that the adapter moves the output in the same
direction on both stacks.

Fixed design:

- **Prompts.** The first 200 examples of
  `datasets/factcheck_full/factcheck_agent_dataset/validation.jsonl` whose prompt is at most 1,500 tokens (skip
  longer ones, keep file order), domain `factcheck`, verdict-first system prompt, `max_passages` 10. Each prompt
  is `tokenizer.apply_chat_template([system, user], add_generation_prompt=True, reasoning_effort="none")`
  followed by the token ids of the forced prefix `{"verdict": "` (encoded without special tokens).
  `parity_common.build_prompts` produces the ids once and saves them to `prompts.json`; both sides read that
  file, so the template cannot differ between them.
- **Quantity.** The next-token distribution at the end of the prompt, restricted to the four labels' first
  tokens (the rule of `AdapterPredictor.verdict_probs`) and renormalised: a 4-vector `p` per prompt. Also saved:
  the label mass.
- **Adapters** (made by `make_random_adapter.py`, seed 0): `full` (all five modules, `--rel-norm 0.05`) and five
  single-module adapters `q`, `k`, `v`, `o`, `gate` (`--rel-norm 0.15`, seed 0), plus `base` (no adapter). The
  five single-module adapters exist to catch the same-shape swaps.
- **Conditions.**
  - `H1`: HF, `experts_impl="fused"`, `mxfp8_storage="nvfp4"` (what training uses after T7): all 7 variants.
  - `H2`: HF, `experts_impl="loop"`, `mxfp8_storage="nf4"` (the reference path): `base` and `full` only.
  - `V`: vLLM through `serve.sh` with the six converted adapters: all 7 variants.
- **Metrics** for two sets of vectors `p`, `q` over the 200 prompts: `agree` = share of prompts with the same
  arg-max label; `tv` = mean over prompts of `0.5 * sum |p - q|`; `tv95` = its 95th percentile. And for an adapter
  `a` on stack `S`, the shift `s_S = p_S(a) - p_S(base)` per prompt; `shift_cos(a)` = mean over the prompts with
  `||s_H1|| >= 0.05` of `cosine(s_H1, s_V)`, and `n_shift(a)` = how many prompts that is.

**Acceptance rule (fixed before any run; do not change it after seeing results):**

- (P1) Stack parity without an adapter: `agree(V, H1; base) >= max(0.90, agree(H2, H1; base) - 0.03)` and
  `tv(V, H1; base) <= 1.5 * tv(H2, H1; base) + 0.01`.
- (P2) The same two inequalities for `full`.
- (P3) The adapter has an effect to compare: `tv(H1 full, H1 base) >= 0.10` and `n_shift(full) >= 50`. If not,
  regenerate `full` with `--rel-norm 0.10`, then `0.20` (each at most once, rerunning only `full`); if it still
  fails, stop and report.
- (P4) Same direction on both stacks: `shift_cos(full) >= 0.80`.
- (P5) Each single-module adapter with `n_shift >= 30` has `shift_cos >= 0.70`. One with `n_shift < 30` is
  reported as "no test" and regenerated once at `--rel-norm 0.30`; if still under 30, it is listed as untested
  (not a pass).
- (P6) vLLM actually applied each adapter: `tv(V a, V base) >= 0.5 * tv(H1 a, H1 base)` for every adapter that
  met its `n_shift` threshold.
- Reported, not gated: `tv95`, label mass per condition, the `V` vs `H2` numbers, and greedy agreement of 24
  generated tokens on the first 50 prompts (`full`, H1 vs V).

Pass = P1-P6 all hold. Reading a failure, in this order: P1 fails: the stacks differ before any adapter (weights
of layers 20-23, or the fork); report `tv(V, H2; base)` to see which HF path vLLM is closer to, and stop. P1
passes and P4/P5/P6 fail for some module: the converter's mapping or vLLM's LoRA path for that module. P2 fails
alone: the adapter amplifies the stack difference; report with the numbers and stop.

When the first trained adapter exists, rerun P1, P2, P4 and P6 with it (300 validation examples) and add:
accuracy of generated replies through `adapter_eval.py --endpoint` within 0.04 of the accuracy from H1 on the
same examples. This rerun is a follow-up, not part of this ticket's acceptance.

Spec:

1. `parity_common.py` (no GPU, no vLLM import): `build_prompts(tokenizer, examples, n=200, max_tokens=1500)`
   returning `{"ids": [[int]], "example_ids": [str], "label_token_ids": [int], "labels": [str]}` (assert the four
   label tokens differ); `metrics(p, q)`; `shift_cos(pH_a, pH_base, pV_a, pV_base, min_norm=0.05)`;
   `evaluate_rule(results) -> dict` implementing P1-P6 exactly as written, returning each check's numbers and a
   boolean. Pure numpy or torch on the CPU.
2. `parity_hf.py --condition {H1,H2} --prompts prompts.json --adapters name=DIR ... --out FILE`: inside
   `gpu_session(min_free_mib=21000)`; `load_inkling_4bit` with the condition's two settings (import only);
   `model.eval()`; for `base`, the bare model; for adapters, `PeftModel.from_pretrained` for the first and
   `load_adapter` for the rest, `set_adapter` per variant. After loading each adapter assert that no LoRA weight
   is missing: every tensor in the adapter file equals the loaded module's `lora_A`/`lora_B` weight
   (`torch.equal` after casting the file tensor to the module's dtype), and that the number of LoRA modules is
   24 x the number of target modules. One forward pass per prompt at batch 1 under `torch.no_grad()`, logits of
   the last position (the model applies its softcap), `softmax` in float32 over the full vocabulary, keep the
   four label probabilities and their sum. Writes JSON `{variant: {"p": [[4 floats]], "mass": [...]}}`. If
   `prompts.json` is missing it builds it first (the tokenizer runs on the CPU). For the 24-token generations
   (H1, `full`, first 50 prompts): `generate(max_new_tokens=24, do_sample=False)`, token ids saved.
3. `parity_vllm.py --base-url http://localhost:8011/v1 --prompts prompts.json --variants base,full,q,k,v,o,gate
   --out FILE`: for each variant (model name `inkling-2-mini` for `base`, else the variant's name) and each
   prompt, `POST /v1/completions` with `{"model", "prompt": <list of token ids>, "max_tokens": 1,
   "temperature": 0, "logprobs": 20, "return_tokens_as_token_ids": true}`; read the first position's
   `top_logprobs` (keys of the form `token_id:N`), take `exp` for the four label ids, 0 if absent; record the mass
   and how many labels were absent. 8 requests in flight (`ThreadPoolExecutor`), results in prompt order. The
   same output format as `parity_hf.py`. Also: for the first 20 examples, `POST {root}/tokenize` with the chat
   messages and `chat_template_kwargs={"reasoning_effort": "none"}`, `add_generation_prompt` true, and assert the
   returned ids equal the HF ids before the forced prefix (this is the template check V2 relies on); and the
   24-token greedy completions for the first 50 prompts with `full`.
4. `parity_report.py H1.json H2.json V.json`: prints the table of every metric and each of P1-P6 with its
   numbers, exits 0 on pass and 1 otherwise, and writes the same as Markdown for section 8.5.
5. Run order on the GPU (the coordinator says when): CPU first: make the 6 adapters, convert them, build
   `prompts.json`. Then `parity_hf.py --condition H1` (about 15 min), `--condition H2` (about 35 min: the loop
   path is slow), then `serve.sh full=... q=... k=... v=... o=... gate=...` in one terminal and, once
   `/v1/models` lists all seven names, `parity_vllm.py` (about 10 min), then stop the server. Never run an HF
   script while the server is up (the lock prevents it). Then `parity_report.py`.
6. Append section 8.5 to this document: the table, pass or fail per check, the `MAX_NUM_SEQS` that worked, the
   server's memory use, and any deviation.

Acceptance:

- CPU (`uv run pytest tests/test_inkling_vllm_parity.py`): `metrics` and `shift_cos` on hand-computed 4-vectors;
  `evaluate_rule` on synthetic results: an all-equal case passes; a case where `V`'s `q` and `gate` shifts are
  exchanged fails P5 for exactly those two; a case where `V` ignores the adapter (`V a == V base`) fails P4 and
  P6; a case with too small a shift returns the "no test" state, not a pass; thresholds at their boundary values
  behave as written (>= and <=).
- CPU: `build_prompts` with a stub tokenizer keeps file order, skips long prompts and appends the prefix ids.
- CPU: all four scripts parse; `parity_vllm.py` runs end to end against a fake server (as in V2's tests) that
  returns canned `top_logprobs`.
- GPU (about 1.5 h in total, the card otherwise idle): steps 5-6 completed; the report exists; the result is
  stated as pass or fail by the rule above, with failures reported as failures.

Review checklist (silent failure modes):

- The rule in `evaluate_rule` is the rule written above, constant for constant, and was not edited after the
  first GPU run (compare the file's history in the working tree with this document).
- Both sides read the same `prompts.json`; the vLLM side sends token ids, not text; no BOS is added twice
  (compare a `/tokenize` round trip for one prompt).
- The HF side really has the adapter active for each variant (the tensor-equality assertion runs, and
  `base` is measured with no adapter loaded or inside `disable_adapter()`), and `model.eval()` is set.
- The vLLM request names the adapter in `model`; a typo falls back to an HTTP 404, which must raise, never to
  the base model. P6 exists to catch a server that ignores the adapter.
- Label ids are the same list on both sides, taken from `prompts.json`; absent labels are counted and reported
  (many absences would make `tv` meaningless).
- H1 uses fused with NVFP4 storage and H2 loop with NF4; the report prints the settings each file was made with
  (saved in the JSON), so two files from the same condition cannot be compared by mistake.
- Single-module adapters use the HF names `q_proj.linear`, `k_proj.linear`, `v_proj.linear`, `o_proj.linear`,
  `o_proj.gate_proj`, one each, and the server was started with the converted copies, not the PEFT originals.
- No `torch.profiler`; HF scripts are inside `gpu_session(21000)`; the server was stopped at the end and the
  lock released.

### 8.4 Downloads, installs, GPU time

- Downloads and installs: none. Everything uses the existing snapshot, `~/inkling-venv` (HF side, PEFT 0.22),
  `~/vllm-venv` (server) and the Windows `uv` environment (CPU tests).
- GPU time: about 1.5 hours with the card otherwise idle (H1 15 min, H2 35 min, vLLM 15 min including the V0
  smoke test and start-up, the rest for model loads and one possible regeneration under P3 or P5). V0-V2 need no
  GPU time of their own.
- Not covered by these tickets, on purpose: FP8 cache, DSpark, more than one active adapter per batch
  (`MAX_LORAS` above 1, needed for the ensemble), weighted adapter stacks on the endpoint, and a verdict-scoring
  mode in the predictor (item 4 of section 6). Each changes numerics or memory and should follow a passed V3.

### 8.5 Results: parity between HF and vLLM (V3, 2026-11-22)

**Result: FAIL** by the rule fixed in V3. Scripts: `scripts/inkling_vllm/parity_{common,hf,vllm,report}.py`, CPU tests
`tests/test_inkling_vllm_parity.py` (19 pass). Run files in WSL `~/v3_parity/` (`prompts.json`, `H1.json`,
`H2.json`, `V.json`, `report_r05.md`, `rule_r05.json`, logs).

Run facts:

- 200 prompts, the first 200 of the validation file (none was over 1,500 tokens; longest 810). Label first tokens
  86995, 31891, 256, 340. Revision ed40e7da.
- Server: `serve.sh` started at `MAX_NUM_SEQS=8` first time; 21.8-22.5 GiB of 24 GiB used while serving; all seven
  names listed. vLLM side 15 s per variant, H1 68 s per variant (+ 276 s for the greedy run), H2 (loop/NF4) about
  805 s per variant. Peak HF allocation 15.5 GiB (H1). The server was stopped and the lock released at the end.
- Input identity, checked before measuring (the script raises on any mismatch): the server's `/tokenize` of the chat
  messages equals the HF prompt ids for the first 20 prompts; the server derives the same assistant header
  (200001, 200004), the same forced-prefix ids and the same label ids; the engine's `prompt_logprobs` for prompt 0
  cover exactly the 489 ids sent. No problem found.
- HF adapter checks: every file tensor equals the loaded `lora_A`/`lora_B` weight; 120 LoRA modules for `full`, 24
  for each single-module adapter; with all adapters disabled the wrappers give the bare model's numbers exactly
  (max difference 0.0).
- Label mass is about 0.9998 everywhere; on V, 95-135 of the 800 label slots per variant were outside the top 20
  (each one a near-zero label; mass stays 0.9998).

| Check | Numbers | Result |
|---|---|---|
| P1 (base) | agree(V,H1) 0.945 >= 0.950 (agree(H2,H1) 0.980); tv(V,H1) 0.0487 <= 0.0696 (tv(H2,H1) 0.0397) | **fail** (agree, by one prompt) |
| P2 (full) | agree(V,H1) 0.935 >= 0.930 (agree(H2,H1) 0.960); tv(V,H1) 0.0470 <= 0.0762 (tv(H2,H1) 0.0442) | pass |
| P3 | tv(H1 full, H1 base) 0.0648 >= 0.10; n_shift(full) 51 >= 50 | **fail** (rel-norm 0.05; not regenerated, see below) |
| P4 | shift_cos(full) 0.247 >= 0.80 (n 51) | **fail** |
| P5 q / k / v / o / gate | shift_cos 0.097 / 0.365 / 0.259 / 0.533 / 0.604 >= 0.70 (n_shift 53 / 53 / 58 / 60 / 55) | **fail** (all five) |
| P6 | tv(V a, V base) >= 0.5 x tv(H1 a, H1 base): full 0.063 vs 0.065, q 0.065 vs 0.062, k 0.059 vs 0.063, v 0.066 vs 0.077, o 0.088 vs 0.079, gate 0.070 vs 0.059 | pass (all six) |

Reported, not gated:

| Pair | Variant | agree | tv | tv95 |
|---|---|---|---|---|
| V vs H1 | base | 0.945 | 0.0487 | 0.320 |
| V vs H1 | full | 0.935 | 0.0470 | 0.280 |
| H2 vs H1 | base | 0.980 | 0.0397 | 0.202 |
| H2 vs H1 | full | 0.960 | 0.0442 | 0.244 |
| V vs H2 | base | 0.960 | 0.0522 | 0.351 |
| V vs H2 | full | 0.955 | 0.0514 | 0.282 |

- Arg-max disagreements at base: V-H1 11, H2-H1 4, V-H2 8 of 200. By tv, V is about as far from H2 (0.052) as from
  H1 (0.049), and both are farther than H1 from H2 (0.040): V is not closer to either HF path.
- Greedy, `full`, H1 vs V, 24 tokens, first 50 prompts: 34 identical; median first difference at token 12.
- Diagnostic for P4/P5 (computed from the same files): shift_cos(full) between H1 and H2, our two HF paths with the
  identical PEFT adapter, is 0.34 (n 51); H2 vs V is 0.25. So at rel-norm 0.05 the per-prompt shift direction is
  mostly stack noise, and the 0.80 bar is out of reach even between HF paths. The low V cosines don't show a wrong
  mapping by themselves. One pattern worth a look: on V every adapter moves the mean probability toward
  `inconclusive` (+0.016 to +0.034) and away from `irrelevant`; on H1 the mean shifts have no common direction.

Deviations from the ticket:

- Prompt ids carry the assistant header: prompt + `<|message_model|><|content_text|>` (200001, 200004) + prefix.
  Inkling's template ignores `add_generation_prompt`, so the ticket's literal recipe would have asked for the
  next message, not for the verdict. The header is derived as `verdict_scoring.assistant_header_ids` does it, and the
  server derives the same one.
- The vLLM side ran first (H1, H2 after it, never at the same time), so a template mismatch would surface before the
  long HF runs.
- The BOS check uses the engine's `prompt_logprobs` (count and ids of what it ran) instead of a `/tokenize` round
  trip, which can re-merge BPE pieces and fail falsely.
- Comparisons in `evaluate_rule` allow 1e-12 of slack, so a value exactly at a decimal bar (194/200 against
  1.0 - 0.03) isn't lost to float rounding. It changes no result here.
- Result JSONs have a `_meta` key next to the variants (settings, prompt digest, adapter hashes, greedy ids).
- P3 regeneration not run: the reading order stops at a P1 failure. `full` at rel-norm 0.10 and 0.20 is made and
  converted (`~/v3_parity/{peft,conv}/full_r10`, `full_r20`), and `~/v3_parity/run_regen.sh 0.10 full_r10` reruns
  only `full` on H1, H2 and V (about 25 min) if the coordinator wants that information anyway.

#### 8.5.1 Bigger-adapter rerun (approved by the user, 2026-11-22 evening)

`full` regenerated at rel-norm 0.10 and 0.20 (seed 0), rerun on H1 (base and full, now on the K2 kernel) and H2
(full; base kept from the morning). H1 base on K2 equals the morning's H1 base exactly (tv 0.000, agree 1.000).
Files: WSL `~/v3_parity/{H1,H2}.full_r10.json`, `{H1,H2}.full_r20.json`; the 0.05 originals are kept as
`*.pre_full_r10.json`.

| rel-norm | tv(H1 full, H1 base) | n_shift | P3 | tv(H2 full, H2 base) | shift_cos H2 vs H1 (P4 floor) | agree / tv, H2 vs H1 on full |
|---|---|---|---|---|---|---|
| 0.05 | 0.065 | 51 | fail | - | 0.34 | 0.960 / 0.044 |
| 0.10 | 0.079 | 61 | fail | 0.110 | 0.56 | 0.940 / 0.060 |
| 0.20 | 0.133 | 84 | **pass** | 0.147 | 0.67 | 0.920 / 0.060 |

Mean shift of p per label, full minus base (supports, contradicts, inconclusive, irrelevant):

| rel-norm | H1 | H2 | V |
|---|---|---|---|
| 0.05 | +0.009, +0.005, -0.027, +0.012 | - | +0.005, -0.005, +0.016, -0.016 |
| 0.10 | +0.012, +0.004, -0.004, -0.012 | +0.015, -0.006, +0.027, -0.035 | not measured |
| 0.20 | +0.022, -0.004, +0.025, -0.042 | +0.025, -0.002, +0.039, -0.062 | not measured |

Readings:

- The move toward `inconclusive` and away from `irrelevant`, which looked vLLM-specific at 0.05, appears on HF too:
  H2 at 0.10, and both HF paths at 0.20. It is the random adapter's own effect, and the stacks pick it up at
  different rates.
- Even between our own two HF paths, shift_cos stays under the P4 bar (0.80) at every scale tried (best 0.67 at
  0.20). With this model's rounding sensitivity, P4 as written is likely out of reach for any stack pair at these
  adapter sizes; a V value near the H2-H1 floor would be the realistic sign of equivalence.
- **vLLM side blocked.** After the WSL restart at 22:18 (driver: KMD 610.88, CUDA UMD 13.3), `serve.sh` fails 4
  times out of 4. It always fails at the same point (MoE weight repack in `humming/ops/weight.py`) with
  `RuntimeError: CUDA driver error: device not ready` and `Failed to create GPU mapping`. 24 GiB is free, the host
  has 30 GB free and the non-paged pool is 1.6 GB. The HF runs on the same card in between work, and the same
  `serve.sh` worked at 14:13. So V at 0.10/0.20, P4, P6 and P5 at 0.20 (the single-module adapters are built and
  converted, `~/v3_parity/conv/*_r20`; `run_p5.sh r20` runs them) are not measured, and **there is no verdict yet**.
  To resume once the server starts: `SKIP_HF=1 ~/v3_parity/run_regen.sh 0.20 full_r20` (3 min), then
  `~/v3_parity/run_p5.sh r20` (about 10 min).
- Retried 23:05 after a clean WSL restart at 23:04 (non-paged pool 1.7 GB, driver unchanged, GPU idle): `serve.sh`
  failed the same way on the first start (MoE weight repack, `device not ready` / `Failed to create GPU mapping`).
  Stopped there, as instructed; still no verdict. The fault survives a WSL restart, so it is on the Windows side
  (driver or host GPU-mapping state) or in the fork's repack allocation, not in WSL's state.

#### 8.5.2 Final results (2026-11-23)

Root cause of the serve failures: the regen scripts export `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for HF,
`serve.sh` inherited it, and the fork's MoE repack fails with it on WSL ("Failed to create GPU mapping"). `serve.sh`
now unsets it. The NVIDIA driver auto-updated to 621.24 around 23:20. After that, one H1 run hung in driver calls on
its first forward; it was killed. On 621.24, `check_kernel.py` passes, and after a fresh start the H1 base pass is
normal (69 s) and the H1 base distribution is identical to the morning's. Input identity was checked again on every
server start: no mismatches.

| rel-norm | Adapter | tv H1 | tv V | P6 | n_shift | shift_cos V vs H1 | HF floor (H2 vs H1, full) |
|---|---|---|---|---|---|---|---|
| 0.05 | full | 0.065 | 0.063 | pass | 51 | 0.247 | 0.34 |
| 0.10 | full | 0.079 | 0.078 | pass | 61 | 0.540 | 0.56 |
| 0.20 | full | 0.133 | 0.145 | pass | 84 | **0.735** | 0.67 |
| 0.20 | q | 0.060 | 0.065 | pass | 58 | 0.509 | - |
| 0.20 | k | 0.059 | 0.068 | pass | 49 | 0.523 | - |
| 0.20 | v | 0.081 | 0.065 | pass | 62 | 0.585 | - |
| 0.20 | o | 0.091 | 0.099 | pass | 66 | 0.750 | - |
| 0.20 | gate | 0.072 | 0.073 | pass | 62 | 0.550 | - |

Swap check (same-shape pairs, shift_cos of H1's adapter against V's other adapter): H1 q vs V gate 0.058, H1 k vs V v
0.141, against 0.51-0.55 on the diagonal. There is no swap.

Mean shift on `inconclusive` / `irrelevant` (full minus base):

| rel-norm | Adapter | H1 | H2 | V |
|---|---|---|---|---|
| 0.05 | full | -0.027 / +0.012 | - | +0.016 / -0.016 |
| 0.10 | full | -0.004 / -0.012 | +0.027 / -0.035 | +0.023 / -0.028 |
| 0.20 | full | +0.025 / -0.042 | +0.039 / -0.062 | +0.052 / -0.063 |
| 0.20 | q, k, v, o, gate | -0.021, +0.014, -0.032, +0.001, +0.013 (inconclusive) | - | +0.005, +0.036, +0.018, +0.042, +0.047 |

Verdict:

- **Strict rule: FAIL.** P1 misses by one prompt (agree 0.945 vs 0.950). P4 (0.735) and P5 (0.51-0.75) are under
  their fixed bars of 0.80 and 0.70.
- **Judged against the HF floor: equivalent.** The 0.80 bar is out of reach even for HF vs HF (floor 0.34 / 0.56 /
  0.67 at 0.05 / 0.10 / 0.20). V's cosine sits at the floor at 0.10 and above it at 0.20. The single-module cosines
  are at about the floor expected for shifts of their size (the full adapter at a similar tv, 0.079, has a floor of
  0.56). The effect sizes match (P6 ratios 0.80-1.17, so no double or missing scaling), and the swap checks are
  clean. The names are mapped right and the scale is right.
- **Residual, to watch:** on V every adapter, including each single-module one, raises `inconclusive` (+0.005 to
  +0.052), while H1's single-module shifts on it go both ways. Part of this is a numerics effect that H2 (loop/NF4)
  shares on `full`; whether H2 shows it per module is not measured. It is small next to the stack noise (base tv
  0.049), but it is systematic. The ticket's follow-up, the accuracy of a trained adapter through `--endpoint`
  within 0.04 of H1, is the real gate before trusting vLLM numbers for model selection.
- Use vLLM for evaluation runs. Run the trained-adapter accuracy rerun first, and compare the inconclusive rate
  there.

Files: WSL `~/v3_parity/{H1,V}.p5_r20.json`, `{H1,H2,V}.full_r20.json`, `{H1,H2}.full_r10.json`, `V_r10.json`.
`run_p5.sh` and `run_v10.sh` now check `nvidia-smi -L` before each GPU step.

## 9. What this does not cover

- Vision input (the tower is not loaded in either setup).
- Multi-GPU or cloud runs; the cloud option is in the hand-off notes, not here.
- The quality of the Inkling judges themselves; no adapter has been evaluated yet.

## 10. Tickets: banded sliding attention, verdict scoring, DSpark (approved 2026-11-22)

Constraints of section 8.2 apply to every ticket here (files, GPU lock, no `torch.profiler` with CUDA activity,
no installs, no commits). GPU steps wait for the coordinator.

### 10.1 Facts checked for these tickets (CPU only)

- **Sliding attention.** `Inkling2LoopSlidingAttention.forward` computes the position bias for all (T, T) pairs
  (`Inkling2LoopRelativePositionBias.forward`: `rel_logits` of shape (batch, heads, T, 1024), then a gather to
  (batch, heads, T, T)) before it calls `eager_attention_forward`, which builds the (T, T) logits, the fp32
  concatenation with the sink and the softmax. So replacing only the attention function would leave the T x T
  bias. The sliding mask allows keys `q - 1023 .. q` (`kv_idx > q_idx - sliding_window`), and the bias is zero
  outside distances 0..1023, the same range. `attention_dropout` is 0.0. The decoder layers discard the returned
  attention weights. With a cache (generation) the shapes differ; training and scoring pass no cache.
- **Why banding cannot lower the peak memory.** Gradient checkpointing keeps one layer's graph at a time, so the
  peak is the largest single layer call. The 8 full-attention calls (layers 5, 11, 17, 23) keep their T x T
  tensors. Banding the other 40 calls makes them cheaper but leaves the peak, and so the longest trainable
  sequence, where it is.
- **How much work banding removes.** With queries in chunks of 256 against the keys they can see (at most 1,279),
  the sliding calls build 1.96 M query-key pairs per head at 2,048 tokens instead of 4.19 M (53% less) and 4.58 M
  at 4,096 instead of 16.8 M (73% less). At or below 1,024 tokens it removes nothing.
- **Reply format.** `templates.completion` writes `json.dumps(payload)` with `verdict` first for verdict-first
  judges, so every such reply starts with exactly `{"verdict": "` followed by the label.
- **DSpark draft weights.** They ship inside the Inkling checkpoint repository as `mtp.safetensors`
  (635,801,416 bytes, 606 MiB; 3 small dense layers and a Markov head). The file is **already in the WSL cache**:
  `~/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/ed40e7da625f594fc9c4990f5f66be24873c22d7/mtp.safetensors`
  (blob `7c3b6513...0a1d2`; the blob name is normally the file's SHA-256). The fork loads it from the target
  model's directory (`nvidia/dspark.py`: `allow_patterns_overrides = ["mtp.safetensors"]`) and shares the
  target's embedding and output layer. No separate draft repository appears in the fork's code or documents.
- The fork's DSpark requires exactly 7 draft tokens, keeps the draft's cache in bf16 on Ampere, and refuses an
  FP16 target cache. Its author tested it with an FP8 target cache; our configuration uses bf16, which the code
  allows but nobody has run. The DSpark code does not mention LoRA either way.
- vLLM exports `vllm:spec_decode_num_accepted_tokens_per_pos` and the related draft counters on `/metrics`.

### 10.2 Ticket list

| ID | Tier | Summary | Depends on | GPU |
|---|---|---|---|---|
| A1 | opus | Exact banded sliding attention as a per-model patch in a new module, CPU and GPU checks, measurements at 1k-4k tokens | none | about 1 h |
| A2 | haiku | Loader switch (`banded_attention` argument and `INKLING_BANDED_ATTENTION`) in `inkling.py`; default on only if A1 passed | A1, and after T7 has finished with `inkling.py` | none |
| S1 | sonnet | Verdict scoring in one forward pass: HF scorer and vLLM scorer in a new module, `--score-verdict` in `adapter_eval.py` | V2 (same script and the HTTP helper) | none |
| S2 | sonnet | Parity of scoring against generate-then-read, with the fixed rule | S1; the vLLM half also V0, V1 | about 0.7 h |
| D1 | sonnet | DSpark in `serve.sh`, benchmark and greedy-identity scripts, results | V0, V1; run after V3 has passed | about 1 h |

Parallel: A1, S1 and D1 have disjoint files, so their code can be written at the same time (S1 and D1 each edit
one file that V2 or V0 creates or edits, so they start after those are merged). A2 is last on `inkling.py`. On
the GPU everything is serial: suggested order V3, S2, D1, A1.

---

### A1. Exact banded sliding attention (opus)

> **Done on the CPU 2026-11-22; GPU check and bench pending.** 41 CPU tests pass (float64 ≤ 3.8e-16; the off-by-one fails both ways). The tiny real model on the CPU matches to 2.9e-7 (logits) and 1.0e-6 (gradients), and is bit-identical at one chunk. The Opus review found no correctness bugs. Fixed and rechecked:
> - the hash pin now also covers get_relative_distance and masking_utils' sliding_window_overlay / causal_mask_function / sliding_window_causal_mask_function (the functions that set the window);
> - it falls back when config.output_attentions is set, or when relative_position_bias has hooks.
>
> Accepted leftovers: create_sliding_window_causal_mask, and_masks and eager_mask aren't pinned (none sets the window width), and global module hooks aren't detected.
>
> **GPU done 2026-11-23: gate PASS** (details in 10.4). check_banded_attention.py passes every bar (bf16 rel 2.5e-4, fp32 3.5e-7, gradients 6.0e-3 / 1.1e-6). Bench: 1,024 tokens within 1.0%; the 2,048 training step is 7.8% faster (forward 12.2%); forward 21% faster at 3,072 and 28% at 4,096; peak memory unchanged. Training at 3,072+ tokens runs out of memory with or without the patch. A2 can set `BANDED_ATTENTION_DEFAULT = True`. The bench now treats a CUDA driver OOM as OOM and has `--modes`/`--banded` to run one case per process.

Tier reason: it replaces the attention arithmetic of 40 of 48 layer calls in training; a wrong slice gives
plausible outputs and plausible gradients.

Files (all new): `domain_verifier_factory/training/inkling_banded.py`, `tests/test_inkling_banded.py`,
`tests/gpu/check_banded_model.py`, `tests/gpu/check_banded_attention.py`, `tests/gpu/bench_banded.py`. A results
section appended to this document (10.4).

Reference behaviour: `Inkling2LoopSlidingAttention.forward`, `Inkling2LoopRelativePositionBias.forward` and
`eager_attention_forward` in the installed `modeling_inkling2_loop.py`. Nothing in site-packages is edited.

Design (fixed):

- **Method.** For each query chunk `[qs, qe)` of `BAND_CHUNK = 256` positions, take the keys
  `[ks, qe)` with `ks = max(0, qs - window + 1)`. Every key outside that range is masked for every query of the
  chunk, by causality or by the window, so dropping those columns removes exact zeros from the softmax. For the
  chunk, call **the module's original `eager_attention_forward`** on slices: `query[:, :, qs:qe]`,
  `key[:, :, ks:qe]`, `value[:, :, ks:qe]`, `attention_mask[:, :, qs:qe, ks:qe]`, the bias block, and
  `sinks[:, :, qs:qe]`. Concatenate the chunk outputs along the query axis. The original function is looked up
  from the transformers module at call time, not copied.
- **Bias block.** `rel_logits` is computed once as the reference does (the three lines before the gather:
  `linear`, `norm`, `@ proj`, transposed to (batch, heads, T, extent)). Per chunk: gather
  `rel_logits[:, :, qs:qe]` with `relative_distance[qs:qe, ks:qe].clamp(0, extent - 1)` and `masked_fill` with 0
  where the distance is negative or at least `extent`, exactly as the reference.
- **The patched forward** `banded_sliding_forward(self, hidden_states, attention_mask, relative_distance,
  conv_mask=None, cache=None, key_hidden_states=None, value_hidden_states=None, cache_offset=0, **kwargs)` repeats
  the reference body (query, key and value projections called as modules, so LoRA wrappers still apply; sinks;
  `o_proj`) and returns `(attn_output, None)`.
- **Fallback to the original forward**, decided per call, when any of these holds: `self.is_sliding` is false;
  `cache is not None`; `attention_mask` is not a 4-D tensor; `relative_distance` is not square or its side is not
  the query length; the query length is at most `self.sliding_window`; `self.training and
  self.attention_dropout > 0`; `kwargs.get("output_attentions")`. So at or below 1,024 tokens the patch is a
  no-op by construction.
- **Applying it.** `enable_banded_attention(model) -> int` sets `module.forward =
  types.MethodType(banded_sliding_forward, module)` on every `Inkling2LoopSlidingAttention` instance with
  `is_sliding` true, keeps the original bound method in `module._banded_original_forward`, and returns the number
  of patched modules (20 for the real model; raise if it is 0). `disable_banded_attention(model)` restores them.
  It is per instance, so no class or module global changes and two models in one process can differ.
- **Guard against a changed transformers build.** The module pins the SHA-256 of `inspect.getsource` of the
  three reference functions (computed by the implementer in `~/inkling-venv` and written as literals with the
  transformers version). `enable_banded_attention` recomputes them and raises `RuntimeError` on a mismatch,
  naming the function. No silent patching of code that has moved.
- **Pure core for CPU tests.** `banded_attention(attn_fn, module, query, key, value, attention_mask,
  relative_distance, rel_logits, sinks, scaling, window, extent, chunk)` holds the chunk loop and imports only
  torch, so the Windows tests can drive it with a literal copy of the reference attention function. The module
  imports transformers lazily inside `enable_banded_attention`.

Acceptance:

- CPU (`uv run pytest tests/test_inkling_banded.py`; the test carries literal copies of
  `eager_attention_forward` and of the bias gather, and a stub module with `num_key_value_groups` and
  `training`): float64, 6 heads, 2 key/value heads, head size 8, window 16, extent 16; for T in {17, 31, 32, 33,
  100} and chunk in {1, 5, 16, 64}: the banded output equals the full masked computation with rel at most 1e-12,
  and so do the gradients with respect to query, key, value, `rel_logits` and sinks. With a batch of 2 where one
  sequence is padded on the left and one on the right (4-D mask built as transformers does: causal, window and
  padding): the same bar, including rows whose keys are all masked.
- CPU: a test that moves `ks` by one in either direction (monkeypatched helper) fails the equality, so the test
  is shown to see an off-by-one window.
- CPU, WSL (`~/inkling-venv/bin/python tests/gpu/check_banded_model.py`, no GPU, no lock): a tiny
  `Inkling2LoopForConditionalGeneration` built from a config with small sizes and `sliding_window_size` 16,
  float32, random weights, T = 50 and a padded batch of 2: logits and the gradient of a scalar loss with respect
  to every parameter agree between patched and unpatched with rel at most 1e-4; `enable` returns the number of
  sliding layers; T = 16 takes the fallback (counted by wrapping the original forward); `disable` restores
  bit-identical outputs; a deliberately wrong pinned hash raises. If a tiny model cannot be built from the config
  class, report it; do not replace this check with a weaker one.
- GPU (`tests/gpu/check_banded_attention.py`, `gpu_session(21000)`, real model, fused experts): hook the inputs
  of every sliding attention module during one no-grad forward of a 2,048-token prompt (validation examples
  concatenated to length) and of a padded batch of two (1,500 and 2,048 tokens). For each captured call run the
  original and the banded forward on the same inputs:
  - bf16: output rel at most 2e-3 and cosine at least 0.9999;
  - the same module deep-copied to fp32 with fp32 inputs: rel at most 1e-5;
  - for layers 0, 9 and 22, the gradient of `output.float().sum()` with respect to `hidden_states`: rel at most
    1e-2 in bf16 and 1e-4 in fp32.
  The full-attention modules are unpatched (`not hasattr(module, "_banded_original_forward")`).
  Whole-model logits are not compared: the model is chaotic at bf16 rounding (plan section 8).
- GPU (`tests/gpu/bench_banded.py`, same session rules, card otherwise idle): at T = 1,024, 2,048, 3,072 and
  4,096, patch off and on: (a) one no-grad forward; (b) one training step with LoRA, gradient checkpointing and
  the loss computed on the last 64 positions only (the script applies the output layer to those positions itself,
  so the logits do not set the peak). 2 warm-up runs, median of 5, `torch.cuda.synchronize()` around each,
  `max_memory_allocated` reset per case. A case that runs out of memory is recorded as such, not retried smaller.
- Gate for A2's default: every bar above passes, the T = 1,024 times differ by at most 2%, and the T = 2,048
  training step is at least 5% faster with the patch.

Expected result, to compare with the measurement (est.): training step 15-20% faster at 2,048 tokens and 30-40%
at 4,096; **peak memory unchanged** (set by the full-attention calls); nothing at or below 1,024.

Review checklist (silent failure modes):

- `ks = max(0, qs - window + 1)`: the window includes the query itself and 1,023 earlier keys. One key too few
  drops a real attention target and still looks plausible.
- The mask, the distance matrix, the keys and the values are sliced with the same `ks:qe`; the sinks and queries
  with `qs:qe`.
- The bias uses the chunk's rows of `rel_logits` and the reference's clamp and zero-fill; `rel_logits` is
  computed from `hidden_states`, not from the key input (the loop layers pass different key and value inputs).
- Projections are called as modules (`self.q_proj(...)`), never through `.weight`, so PEFT's LoRA layers and the
  convolutions inside the key and value projections run.
- Every fallback condition is present; with a cache the original path runs (generation is unchanged).
- The pinned hashes are of the installed functions, and a mismatch raises instead of warning.
- The CPU test's reference is a literal copy, not an import of the code under test; the padded cases include
  fully masked rows.
- No `torch.profiler`; GPU scripts inside `gpu_session`.

---

### A2. Loader switch (haiku)

Tier reason: three mechanical edits with a fixed specification.

Ordering: only after A1's GPU results are in section 10.4 and T7 has finished its edits to `inkling.py`.

Files: `domain_verifier_factory/training/inkling.py`, `tests/test_inkling.py` (extend).

Spec:

- `resolve_banded_attention(banded_attention: bool | None) -> bool` next to `resolve_experts_impl`: `None` reads
  `INKLING_BANDED_ATTENTION` (`"1"` true, `"0"` false, anything else `ValueError`); the default when the variable
  is unset is the constant `BANDED_ATTENTION_DEFAULT`, set to `True` if A1's gate passed and `False` otherwise.
- `load_inkling_4bit(..., banded_attention: bool | None = None)`: after the experts are loaded, if resolved true,
  call `inkling_banded.enable_banded_attention(model)` (lazy import). Document the argument in the docstring.
- No change to `configs.py` or `qlora.py`: the switch is the environment variable, so the B tickets' files are
  untouched. The value is not recorded in `agent_config.json`; that is acceptable because the patch is exact and
  a no-op at our training lengths.

Acceptance (CPU): `resolve_banded_attention` for unset, `"0"`, `"1"`, an invalid value and explicit arguments; the
existing tests pass. Review checklist: the default constant matches A1's recorded gate result; the import is lazy
(Windows has no Inkling transformers build); nothing else in `inkling.py` changed.

---

### S1. Verdict scoring in one forward pass (sonnet)

> **Done 2026-11-22 (CPU; live checks in S2).** The Opus review found a HIGH bug, now fixed: Inkling's chat template ignores `add_generation_prompt`, so the forced prefix lacked the assistant header `<|message_model|><|content_text|>` (200001, 200004) and both scorers read the wrong position. Now `assistant_header_ids()` derives the header from the prompt vs prompt+sentinel renderings (HF via apply_chat_template, vLLM via the server's /tokenize), and raises if the template isn't prefix-consistent. tests/test_verdict_scoring_tokenizers.py (runnable as a plain script in WSL) confirms the forced ids are a prefix of the to_chat_record training sequence, with the label's first token next, for Inkling, Qwen3.6-27B and Qwen2.5-0.5B (Qwen headers are empty). Also: position_ids from the attention mask (batch 1 = batch 3), and a one-time log when the server ignores token ids. vLLM's template/kwargs equivalence was verified by reading code only; S2's live run confirms it.

Tier reason: well specified and CPU-testable with stubs; the numerics are checked in S2.

Files: `domain_verifier_factory/evaluation/verdict_scoring.py` (new), `tests/test_verdict_scoring.py` (new),
`scripts/adapter_eval.py` (edit, after V2's edit is merged). `predictors.py` needs **no** change: the HF scorer
takes an already loaded model and tokenizer, which `AdapterPredictor` exposes as attributes.

Reference behaviour: `AdapterPredictor.verdict_probs` (the label's first token, encoded alone; `label_probs`
renormalised over the labels; `label_mass`; `margin`).

Spec:

- `FORCED_PREFIX = '{"verdict": "'`.
- `check_verdict_first(rationale_first: bool) -> None`: raises `ValueError("verdict scoring needs a verdict-first
  judge: a reason-first judge writes its rationale before the verdict, so the verdict cannot be read from one
  forward pass; use generation")`. Both scorers call it in `__init__`.
- `scored_prediction(labels: list[str], probs: list[float]) -> dict`: `mass = sum(probs)`; if `mass < 0.5` the
  verdict is `None` (the model did not want a label there, the analogue of an unparseable reply); else the
  arg-max label. Returns `{"verdict", "confidence": None, "rationale": None, "domain_relevance": None,
  "label_probs": {label: p / mass}, "label_mass": mass, "margin": top - runner_up, "scored": True}`
  (`label_probs` and `margin` are omitted when `mass == 0`). `confidence` stays `None` on purpose: that field is
  the judge's own stated confidence, and a probability must not be passed off as it.
- `class HFVerdictScorer`: `__init__(self, model, tokenizer, *, domain, scheme="four_way", display_name=None,
  max_passages=10, rationale_first=False, chat_template_kwargs=None, batch_size=8, max_batch_tokens=6000)`.
  `name = "score:hf"`. It calls `model.eval()`. `predict(examples)`:
  1. per example, ids = `apply_chat_template([system, user], add_generation_prompt=True, **chat_template_kwargs)`
     followed by `tokenizer.encode(FORCED_PREFIX, add_special_tokens=False)`;
  2. sort by length, form batches of at most `batch_size` examples and at most `max_batch_tokens` padded tokens,
     pad on the **left** with the pad token (or EOS if there is none) and build the attention mask;
  3. one forward under `torch.no_grad()` with `use_cache=False`; `logits[:, -1].float()`, softmax over the whole
     vocabulary, pick the label first-token ids (computed once; raise if two labels share one);
  4. results returned in input order.
  `classmethod from_adapter_predictor(pred, batch_size=8)` copies the model, tokenizer, scheme, prompts and
  `chat_template_kwargs` from an `AdapterPredictor` and passes its `rationale_first` (so a reason-first judge is
  refused).
  `classmethod for_inkling(agent_dir, *, domain, ...)` loads with `load_inkling_4bit` using
  `inkling_experts_impl` and `inkling_mxfp8_storage` from `agent_dir/agent_config.json` (defaults `"fused"`,
  `"nvfp4"` if absent), applies the PEFT adapter if `adapter_config.json` exists, and uses
  `{"reasoning_effort": "none"}` unless the agent config has other template options.
- `class VLLMVerdictScorer(VLLMJudgePredictor)` (from V2): `name = f"score:vllm:{model}"`. Per example:
  1. `POST {root}/tokenize` with `{"model", "messages": [system, user], "add_generation_prompt": true,
     "chat_template_kwargs": {...}}`, take `tokens`;
  2. append the prefix ids (from one `/tokenize` call with `{"prompt": FORCED_PREFIX, "add_special_tokens":
     false}`, cached);
  3. `POST {base_url}/completions` with `{"model", "prompt": <ids>, "max_tokens": 1, "temperature": 0,
     "logprobs": 20, "return_tokens_as_token_ids": true}`; read the first position's alternatives (keys
     `token_id:N`), `exp` for each label's first-token id (from `/tokenize`, cached), 0 if absent.
  Concurrency and retries as in V2.
  **Why `max_tokens=1` and not prompt log-probabilities:** scoring needs the distribution over the next token
  after the prefix. `prompt_logprobs` returns alternatives for every prompt position (about 500 positions x 20
  entries per request, computed and serialised for nothing), and reading four labels from it would need a label
  appended, so four prompts per example. One generated token gives the whole distribution in one request.
  **Why token ids and not a prefilled assistant message:** how the template renders a partial assistant turn is
  not under our control; tokenising the prompt on the server and appending the prefix ids fixes the exact
  context, and it is the method V3 validates.
  Known limit, in the docstring: only the 20 most likely tokens are visible, so `label_mass` is a lower bound.
- `scripts/adapter_eval.py`: new flag `--score-verdict`. Without `--endpoint` it builds the `AdapterPredictor` as
  today and wraps it with `HFVerdictScorer.from_adapter_predictor` (so `--stock` and `--stack` keep working
  through the wrapped model); with `--endpoint` it builds `VLLMVerdictScorer`. `--score-verdict` with `--claim`
  or `--windows` is allowed (the printed rationale is empty). With `--margins` it is an error ("scoring already
  returns the margins"). The report label gets the suffix `" [verdict scoring]"`.

Acceptance (CPU, `uv run pytest tests/test_verdict_scoring.py`):

- `FORCED_PREFIX` is a prefix of `templates.completion(ex, rationale_first=False)` for an example of every
  label, and not of the reason-first reply.
- Both scorers raise the `ValueError` for `rationale_first=True`; `from_adapter_predictor` raises for a stub
  predictor with `rationale_first` true.
- `scored_prediction`: hand-computed cases, including mass below 0.5 (verdict `None`), mass 0, and a tie.
- `HFVerdictScorer` with a stub tokenizer and a stub model whose logits depend on the last real token and on the
  attention mask: results in input order for mixed lengths; left padding (a model that reads position -1 gets
  the real last token for every row); the batch limits hold; two labels with one first token raise.
- `VLLMVerdictScorer` against the fake server of V2's tests: the `/tokenize` body carries the messages and
  template options; the completion prompt is the returned ids plus the prefix ids; a label absent from the
  alternatives gets 0; HTTP errors raise.
- `adapter_eval.py`: `--score-verdict --margins` errors; with a fake predictor the written prediction rows carry
  `label_probs`, `label_mass`, `margin`, `scored`.

Review checklist (silent failure modes):

- Left padding with a correct attention mask. Right padding would read a pad position and still return four
  numbers.
- The prefix ids are appended after the generation prompt, encoded without special tokens (no second BOS).
- The label ids come from the same tokenizer as the prompt; the rule is the reference's (first token of the
  label encoded alone).
- A reason-first judge cannot reach either scorer by any constructor path.
- `confidence` is `None`, never the probability; a verdict of `None` when the mass is low, not an arg-max over
  noise.
- Results are in input order after the length sort.
- The scorer calls `model.eval()` itself (dropout in LoRA layers would change the result).

---

### S2. Parity of scoring against generate-then-read (sonnet)

Tier reason: a scripted comparison with a rule fixed here.

Files (new): `scripts/inkling_vllm/score_parity.py`, `tests/test_score_parity.py`; results appended as 10.5.

Fixed design. Two runs, because no trained Inkling adapter exists yet:

- **Run Q (a real trained judge).** A verdict-first Qwen judge named by the coordinator, the first 300 examples
  of its test set. Generation: `AdapterPredictor` with `record_margin` (it stops after 24 tokens). Scoring:
  `HFVerdictScorer.from_adapter_predictor` on the same loaded model, once with `batch_size=1` and once with 8.
- **Run I (Inkling on vLLM).** The server from `serve.sh` with V3's converted `full` adapter; the first 300
  factcheck validation examples. Generation: `VLLMJudgePredictor` with `record_margin`. Scoring:
  `VLLMVerdictScorer`. Both for the base model and for `full`.

An example **qualifies** if its generated reply starts with exactly `FORCED_PREFIX` and the generation produced
margin fields.

**Rule (fixed before running):**

- Run Q, batch 1: at least 98% of examples qualify; on the qualifying ones verdict agreement at least 99%, mean
  `|margin difference|` at most 0.02, and the 95th percentile of the total-variation distance between the two
  `label_probs` at most 0.05.
- Run Q, batch 8: agreement at least 98% and mean `|margin difference|` at most 0.03 (batching changes rounding).
- Run I, per model: if fewer than 100 examples qualify the result is "no test" (expected for an untrained model;
  rerun with the first trained Inkling adapter). Otherwise agreement at least 97% and mean `|margin difference|`
  at most 0.05 (looser: the generation path sees only the 20 most likely tokens, and the two requests may be
  batched differently).
- Reported, not gated: seconds per example for each path, and the label mass distributions.

Pass = Run Q passes and Run I passes or is "no test". The script exits 0 for a pass with Run I tested, 3 for a
pass with Run I not tested, 1 otherwise.

Acceptance: CPU: the rule function on synthetic prediction lists (boundaries, the qualifying filter, the "no
test" state); the script parses. GPU: Run Q about 25 minutes (model load, 300 generations of 24 tokens, two
scoring passes) inside `gpu_session`; Run I about 15 minutes against the running server.

Review checklist: both paths see the same examples and the same prompts (`max_passages`, scheme, template
options); agreement is computed only over qualifying examples and the qualifying share is itself gated; Run Q
uses one loaded model for both paths; the thresholds in code equal the ones above.

> **Status 2026-11-23: code done, Run Q NOT finished (GPU/WSL blocked).** Written: `scripts/inkling_vllm/score_parity.py` (rule thresholds as above; caches generations to `results/s2/score_parity.gen.jsonl` and resumes) and `tests/test_score_parity.py` (4 CPU tests pass). Judge: `~/agents_factcheck_full/factcheck_agent` (Qwen3.6-27B, `enable_thinking: false`), first 300 of `datasets/factcheck_full/factcheck_agent_dataset/test.jsonl`. Attempt 1 generated all 300 (about 4.4 s/example) but crashed in the first scoring pass with `CUBLAS_STATUS_EXECUTION_FAILED`, before anything was saved (log `results/s2/run1_crash.log`); a likely cause was the full-vocabulary logits (up to 6000 tokens x 248k) on a card holding a 20 GB model, so `HFVerdictScorer` now passes `logits_to_keep=1` (the test stubs accept it). Attempt 2 generated 170/300 (cached) and then died with `cudaErrorUnknown` (`run2_crash170.log`). Since then WSL's `nvidia-smi` fails ("couldn't find libnvidia-ml"; `/usr/lib/wsl/lib` carries 610.53 libs, ld.so assertion), so `gpu_session` exits 75. Needs a WSL restart (`wsl --shutdown`), which the agent was not permitted to do. Resume: rerun the same command; the 170 cached generations are reused. Run I: deferred to the first trained Inkling adapter ("no test"). No pass/fail yet.

---

### D1. DSpark speculative decoding in the fork (sonnet)

> **Code done 2026-11-22; GPU run pending.** The draft weights were verified (mtp.safetensors 635,801,416 bytes, sha256 7c3b6513…a1d2), so no download is needed. 10 CPU tests pass. The Opus review found a HIGH bug, now fixed: serve.sh's snapshot pick died silently under `set -euo pipefail` when the last snapshot listed lacked weights. It's now plain loops with a clear exit 2, preferring refs/main if that snapshot has weights, else the newest by mtime; rechecked over 8 temp-dir cases. compare_greedy uses `templates.parse_prediction`, and unparseable replies count as disagreements. Verified: the DSPARK flags match the fork's doc, the draft KV stays bf16 on sm86, the lock is held for the server's lifetime, and bench_decode detects a server that ignores DSpark.

Tier reason: flags and measurement scripts with a fixed rule; no numerics are written.

Files: `scripts/inkling_vllm/serve.sh` (edit, after V0), `scripts/inkling_vllm/check_draft_weights.sh` (new),
`scripts/inkling_vllm/bench_decode.py` (new), `scripts/inkling_vllm/compare_greedy.py` (new),
`tests/test_inkling_vllm_bench.py` (new); results appended as 10.6.

**Step 1: the draft weights (its own step, reported before anything else).** `check_draft_weights.sh` verifies
that `mtp.safetensors` exists in the snapshot directory `serve.sh` uses, is 635,801,416 bytes, and that
`sha256sum` of the file equals its blob name
(`7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2`). The file was present with that size on
2026-11-22, so **no download is expected**. Only if the check fails, the approved download is exactly this one
file:

```bash
HF_HUB_OFFLINE=0 ~/vllm-venv/bin/huggingface-cli download thinkingmachines/Inkling-2-Mini mtp.safetensors
```

That is 606 MiB from the Hugging Face repository `thinkingmachines/Inkling-2-Mini`, into
`~/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/` in WSL. The script prints this command and
exits 2 instead of running it; the implementer tells the coordinator, who reports it to the user, and runs it
only then. (If `huggingface-cli` is not in `~/vllm-venv`, report that too; do not install it.) Nothing else may
be downloaded.

Spec:

- `serve.sh`: two environment switches. `DSPARK=1` appends `--speculative-config.method=dspark
  --speculative-config.num_speculative_tokens=7`. `KV_CACHE_BYTES` (default `2684354560`) replaces the literal in
  `--kv-cache-memory`. Nothing else changes; the cache stays bf16 and LoRA stays enabled.
- **Memory fit at the 8k configuration.** Start with the defaults. If the server fails to start for lack of
  memory, retry with `KV_CACHE_BYTES=2147483648` (2.0 GiB), then `1610612736` (1.5 GiB), each once; then
  `MAX_NUM_SEQS=1` at 1.5 GiB. Record the first setting that starts, the server's reported cache size in tokens,
  and the `nvidia-smi` memory. If none starts, the result is "DSpark does not fit at 8k in WSL" and the ticket
  ends there with that documented. (An 8k sequence needs about 0.43 GiB of cache, so 1.5 GiB still serves it.)
- **With an adapter.** Start with V3's converted `full` adapter loaded. If the server refuses the combination or
  a request to the adapter fails, record the exact error, rerun everything for the base model only (server
  started without `--enable-lora` is not needed: just do not request the adapter), and document "DSpark cannot be
  used with a LoRA adapter in this fork". Do not patch the fork.
- `bench_decode.py --base-url ... --model NAME --workload {judge,long} --concurrency {1,8} --out FILE`: `judge`
  = the first 100 factcheck validation prompts through chat completions (`max_tokens` 200, temperature 0, the
  template options of V2); `long` = 10 fixed free-text prompts (literals in the script) with `max_tokens` 1,000.
  It measures wall-clock per request, completion tokens from `usage` and total tok/s, and reads `/metrics` before
  and after for the `vllm:spec_decode_*` counters; it reports accepted tokens per draft position and the mean
  accepted per draft (the acceptance rate is that mean over 7). 1 warm-up request per model. It saves token ids
  (`return_tokens_as_token_ids`) and, for the non-speculative runs, `logprobs` with 2 alternatives.
- `compare_greedy.py A.json B.json`: per prompt, whether the token-id sequences are identical, and the index of
  the first difference.
- Runs: server without DSpark: `judge` and `long`, models base and `full`, concurrency 1, plus `judge` at 8.
  Server with DSpark: the same.

**Rule for "identical to non-speculative greedy decoding" (fixed before running).** Exact identity is the
target. Because this model's arg-max can flip under rounding when two tokens are nearly tied, and speculation
verifies 8 positions in one batch, the gate is stated against a measured floor:

- floor = the share of prompts whose output is identical between the non-speculative runs at concurrency 1 and
  at concurrency 8 (`judge`, per model);
- pass if, per model and workload at concurrency 1, the share of prompts identical between DSpark and
  non-speculative is at least `floor - 0.05`, **and** at every first difference the non-speculative run's top-2
  log-probability gap is below 0.5 (a near-tie, not a wrong token), **and** the parsed verdict agrees on at
  least 99 of the 100 `judge` prompts;
- a first difference at a position with a gap of 0.5 or more is a failure regardless of the shares.

Acceptance: CPU: `bash -n` on both shell scripts; `check_draft_weights.sh` passes in WSL today;
`compare_greedy` and the rule on synthetic inputs (identical, a near-tie difference, a clear difference);
`bench_decode.py` against the fake server with canned `/metrics` text. GPU (about 1 h: two to five server starts
at about 4 minutes each, ten benchmark runs): the results table in 10.6 with tok/s and acceptance with and
without DSpark for base and adapter, the memory setting that worked, and the identity rule's outcome.

Expected, to compare with the measurement (est.): `long` 1.2-1.6x at concurrency 1 for the base model (the
fork's document), possibly less with an adapter because the draft was trained on the base model's features;
`judge` replies are about 53 tokens, so 10-25% per example at concurrency 1 and little or nothing at 8; nothing
for verdict scoring, which generates one token.

Review checklist (silent failure modes):

- The draft-weight check ran first and its result is in the report; no download happened without the
  coordinator's word.
- With `DSPARK` unset, `serve.sh` produces the same command line as before this ticket (diff the echoed
  command).
- Speculation was really active in the DSpark runs: the `/metrics` counters increased (a server that silently
  ignored the flag would show equal speed and "identical" outputs).
- Requests to the adapter name really used the adapter: its outputs differ from the base model's in these runs.
- tok/s uses completion tokens and client wall-clock at the stated concurrency, warm-up excluded, the card
  otherwise idle.
- The identity rule's constants in code equal the ones above; near-tie differences are listed with their gaps.

### 10.3 Downloads, installs, GPU time, expected gain

- Downloads: none expected. The DSpark draft is `mtp.safetensors` (606 MiB) inside the Inkling checkpoint
  repository and is already in the WSL cache; D1's first step verifies it and names the single approved download
  if it is missing. Installs: none.
- GPU time: A1 about 1 h, S2 about 0.7 h, D1 about 1 h; about 2.7 h in total, serial, after V3.

| Item | Gain for our judges | Gain for general use |
|---|---|---|
| Banded sliding attention (A1, A2) | None: the 40k mixture has p99 712 and max 1,861 tokens, and the patch does nothing at or below 1,024. Under 1% of examples are touched | Time only: about 15-20% of a training step at 2,048 tokens and 30-40% at 4,096 (est.). Peak memory and the longest trainable sequence do not change; that needs the full-attention calls fixed (item 11 of section 6) |
| Verdict scoring (S1, S2) | Large: evaluation of verdict-first judges at about 0.2 s per example on vLLM instead of 0.8 s, and about 0.4 s instead of 13 s on HF. It is also the fast path for evaluation during training, and it works for the Qwen judges | Only for tasks whose answer is one of a few fixed first tokens; none for free text or reason-first judges |
| DSpark (D1) | Small: 10-25% per generated reply at concurrency 1, nothing for scoring, unknown with an adapter | 1.2-1.6x on long outputs for one stream, if it fits in memory beside the 8k configuration |

### 10.4 Results: banded sliding attention (A1)

Code: `domain_verifier_factory/training/inkling_banded.py` (`enable_banded_attention`, `disable_banded_attention`,
`banded_attention`), pinned to transformers 5.24.0.dev0 by the SHA-256 of the three reference functions' sources.

CPU (2026-11-22):

- `tests/test_inkling_banded.py`, float64, 6 heads, 2 key/value heads, head size 8, window and extent 16, T in
  {17, 31, 32, 33, 100}, chunk in {1, 5, 16, 64}, single sequence and a left- plus right-padded batch of 2 (with
  fully masked rows): worst relative difference over the output and the gradients of query, key, value,
  `rel_logits` and sinks 3.8e-16 (bar 1e-12). This needs the reference attention with its one fp32 cast made fp64:
  `eager_attention_forward` runs the sink softmax in fp32 (`.float()`) even for fp64 inputs, so with the literal
  copy banded and full differ by fp32 rounding alone (worst 1.8e-7, tested at a 1e-6 bar). Fully masked rows are
  bit-identical. A band start moved by one key in either direction fails (chunk 1 with a causal-only mask, where
  the band alone is the window; a key too few also fails with the full mask).
- `tests/gpu/check_banded_model.py` (WSL, CPU), tiny `Inkling2LoopForConditionalGeneration` (6 layers, 5 sliding,
  loop layers 2-3 run twice), float32, T = 50 and a padded batch of 2, chunk 5 and 16: logits rel 2.9e-7, worst of
  159 parameter gradients rel 1.0e-6 (bar 1e-4); with the default chunk (one chunk) bit-identical. `enable` returns
  5; T = 16 and cached calls take the fallback (7 of 7 calls), T = 50 never; `disable` restores bit-identical
  logits; a wrong pinned hash raises.

GPU (2026-11-23, driver 621.24, fused experts, MXFP8 layers as NVFP4, under `gpu_session`):

- `tests/gpu/check_banded_attention.py`: PASS. 40 sliding calls captured per forward (2,048 tokens; right-padded
  batch of 1,500 and 2,048), 20 modules patched, the 4 full-attention modules unpatched, no call took the fallback.
  Worst: bf16 output rel 2.5e-4 (bar 2e-3), cosine 1.0000 (bar 0.9999); fp32 rel 3.5e-7 (bar 1e-5); input gradients
  of layers 0, 9, 22 rel 6.0e-3 in bf16 (bar 1e-2), 1.1e-6 in fp32 (bar 1e-4). Numbers in `~/banded_check.json`.
- `tests/gpu/bench_banded.py`, median of 5 after 2 warm-ups, time in s (peak GiB), off / on, time saved:

| T | forward off | forward on | saved | train step off | train step on | saved |
|---|---|---|---|---|---|---|
| 1,024 | 0.707 (15.53) | 0.714 (15.53) | -1.0% | 2.381 (18.04) | 2.356 (18.04) | +1.0% |
| 2,048 | 1.471 (16.74) | 1.292 (16.73) | +12.2% | 5.040 (21.38) | 4.645 (21.38) | +7.8% |
| 3,072 | 2.443 (18.63) | 1.926 (18.62) | +21.2% | OOM | OOM | - |
| 4,096 | 3.637 (21.21) | 2.623 (21.19) | +27.9% | OOM | OOM | - |

  Peak memory is unchanged by the patch, as expected (the full-attention calls set it). The training step does not
  fit on the 24 GiB card at 3,072 tokens or more with or without the patch. The OOM came as a CUDA driver error
  under expandable segments and left the allocator unusable (a later internal assert), so the 3,072 training case
  and the 4,096 cases were rerun one per process (`--lengths/--modes/--banded`); logs `~/bench_banded*.log`,
  `~/bench_3072_*.log`, `~/bench_4096_*.log`.
- **Gate for A2's default: PASS.** All correctness bars pass; at 1,024 tokens the times differ by 1.0% (bar 2%);
  the 2,048-token training step is 7.8% faster (bar 5%). The gain is below the estimate (15-20% at 2,048): the
  attention is a smaller share of the step than assumed. So `BANDED_ATTENTION_DEFAULT = True`.

### 10.6 Results: DSpark speculative decoding (D1)

GPU (WSL, 2026-11-23, driver 621.24, RTX 3090). Draft weights verified first (`check_draft_weights.sh`: 635,801,416
bytes, sha256 equals the blob name; no download). Server: `serve.sh` with `full=~/v3_parity/conv/full` (V3's converted
random rank-16 adapter) loaded in both runs; "full" requests used it (95 of 100 `judge` outputs differ from the base
model's). Raw runs: WSL `~/d1/*.json`, driver `~/d1/run_cfg.sh`.

**Memory.** The default `KV_CACHE_BYTES=2684354560` (2.5 GiB) does not start with DSpark: the draft layers also need
cache, 2.64 GiB is needed for one 8k sequence ("estimated maximum model length 3904"). The ladder in the ticket goes
the wrong way (down). Up one step, `KV_CACHE_BYTES=2952790016` (2.75 GiB), starts and serves. nvidia-smi peak:
23,599 MiB without DSpark (2.5 GiB cache), 23,253 MiB with DSpark (2.75 GiB). DSpark fits at 8k with LoRA enabled.

**LoRA.** DSpark and a LoRA adapter work together in this fork: no refusal, no request error, speculation active for
adapter requests (acceptance below).

Throughput (client wall-clock, completion tokens, 1 warm-up, c = concurrency):

| Model | Workload | No DSpark tok/s | DSpark tok/s | Ratio | Mean accepted / draft | Rate (/7) |
|---|---|---|---|---|---|---|
| base | judge c1 | 83.1 | 147.4 | 1.77x | 2.57 | 0.367 |
| base | long c1 | 88.6 | 110.8 | 1.25x | 1.38 | 0.197 |
| base | judge c8 | 312.7 | 499.7 | 1.60x | 2.59 | 0.370 |
| full | judge c1 | 74.5 | 136.5 | 1.83x | 2.51 | 0.359 |
| full | long c1 | 77.4 | 100.7 | 1.30x | 1.29 | 0.185 |
| full | judge c8 | 294.8 | 475.8 | 1.61x | 2.52 | 0.360 |

Accepted per draft position, base judge c1: 0.77 0.58 0.42 0.30 0.22 0.16 0.12; base long c1: 0.58 0.32 0.19 0.12
0.08 0.05 0.04; the adapter's are within 0.02 of these. The `/metrics` draft counters increased in every DSpark run
(speculation was active). Note: the `judge` replies are not about 53 tokens as assumed. With `reasoning_effort: none`
and the plain verdict-first system prompt this model still writes visible reasoning ("The user wants me to verify..."),
and 99 to 100 of 100 replies hit `max_tokens` 200. So `judge` here is a 200-token generation workload, not a verdict
call, which is why DSpark helps it more than the ticket expected.

**Identity rule (fixed in the ticket): FAIL as literally stated, but the noise floor explains it.** Non-speculative
runs are not even identical to themselves across concurrency: `judge` c1 vs c8 identical share 0.08 (base) and 0.10
(adapter). DSpark vs non-speculative c1:

| Model | Workload | Identical share | Floor | Share gate | First differences with gap >= 0.5 | Verdict disagreements |
|---|---|---|---|---|---|---|
| base | judge | 0.22 | 0.08 | pass | 0 (max gap 0.42) | 27 of 100 |
| base | long | 0.00 | 0.08 | fail | 0 (max 0.44) | n/a |
| full | judge | 0.15 | 0.10 | pass | 6 (max 0.82) | 23 of 100 |
| full | long | 0.00 | 0.10 | fail | 0 (max 0.12) | n/a |

The 99-of-100 verdict gate fails for both models, but the non-speculative c1 vs c8 pair itself disagrees on 23 (base)
and 16 (adapter) of 100 verdicts, because the verdict is parsed from truncated free-text reasoning by keyword fallback
(`parse_prediction`), not from a verdict-first JSON. So the verdict gate does not test speculation here. Every first
difference for the base model is a near-tie (gap below 0.5); the adapter has 6 first differences with a gap above 0.5
(0.5 to 0.82), which the rule counts as failures. `long` shares are 0 of 10 against a floor taken from `judge`, which
is not comparable for 1,000-token outputs. Conclusion: DSpark output is not token-identical to non-speculative
decoding, in the same way concurrency alone changes the output; at the base model every divergence is a near-tie.
A clean test needs a verdict-first one-token or short-JSON workload and a floor measured on the same workload.
Not applicable to verdict scoring (one generated token).
