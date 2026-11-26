# Inkling-2-Mini on one RTX 3090: unused ways to lower GPU memory

Written 2026-11-23. Analysis only: nothing was run on the GPU (a training run holds it). Sources: the code, the
three existing documents, the vLLM fork in WSL (`~/src/vllm-ampere`), the checkpoint header and `config.json`, and
the serve logs in WSL (`~/d1/server_*.log`, `~/vllm_serve.log`). "Measured" means a number from an existing log or
document. "Derived" means computed from the fork's code and checked against a measured number. "Est." is a guess.

Priority, as set by the user: (B) serving stock Inkling with as much context as possible comes first; (A) training
is short and lists only cheap options or ones that also help serving.

## 1. Answer

- **300k context is reachable on paper on this card, and without new kernels.** It needs the FP8 KV cache, a small
  prefill chunk, one sequence, no DSpark, and about 1 GiB more from one of: the display moved to the AMD graphics,
  the embedding table on the CPU, or expert layers 20-23 stored as NVFP4. Two of the three give a safe margin.
  None of this has been run here, and FP8 KV quality at that length is unmeasured.
- **The biggest unused saving is a flag, not a kernel.** `serve.sh` passes `--max-num-batched-tokens 1024`. The KV
  planner reserves room for 2 x that many "in-flight" tokens in every sliding-window cache, and the short-convolution
  state caches cost 16 KiB per token per layer call. That reservation is 1.97 GiB of the 2.25 GiB that an 8k context
  "needs". At 256 (the fork author's value) it is 0.61 GiB. With the same 2.5 GiB pool that is about 60k tokens
  of context in bf16 instead of 9k.
- **Per token of context the cost is fixed by 8 full-attention caches:** 32 KiB per token in bf16, 16 KiB in FP8.
  So 300k tokens is 9.2 GiB in bf16 (never fits) and 4.6 GiB in FP8 (fits if the pool reaches about 5 GiB).
- Not available: sharing one cache across the three loop passes (the checkpoint is not a variable-loop one), a
  4-bit KV cache (no kernel), KV on the CPU for a live sequence (vLLM's KV offload is a prefix-cache tier).

## 2. Where the memory goes when serving (facts)

### 2.1 Weights: 17.64 GiB (measured), by part (from the checkpoint header)

| Part | GiB | Format | Note |
|---|---|---|---|
| Experts, layers 0-19 | 10.68 | NVFP4, 4 bits + scales | 0.534 GiB per layer |
| Experts, layers 20-23 | 3.92 | MXFP8, 8 bits + scales | 0.98 GiB per layer; as NVFP4 they would be 2.14 |
| Input embedding (`embed`) | 0.86 | bf16, 201,024 x 2,304 | read by rows |
| Output layer (`unembed`) | 0.86 | bf16 | not tied to the embedding; read in full for every token |
| Attention (`wq`, `wo`, `gate_attn`, `wk`, `wv`, `wr`) | 1.20 | bf16 | the LoRA targets |
| Router, loop controllers, norms, convolutions | 0.1 | bf16 | |
| Vision tower | 0 (0.6 on disk) | | already dropped by `--limit-mm-per-prompt.image=0` |
| DSpark draft (`mtp.safetensors`) | +0.73 | bf16 | only with DSpark (18.37 GiB loaded, measured) |

The fork's README says "MXFP8 dense", but `quant_config.py` says only the experts are quantized and the header
agrees: every non-expert tensor is bf16.

### 2.2 KV cache: what one sequence costs (derived from the fork's planner)

The fork registers 96 cache "layers" (`nvidia/attention.py`, `inkling/nvidia/sconv_swa_attn.py`):

| Cache | Count | Bytes per token | Length kept |
|---|---|---|---|
| Full attention: layers 5 and 23 once, layers 11 and 17 once per loop pass | 8 | 4,096 bf16 / 2,048 FP8 | the whole context |
| Sliding attention: 20 layers, 10 of them three times | 40 | 4,096 bf16 / 2,048 FP8 | 1,023 + in-flight tokens |
| Short-convolution state, one per layer call (K, V, attention-out, MLP-out packed) | 48 | 16,384 (bf16 only) | 3 + in-flight tokens |
| DSpark draft: 1 full layer, 2 sliding (padded to 9) | 3 | 4,096 (bf16 only) | context / window |

- **The three loop passes do not share a cache.** Each pass has its own KV and its own convolution state
  (`Inkling2AttentionPassCache`). The fork has a `loop_share_cache` option that would halve the full-attention cost,
  but `configs.py` refuses it unless `loop_config.is_varloop` is true, and this checkpoint has `is_varloop: false`.
  Forcing it would change the model's output (later tokens would see the last pass's keys), so it is not an option.
- In-flight tokens F = `max_num_batched_tokens` x 2 with async scheduling (the default), x 1 with
  `--no-async-scheduling` (`config/vllm.py`, `max_in_flight_tokens`).
- Bytes for one sequence of T tokens, b = 2 for bf16 and 1 for FP8:
  `T x 8 x 2048 x b  +  (1023 + F) x 40 x 2048 x b  +  (3 + F) x 48 x 16384`.

| Setting | Fixed part, GiB | Per 100k tokens, GiB |
|---|---|---|
| bf16, F = 2,048 (today) | 1.97 | 3.05 |
| bf16, F = 512 | 0.61 | 3.05 |
| FP8, F = 512 (the fork author's setting) | 0.49 | 1.53 |
| FP8, F = 256 | 0.29 | 1.53 |
| DSpark on | +0.1 to +0.2 | +0.38 |

Checks against measurements: the formula gives 2.22 GiB for 8k today (log: 2.5 GiB pool is 1.11 x one request, so
2.25); 2.46 GiB with DSpark (measured 2.64, so the draft part is underestimated by about 0.2 GiB); and 220k tokens
for the fork author's 4.8 GiB FP8 pool with DSpark (he reports 224k for 2 sequences).

### 2.3 Everything else

- WSL sees 22.68 GiB free at start (measured); Windows and its desktop hold the other 1.3 GiB. The 3090 drives the
  3840 x 2160 display. The PC also has AMD Radeon integrated graphics (Ryzen 9 7950X) with no display attached.
- Run-time overhead above weights and pool: 1.6 GiB today (nvidia-smi peak 23,599 MiB with 17.66 GiB of weights
  and a 2.5 GiB pool; 8 sequences, chunk 1,024, LoRA on). It holds the CUDA context, compiled kernels, CUDA graphs
  (0.05-0.14 GiB, 0.36 with DSpark) and prefill activations. The fork author's run implies about 1.5 GiB with
  DSpark on native Linux. With chunk 256, one sequence and no LoRA it should be lower (est. 1.2-1.4 GiB).
- Budget used below for weights + pool: **21.0 GiB on WSL as it is now**, 22.0 GiB with the display on the AMD
  graphics, 22.4 GiB on headless native Linux. These are on the careful side (they keep today's 1.6 GiB overhead).

## 3. (B) Serving: options ranked by GiB per effort

Context gained is for one sequence with the FP8 cache (65.5k tokens per GiB; half that in bf16).

| # | Option | Saves | Context gained | Effort | Risk | Confidence |
|---|---|---|---|---|---|---|
| 1 | `--max-num-batched-tokens 256` (from 1,024) | 1.36 GiB of pool in bf16 | 8k -> about 60k in bf16 with today's 2.5 GiB pool | One flag | Prefill maybe slightly slower per chunk; the fork author measured 2,500-2,700 tok/s at 256 | High (derived, matches two logs) |
| 2 | `--kv-cache-dtype fp8` | Halves all attention caches: 1.53 GiB per 100k tokens | x2 | One flag | Precision change, emulated E4M3 on this card. The author did not rerun quality benchmarks. Decode pays for the bit-level decode (he still got 91 tok/s) | High that it runs (it is his tested setup); medium on quality |
| 3 | DSpark off for the long-context server, `--max-num-seqs 1`, no `--enable-lora` | 0.73 GiB weights + 0.2-0.3 GiB graphs + 0.38 GiB per 100k tokens | +50k to +100k | Flags (`DSPARK=0`; LoRA flags need a switch in `serve.sh`) | Decode 1.3x slower on long outputs (measured gain of DSpark) | High |
| 4 | Raise the pool to what fits (`KV_CACHE_BYTES`; today 2.5 of about 3.4-3.7 GiB possible) | 0.9-1.2 GiB unused today | +60k | One variable; find the limit by steps of 0.25 GiB | OOM at the first long prefill if set too high. Set the NVIDIA "CUDA sysmem fallback policy" to "prefer no sysmem fallback" so an overshoot fails instead of silently paging to system RAM | High |
| 5 | Display on the AMD integrated graphics | About 1.0 GiB (of the 1.3 Windows holds) | +65k | Move the monitor cable to the motherboard output; enable the iGPU in the BIOS if needed; set Windows' default graphics to the AMD adapter | Desktop and browser render on the iGPU (fine for 4K desktop work, not for games). Some apps may still open a context on the 3090 | Medium-high; the size is est. until measured with `nvidia-smi` |
| 6 | Embedding on the CPU with `VLLM_WSL2_ENABLE_PIN_MEMORY=1` | 0.86 GiB | +56k | One environment variable plus `--cpu-offload-gb 1 --cpu-offload-params embed_tokens` | The fork disables pinned memory on WSL by default (`platforms/cuda.py`) and has this switch for exactly this case. `serve.sh` never set it, so the 2026-11-21 failure may only be that default. WSL pinned memory is limited; 0.86 GiB may or may not be granted. Speed cost measured by the author: 1-3% | Medium (untested here) |
| 7 | Expert layers 20-23 re-stored as NVFP4 in a converted checkpoint | 1.78 GiB | +117k | Medium: an offline CPU script (the codec exists, `inkling_nvfp4.py`), new tensors and an edited `hf_quant_config.json`; the fork picks the format per layer from that file (`is_mxfp8_layer`) | Quality: the authors kept these four layers at 8 bits. Our training loader already does this; measured weight error 8.7% relative on layer 20 (NVFP4 vs the MXFP8 original), no end-task measurement. Needs a coding benchmark before trusting it | High that it loads and fits; low-medium on quality |
| 8 | Attention weights in 8 bits (stock only; blocks or complicates LoRA) | 0.6 GiB | +39k | High: the fork runs these through its own Triton linear kernels in bf16; needs an 8-bit decode in those kernels | Low-medium quality risk; some speed cost | Medium |
| 9 | Output layer in 8 bits | 0.43 GiB | +28k | High: new GEMV path on Ampere | Logit noise; could speed decode (it reads 0.86 GiB per token today) | Medium |
| 10 | Smaller convolution-state cache (the per-head width is padded from 832 to 1,024; state for 3 tokens is held in blocks sized for more) | 0.05-0.1 GiB at F = 512 | small | Medium, in the fork | None | Medium |
| 11 | Native Linux, headless | 0.4 GiB beyond option 5, and option 6 works for certain | +26k | Dual boot | None | Medium |

Not options:

- **Sharing the loop passes' cache**: refused for this checkpoint and would change outputs (section 2.2).
- **4-bit KV**: no kernel in the fork; quality unknown. It would halve the per-token cost again; high effort.
- **KV on the CPU**: `--kv-offloading-size` stores finished or evicted prefix blocks in RAM and loads them back
  into the GPU pool on a prefix hit. It does not extend a live sequence, because the 8 full-attention calls read
  every key and value for every token (4.9 GB per token at 300k in FP8; over PCIe that is 2-3 tok/s). It is useful
  for a different thing: keeping several long sessions warm without holding them in VRAM. Not tested with this
  fork's cache groups.
- **Dropping the vision tower**: already done.

### 3.1 What context each combination gives (derived; one sequence, FP8, no DSpark, chunk 256)

| Weights | WSL as now (21.0) | Display on AMD (22.0) | Native headless (22.4) |
|---|---|---|---|
| Stock, 17.64 GiB (options 1-4 only) | 188k | 253k | 280k |
| + embedding on CPU (6) | 244k | 310k | 336k |
| + layers 20-23 as NVFP4 (7) | 304k | 370k | 396k |
| + both | 361k | 426k | 453k |
| + attention and output layer in 8 bits (8, 9) | 428k | 494k | 520k |

In bf16 these are roughly halved (97k for the first cell). With DSpark on, the first row is 98k / 150k / 171k.
Take 10% off for safety: the budget is an estimate and activation memory at long T is not measured here (the
author prefilled 126k inside 23.8 GiB).

**Realistic maximum on this card after the cheap options (1-6): 240k to 310k.** With the converted checkpoint (7)
as well: 360k to 430k.

### 3.2 300k, directly

- Flags alone on today's WSL setup: no, about 190k.
- Flags + display on the AMD graphics + embedding offload working: yes on paper (310k), with no margin.
- Flags + display on AMD + layers 20-23 as NVFP4: yes with margin (370k), at an unmeasured quality cost.
- What 300k costs in use even when it fits: a cold prefill of several minutes (the author measured 1,870 tok/s at
  126k and falling, so est. 4-6 minutes for 300k; prefix caching makes later turns cheap), slower decode (the full
  layers read about 4.9 GB of cache per token; est. 40-60 tok/s instead of 91), and one sequence only.
- The model's own limit is 1,048,576 tokens, so `--max-model-len 300000` is accepted.

### 3.3 Suggested order (all GPU steps wait for the training run)

1. `serve.sh`: chunk 256, FP8, `DSPARK=0`, one sequence, no LoRA; raise `KV_CACHE_BYTES` in steps and read
   "GPU KV cache size" and the nvidia-smi peak. Expect about 190k. Half an hour.
2. Move the display to the AMD output and read `nvidia-smi` on Windows with nothing running. Expect under 0.3 GiB
   used instead of 1.3.
3. Try `VLLM_WSL2_ENABLE_PIN_MEMORY=1` with the embedding offload.
4. Quality check of FP8 KV on a coding task set against bf16 at a context both can hold (for example 60k).
5. Only if 300k is still wanted with margin: the converted checkpoint (7), with the same quality check.

## 4. (A) QLoRA training, briefly

After load 15.0 GiB is allocated (experts all NVFP4 12.8, attention 1.2, output layer 0.86; embedding on the CPU);
the batch-2 peak is 21.5 GiB.

| Option | Saves | Effort | Risk | Confidence |
|---|---|---|---|---|
| Display on the AMD graphics (same as B-5) | About 1.0 GiB of headroom | User, minutes | None | Medium-high |
| NVIDIA "prefer no sysmem fallback" | Nothing, but an overshoot fails fast instead of slowing down | User, minutes | None | High |
| Attention base weights in 8 bits under the LoRA | 0.6 GiB | Medium | The adapter is trained on a base that the vLLM server (bf16 attention) does not have; small mismatch | Medium |
| Output layer in 8 bits | 0.43 GiB | Medium (TRL's fused head assumes bf16) | Low | Medium |
| Full-attention calls without T x T tensors | Only matters above about 2,000 tokens | High | As in `inkling_general_speed.md` 4.1 | - |

Already used, nothing left: per-layer gradient checkpointing, fused loss, packed experts, K1/K2 (the saved `a` and
`dfac` are part of the 0.31-0.56 GiB per call), optimizer state (17 MB). Checkpoints to the CPU would save about
0.2 MB per token, which is nothing at 1,024 tokens. Long-sequence training is out of scope per the user.

## 5. What is uncertain

- The budget (21.0 / 22.0 / 22.4 GiB) rests on one measured overhead of 1.6 GiB; it could be 0.3 GiB better.
- Whether pinned memory works on this WSL for 0.86 GiB, and whether the 2026-11-21 failure was only the default.
- FP8 KV quality (emulated on Ampere) and NVFP4 for layers 20-23, both for coding at long context.
- Activation memory and speed of prefill and decode beyond 131k; nobody has reported a run that long on a 3090.
- How much of the 1.3 GiB Windows releases when the display moves.

## 9. Item 7 assessed: layers 20-23 as NVFP4 in a converted checkpoint (2026-11-23)

Read from the fork (`vllm/models/inkling2/quant_config.py`, `nvidia/model.py`, `nvidia/moe.py`) and the checkpoint
header. Nothing run on the GPU. The converter's CPU dry run on layer 20 (4 of 128 experts) matched layer 0's names,
shapes and dtypes and gave 8.6-8.9% relative weight error, as the review's measured 8.7%.

**(a) Feasible: yes.** The fork chooses a layer's format only from `hf_quant_config.json`: `is_mxfp8_layer(l)` is true
if `model.llm.layers.{l}.mlp.experts.w13_weight` is in the `mxfp8` parameter group; otherwise the layer is NVFP4.
Per converted layer and projection (w13: N=2304, K=2304; w2: N=2304, K=1152; 128 experts):

| Tensor | MXFP8 now | NVFP4 needed |
|---|---|---|
| `w13_weight` | F8_E4M3 (128,2304,2304) | U8 (128,2304,1152) |
| `w13_weight_scale` | U8 (128,2304,72) | F8_E4M3 (128,2304,144) |
| `w13_weight_amax` | absent | F32 (128,) |
| `w13_weight_outer_scale_denominator` | absent | F32 (1,), value 2688 (= 6 x 448, as in layers 0-19) |
| `w2_weight` / `_scale` / `_amax` / `_denominator` | (128,2304,1152) / (128,2304,36) | U8 (128,2304,576) / F8_E4M3 (128,2304,72) / F32 (128,) / F32 (1,) |

The fork's per-expert global scale is `amax / denominator`; our `quantize_nvfp4` factor is `amax / 2688`, so
`amax = factor x 2688`. Row order is the checkpoint's (gate/up interleaved) in both formats; the loader copies as is
(it only reshuffles on sm100) and its dtype and shape check would fail loudly on a mistake. Edits: move the 8 weight names
(layers 20-23, w13 and w2) from the `mxfp8` group to the `nvfp4` group in `hf_quant_config.json` (the empty mxfp8 group is
dropped); add 16 keys to `model.safetensors.index.json` and update `total_size`. `config.json` needs nothing (its
`quantization_config` says `modelopt`/`NVFP4`; the fork overrides it by architecture). MTP/DSpark weights are dense bf16 and untouched.
One risk: the fork's per-layer NVFP4 path is exercised by layers 0-19, but a model with 24 NVFP4 layers has not been run by anyone; the
code has no all-layers-equal assumption that I found.

**(b) Disk.** The snapshot is ONE 19.53 GB shard (`model-00001-of-00001.safetensors`, every tensor in it), so symlinks
cannot share it: the shard must be rewritten (the fork would otherwise also read the old MXFP8 tensors from it).
New shard: 19,526,019,168 B - 4 x 1,051,066,368 + 4 x 573,309,960 = 17.62 GB (16.4 GiB). Everything else (`mtp.safetensors`
0.64 GB, tokenizer, config, template) is symlinked. The WSL disk has 629 GiB free. Written to `~/inkling_nvfp4_all/`; the
original is only read.

**(c) VRAM saved: 1.78 GiB, verified.** Per layer 1,051,066,368 - 573,309,960 = 477,756,408 B; x 4 = 1,911,025,632 B = 1.780 GiB
(0.979 -> 0.534 GiB per layer). About +117k tokens of FP8 context, as in the table of section 3.

**(d) Quality plan (bar fixed before any run).** Same prompts, same vLLM flags (`serve_long.sh`, FP8 KV, DSpark off, greedy,
temperature 0), stock vs `INKLING_MODEL_DIR=$HOME/inkling_nvfp4_all`.
1. Logprob agreement: 150 fixed texts of about 1,000 tokens (70 code, 50 prose, 30 QA/long-form); stock generates 256 greedy
   tokens each; both models then score the stock continuation teacher-forced (`prompt_logprobs`). Pass: mean NLL rise
   <= 0.03 nats/token, and top-1 agreement with the stock model >= 95%; no single text with a rise over 0.15.
2. Coding: HumanEval (164, greedy, pass@1) on both. Pass: converted drops by at most 2 points (3 problems).
3. QA/long context: 100 fixed questions with passages (or the probe's needle test at 64k and 128k). Pass: accuracy drop
   <= 2 points and the needle found at both lengths.
4. Controls: stock bf16 KV against stock FP8 KV on item 1 gives the size of the "ordinary" noise; a converted result worse than
   that noise plus the bars above fails. Fail on any bar = keep stock layers 20-23 (the authors kept them at 8 bits for a reason).

**Files and GPU commands for later**
- `scripts/inkling_vllm/convert_layers_nvfp4.py`: `--dry-run-layer N [--max-experts K]` (done for 20), then the full run on the CPU:
  `~/vllm-venv/bin/python scripts/inkling_vllm/convert_layers_nvfp4.py` (about 3 min per layer, 12-15 min, 16.4 GiB written).
  The write path (header, byte copy, config and index edits) is written but was not run, by instruction; check its output with
  a `safe_open` header read and the fork's loader before trusting it.
- `scripts/inkling_vllm/serve_long.sh` (new; `INKLING_MODEL_DIR` picks the converted copy) and `long_context_probe.sh`.
- Serve test: `INKLING_MODEL_DIR=$HOME/inkling_nvfp4_all DSPARK=0 MAX_MODEL_LEN=300000 scripts/inkling_vllm/serve_long.sh`.
