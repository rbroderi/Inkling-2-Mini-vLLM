# Inkling-2-Mini: fused frozen-expert kernels for QLoRA on one RTX 3090

Status: plan only, written 2026-11-21. Nothing here has been run on the GPU. Every performance number in
sections 2 and 3 is an estimate from operation counts, not a measurement; the measurements in section 1 are the
user's A/B timings.

## 1. Facts the plan rests on (checked on disk, CPU only)

Model (config.json and safetensors headers of the local snapshot):

- hidden H = 2304, **moe_intermediate I = 1152** (the brief guessed 2304; the config class defaults
  `moe_intermediate_size` to 1152 and the tensor shapes confirm it), 128 experts, top-8, 24 layers, layers 6-17 run
  3 times, so **48 expert-layer invocations per forward pass**.
- `top_k_weights` is float32 (the router runs in fp32); hidden states are bf16.
- Layers 0-19 are NVFP4, per layer:
  - `w13_weight` uint8 (128, 2304, 1152): rows = 2I outputs (gate/up interleaved: even row = gate_i, odd row =
    up_i), 2 codes per byte along the H input columns;
  - `w13_weight_scale` float8_e4m3 (128, 2304, 144): one per 16 input columns;
  - `w2_weight` uint8 (128, 2304, 576): rows = H outputs, columns = I inputs; `w2_weight_scale` (128, 2304, 72);
  - `*_amax` float32 (128,), `*_outer_scale_denominator` float32 (1,), so `factor` is one scalar per expert.
- Layers 20-23 are MXFP8: values float8_e4m3 (128, 2304, 2304) and (128, 2304, 1152), scales uint8 per 32 columns.

Expert memory:

| Storage | Per layer | Layers 20-23 | All 24 layers |
|---|---|---|---|
| NVFP4 as in the checkpoint (layers 0-19) | 0.534 GiB | - | 10.68 GiB for 20 layers |
| MXFP8 re-quantized to NF4 (today) | about 0.48 GiB | about 1.9 GiB | about 12.6 GiB |
| MXFP8 kept native | 0.979 GiB | 3.92 GiB | 14.6 GiB (+2.0 GiB) |
| MXFP8 re-quantized to the NVFP4 layout | 0.534 GiB | 2.14 GiB | 12.8 GiB (+0.24 GiB) |
| A second, transposed packed copy of everything | - | - | +10.7 to +12.8 GiB (does not fit) |

Environment (`~/inkling-venv`): torch 2.14.1+cu130, **triton 3.8.0** (has `tl.interleave` and `tl.bfloat16`),
bitsandbytes 0.50.2, peft 0.22.0. **No pytest, no humming, no vllm** in that environment. The Windows `uv` environment
has no Triton and no Inkling transformers build, so CPU tests must not import either.

Work per example at 750 tokens (6,000 token-expert pairs per layer invocation):

- GEMM: 7.96 M multiply-adds per pair, so about 4.6 TFLOP per forward pass. With gradient checkpointing the forward
  runs twice and the input-gradient pass costs the same again: about 13.8 TFLOP per example.
- Dequantization: nearly all 128 experts are hit, so each layer invocation decodes about 1.02 G weight elements;
  x 48 x 3 passes = about 147 G element decodes per example. This is the same order of cost as the GEMM itself, so
  the kernel must decode in registers and never write bf16 weights to memory.

## 2. Approach

### 2.1 Decision: one new Triton grouped-GEMM kernel with in-register NVFP4 decode, used for forward and backward

A single kernel `nvfp4_grouped_gemm` with a compile-time `TRANSPOSE` switch computes, for rows sorted by expert:

- `TRANSPOSE=False`: `out[p, r] = sum_c inp[p, c] * W[e(p), r, c]` (the forward `F.linear`);
- `TRANSPOSE=True`: `out[p, c] = sum_r inp[p, r] * W[e(p), r, c]` (the input gradient `dY @ W`).

Both read the **same packed checkpoint layout**. A Triton tile is addressed by arbitrary 2-D pointer arithmetic, so
the transposed product only changes which axis is the reduction axis; no transposed copy and no bf16 scratch copy is
needed. Memory cost of the backward: 0 bytes of extra weights.

Why not the alternatives:

- **humming 0.1.20** (Apache-2.0, so the licence is fine): forward-only. It has no transposed GEMM and no autograd.
  It repacks weights into its own padded layout, so using it for the forward would mean either a second copy of the
  weights for the backward (+10.7 GiB, does not fit) or a backward kernel that reads humming's internal layout
  (undocumented). An exact transposed NVFP4 copy does not exist anyway: the block scales run along the input axis, so
  a transposed copy would need re-quantization. It is also not installed in `~/inkling-venv` and installing is out of
  scope. Rejected.
- **vLLM `fused_moe` Triton kernel and `moe_align_block_size`**: the right structure (rows sorted by expert, padded to
  the row block, `tl.dot` with fp32 accumulation), but it takes dense bf16 or fp16 weights, and `moe_align_block_size`
  is a compiled vLLM op that the training environment does not have. We copy the structure, not the code; the
  alignment is a few vectorized torch operations.
- **The vLLM fork's `ops/grouped_gemm.py`, `ops/nvfp4.py`, `ops/mxfp8.py`**: these quantize activations and run
  dense grouped GEMMs for serving; useful as reading for the E2M1 arithmetic, not reusable for a frozen-weight
  backward.
- **Dequantize a chunk of experts to bf16 scratch, then `torch.bmm`**: simpler, but it writes and re-reads about
  2 bytes per weight element per pass. Kept as the contingency if the fused kernel misses its gate (section 2.6).

### 2.2 Layers 20-23 (MXFP8)

Decision: **re-quantize layers 20-23 to the NVFP4 layout at load**, so one kernel and one layout serve all 24 layers.

- Native MXFP8 costs +2.0 GiB on a 19.9 GiB peak; that leaves about 2 GiB on a 24 GiB card that also drives the
  desktop. Rejected on memory.
- Keeping NF4 with the current loop for these layers costs 4 of 48 invocations of the old path, about
  (7.65 + 5.08) x 4 / 48 = 1.06 s per example. Rejected on time.
- The NVFP4 layout costs +0.24 GiB over NF4. Both are lossy 4-bit re-quantizations of 8-bit weights. Ticket T5
  measures the error of both against the MXFP8 original and applies a fixed gate (NVFP4 error at most 1.15 x the NF4
  error); if the gate fails the default stays NF4 and the question goes back to the user.
- This changes the weights of layers 20-23 slightly relative to today's NF4 path, so it is a load option
  (`mxfp8_storage`), recorded with each training run. See the questions at the end.

### 2.3 Forward and backward as one `torch.autograd.Function`

Notation: T tokens, K = 8, P valid pairs, d = dtype of hidden states (bf16 in training), f = float32 when d is bf16
(else d).

Routing plan (torch, no kernel):

1. `flat = top_k_index.reshape(-1)`; `order = argsort(flat, stable=True)`; keep the first P entries, where P is the
   number of entries with expert id < 128 (id 128 is the "no expert" sentinel of the original loop and is dropped).
2. `slot = order[:P]` (index into T x K), `token_idx = slot // K`, `counts = bincount(flat, minlength=129)[:128]`.
3. Row blocks of `BLOCK_M` rows per expert: `expert_of_block`, `block_row0`, `block_rows` (int32), built with
   `repeat_interleave`; an expert with n rows gets `ceil(n / BLOCK_M)` blocks, the last one partial.

Forward, with the same rounding points as the reference loop:

```
xs  = hidden.index_select(0, token_idx)                      # (P, H)  d
gu  = gemm(xs, "gate_up", transpose=False)                   # (P, 2I) d, gate = gu[:, 0::2], up = gu[:, 1::2]
a   = swiglu(gate.to(f), up.to(f)).to(d)                     # (P, I)
y   = gemm(a, "down", transpose=False)                       # (P, H)  d
ws  = top_k_weights.reshape(-1).index_select(0, slot)        # (P,)    float32
out = zeros(T, H, f).index_add_(0, token_idx, y.to(f) * ws[:, None]).to(d)
```

The kernel writes gate and up in checkpoint row order (interleaved), so the weights are never de-interleaved; the
activation reads two strided views.

Backward for `grad_out` (T, H):

```
go  = grad_out.to(f).index_select(0, token_idx)              # (P, H)
dws = (go * y.to(f)).sum(-1)                                 # (P,)  -> dW = zeros(T*K, f).index_copy_(0, slot, dws)
dy  = (go * ws[:, None]).to(d)
da  = gemm(dy, "down", transpose=True).to(f)                 # (P, I)
dgu = stack((da * dact_dgate, da * dact_dup), -1).reshape(P, 2I).to(d)
dxs = gemm(dgu, "gate_up", transpose=True)                   # (P, H)
dX  = zeros(T, H, f).index_add_(0, token_idx, dxs.to(f)).to(d)
return dX, None, dW.view(T, K).to(top_k_weights.dtype)
```

The gradient with respect to `top_k_weights` is required (the router input depends on upstream LoRA), so it is
always returned when requested. The expert weights get no gradient.

Softcapped SwiGLU, with caps cg = 4.0 and cu = 25.0 from the config: `act = cg*tanh(g/cg) * sigmoid(g) * cu*tanh(u/cu)`.
With `tg = tanh(g/cg)`, `s = sigmoid(g)`, `tu = tanh(u/cu)`:

- `dact/dg = cu*tu * ((1 - tg^2) * s + cg*tg * s * (1 - s))`
- `dact/du = cg*tg * s * (1 - tu^2)`

Saved for the backward: `gu` and `y` (bf16, 37 MB each at P = 8,000), `ws`, and the plan. They live only while one
layer's graph is alive.

Gradient checkpointing (`use_reentrant=False`): each layer's forward runs twice. (Corrected after the T2 review:
the first run is NOT under no_grad; non-reentrant checkpointing runs it with grad enabled and its hooks discard the
saved tensors, so the autograd Function is called twice, and the gradients match the run without checkpointing
exactly. Only the plan's small index tensors, ~16 bytes per pair per layer, stay alive.) The module calls a plain no-grad function (nothing saved) when grad is disabled or the input
does not require grad, and the autograd Function otherwise. Per example the kernel therefore runs 2 forward passes
and 1 transposed pass; this cannot be reduced without the memory that checkpointing saves.

### 2.4 Kernel design (fixed)

- Inputs: dense, already gathered and sorted `inp` (P, K_dim) bf16 or fp32; `codes` uint8 (E, R, C/2); `scale` uint8
  view of the e4m3 bytes (E, R, C/16); `factor` fp32 (E,); a 256-entry fp32 table that decodes e4m3 bytes; the block
  table. Output (P, N_dim) in the dtype of the supplied output tensor (bf16 in training, fp32 for tight tests).
- Grid `(num_blocks, N_dim / BLOCK_N)`. Each program loads its expert id, row start and row count, loops over the
  reduction axis in steps of `BLOCK_K`, and accumulates `tl.dot(inp_tile, w_tile)` in fp32.
- Weight tile decode, per step:
  1. load the byte tile (contiguous along C), widen to int32, `lo = b & 15`, `hi = b >> 4`, `nib = tl.interleave(lo, hi)`;
  2. E2M1 by arithmetic: `m = nib & 7; e = m >> 1; fbit = m & 1; mag2 = where(e == 0, fbit, (2 + fbit) << (e - 1))`
     (this is 2 x |value|, in {0,1,2,3,4,6,8,12}); `sign = 1 - 2 * (nib >> 3)`;
  3. scale tile: gather the e4m3 bytes through the 256-entry table, multiply by `0.5 * factor[e]`, broadcast over the
     16 columns of each block;
  4. `w = (sign * mag2).to(fp32) * scale`, cast to bf16 (the reference rounds the weight to bf16), then to the
     compute dtype of `inp`.
  For `TRANSPOSE=False` the tile is `[BLOCK_N, BLOCK_K]` and is transposed in registers before `tl.dot`; for
  `TRANSPOSE=True` it is `[BLOCK_K, BLOCK_N]` and used directly.
- Rows beyond a block's row count are loaded as zero and not stored.
- Start values: `BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=3`. T5 tunes over
  `BLOCK_M in {32, 64, 128}`, `BLOCK_N in {64, 128}`, `BLOCK_K in {32, 64}`, `num_warps in {4, 8}` and hard-codes the
  fastest combination per (projection, transpose).
- The activation, the routing-weight multiply and the two scatter-adds stay in torch (about 40 small operations per
  layer call; estimated 0.1-0.2 s per example in total).

### 2.5 Targets and how they are measured

Estimate (not measured): 13.8 TFLOP of GEMM plus 147 G decodes per example. At an effective 15-30 TFLOP/s for a
Triton bf16 kernel that also decodes its weights, the experts should cost about 0.8-1.5 s per example, plus about
0.15 s of torch glue, plus the measured 0.98 s for everything else.

| | Per example | Speed-up vs 13.7 s |
|---|---|---|
| Must (acceptance gate) | at most 3.0 s | 4.6x |
| Goal | at most 2.3 s | 6x |
| Floor without touching the rest of the model | about 1.0 s + experts | - |

Qwen parity (1.9 s) needs the experts under about 0.9 s and is not promised by this plan.

Measurement: `scripts/inkling/ab_step.py` (the scratchpad `ab_inkling_step.py`, copied into the repo by T0 and given
a `fused` mode by T5): same 4 examples, 2 warm-up steps, `torch.cuda.synchronize()` around each step, peak memory
from `torch.cuda.max_memory_allocated()`. No `torch.profiler`. The full-model run needs about 21 GiB free, so it can
only run when both long jobs are finished.

Correctness bars. `rel(a, b) = ||a - b||_F / ||b||_F`, computed in float64.

| Check | Where | Bar |
|---|---|---|
| NVFP4 decode vs `Quant4bitExperts.weight` formula, fp32 | CPU | exactly equal |
| Function with torch backend vs eager loop, float64, outputs and both gradients | CPU | rel at most 1e-12; `gradcheck` passes |
| Triton kernel vs torch backend, fp32 output, both directions, both projections | GPU | rel at most 1e-5 |
| Triton kernel vs torch backend, bf16 output | GPU | rel at most 2e-3 |
| Fused vs loop on a real layer, bf16: output, dX, d(top_k_weights) | GPU | rel at most 5e-3 output, 1e-2 gradients; cosine at least 0.9999 |
| Fused and loop vs an fp32 golden path | GPU | fused error at most 1.25 x loop error + 1e-4 |
| Full model, 4 fixed examples: loss | GPU, 21 GiB | relative difference at most 1e-2 |
| Full model: all LoRA gradients concatenated | GPU, 21 GiB | cosine at least 0.99 |
| 20 LoRA steps, same seed: mean loss of the last 5 steps | GPU, 21 GiB | within 5% |
| Peak memory | GPU, 21 GiB | at most 20.5 GiB allocated |

bf16 comparisons are GPU-only: CPU bf16 matmul does not accumulate in fp32, so it is not a valid reference.

### 2.6 Contingencies (not ticketed unless a gate fails)

- If the fused kernel is correct but the experts cost more than 2.0 s per example after tuning: add a kernel that
  decodes a chunk of 16 experts to a 255 MB bf16 scratch buffer and use padded `torch.bmm` for chunks where the mean
  rows per expert exceed a measured threshold.
- If glue dominates: fold the SwiGLU and the routing-weight multiply into kernel epilogues.

## 3. Integration and switch/rollback strategy

New modules, so the kernel code is separate from loading:

- `domain_verifier_factory/training/inkling_nvfp4.py`: pure-torch codec (decode, quantize, MXFP8 decode). CPU.
- `domain_verifier_factory/training/inkling_kernels.py`: routing plan, softcapped SwiGLU, the autograd Function, the
  torch reference backend. CPU. No Triton import.
- `domain_verifier_factory/training/inkling_triton.py`: the Triton kernel and `TritonBackend`. Imported lazily, only
  on the CUDA fused path.
- `inkling.py` keeps loading and `Quant4bitExperts`. The current per-expert loop stays, unchanged, as the reference
  implementation.

Switches:

- `Quant4bitExperts.impl`: `"loop"` (today's path) or `"fused"`. Set by `load_inkling_4bit(experts_impl=...)`; when
  that argument is None it comes from the environment variable `INKLING_EXPERTS_IMPL`, default `"loop"`.
- `load_inkling_4bit(mxfp8_storage=...)`: `"nf4"` (today) or `"nvfp4"`. A layer stored as NF4 always uses the loop.
- `"fused"` uses `TritonBackend` on CUDA and the torch backend on CPU (tests only).
- Training config: `inkling_experts_impl` and `inkling_mxfp8_storage` next to `loader` in `configs.py`, passed
  through in `qlora.py` and written into the run's saved hyperparameters.
- Defaults stay `"loop"` and `"nf4"` until T5's gates pass; T5 then changes the two config defaults. Rollback is
  setting them back, or `INKLING_EXPERTS_IMPL=loop`: no code is removed.

## 4. Constraints for every ticket

- **GPU lock.** GPU work runs only through `tests/gpu/gpu_guard.py` (T0): it takes `/tmp/inkling_gpu.lock` in WSL
  with `flock` (exclusive, non-blocking), then checks that `nvidia-smi` reports at least 8 GiB free (21 GiB for the
  full-model checks), and releases the lock in `finally`. If either check fails the script exits with code 75 and
  the implementer waits; it must not retry in a loop or touch the running jobs.
- No `torch.profiler` with CUDA activities. Time with `torch.cuda.synchronize()` and `time.perf_counter()`.
- No commits, no installs, no downloads.
- CPU tests: `uv run pytest tests/<file>` on Windows. GPU checks: plain scripts with asserts, run in WSL as
  `cd /mnt/e/code/aarr && PYTHONPATH=. ~/inkling-venv/bin/python tests/gpu/<script>.py` (there is no pytest in that
  environment). GPU scripts are named `check_*.py` so pytest does not collect them.
- Tier rule: haiku only for fully mechanical work, sonnet by default, opus for kernel numerics, autograd or anything
  that can fail silently. Each ticket's diff is reviewed by a fresh Opus reviewer using the ticket's checklist.

## 5. Tickets

Dependency graph: `T0` and `T1` in parallel; then `T2` (after T1) and `T3` (after T0, T1) in parallel; `T4` after T2;
`T5` after T3 and T4. Files are disjoint between tickets that run in parallel.

| ID | Tier | Summary | Depends on | Effort |
|---|---|---|---|---|
| T0 | sonnet | GPU guard, environment probe, harness copied into the repo | - | 1-2 h |
| T1 | sonnet | Pure-torch NVFP4/MXFP8 codec with CPU tests | - | 2-3 h |
| T2 | opus | Routing plan, SwiGLU, autograd Function, torch backend, CPU tests | T1 | 4-5 h |
| T3 | opus | Triton `nvfp4_grouped_gemm` and `TritonBackend`, GPU check on synthetic weights | T0, T1 | 6-8 h plus GPU time |
| T4 | sonnet | Integration into `inkling.py`, config plumbing, MXFP8 re-quantization at load | T1, T2 | 2-3 h |
| T5 | opus | Real-layer numerics, re-quantization gate, tuning, full-model A/B, flip defaults | T3, T4 | 4-6 h plus GPU windows |

---

### T0. Environment: GPU guard, probe, harness (sonnet)

> **Done 2026-11-21, except the GPU run of check_env.py (the GPU was full).** The Opus review passed the lock, the exit codes, pytest collection and the copied scripts. Its 4 findings were fixed and rechecked: an fp32 `tl.dot` probe at ieee (tf32 shown alongside), a 2-D interleave→trans→dot probe, nvidia-smi failures exit 75, and os.utime on the lock file. Divergences: the copies were ruff-formatted, with import sorting and `# noqa: E501` on 3 lines; the guard is imported as `tests.gpu.gpu_guard` (PYTHONPATH=., no `__init__.py`).

Tier reason: small and well specified, but the lock must be correct, so not haiku.

Files (all new): `tests/gpu/gpu_guard.py`, `tests/gpu/check_env.py`, `scripts/inkling/ab_step.py`,
`scripts/inkling/lora_smoke.py`.

Spec:

1. `gpu_guard.py` exposes `gpu_session(min_free_mib: int = 8192)` as a context manager:
   - open `/tmp/inkling_gpu.lock` (create if missing), `fcntl.flock(fd, LOCK_EX | LOCK_NB)`; on `BlockingIOError`
     print a message and `sys.exit(75)`;
   - after the lock is held, run `nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits`; if the value is
     below `min_free_mib`, release and `sys.exit(75)`;
   - release (`LOCK_UN`, close) in `finally`. Do not delete the lock file.
2. `check_env.py`, inside `gpu_session()`: print torch, triton and CUDA versions and the device capability; compile
   and run one trivial Triton kernel that does `tl.dot` on two bf16 64x64 tiles with fp32 accumulation and one that
   uses `tl.interleave` on uint8-derived int32 tiles; compare both with torch (`rel <= 1e-5` for the dot, exact for
   the interleave). Uses under 100 MB.
3. Copy the two scratchpad scripts byte-for-byte except for their module docstring's first line:
   `ab_inkling_step.py` (a local scratch copy)
   to `scripts/inkling/ab_step.py`, and `test_inkling_gpu.py` in the same folder to `scripts/inkling/lora_smoke.py`.
   Wrap each script's body in `with gpu_session(min_free_mib=21000):`. No other change.

Acceptance:

- CPU, any time: `python -c "import ast,sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]"` on the four files;
  `uv run pytest` still collects the same tests as before (no new collection).
- CPU, WSL: two concurrent runs of a 5-second `with gpu_session(min_free_mib=0)` block: the second exits 75.
- Needs GPU, run after 02:00 when at least 8 GiB is free: `check_env.py` prints both comparisons as passed.

Review checklist:

- The free-memory check happens after the lock is taken, not before.
- The lock is released when the body raises and when `sys.exit` is called inside it.
- The free-memory query reads GPU 0 only and parses MiB, not a percentage.
- The copied scripts differ from the originals only by the guard and indentation (diff them).
- No `torch.profiler` import anywhere.

---

### T1. NVFP4 and MXFP8 codec in pure torch (sonnet)

> **Done 2026-11-21.** 11 tests pass (plus 1 optional real-checkpoint test, skipped by default). The Opus review found no correctness bugs: torch.equal to the reference on real layer-0 NVFP4 and layer-20 MXFP8 experts. Its 4 low findings were fixed and rechecked: a TypeError for non-uint8 integer scales, plus tests for the scale-candidate choice and the tie rules, each shown to catch its mutation. Divergences: a factor must match the codes' leading dims; zeros always encode as nibble 0; equal-error candidates pick /6.

Tier reason: exact, CPU-testable arithmetic with a clear specification.

Files (new): `domain_verifier_factory/training/inkling_nvfp4.py`, `tests/test_inkling_nvfp4.py`.

Spec (all functions work on CPU and CUDA, import only torch):

- `E2M1_VALUES`: the 16-tuple now in `inkling.py` (do not edit `inkling.py` in this ticket).
- `e4m3_table(device=None) -> Tensor`: float32 (256,), built as
  `torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()` with non-finite entries set to 0.
- `dequantize_nvfp4(codes, scale, factor, dtype=torch.float32) -> Tensor`: `codes` uint8 (..., R, C/2), low nibble =
  even column; `scale` uint8 or float8_e4m3fn (..., R, C/16); `factor` float tensor broadcastable to the leading
  dimensions. Returns (..., R, C) in **checkpoint row order** (no gate/up de-interleave):
  `value[nibble] * scale * factor`, computed in float32 in exactly this order
  `(byte_to_values[codes] * scale[..., None]) * factor` so it equals `Quant4bitExperts.weight` before its bf16 cast;
  for `dtype=float64` compute in float64.
- `quantize_nvfp4(w) -> (codes, scale, factor)` for one matrix `w` (R, C) float, C divisible by 16:
  - `factor = amax(|w|) / (6.0 * 448.0)` as a float32 0-dim tensor (if `amax == 0`, factor = 1.0);
  - per block of 16 columns, two candidate scales `blockmax / 6.0 / factor` and `blockmax / 4.0 / factor`, each
    rounded to the nearest finite positive e4m3 value by `torch.bucketize` over the sorted positive entries of
    `e4m3_table()` (ties to the lower code; clamp to the table's range; an all-zero block gets the smallest positive
    scale and all-zero codes);
  - codes: nearest of the 8 magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6} to `|w| / (scale * factor)`, ties to the smaller
    magnitude, sign bit 8 for negative values; a zero is encoded as nibble 0;
  - keep, per block, the candidate with the smaller sum of squared reconstruction errors;
  - pack: byte = `even_nibble | (odd_nibble << 4)`; `scale` returned as float8_e4m3fn via `.view` of the uint8 codes.
- `dequantize_mxfp8(values, scale) -> Tensor`: `values` float8_e4m3fn (..., R, C), `scale` uint8 (..., R, C/32);
  returns float32 `values.float().view(..., R, C/32, 32) * exp2(scale.float() - 127)[..., None]` reshaped.

Acceptance (CPU, `uv run pytest tests/test_inkling_nvfp4.py`):

- `dequantize_nvfp4` on random codes/scales/factor (R=8, C=64, 3 experts) is exactly equal (`torch.equal`) to a
  literal copy of the `Quant4bitExperts.weight` NVFP4 branch without its de-interleave and bf16 cast.
- Every one of the 16 nibbles decodes to `E2M1_VALUES`; nibble order within a byte is low-then-high.
- Round trip: `quantize_nvfp4(dequantize_nvfp4(codes, scale, factor))` reproduces the matrix exactly when the input
  was produced with `factor` from the function's own rule; for Gaussian `w` (256 x 256) the relative error is below
  0.12 and every `scale` byte is a finite positive e4m3 code.
- All-zero matrix and a matrix with one all-zero block quantize without NaN and decode to zeros there.
- `dequantize_mxfp8` equals the formula in `transformers.core_model_loading.Inkling2LoopDequantizeExperts` (copied
  into the test as a literal, since that module is not in the Windows environment).

Review checklist:

- Nibble order (low nibble = even column) and that negative zero (nibble 8) decodes to 0.
- `scale` is interpreted as e4m3 bytes, never cast numerically from uint8 (`.view`, not `.to`).
- The e4m3 encoder never emits 0x7F/0xFF (NaN) or a negative code.
- Multiplication order matches the reference so the exact-equality test is meaningful, and the test's reference is
  an independent literal, not a call to the function under test.
- Output is in checkpoint row order; nothing de-interleaves.

---

### T2. Routing plan, SwiGLU, autograd Function, torch backend (opus)

> **Done 2026-11-21.** 19 tests pass: float64 vs an independent eager loop within rel 1e-12, and gradcheck passes. The Opus review found no bugs; it checked the SwiGLU, the interleave, the routing, the top-k gradient and the dtypes against the transformers module, and its injected bugs failed the tests. Divergences: `ExpertsBackend.num_experts` was added; f is float32 for fp16 too, and the input dtype for non-half; the transposed GEMMs are skipped when only top_k_weights needs a gradient. bf16 parity vs a bf16 eager loop waits for T5.

Tier reason: hand-written backward; errors here give plausible but wrong gradients.

Files (new): `domain_verifier_factory/training/inkling_kernels.py`, `tests/test_inkling_kernels.py`.

Spec (imports torch and `inkling_nvfp4` only; no Triton, no transformers):

- `class SoftcappedSwiGLU(nn.Module)` with `__init__(gate_cap: float, up_cap: float)` and
  `forward(gate, up)`: computes in float32 when the input is bf16/fp16 (else in the input dtype) and returns the
  input dtype: `gate_cap*tanh(gate/gate_cap) * sigmoid(gate) * up_cap*tanh(up/up_cap)`.
- `softcapped_swiglu_grads(gate, up, gate_cap, up_cap) -> (dgate, dup)`: the two formulas of section 2.3, inputs
  already in the float compute dtype.
- `@dataclass RoutePlan`: `num_pairs: int`, `slot` int64 (P,), `token_idx` int64 (P,), `expert_of_block` int32 (NB,),
  `block_row0` int32 (NB,), `block_rows` int32 (NB,), `counts` int64 (E,).
- `build_route_plan(top_k_index, num_experts, block_m) -> RoutePlan` exactly as in section 2.3 (stable argsort,
  sentinel id `num_experts` dropped, vectorized, no Python loop over experts).
- Backend protocol: an object with
  `gemm(inp: Tensor, proj: str, transpose: bool, plan: RoutePlan) -> Tensor`, `proj in {"gate_up", "down"}`, rows in
  plan order, output dtype = input dtype, shapes as in section 2.1/2.3 (gate_up forward output and gate_up transposed
  input are in interleaved checkpoint order).
- `class TorchBackend`: `__init__(gate_up_codes, gate_up_scale, gate_up_factor, down_codes, down_scale, down_factor)`.
  `gemm` loops over experts with rows, dequantizes with `dequantize_nvfp4`; for bf16/fp16 input it rounds the weight
  to the input dtype, then multiplies in float32 and rounds the result to the input dtype; for float32/float64 input
  it computes in that dtype with the weight dequantized in that dtype and no bf16 rounding.
- `class FusedExpertsFunction(torch.autograd.Function)`:
  `forward(ctx, hidden, top_k_index, top_k_weights, backend, gate_cap, up_cap, block_m)` and
  `backward` (decorated `once_differentiable`) exactly as in section 2.3, including dtypes and rounding points.
  Returns `None` for inputs whose `ctx.needs_input_grad` is false and for all non-tensor arguments. With P = 0 it
  returns zeros and zero gradients.
- `fused_experts(hidden, top_k_index, top_k_weights, backend, gate_cap, up_cap, block_m=64) -> Tensor`: if grad is
  disabled, or neither `hidden` nor `top_k_weights` requires grad, run the same forward without the Function and
  without saving anything; otherwise call the Function. Both paths share one private forward helper.

Acceptance (CPU, `uv run pytest tests/test_inkling_kernels.py`), tiny sizes E=4, H=32, I=16, T=5, K=2, random packed
weights from `quantize_nvfp4`, with at least one sentinel index and one expert that receives no token:

- The test contains its own 12-line eager loop (the original algorithm, de-interleaving gate/up as `inkling.py`
  does, dtype-parametrized). Float64: output, d(hidden) and d(top_k_weights) from `fused_experts` match the loop's
  autograd results with rel at most 1e-12.
- `torch.autograd.gradcheck` on `fused_experts` in float64 with respect to `hidden` and `top_k_weights` passes
  (`eps=1e-6, atol=1e-5`).
- `softcapped_swiglu_grads` matches autograd of `SoftcappedSwiGLU` in float64 (rel at most 1e-12), with inputs
  spanning -40..40.
- `build_route_plan`: for random indices and `block_m in {1, 2, 64}`, the blocks cover rows `0..P-1` exactly once,
  every row of a block belongs to `expert_of_block`, and `token_idx`/`slot` reproduce the eager loop's pairs.
- Under `torch.no_grad()` the result equals the grad-enabled result exactly and the Function is not invoked (assert
  via a counter monkeypatched onto `FusedExpertsFunction.forward`).
- All-sentinel input returns zeros with zero gradients.

Review checklist:

- d(top_k_weights) is scattered back with `slot` (pair order), not `token_idx`, and uses `y` before the weight
  multiply.
- The gradient uses the transposed GEMM with the matching projection (down first, then gate_up) and re-interleaves
  `(dgate, dup)` in the same order the forward de-interleaved (even = gate).
- `index_add_` (not `index_copy_`) for per-token accumulation; `index_copy_` only where indices are unique (`slot`).
- No tensor is silently saved on the no-grad path; saved intermediates are not modified in place afterwards.
- The test's eager loop is independent of the code under test and includes the gate/up row de-interleave.
- Sentinel handling: pairs with id == num_experts contribute nothing and receive zero weight gradient.

---

### T3. Triton `nvfp4_grouped_gemm` and `TritonBackend` (opus)

> **GPU check passed 2026-11-22 02:21** (beside the plasticity job, 11 GB free; peak 630 MiB). Real shapes (2304×2304, 2304×1152), both transposes, two block configs: fp32/bf16-in-fp32-out rel ≤ 2.5e-6 (bar 1e-5), bf16 out 1.66e-3 (bar 2e-3), canaries intact, deterministic. fused_experts Triton vs Torch: fp32 ~1.5e-6, bf16 out 4.8e-4 / dX 6.1e-4 / dW 4.0e-4. Timing at 6,000 rows over 128 experts: gate_up 11.4 ms forward / 7.5 ms transposed, down 3.9 / 3.2 ms (5.6–10 TFLOP/s). T0's check_env passed too (bf16 dot, ieee fp32 dot 1.5e-7, tf32 7.6e-4, 2-D interleave→trans→dot).
> **Done on the CPU 2026-11-21.** Under TRITON_INTERPRET=1: kernel vs float64 rel ~1e-7 over 16 cases (empty, partial and multi-block experts, both transposes), and fused_experts with TritonBackend vs TorchBackend at ~3e-7 on out, dX and dW. The Opus review found no kernel or backend bugs (decode, transposed indexing, ieee precision, bounds, int64 offsets). Its findings were fixed and rechecked: finite exact canaries plus TAIL rows in check_kernel.py, now shown to catch an off-by-one mask; the lock threshold is 4096 MiB (lead's choice). Divergences: `round_weight_bf16` and `out=` kwargs; fp16/fp64 raise; a block_m guard with 1 host sync per plan. **For T5:** keep each projection's BLOCK_M ≥ the plan's block_m; a larger BLOCK_M than the plan's block_m wastes tile rows, so tune them together.

Tier reason: kernel numerics and indexing; mistakes produce plausible numbers.

Files (new): `domain_verifier_factory/training/inkling_triton.py`, `tests/gpu/check_kernel.py`.
Can run in parallel with T2: it depends only on T1's codec and on the `gemm` protocol and `RoutePlan` fields fixed
in T2's spec (the check script builds the block table itself with a local helper, so it does not import T2's code).

Spec:

- One `@triton.jit` kernel as designed in section 2.4, with constexprs `R, C, BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE,
  OUT_BF16`. Assert on the Python side: `C % 16 == 0`; N_dim divisible by `BLOCK_N`; for `TRANSPOSE=False`,
  `BLOCK_K % 16 == 0` and `C % BLOCK_K == 0`; for `TRANSPOSE=True`, `BLOCK_N % 16 == 0` and `R % BLOCK_K == 0`.
- `nvfp4_grouped_gemm(inp, codes, scale_u8, factor, e4m3_lut, expert_of_block, block_row0, block_rows, transpose,
  out_dtype=None, block_m=64, block_n=64, block_k=32, num_warps=4, num_stages=3) -> Tensor`. `inp` must be
  contiguous bf16 or fp32 on CUDA; the block table must have been built with the same `block_m`. With zero blocks it
  returns an empty-row-safe zero tensor without launching.
- `class TritonBackend` with the same constructor as `TorchBackend` (it stores `scale.view(torch.uint8)`, a
  `e4m3_table` on the device, and a per-(proj, transpose) dict of block sizes initialised to the start values) and
  the same `gemm` signature. It holds references to the module's buffers; it must not copy the codes.
- No in-kernel gather, no stream-K, no atomics: each output element is written by exactly one program, so results
  are deterministic.

Acceptance:

- CPU, any time: the module parses (`ast.parse`); importing `domain_verifier_factory.training.inkling_kernels` on
  Windows still works (nothing imports `inkling_triton` at module import time).
- Needs GPU, run after 02:00 when at least 8 GiB is free (`tests/gpu/check_kernel.py`, inside `gpu_session()`, uses
  under 1 GiB): synthetic weights from `quantize_nvfp4` with E=8; shapes (R, C) in {(128, 64), (2304, 2304),
  (2304, 1152)}; row counts per expert in {0, 1, 63, 64, 65, 200}; both `transpose` values; reference = dense
  dequantized weights (rounded to bf16) and a float32 matmul per expert:
  - fp32 output: rel at most 1e-5; bf16 output: rel at most 2e-3;
  - rows of an expert with 0 rows do not exist; the row after a partial block is not overwritten (fill the output
    with a canary value first and check untouched padding in an over-allocated output tensor);
  - two runs give bit-identical outputs;
  - prints time per call for the two real shapes at 6,000 rows over 128 experts (synchronize timers), for the record
    only; no gate here.

Review checklist:

- The decode uses the low nibble for the even column and the scale of block `c // 16` of the same row `r`.
- `TRANSPOSE=True` reduces over rows r and indexes scale by `(r, c // 16)`; it is not just a relabelled forward.
- uint8 values are widened before shifts; the sign is taken from bit 3; negative zero gives 0.
- The weight is rounded to bf16 before the multiply (matches the reference), and accumulation is fp32.
- Masked loads use `other=0`; stores are masked by the row count; no out-of-bounds read when the last block of the
  last expert is partial.
- The check script's reference does not call the kernel or share its decode helper.
- No `torch.profiler`; every GPU script runs inside `gpu_session()`.

---

### T4. Integration into `inkling.py` and the training config (sonnet)

> **Done 2026-11-22.** The full suite passes. The Opus review verified `_forward_loop` and `weight()` torch.equal to the original transformers activation and weight math (bf16 and fp32, several shapes and seeds); the defaults stay loop/nf4. Fixed over 2 review rounds:
> - INKLING_EXPERTS_IMPL overrides the config in qlora, and the effective value is saved in agent_config.json.
> - `_apply` clears the backend cache and keeps the packed buffers' dtypes under casts.
> - `resolve_experts_impl` is factored out and tested; the docstrings are updated.
>
> Real-weight CPU checks: layer-20 MXFP8 requantized to NVFP4 has rel 0.086 vs the MXFP8 values; fused vs loop in fp32 differ by 3.4e-3, as expected since the loop rounds weights to bf16 (bf16 parity is a T5 check).

Tier reason: clear specification with CPU tests; the numerics live in T1-T3.

Files: `domain_verifier_factory/training/inkling.py`, `domain_verifier_factory/training/configs.py`,
`domain_verifier_factory/training/qlora.py`, `tests/test_inkling.py` (extend).

Spec:

1. `Quant4bitExperts.__init__`: replace the transformers `Inkling2LoopSoftcappedSwiGLU` with
   `inkling_kernels.SoftcappedSwiGLU(config.mlp_gate_softcap, config.mlp_up_softcap)` (same math; makes the module
   constructible without the Inkling transformers build). Import `E2M1_VALUES` from `inkling_nvfp4`. Add
   `self.impl = "loop"`.
2. Rename the current `forward` body to `_forward_loop` without changing it. New `forward`: if
   `self.impl == "fused" and self.storage == "nvfp4"`, return
   `fused_experts(hidden_states, top_k_index, top_k_weights, self._backend(hidden_states.device), gate_cap, up_cap)`;
   otherwise `_forward_loop`. `_backend` builds once and caches (not as a submodule or buffer): `TritonBackend` when
   the device is CUDA (lazy import of `inkling_triton`), else `TorchBackend`, from the existing `*_codes`, `*_scale`,
   `*_factor` buffers.
3. `set_nvfp4_packed(self, packed: dict[str, tuple[Tensor, Tensor, Tensor]], device)`: registers codes, scale and
   factor (E,) for `gate_up` and `down` and `byte_to_values`, sets `storage = "nvfp4"`. `set_nvfp4` is rewritten to
   compute the factors and call it; its behaviour is unchanged.
4. `load_inkling_4bit(..., experts_impl: str | None = None, mxfp8_storage: str = "nf4")`:
   - `experts_impl` None means `os.environ.get("INKLING_EXPERTS_IMPL", "loop")`; validate in {"loop", "fused"}; set
     `impl` on every `Quant4bitExperts`.
   - for MXFP8 layers with `mxfp8_storage == "nvfp4"`: `read_layer(..., device="cpu")`, then per expert and per
     projection move the raw values and scales to `device`, `dequantize_mxfp8`, `quantize_nvfp4` (rows stay in
     checkpoint order: no `Interleave`), collect, and call `set_nvfp4_packed`. With `"nf4"`: the current code path,
     unchanged.
5. `configs.py`: next to `loader`, add `inkling_experts_impl: Literal["loop", "fused"] = "loop"` and
   `inkling_mxfp8_storage: Literal["nf4", "nvfp4"] = "nf4"`. `qlora.py`: pass both to `load_inkling_4bit` and add
   both to the saved run metadata next to the existing hyperparameters.

Acceptance (CPU, `uv run pytest tests/test_inkling.py tests/test_training.py`):

- With a `SimpleNamespace` config (E=4, H=32, I=16, caps 4.0/25.0) and random packed weights set through
  `set_nvfp4_packed`, in float32: `impl="fused"` and `impl="loop"` outputs and input gradients agree with rel at most
  1e-5. The loop rounds each weight to bf16 and the float32 torch backend does not, so the test sets every
  `factor` to a power of two (2^-6): code x e4m3 scale x factor then has at most 6 significant bits and is exact in
  bf16, and both paths see identical weights.
- `set_nvfp4` with synthetic `amax`/`outer_scale_denominator` gives the same buffers as before the refactor
  (compare against values computed in the test).
- `storage == "nf4"` with `impl == "fused"` falls back to the loop (assert via monkeypatched `_forward_loop`).
- `INKLING_EXPERTS_IMPL` is honoured when `experts_impl` is None, and an invalid value raises `ValueError`.
- A fake MXFP8 layer (values and scales built in the test) loaded through the `"nvfp4"` branch helper reconstructs
  the original with relative error below 0.12, with gate rows still on even indices.
- Existing tests in `tests/test_inkling.py` and `tests/test_training.py` still pass; the config defaults are
  `"loop"` and `"nf4"`.

Review checklist:

- `_forward_loop` is byte-identical to the old `forward` body (diff it); the default behaviour is unchanged.
- The backend cache does not register tensors twice or break `.to(device)`; the backend references the same buffers.
- The re-quantized MXFP8 `gate_up` keeps checkpoint (interleaved) row order; `weight("gate_up", e)` of the loop path
  still de-interleaves it correctly, so loop and fused agree on re-quantized layers.
- Peak memory during the re-quantization at load is one expert in float32 on the device, not a whole layer in bf16.
- The two new settings reach `load_inkling_4bit` and the saved metadata; nothing else in `qlora.py` changes.
- No Triton or Inkling-transformers import at module import time.

---

### T5. GPU validation, re-quantization gate, tuning, full-model A/B, defaults (opus)

Tier reason: judging numerics and performance, and it flips the defaults.

Files: `tests/gpu/check_real_layer.py` (new), `tests/gpu/check_requant.py` (new), `tests/gpu/tune_blocks.py` (new),
`scripts/inkling/ab_step.py` (add a mode), `domain_verifier_factory/training/inkling_triton.py` (block-size table
only), `domain_verifier_factory/training/configs.py` (two defaults only), this document (append a results section).

Spec, in this order:

1. `check_real_layer.py` (at least 8 GiB free; uses under 2 GiB): build one `Quant4bitExperts` from the real
   config values, load layer 0 with `read_layer` + `set_nvfp4`; inputs: 750 random bf16 hidden rows scaled to unit
   RMS, top-8 indices from a random permutation per token, float32 weights summing to 8 per token. Compare `fused`
   with `loop`, and both with an fp32 golden path (weights dequantized to fp32 per expert, everything in fp32), for
   the output, d(hidden) and d(top_k_weights), against the bars in section 2.5. Also assert, on CPU inside the same
   script, that `SoftcappedSwiGLU` equals transformers' `Inkling2LoopSoftcappedSwiGLU` on random bf16 input
   (`torch.equal`).
2. `check_requant.py` (at least 8 GiB free; under 3 GiB): for layer 20, experts 0-15, both projections: relative
   error of (a) NF4 via bitsandbytes and (b) `quantize_nvfp4`, each against `dequantize_mxfp8` of the original.
   Gate: mean (b) at most 1.15 x mean (a). Repeat step 1's comparison on layer 20 loaded with
   `mxfp8_storage="nvfp4"`.
3. `tune_blocks.py` (at least 8 GiB free): for each (projection, transpose), time the kernel over the grid of
   section 2.4 on layer 0 with 6,000 rows over 128 experts drawn as in step 1 (5 warm-up calls, median of 20,
   synchronize timers); write the fastest combination into the table in `inkling_triton.py`; rerun
   `tests/gpu/check_kernel.py`.
4. `ab_step.py`: add a `fused` mode (same loaded model, `impl` switched on every `Quant4bitExperts`) and print its
   time and peak memory next to `normal`; when `--mxfp8-storage nvfp4` is given, load with it. Full-model checks
   (at least 21 GiB free, both long jobs finished): the A/B run; loss and LoRA-gradient comparison on the 4 fixed
   examples between `loop` and `fused`; `scripts/inkling/lora_smoke.py` once per implementation with the same seed.
5. Defaults: if every bar in section 2.5 passes and `fused` is at most 3.0 s per example, set
   `inkling_experts_impl` default to `"fused"`. If, in addition, the step-2 gate passed, set
   `inkling_mxfp8_storage` default to `"nvfp4"`. Otherwise leave the defaults and report which gate failed. Append
   the measured table (times, peak memory, every bar) to this document.

Acceptance:

- CPU, any time: the new scripts parse; `uv run pytest tests/test_inkling.py tests/test_training.py` passes after any
  default change.
- Needs GPU, run after 02:00 when at least 8 GiB is free: steps 1-3 pass their bars.
- Needs GPU with at least 21 GiB free (after the overnight training job): step 4 results recorded; time target and
  memory bar evaluated; defaults changed only as step 5 allows.

Review checklist:

- The golden path is independent fp32 (no bf16 rounding of weights) and the bars are applied to the right pairs.
- Timings use `torch.cuda.synchronize()` on both sides, exclude JIT compilation, and are not taken while another job
  shares the GPU (the 21 GiB guard enforces this for the full-model run; microbenchmarks taken alongside the
  training job are labelled as such and not used for the acceptance gate).
- Loss comparison uses the same model object, same examples and no optimizer step between the two implementations.
- Defaults were flipped only when the documented gates passed; the results section reports failures as failures.
- No `torch.profiler`; every GPU script is inside `gpu_session()` with the right `min_free_mib`.

## 6. Risks

- **Kernel speed is unmeasured.** The estimate assumes Triton emits bf16 tensor-core `tl.dot` on sm86 and that the
  nibble decode vectorizes well. T0 proves the former; T3 prints the first real timings. Contingencies are in 2.6.
- **Experts hit by few rows.** An expert with 5 rows still decodes its whole 8 M-element weight; with 128 experts
  and about 47 rows each this is the dominant cost and it does not shrink with shorter examples.
- **Triton features.** `tl.interleave`, uint8 loads and bf16 dot exist in 3.8.0 but have not been run here on
  Ampere; T0 checks them before T3 starts GPU work.
- **Re-quantizing layers 20-23** changes those weights relative to today's NF4. Adapters must be trained and
  evaluated with the same `mxfp8_storage`.
- **GPU availability.** Only small checks fit beside the overnight job; the acceptance run needs the whole card.
- **First-call compile time.** Four kernel variants compile on first use (cached under `~/.triton`); timings must
  exclude it.

## 7. Questions for the user

1. Layers 20-23: is it acceptable to store them in the NVFP4 layout instead of NF4 (gated on measured error at most
   1.15 x NF4's)? Has any Inkling adapter already been trained on the NF4 variant that must stay comparable?
2. When is the card fully free (about 21 GiB) for the full-model A/B and the 20-step comparison? Is the overnight
   15 GB job itself an Inkling run that imports `inkling.py` (T4 edits that file; the default path is unchanged, but
   a restart would pick up the new code)?
3. Is "at most 3.0 s per example, goal 2.3 s" an acceptable gate, given that Qwen parity (1.9 s) would also need
   work on the 0.98 s outside the experts?
4. The plan does not install humming into `~/inkling-venv`. Is that right, or should a forward-only humming path be
   evaluated later as a follow-up?
5. Are `tests/gpu/` (guarded check scripts) and `scripts/inkling/` (the copied harness) acceptable locations?

**Answers (2026-11-21):**
1. Yes (the user): NVFP4 storage for layers 20-23, gated on error at most 1.15x NF4's. No Inkling adapter exists yet, so there's nothing to keep comparable.
2. The card is fully free after the overnight plasticity seeds (about mid-morning 2026-11-22). That job is plasticity on Qwen2.5-0.5B, run from a separate worktree; it doesn't import inkling.py. Until then GPU tests use the ~8 GB left during plasticity, under the lock.
3. Yes (the user): a gate of at most 3.0 s per example, goal 2.3 s.
4. Right: no humming install. A forward-only humming path isn't planned.
5. Yes: `tests/gpu/` and `scripts/inkling/`.

## 8. T5 results (2026-11-22, RTX 3090, GPU otherwise idle)

**Defaults not flipped.** Speed and memory pass by a wide margin, every kernel and real-layer bar passes, and the
MXFP8 re-quantization gate passes, but the three full-model agreement bars (loss, LoRA-gradient cosine, 20-step
loss) fail. The measurements below show that these bars, as written, cannot be met by any implementation that is not
bit-identical to the loop: the full model's loss and gradients move just as much when the loop's own expert outputs
get bf16-rounding-sized noise. `inkling_experts_impl` stays `"loop"` and `inkling_mxfp8_storage` stays `"nf4"`
until the user decides how to judge full-model agreement.

| Check (section 2.5) | Measured | Bar | Result |
|---|---|---|---|
| Real layer 0, fused vs loop, bf16: out / dX / d(top_k_weights) rel | 1.86e-3 / 5.47e-3 / 1.39e-3 | 5e-3 / 1e-2 / 1e-2 | pass |
| Same, cosine | 0.9999983 / 0.9999850 / 0.9999990 | >= 0.9999 | pass |
| Layer 0 vs fp32 golden: fused / loop error (out, dX, dW) | 4.68e-3/4.72e-3, 5.07e-3/6.29e-3, 4.63e-3/4.66e-3 | fused <= 1.25 x loop + 1e-4 | pass |
| Layer 20 (MXFP8 re-quantized to NVFP4): same three checks | rel 1.91e-3 / 5.49e-3 / 1.46e-3; fused at or below loop vs golden | as above | pass |
| `SoftcappedSwiGLU` == `Inkling2LoopSoftcappedSwiGLU` (CPU, bf16) | torch.equal | exact | pass |
| Re-quantization, layer 20, experts 0-15, both projections: rel error vs MXFP8 | NF4 0.0926, NVFP4 0.0872 (ratio 0.94) | NVFP4 <= 1.15 x NF4 | pass |
| `check_kernel.py` rerun with the tuned table | all cases pass | T3 bars | pass |
| Full model, 4 examples, s/example (`ab_step.py --modes normal,fused --mxfp8-storage nvfp4`) | loop 12.98 s, fused 1.72 s (7.5x) | <= 3.0 s (goal 2.3 s) | pass |
| Peak allocated, A/B (no optimizer) | loop 20.08 GiB, fused 20.08 GiB | <= 20.5 GiB | pass |
| Peak allocated, 20 LoRA steps with AdamW (`lora_smoke.py`) | loop 20.14 GiB, fused 20.15 GiB | <= 20.5 GiB | pass |
| Full model loss, rel diff per example | 3.3e-2, 1.6e-2, 1.6e-1, 9.2e-3 | <= 1e-2 | **fail** |
| All LoRA gradients concatenated, cosine | -0.068 (per example -0.06 .. 0.34) | >= 0.99 | **fail** |
| 20 LoRA steps, seed 0, mean loss of steps 16-20 | loop 0.879, fused 0.812 (-7.6%) | within 5% | **fail** |

Other numbers: 20-step `lora_smoke.py` mean step (excluding the first) loop 13.3 s, fused 1.5 s; greedy generation
gave the same 65-token answer on both paths (loop 2.2 tok/s, fused 4.2 tok/s). The 20-step loss curves fall the same
way (step 1 2.95 vs 3.03, step 10 1.32 vs 1.37, step 20 0.53 vs 0.56). The fused peak above the layer's weights in
`check_real_layer.py` is 0.42 GiB (the loop's is 2.18 GiB because its autograd keeps every used expert's bf16 weights).

Why the full-model bars fail (scratchpad diagnostics, same model, examples, LoRA init, no optimizer step):

| Comparison | Loss rel diff (4 examples) | LoRA-gradient cosine |
|---|---|---|
| loop vs loop | 0, 0, 0, 0 | 1.000 each |
| loop with x(1 + 1e-3 N(0,1)) noise on every experts output vs loop | 2.8e-2, 1.2e-1, 1.0e-2, 2.4e-3 | 0.08, 0.38, -0.77, -0.06 |
| fused vs fused (only the order of `index_add_`'s fp32 atomics differs) | 2.0e-2, 6.0e-2, 5.1e-2, 2.7e-2 | 0.03, 0.32, 0.76, 0.25 |
| fused vs loop | 8.2e-3, 9.2e-3, 2.3e-2, 4.3e-2 | 0.02, 0.26, 0.81, 0.25 |

A relative noise of 1e-3 is a quarter of a bf16 ulp, so the loss of this model (with gradient checkpointing and LoRA
in train mode, and also in eval mode with no grad) is chaotic at the bf16 rounding level. Captured per-layer expert
inputs from a real forward give fused vs loop rel 1.7e-3 to 3.6e-3 in all 24 layers, the same as the synthetic
real-layer check. Two consequences:

- The full-model loss and gradient bars measure that chaos, not the kernels. A usable bar would compare fused vs loop
  against the loop-vs-noisy-loop spread above (fused vs loop is inside it), or compare training curves over more
  steps and seeds.
- The fused path is not bit-reproducible run to run: the per-token `index_add_` over all P pairs (torch glue, section
  2.3) uses fp32 atomics on CUDA. The kernel itself is deterministic. A deterministic scatter (e.g. a sorted
  segment sum) would make runs repeatable but would not make the loss bars pass.

Tuned block sizes (`tests/gpu/tune_blocks.py`, layer 0, 6,000 rows over 128 experts, median of 20 calls), now
`inkling_triton.TUNED_BLOCK_SIZES`. block_m stays 64, the routing plan's default, so no plan change was needed
(per-layer per-example cost 2 x forward + transposed: block_m 32 12.3 ms, 64 8.9 ms, 128 14.1 ms):

| (projection, transpose) | BLOCK_M, BLOCK_N, BLOCK_K, warps, stages | ms (start values 64, 64, 32, 4, 3) |
|---|---|---|
| gate_up, forward | 64, 128, 64, 8, 3 | 1.99 (5.38) |
| down, forward | 64, 128, 64, 8, 3 | 1.03 (1.88) |
| gate_up, transposed | 64, 128, 64, 8, 3 | 1.93 (3.79) |
| down, transposed | 64, 128, 32, 4, 3 | 0.96 (1.59) |

Not run: the A/B with `--mxfp8-storage nf4` (layers 20-23 looping), since the NVFP4 gate passed.

## 9. T5 review (2026-11-22) and the proposed acceptance change

A fresh Opus review found **no bug in the fused path**. Evidence (scratchpad rv_route_both.log, rv_drift.log):
- With routing replayed from the loop, fused vs loop still differs (gradient cosine 0.09–0.81). But fused vs fused, also with replayed routing (only the fp32 atomic order differs), gives cosine −0.49 to 0.92. So no gradient-cosine bar of 0.99 is reachable by any implementation that isn't bit-identical.
- Free routing: fused picks a different expert for 4–7% of (token, slot) pairs.
- Per-layer-call drift with forced routing follows loop + 1e-3 noise within ~10% at all 48 calls. There is no step at any layer (a missing term or a recompute mismatch would show as one).
- Gradient norms are the same order, so no gradient path is missing.

Findings:
1. (medium) `index_add_` at inkling_kernels.py:186/:242 makes the fused path nondeterministic, so the checkpoint recompute can differ from the original forward. Fix: index_copy into a (T*K, H) buffer by slot, then sum over K.
2. (low) The section 8 loop+noise gradient row is inflated (the noise isn't reseeded for the recompute).
3. (low) check_requant compares NF4 with a bf16 round-trip against NVFP4 without one (~2e-3 bias; the ratio of 0.94 is unaffected).

Proposed acceptance criterion, replacing the three full-model bars in section 2.5 (pending the user's decision):
- (a) per-layer agreement on captured real inputs in all 24 layers, with the existing bars;
- (b) full-model drift with forced routing at most 1.5× loop vs loop+1e-3 noise at every layer call;
- (c) mean |Δloss| over 16 or more examples at most 1.5× loop+noise vs loop;
- (d) 3 seeds × 100 or more steps per implementation: the fused last-20-step mean loss falls within the loop's seed spread;
- drop the gradient-cosine bar.

**Decision (the user, 2026-11-22):** adopt criteria (a)–(d) above in place of section 2.5's full-model bars, drop the gradient-cosine bar, make the fused path deterministic, then flip the defaults if (a)–(d) pass. Follow-up tickets:
- **T6 (sonnet):** replace `index_add_` in inkling_kernels.py (forward combine and backward dX) with a deterministic slot-indexed copy into a (T*K, H) buffer plus a sum over K. The CPU tests must still pass; add a test that two runs give torch.equal results.
- **T7 (opus):** run (a)–(d) on the GPU (a forced-routing harness for (b), 16+ examples for (c), 3 seeds × 100 steps per implementation for (d)), and flip the defaults (impl="fused", mxfp8_storage="nvfp4") if all pass.

**T6 done 2026-11-22.** `_combine` (index_copy_ into a (T*K, H) buffer, then a sum over K) replaces index_add_ in the forward combine and the backward dX. The Opus review found no bugs and confirmed the GPU check catches the regression: the old index_add_ version failed tests/gpu/check_determinism.py in 4 of 4 runs; the new one is bit-identical in every run and also passes under torch.use_deterministic_algorithms(True). Memory +17 MiB at T=1536. Accepted as is: the CPU two-runs test can't catch this regression (CPU index_add_ is already sequential); the GPU check is the guard. No other nondeterminism on the fused path.

## 10. T7 results (2026-11-22, RTX 3090, GPU otherwise idle)

**Update (2026-11-22): the user decided to judge (b) on calls 2-48 (max ratio 1.20 there, pass), so all of (a)-(d) pass and
the defaults are now flipped: `inkling_experts_impl="fused"`, `inkling_mxfp8_storage="nvfp4"` (configs.py, `load_inkling_4bit`,
`Quant4bitExperts`, `INKLING_EXPERTS_IMPL` fallback). Rollback: set them back or `INKLING_EXPERTS_IMPL=loop`.** The text below
is the original T7 verdict, before that decision.

**Defaults not flipped: criterion (b) fails as written, at the first layer call only.** (a), (c) and (d) pass.
`inkling_experts_impl` stays `"loop"` and `inkling_mxfp8_storage` stays `"nf4"`.

Kernel version: every run used the pre-K1 kernels (T6's deterministic `_combine`). (d) was started before K1 landed;
(a), (b) and (c) ran with a scratchpad copy of the package with the pre-K1 `inkling_kernels.py` / `inkling_triton.py`
first on `PYTHONPATH` (the log of (b)/(c) prints the kernel file paths). K1's kernels are not covered by these numbers.
All runs load with `mxfp8_storage="nvfp4"`.

| Criterion | Script | Measured | Bar | Result |
|---|---|---|---|---|
| (a) captured real inputs, all 24 layers (every experts call of one 1,002-token example), fused vs loop rel | `tests/gpu/check_captured_layers.py` | worst out 4.06e-3, dX 5.76e-3, dW 3.39e-3; min cosine 0.9999834 | 5e-3 / 1e-2 / 1e-2; cosine >= 0.9999 | pass |
| (a) same, vs fp32 golden: worst fused error / loop error | same | out 0.99, dX 0.83, dW 0.98 | <= 1.25 (+1e-4) | pass |
| (b) forced routing, eval, 8 examples: D_fused(c) / D_noise(c), per decoder-layer call | `tests/gpu/check_full_model_drift.py` | call 1: **3.60**; call 2: 1.20; call 3: 1.08; calls 4-48: 0.92-1.04 | <= 1.5 at all 48 calls | **fail** (call 1) |
| (c) free routing, train mode + checkpointing + LoRA, 32 examples: mean abs loss diff | same | fused vs loop 0.182, loop+noise vs loop 0.168 (ratio 1.08) | ratio <= 1.5 | pass |
| (d) 3 seeds x 100 LoRA steps, mean loss of steps 81-100 | `tests/gpu/check_lora_seeds.py` | loop per seed 0.0155 / 0.0168 / 0.0544 (spread [0.0155, 0.0544]); fused 0.0149 / 0.0158 / 0.0362, mean 0.0223 | fused mean inside the loop spread | pass |

Definitions fixed before running. (b): D_X(c) = mean over the 8 examples of rel(layer-c output, the loop's), with the
loop's top-k indices recorded per router call and replayed for fused and for loop+noise (routing weights still
computed live); sanity: the loop with replayed routing reproduces the loop exactly at all 48 calls (it did). (c):
mean over examples of |L_X - L_loop|. (d): M(impl, s) = mean loss of steps 81-100 of seed s; spread = [min_s, max_s]
of M(loop, s); pass iff mean_s M(fused, s) is inside. Noise everywhere: experts output x (1 + 1e-3 N(0,1)) in fp32,
rounded to bf16; the generator is reseeded from a hash of the call's bf16 input and expert indices, so the
checkpoint recompute draws the same noise (checked: 1,536 recomputed calls, all bit-identical to the original
forward). (d) setup: batch 1, AdamW lr 1e-4, no weight decay or schedule, clip 0.3, r 16 / alpha 32 / dropout 0,
examples of at most 1,536 tokens shuffled with `random.Random(seed)`, LoRA re-initialized under `torch.manual_seed(seed)`
for each implementation; same model object, `impl` switched.

Why (b) fails, and only at call 1. Call 1's drift is just layer 0's own experts difference: fused vs loop there is
the per-layer kernel rounding difference that (a) bounds (rel 1.9e-3), while the 1e-3 multiplicative noise, after
rounding back to bf16, changes the experts output by only rel ~5e-4 (a quarter of a bf16 ulp; most elements round
back to the same value). From call 2 on, both perturbations are swamped by the model's own amplification and track
each other within 0.92-1.20. So the bar as written compares a one-ulp-scale kernel difference against a
sub-ulp noise at the first call; calls 2-48 show no step anywhere (no missing term, no recompute mismatch). Whether
to judge (b) from call 2 on, or with noise matched to the per-layer difference (about 4e-3), is the user's call.
The same pattern was in the T5 review's rv_drift.log (call 1: 2.1e-3 vs 4.8e-4).

Other numbers:
- Determinism (T6): fused vs fused gave bit-identical losses in 32 of 32 train-mode examples and LoRA-gradient
  cosine 1.000000 in each.
- Corrected section 8 row (finding 2): with the noise reseeded identically for the recompute, LoRA-gradient cosine
  loop+noise vs loop per example ranges -0.44 to 0.70 (fused vs loop -0.22 to 0.72): still chaotic, so the dropped
  gradient-cosine bar stays dropped.
- (d): fused's last-20 mean is below the loop's in all 3 seeds (by 0.0006, 0.0010, 0.0182); first-10-step means
  2.10/2.39/2.16 fused vs 2.15/2.41/2.18 loop. Peak allocated 20.90 GiB (both), 20.08 GiB in (c).
- Step time in (d), examples averaging about 520 tokens: loop about 20-28 s, fused about 1.5 s.
- The local Hugging Face cache's `refs/main` for Inkling-2-Mini pointed at a snapshot with only config and
  tokenizer files (created 2026-11-22 11:08), so offline loading fails to find the weights. The cache was not
  changed; the new GPU scripts (and `check_real_layer.py`) call `tests/gpu/inkling_snapshot.pin_full_snapshot()`,
  which uses the cached snapshot that has the weights. Other scripts (`ab_step.py`, `lora_smoke.py`, training) will
  fail offline until `refs/main` points at `ed40e7da625f594fc9c4990f5f66be24873c22d7` again. Since fixed: `refs/main` was
  restored (backup `refs/main.bak-2026-11-22`), and the loader now pins `inkling.INKLING_REVISION` (env `INKLING_REVISION`
  overrides) for both `from_pretrained` and `snapshot_download`; `pin_full_snapshot` is now a no-op.

Files: `tests/gpu/check_captured_layers.py`, `tests/gpu/check_full_model_drift.py`, `tests/gpu/check_lora_seeds.py`,
`tests/gpu/inkling_snapshot.py` (new); `tests/gpu/check_real_layer.py` (`compare` split out of `check_layer`, snapshot pin).

**Decision (the user, 2026-11-22):** judge criterion (b) from layer call 2 onward. Call 1 only measures layer 0's own kernel difference, which (a) already bounds, and the 1e-3 noise mostly rounds away in bf16 there. Calls 2–48 pass (1.20, 1.08, then 0.92–1.04). With (a), (c) and (d) passing, all criteria pass, so the defaults flip to impl="fused", mxfp8_storage="nvfp4".
