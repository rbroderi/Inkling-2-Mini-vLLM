# Inkling-2-Mini on one RTX 3090: QLoRA training and vLLM serving

Tools for fine-tuning [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
with LoRA on a **single 24 GB Ampere GPU** (RTX 3090, A5000, A6000, A100), and serving the adapters with vLLM on the
same card. The official checkpoint stores its MoE experts in NVFP4/MXFP8, which Ampere cannot compute in, and loading
it through transformers dequantizes it to about 53 GB. This repo keeps the experts packed and runs them through custom
Triton kernels instead.

It came out of training claim-verification judges (40k examples per adapter, about 12 h per run on a 3090). It
has been used on that task only. Expect to check things for your own data.

## What's here

| Path | What it is |
|---|---|
| `inkling_qlora/inkling.py` | **Loader** `load_inkling_4bit()`. Experts are read one layer at a time and stay packed (NVFP4; MXFP8 layers re-quantized to NVFP4 or NF4) and frozen. Attention, router and embeddings are bf16. The vision tower is dropped and the embedding table sits in CPU memory. The checkpoint revision is pinned (`INKLING_REVISION` overrides it). |
| `inkling_qlora/inkling_kernels.py`, `inkling_triton.py`, `inkling_nvfp4.py` | **Fused experts.** Grouped NVFP4 GEMMs in Triton (tuned for SM86) with a fused softcapped SwiGLU and its backward pass, plus NVFP4/MXFP8 (de)quantization. `INKLING_EXPERTS_IMPL=loop` falls back to the reference per-expert loop. |
| `inkling_qlora/inkling_banded.py` | **Exact banded sliding-window attention:** only the 1,024-token band is computed instead of a full masked matrix. It changes only sequences longer than the window. `INKLING_BANDED_ATTENTION=0` turns it off. |
| `inkling_qlora/bucketing.py` | **Length-bucketed sampler**, so batch 2 wastes little padding. |
| `inkling_qlora/train.py` | **Trainer CLI** (TRL SFT). LoRA on the attention projections, loss on the completion only, TRL's fused labelled-positions LM head (asserted), wall-clock checkpoints, automatic resume. |
| `inkling_qlora/inkling_vllm_adapter.py`, `scripts/convert_adapter.py` | **PEFT → vLLM adapter converter.** It renames keys only (HF `self_attn.*` → checkpoint `attn.*`); no tensors change. |
| `scripts/merge_adapters_cat.py` | **Merge** several adapters into one by concatenating ranks with weights (same result as PEFT `add_weighted_adapter(..., "cat")`), so vLLM can serve the merge as a single adapter. |
| `scripts/serve.sh`, `scripts/serve_long.sh` | **vLLM serve scripts** for a 3090: 8k context with LoRA and DSpark speculative decoding (`serve.sh`), or one sequence of up to 131k tokens (`serve_long.sh`). |
| `scripts/check_draft_weights.sh` | Checks the DSpark draft weights (`mtp.safetensors`) in the HF cache. |
| `scripts/convert_layers_nvfp4.py` | Experimental: an all-NVFP4 copy of the checkpoint for vLLM (memory headroom for long context). |
| `scripts/make_random_adapter.py` | Random adapters for testing the converter and serving path. |
| `gpu_checks/` | GPU validation scripts: kernel vs reference, re-quantization, banded attention, fused head, determinism, block tuning. `gpu_guard.py` holds a GPU lock and checks free VRAM. |
| `tests/` | CPU unit tests (`pytest`). |
| `docs/` | Design notes and measurements from the original project: fused-experts plan, memory review, speed headroom, general speed. File paths in them refer to that project. |

## Setup

Everything was tested in **WSL2 Ubuntu on Windows 11** with an RTX 3090 and Python 3.12. Native Linux should work the
same way, without the WSL caveats below. It uses two separate virtual environments, because training needs the
transformers PR build and serving needs the vLLM fork.

You also need to accept the model's terms on Hugging Face and download the checkpoint, plus `mtp.safetensors` if you want DSpark.

### Training environment

```bash
python3.12 -m venv ~/inkling-venv
~/inkling-venv/bin/pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.14.1
~/inkling-venv/bin/pip install -r requirements-train.txt
~/inkling-venv/bin/pip install -e .
```

Tested versions: torch 2.14.1+cu130, Triton 3.8.0, transformers 5.24.0.dev0 (PR #51050, commit `6173a2f`),
PEFT 0.22.0, TRL 1.18.2, accelerate 1.16.0, bitsandbytes 0.50.2, datasets 5.2.0.

### Serving environment (vLLM Ampere fork)

Inkling-2 support in vLLM is [PR #69028](https://github.com/vllm-project/vllm/pull/69028), which targets Hopper and
Blackwell. The [`Hiutaky/vllm`](https://github.com/Hiutaky/vllm) fork's `ampere-port` branch adds the Ampere paths:
Triton fallbacks, emulated FP8 KV cache, DSpark draft KV in bf16, and more; see `INKLING2-RTX3090.md` in that fork.
This repo was tested with fork commit `6e9d1cc`, installed editable with precompiled wheels:

```bash
git clone -b ampere-port https://github.com/Hiutaky/vllm ~/src/vllm-ampere
cd ~/src/vllm-ampere && git checkout 6e9d1cc
python3.12 -m venv ~/vllm-venv
VLLM_USE_PRECOMPILED=1 ~/vllm-venv/bin/pip install -e .
```

## Training

Data is JSONL in TRL's conversational prompt/completion format, one example per line:

```json
{"prompt": [{"role": "system", "content": "You verify claims..."}, {"role": "user", "content": "Claim: ...\nEvidence: ..."}],
 "completion": [{"role": "assistant", "content": "{\"label\": \"supports\", ...}"}]}
```

```bash
cd /path/to/Inkling-2-Mini-vLLM
HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ~/inkling-venv/bin/python -m inkling_qlora.train \
    --train data/train.jsonl --validation data/validation.jsonl --output out/my-adapter
```

The defaults are the settings the judges were trained with:

| Setting | Default |
|---|---|
| LoRA | r 16, alpha 32, dropout 0.05, on q/k/v/o and the attention output gate |
| Learning rate | 2e-4, cosine schedule, 3% warmup |
| Optimizer | `paged_adamw_8bit` |
| Batch | 2 × gradient accumulation 8, length bucketing on |
| Sequence length | max 1,024 tokens |
| Epochs | 1 |
| Experts | fused NVFP4 |
| Attention | banded |

Speed on a 3090 at a 300 W power cap: about **17 s per optimizer step** (16 examples), so 40k examples take about
12 h. Use `--max-steps 20` for a smoke test. Run `python -m inkling_qlora.train --help` for every option.

Checkpoints are written every 250 steps and every 30 minutes. Re-running the same command resumes from the latest
`checkpoint-N` (`--no-resume` starts over).

### Training tips

- **Leave about 21 GB of VRAM free** before starting. A desktop on the same GPU can push training into shared memory and slow it badly.
- **GPU hangs.** On our 3090, long runs sometimes froze with a GPU engine hang (Windows Event ID 153), not an out-of-memory error.
  Capping the card at 300 W (`nvidia-smi -pl 300`) helped. So did a watchdog that kills and restarts the run when the
  log stops growing for 25 minutes; with automatic resume, that loses at most one checkpoint interval.
- **Don't use `torch.profiler` with CUDA tracing (CUPTI) under WSL.** It leaked Windows non-paged pool memory until the machine needed a reboot.

## Serving the adapters with vLLM

```bash
# 1. PEFT adapter -> vLLM names (rename only)
~/inkling-venv/bin/python scripts/convert_adapter.py out/my-adapter out/my-adapter-vllm

# 2. (optional) check the DSpark draft weights for speculative decoding
scripts/check_draft_weights.sh

# 3. serve (OpenAI-compatible API on :8011, model name inkling-2-mini, adapter name "judge")
scripts/serve.sh judge=out/my-adapter-vllm
```

What `serve.sh` sets up:
- Text only, bf16 KV cache, 8,192-token context, up to 8 concurrent sequences.
- CUDA graphs for decode.
- DSpark speculative decoding with 7 draft tokens, which was 1.6-1.8× faster on short judge-style replies and 1.3× on
  long ones, and works with LoRA. `DSPARK=0` turns it off.

Environment variables:

| Variable | Default | Effect |
|---|---|---|
| `MAX_LORA_RANK` | 16 | must be at least the adapter's rank (32 for a merge of two r16 adapters) |
| `MAX_LORAS` | 1 | |
| `MAX_NUM_SEQS` | 8 | |
| `KV_CACHE_BYTES` | | override the KV cache size |
| `PORT` | 8011 | |
| `INKLING_CACHE` | | HF cache directory of the model |

The script exits with code 75 if another job holds the GPU lock or less than 21 GB of VRAM is free.

Ask for the adapter by name with no reasoning:

```bash
curl localhost:8011/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "judge",
  "messages": [{"role": "user", "content": "..."}],
  "temperature": 0,
  "chat_template_kwargs": {"reasoning_effort": "none"}
}'
```

The adapters were trained with `reasoning_effort: "none"`; serve them the same way. In our tests, the default
reasoning effort was slightly less accurate and about 4× slower for classification-style answers.

**Long context:** `scripts/serve_long.sh` serves the stock model for one sequence of up to 131,072 tokens. It uses an
FP8 KV cache (emulated on Ampere) and a smaller batched-token budget. The header comments explain how the memory is worked out.

**Merging adapters:**

```bash
python scripts/merge_adapters_cat.py out/merged out/a out/b --weights 0.5 0.5
```

Then convert `out/merged` and serve it with `MAX_LORA_RANK=32`.

### Serving caveats (WSL)

- `--cpu-offload-*` fails under WSL: the fork turns off pinned memory there, and CUDA graphs need it.
- Unset `PYTORCH_CUDA_ALLOC_CONF` for vLLM; the serve scripts do this. With `expandable_segments` the MoE repack failed with
  "Failed to create GPU mapping".
- Measured on a 3090: about 93 tok/s decode without DSpark and 17.6 GiB of weights, leaving about 2.5-2.75 GiB for the KV cache at 8k context.

## Tests

```bash
pip install pytest && pytest          # CPU unit tests; the kernels run through their torch reference paths
```

The GPU checks run in the training environment, for example:

```bash
PYTHONPATH=. ~/inkling-venv/bin/python gpu_checks/check_kernel.py
```

Some checks read sample data from `INKLING_CHECK_DATA` (a JSONL in the training format).

## Credits

- Model: Thinking Machines, Inkling-2-Mini.
- Inkling-2 support in transformers: huggingface/transformers PR #51050.
- Inkling-2 support in vLLM: vllm-project/vllm PR #69028.
- Ampere port of vLLM: Hiutaky/vllm, branch `ampere-port`.

## License

The code in this repository is licensed under the GNU General Public License v3.0 or later; see [LICENSE](LICENSE).
The Inkling-2-Mini model weights, transformers and vLLM are covered by their own licenses.
