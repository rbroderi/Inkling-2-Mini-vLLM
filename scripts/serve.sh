#!/usr/bin/env bash
# Serve Inkling-2-Mini with the vLLM Ampere fork (WSL): text only, bf16 KV cache, LoRA on.
# Speculative decoding (DSpark, 7 draft tokens; needs mtp.safetensors, see check_draft_weights.sh) is ON by default since
# 2026-11-23 (D1: 1.6-1.8x faster judge-style generation, 1.3x on long outputs, works with LoRA). DSPARK=0 turns it off.
# KV_CACHE_BYTES overrides the KV cache size (default 2.75 GiB with DSpark, which its draft layers need for one 8k
# sequence; 2.5 GiB without).
# Usage: scripts/serve.sh [NAME=ADAPTER_DIR ...]   (converted adapters, see convert_adapter.py)
# MAX_LORA_RANK (16) must be at least the adapter rank: 32 for a merge of two r16 adapters (merge_adapters_cat.py).
# No --cpu-offload: on WSL it needs pinned memory for CUDA graphs, which WSL lacks (failed 2026-11-21).
set -euo pipefail
# The vLLM fork must not inherit PYTORCH_CUDA_ALLOC_CONF=expandable_segments (set by the HF parity scripts): on WSL
# its cuMemMap path fails in the MoE repack with "device not ready / Failed to create GPU mapping" (2026-11-22).
unset PYTORCH_CUDA_ALLOC_CONF
exec 9>/tmp/inkling_gpu.lock
flock -n 9 || { echo "GPU lock /tmp/inkling_gpu.lock is held; exiting 75" >&2; exit 75; }
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0) || exit 75
[ "$free" -ge 21000 ] || { echo "only $free MiB free on GPU 0, need 21000; exiting 75" >&2; exit 75; }
# The snapshot with the weights: the one refs/main points to if it has them, else the newest by mtime (the cache can
# also hold a weightless snapshot). Plain loops, no pipeline, so set -e / pipefail cannot kill the script here.
C="${INKLING_CACHE:-$HOME/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini}"
M=""
if [ -r "$C/refs/main" ]; then
  ref=$(<"$C/refs/main")
  [ -e "$C/snapshots/$ref/model-00001-of-00001.safetensors" ] && M="$C/snapshots/$ref/"
fi
if [ -z "$M" ]; then
  newest=0
  for s in "$C"/snapshots/*/; do
    [ -e "${s}model-00001-of-00001.safetensors" ] || continue
    t=$(stat -L -c %Y "$s")
    if [ "$t" -gt "$newest" ]; then newest=$t; M=$s; fi
  done
fi
[ -n "$M" ] || { echo "no snapshot with model-00001-of-00001.safetensors in the Inkling-2-Mini cache" >&2; exit 2; }
spec=()
kv_default=2684354560
if [ "${DSPARK:-1}" = 1 ]; then
  spec=(--speculative-config.method=dspark --speculative-config.num_speculative_tokens=7)
  kv_default=2952790016
fi
lora=()
[ "$#" -gt 0 ] && lora=(--lora-modules "$@")
HF_HUB_OFFLINE=1 exec ~/vllm-venv/bin/vllm serve "$M" --served-model-name inkling-2-mini --dtype bfloat16 \
  --max-model-len 8192 --max-num-seqs "${MAX_NUM_SEQS:-8}" --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.94 --kv-cache-memory="${KV_CACHE_BYTES:-$kv_default}" \
  --limit-mm-per-prompt.image=0 \
  --compilation-config.cudagraph_mode=FULL_DECODE_ONLY \
  --enable-lora --max-lora-rank "${MAX_LORA_RANK:-16}" --max-loras "${MAX_LORAS:-1}" --max-cpu-loras 8 \
  --port "${PORT:-8011}" "${spec[@]}" "${lora[@]}"
