#!/usr/bin/env bash
# Serve STOCK Inkling-2-Mini with the vLLM Ampere fork (WSL) for ONE long sequence (default 131072 tokens). Based on
# serve.sh (same GPU lock, snapshot pick, unset); see docs/inkling_memory_review.md (items 1-3).
#   --max-num-batched-tokens 256   (item 1) shrinks the in-flight reservation of the sliding-window / conv-state caches
#   --kv-cache-dtype fp8           (item 2) halves the attention caches (quality at length unmeasured; emulated on Ampere)
#   MAX_NUM_SEQS default 1; no LoRA unless adapters are passed; DSpark ON by default, DSPARK=0 turns it off (frees about
#   0.73 GiB of draft weights, 0.2-0.3 GiB of graphs and the draft cache, at the price of 1.3x slower long decodes).
# Env: MAX_MODEL_LEN (131072), KV_CACHE_BYTES (computed below), MAX_NUM_SEQS (1), DSPARK (1), PORT (8011), MAX_LORAS.
# Usage: scripts/serve_long.sh [NAME=ADAPTER_DIR ...]
#
# KV_CACHE_BYTES default, one sequence of L = MAX_MODEL_LEN tokens (review sections 2.2 and 3):
#   F = in-flight tokens = 2 x 256 = 512 (async scheduling)
#   fixed     = (1023 + F) x 40 x 2048  (40 sliding caches, FP8)  +  (3 + F) x 48 x 16384 (48 conv-state caches, bf16)
#             = 125,747,200 + 405,012,480 = 530,759,680 B = 0.494 GiB
#   per token = 8 full-attention caches x 2048 B (FP8) = 16,384 B = 16 KiB   (1.53 GiB per 100k tokens)
#   DSpark adds, bf16 only: about 0.36 GiB fixed (measured: 2.64 - 2.25 GiB for 8k at chunk 1024, minus 8k x 4 KiB)
#   and 4,096 B per token (1 full draft layer).
#   default = (fixed + L x per_token [+ DSpark parts]) x 1.05 (page rounding, estimate error), in whole MiB.
#   L=131072: 2.68 GB without DSpark -> about 2.62 GiB; with DSpark about 3.5 GiB. The pool is the only thing the
#   flag sets: DSpark's 0.73 GiB of draft weights come out of the rest of the 24 GiB, not out of this number.
# Not done here (user will do later): embedding offload (--cpu-offload-gb 1 --cpu-offload-params embed_tokens), display
# on the AMD graphics, and VLLM_WSL2_ENABLE_PIN_MEMORY=1 (future option: the fork disables pinned memory on WSL; this
# switch is what the embedding offload needs, review item 6). serve.sh's note: --cpu-offload failed on WSL 2026-11-21.
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
# INKLING_MODEL_DIR (e.g. a converted checkpoint) overrides the snapshot pick.
[ -n "${INKLING_MODEL_DIR:-}" ] && M="$INKLING_MODEL_DIR"
L="${MAX_MODEL_LEN:-131072}"
CHUNK=256
F=$((2 * CHUNK))
fixed=$(( (1023 + F) * 40 * 2048 + (3 + F) * 48 * 16384 ))
per_token=$(( 8 * 2048 ))
spec=()
if [ "${DSPARK:-1}" = 1 ]; then
  spec=(--speculative-config.method=dspark --speculative-config.num_speculative_tokens=7)
  fixed=$(( fixed + 386547057 ))      # 0.36 GiB
  per_token=$(( per_token + 4096 ))
fi
kv_default=$(( (fixed + L * per_token) * 105 / 100 / 1048576 * 1048576 ))
lora=()
[ "$#" -gt 0 ] && lora=(--enable-lora --max-lora-rank 16 --max-loras "${MAX_LORAS:-1}" --max-cpu-loras 8 --lora-modules "$@")
echo "serve_long: len=$L kv_cache=${KV_CACHE_BYTES:-$kv_default} B dspark=${DSPARK:-1} seqs=${MAX_NUM_SEQS:-1}" >&2
HF_HUB_OFFLINE=1 exec ~/vllm-venv/bin/vllm serve "$M" --served-model-name inkling-2-mini --dtype bfloat16 \
  --max-model-len "$L" --max-num-seqs "${MAX_NUM_SEQS:-1}" --max-num-batched-tokens "$CHUNK" \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.94 --kv-cache-memory="${KV_CACHE_BYTES:-$kv_default}" \
  --limit-mm-per-prompt.image=0 \
  --compilation-config.cudagraph_mode=FULL_DECODE_ONLY \
  --port "${PORT:-8011}" "${spec[@]}" "${lora[@]}"
