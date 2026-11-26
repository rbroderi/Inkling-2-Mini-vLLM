#!/usr/bin/env bash
# Check the DSpark draft weights (mtp.safetensors) in the snapshot serve.sh uses: present, 635,801,416 bytes, and
# sha256 equal to the blob name. Exit 0 if good; exit 2 (printing the one approved download command, NOT running it) if not.
set -uo pipefail
WANT_SIZE=635801416
WANT_SHA=7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2
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
f="${M}mtp.safetensors"
fail() {
  echo "FAIL: $1" >&2
  echo "Approved download (not run; ask the user first, 606 MiB from thinkingmachines/Inkling-2-Mini):" >&2
  echo "  HF_HUB_OFFLINE=0 ~/vllm-venv/bin/huggingface-cli download thinkingmachines/Inkling-2-Mini mtp.safetensors" >&2
  echo "  (huggingface-cli may be absent from ~/vllm-venv; its binary there is 'hf': ~/vllm-venv/bin/hf download ...)" >&2
  exit 2
}
[ -n "$M" ] || fail "no snapshot with the model weights"
[ -e "$f" ] || fail "$f is missing"
size=$(stat -L -c %s "$f")
[ "$size" = "$WANT_SIZE" ] || fail "size $size, expected $WANT_SIZE"
sha=$(sha256sum -- "$f" | cut -d' ' -f1)
[ "$sha" = "$WANT_SHA" ] || fail "sha256 $sha, expected $WANT_SHA"
echo "OK: $f $size bytes sha256 $sha"
