#!/usr/bin/env bash
# Probe how long a context stock Inkling-2-Mini serves on this GPU with serve_long.sh. NOT run yet: needs the GPU free
# (the 12 h training run must be finished). Usage: scripts/long_context_probe.sh   (inside WSL)
# For DSPARK in 1 (on) then 0 (off), for MAX_MODEL_LEN in 65536 131072 196608 262144 300000: start serve_long.sh; if it
# refuses because the KV pool is too small for one sequence of that length, step KV_CACHE_BYTES up by STEP_MIB (default
# 256) at most MAX_STEPS (6) times from the script's computed default; any other start failure (OOM) ends that mode.
# When the server is up it records: peak nvidia-smi memory, prefill seconds of a synthetic prompt of about 90% of the
# length, decode tok/s for 300 tokens at that context, and a needle-in-haystack check (a passcode hidden mid-prompt).
# The first failure of any kind ends that mode; the summary gives the largest passing length for each mode.
# Output: $OUT (default ~/long_probe_<timestamp>/): per-run server logs, results.tsv, summary.txt.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-$HOME/long_probe_$(date +%Y%m%d_%H%M%S)}"; mkdir -p "$OUT"
PORT="${PORT:-8011}"; STEP_MIB="${STEP_MIB:-256}"; MAX_STEPS="${MAX_STEPS:-6}"
LENGTHS="${LENGTHS:-65536 131072 196608 262144 300000}"
PY="$HOME/vllm-venv/bin/python"
TAB=$'\t'
printf 'dspark\tlen\tkv_bytes\tstarted\tpeak_mib\tprefill_s\tprompt_tokens\tdecode_tok_s\tneedle\tnote\n' > "$OUT/results.tsv"

SPID=""; MPID=""
stop_server() {
  [ -n "$SPID" ] && kill -TERM -- "-$SPID" 2>/dev/null
  if [ -n "$SPID" ]; then
    for _ in $(seq 60); do kill -0 "$SPID" 2>/dev/null || break; sleep 1; done
    kill -KILL -- "-$SPID" 2>/dev/null
  fi
  SPID=""
  [ -n "$MPID" ] && kill "$MPID" 2>/dev/null
  MPID=""
  sleep 5
}
trap stop_server EXIT

declare -A best
for D in 1 0; do
  best[$D]=0
  failed=0
  for L in $LENGTHS; do
    [ "$failed" = 1 ] && break
    kv=""; step=0; ok=0; cur=""
    while [ "$step" -le "$MAX_STEPS" ]; do
      log="$OUT/server_d${D}_L${L}_s${step}.log"
      if [ -z "$kv" ]; then
        env DSPARK=$D MAX_MODEL_LEN=$L PORT=$PORT setsid "$HERE/serve_long.sh" > "$log" 2>&1 & SPID=$!
      else
        env DSPARK=$D MAX_MODEL_LEN=$L PORT=$PORT KV_CACHE_BYTES=$kv setsid "$HERE/serve_long.sh" > "$log" 2>&1 & SPID=$!
      fi
      : > "$OUT/mem.txt"
      ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 >> "$OUT/mem.txt"; sleep 1; done ) & MPID=$!
      up=0
      for _ in $(seq 900); do   # up to 15 minutes to load, compile and capture graphs
        curl -sf "http://localhost:$PORT/v1/models" > /dev/null && { up=1; break; }
        kill -0 "$SPID" 2>/dev/null || break
        sleep 1
      done
      # the KV size actually used: serve_long.sh prints it on its first line (the stderr is in the log)
      cur=$(sed -n 's/.*kv_cache=\([0-9]*\) B.*/\1/p' "$log" | head -1)
      if [ "$up" = 1 ]; then ok=1; break; fi
      too_small=0
      grep -qiE "larger than the available KV cache|maximum model length|KV cache is needed|increase .*kv" "$log" && too_small=1
      stop_server
      if [ "$too_small" = 1 ]; then
        kv=$(( ${cur:-0} + STEP_MIB * 1048576 )); step=$((step + 1)); continue
      fi
      printf '%s\t%s\t%s\tno\t-\t-\t-\t-\t-\tstart failed, not a KV-size error (see %s)\n' "$D" "$L" "${cur:-?}" "$log" >> "$OUT/results.tsv"
      failed=1; break
    done
    if [ "$ok" != 1 ]; then
      [ "$failed" = 1 ] || printf '%s\t%s\t%s\tno\t-\t-\t-\t-\t-\tKV pool still too small after %s steps\n' "$D" "$L" "${cur:-?}" "$MAX_STEPS" >> "$OUT/results.tsv"
      failed=1; break
    fi
    # client prints: prefill_s TAB prompt_tokens TAB decode_tok_s TAB needle(pass|fail)
    res=$("$PY" "$HERE/long_context_client.py" --port "$PORT" --length "$L" --decode-tokens 300 2> "$OUT/client_d${D}_L${L}.err") || res=""
    peak=$(sort -n "$OUT/mem.txt" | tail -1)
    kill -0 "$SPID" 2>/dev/null || res=""   # the server died during the requests (OOM in prefill)
    if [ -z "$res" ]; then
      printf '%s\t%s\t%s\tyes\t%s\t-\t-\t-\t-\trequests failed or server died (see client .err and server log)\n' "$D" "$L" "$cur" "$peak" >> "$OUT/results.tsv"
      failed=1
    else
      IFS="$TAB" read -r pf pt dt nd <<< "$res"
      printf '%s\t%s\t%s\tyes\t%s\t%s\t%s\t%s\t%s\t\n' "$D" "$L" "$cur" "$peak" "$pf" "$pt" "$dt" "$nd" >> "$OUT/results.tsv"
      if [ "$nd" = pass ]; then best[$D]=$L; else failed=1; fi
    fi
    stop_server
  done
  stop_server
done
{
  echo "max passing MAX_MODEL_LEN with DSpark ON:  ${best[1]}"
  echo "max passing MAX_MODEL_LEN with DSpark OFF: ${best[0]}"
  echo
  column -t -s "$TAB" "$OUT/results.tsv"
} | tee "$OUT/summary.txt"
