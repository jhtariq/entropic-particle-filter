#!/bin/bash
# One node's shard of the Mellow EPF sweep (self-certainty signal).
#   BENCH=mmar|mmau bash mmar_epf/run_mellow_node.sh <shard_i> <shard_n> [ngpu=4] [shims_per_gpu=6]
#
# Mellow is not servable by vLLM (run_node.sh is vLLM-specific end-to-end), so this
# driver serves ngpu x shims_per_gpu run17 shim processes instead. Each shim decodes
# ONE request at a time, so parallelism = process count; the model is 167M fp32
# (~1.6 GB VRAM), so 6 replicas per A40 is comfortable. Resumable: re-running the
# identical command skips (item,prompt,signal,budget) cells already in the shard JSONL.
set -uo pipefail

SHARD_I="${1:?want shard index}"; SHARD_N="${2:?want shard count}"
NGPU="${3:-4}"; PER_GPU="${4:-6}"

# cu130 wheels vs Delta's CUDA-12.8 driver (idempotent; harmless for the fp32 shim
# but keeps the environment identical to every other GPU launcher in this campaign)
module load cuda-compat/13.0 2>/dev/null || true

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
cd "$REPO_ROOT"

EPF_PY=/u/awaheed/envs/epf/bin/python
export HF_HOME=/u/awaheed/epf_data/hf_cache HF_HUB_OFFLINE=1
BENCH="${BENCH:-mmar}"
case "$BENCH" in
  mmar) DATA_ROOT=/u/awaheed/epf_data/mmar; OUT_ROOT=/u/awaheed/mmar_epf_out
        PROBE=benchmarking.mmar.diversity_probe
        LOADER="from benchmarking.mmar.loader import load_mmar_mcq as load" ;;
  mmau) DATA_ROOT=/u/awaheed/epf_data/mmau; OUT_ROOT=/u/awaheed/mmau_epf_out
        PROBE=benchmarking.mmau.diversity_probe
        LOADER="from benchmarking.mmau.loader import load_mmau_mcq as load" ;;
  *) echo "unknown BENCH='$BENCH' (want mmar|mmau)" >&2; exit 1 ;;
esac
MEDIA_ROOT=/u/awaheed/epf_data
CODE_DIR=/u/awaheed/epf_data/mellow_code
MELLOW_REV=83672db0dae28764e283210d5bb732621e903d8a
SMOLLM2_REV=93efa2f097d58c2a74874c7e644dbc9b0cee75a2
CODE_COMMIT=349f9b2be84bec713ac71e54fcbcac9bf4d116e5
BASE_PORT="${BASE_PORT:-8100}"
SERVE_TIMEOUT="${SERVE_TIMEOUT:-900}"

# probe knobs (run17 precedent: native prompts, stop-regex ends a trajectory at the
# first native answer -> effectively best-of-N + self-certainty selection)
PROMPTS="${PROMPTS:-10,11}"
SIGNALS="${SIGNALS:-mean_logprob}"
BUDGETS="${BUDGETS:-1,8,16,32,64,128}"
TEMP="${TEMP:-0.8}"
STOP_REGEX="${STOP_REGEX:-[a-k]\\)}"
MAX_INFLIGHT="${MAX_INFLIGHT:-8}"
SEED="${SEED:-1234}"
LIMIT="${LIMIT:-1000}"

TAG="mellow__s${SHARD_I}of${SHARD_N}"
mkdir -p "$OUT_ROOT/samples" "$OUT_ROOT/logs" "$OUT_ROOT/servers"
echo "=== $(date '+%F %T') [$BENCH] $TAG on $(hostname) — ${NGPU}x${PER_GPU} shims ==="

got=$(git -C "$CODE_DIR" rev-parse HEAD 2>/dev/null || echo none)
[ "$got" = "$CODE_COMMIT" ] || { echo "FATAL: mellow code clone missing/wrong -> $CODE_DIR @ $got" >&2; exit 1; }
for snap in "models--soham97--mellow/snapshots/$MELLOW_REV" \
            "models--HuggingFaceTB--SmolLM2-135M/snapshots/$SMOLLM2_REV"; do
  [ -d "$HF_HOME/hub/$snap" ] && [ -n "$(ls -A "$HF_HOME/hub/$snap" 2>/dev/null)" ] \
    || { echo "FATAL: snapshot missing -> $HF_HOME/hub/$snap" >&2; exit 1; }
done
n=$("$EPF_PY" -c "
$LOADER
print(len(load('$DATA_ROOT', subset='full')))" | tail -1)
[ "$n" = "1000" ] || { echo "FATAL: $BENCH loaded $n records, expected 1000" >&2; exit 1; }
echo "preflight OK: code@${got:0:8}, snapshots present, $n records"

PIDFILES=()
EPS=""
for g in $(seq 0 $((NGPU - 1))); do
  for k in $(seq 0 $((PER_GPU - 1))); do
    port=$((BASE_PORT + g * PER_GPU + k))
    log="$OUT_ROOT/servers/mellow_$(hostname -s)_g${g}k${k}.log"
    pidf="$OUT_ROOT/servers/mellow_$(hostname -s)_g${g}k${k}.pid"
    rm -f "$pidf"
    CUDA_VISIBLE_DEVICES=$g HF_HOME="$HF_HOME" HF_HUB_OFFLINE=1 \
    nohup setsid "$EPF_PY" -m benchmarking.mmau_pro.serve_mellow \
      --port "$port" --model-name mellow --variant v0 \
      --revision "$MELLOW_REV" --smollm2-revision "$SMOLLM2_REV" \
      --code-dir "$CODE_DIR" --max-text-tokens 0 --single-audio-fill silence \
      --allowed-media-root "$MEDIA_ROOT" --waveform-cache 1100 \
      --pid-file "$pidf" > "$log" 2>&1 &
    PIDFILES+=("$pidf")
    EPS="${EPS:+$EPS,}http://localhost:$port/v1"
    echo "  launching shim :$port (GPU $g replica $k) -> $log"
    sleep 2   # stagger 670 MB ckpt loads
  done
done

# Kill by RECORDED pid (the shim writes its own; setsid-safe) — never pkill by pattern.
cleanup() {
  local f pid
  for f in "${PIDFILES[@]}"; do
    pid=$(cat "$f" 2>/dev/null) || continue
    [ -n "$pid" ] && { kill -- -"$pid" 2>/dev/null || kill "$pid" 2>/dev/null; }
    rm -f "$f"
  done
  sleep 5
}
trap cleanup EXIT

for g in $(seq 0 $((NGPU - 1))); do
  for k in $(seq 0 $((PER_GPU - 1))); do
    port=$((BASE_PORT + g * PER_GPU + k)); t=0
    while [ "$t" -lt "$SERVE_TIMEOUT" ]; do
      curl -sf "http://localhost:$port/v1/models" >/dev/null 2>&1 && break
      sleep 5; t=$((t + 5))
    done
    if [ "$t" -ge "$SERVE_TIMEOUT" ]; then
      echo "FATAL: shim :$port not healthy after ${SERVE_TIMEOUT}s" >&2
      tail -20 "$OUT_ROOT/servers/mellow_$(hostname -s)_g${g}k${k}.log" >&2
      exit 1
    fi
    echo "  :$port healthy after ${t}s"
  done
done

echo "--- probe: [$BENCH] $TAG prompts=$PROMPTS signals=$SIGNALS budgets=$BUDGETS ---"
"$EPF_PY" -m "$PROBE" \
  --endpoints "$EPS" --model-name mellow \
  --data-root "$DATA_ROOT" --subset full \
  --select all --limit "$LIMIT" --shard "${SHARD_I}/${SHARD_N}" --seed "$SEED" \
  --prompts "$PROMPTS" --signals "$SIGNALS" --budgets "$BUDGETS" \
  --temp "$TEMP" --max-steps 6 --max-tokens-per-step 300 \
  --stop-regex "$STOP_REGEX" --max-inflight "$MAX_INFLIGHT" --store-text \
  --jsonl "$OUT_ROOT/samples/${TAG}.jsonl" \
  --csv   "$OUT_ROOT/samples/${TAG}.csv" \
  --log   "$OUT_ROOT/logs/${TAG}.report.txt" 2>&1 | tee -a "$OUT_ROOT/logs/${TAG}.log"
rc=${PIPESTATUS[0]}
echo "=== $(date '+%F %T') $TAG finished rc=$rc ==="
exit $rc
