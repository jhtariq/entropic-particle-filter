#!/bin/bash
# One node's share of the random-weight EPF sweep.
#   bash mmar_epf/run_node.sh <3b|7b> <shard_i> <shard_n> [ngpu] [limit]
# Serves one vLLM replica per local GPU, runs the probe over records[i::N], tears down.
# Resumable: re-running the identical command skips rows already in this shard's JSONL,
# so a shard begun on one node can be finished on another (/u is shared).
set -uo pipefail

MODEL="${1:?want 3b|7b|q2a|phi4mm}"; SHARD_I="${2:?want shard index}"; SHARD_N="${3:?want shard count}"
NGPU="${4:-4}"; LIMIT_OVERRIDE="${5:-}"

# cu130 wheels vs Delta's CUDA-12.8 driver — without this every engine dies with
# "NVIDIA driver is too old (found version 12080)".
module load cuda-compat/13.0 2>/dev/null || true

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"
cd "$REPO_ROOT"
[ -n "$LIMIT_OVERRIDE" ] && LIMIT="$LIMIT_OVERRIDE"

model_cfg "$MODEL" || exit 1
model_post
export TMPDIR="/tmp/$USER/${SLURM_JOB_ID:-0}"; mkdir -p "$TMPDIR"
TAG="${TAG_PREFIX:-}${M_NAME}__s${SHARD_I}of${SHARD_N}"
mkdir -p "$OUT_ROOT/samples" "$OUT_ROOT/logs" "$OUT_ROOT/servers"

echo "=== $(date '+%F %T') [$BENCH] $TAG on $(hostname) — $NGPU GPU(s) ==="

SNAP="$HF_HOME/hub/models--${M_ID//\//--}/snapshots/$M_REV"
[ -d "$SNAP" ] && [ -n "$(ls -A "$SNAP" 2>/dev/null)" ] || { echo "FATAL: checkpoint missing -> $SNAP" >&2; exit 1; }
echo "checkpoint OK: $SNAP"

n=$("$EPF_PY" -c "
import importlib,sys
m=importlib.import_module('$B_MODULE'.rsplit('.',1)[0]+'.loader')
# pick the MCQ loader explicitly: 'load_metadata' sorts first and has a different
# signature, and it silently returned a plausible count instead of failing loudly.
fn=[getattr(m,x) for x in dir(m) if x.startswith('load_') and x.endswith('_mcq')][0]
ar='$AUDIO_ROOT' or None
print(len(fn('$DATA_ROOT', subset='$SUBSET', audio_root=ar)))" 2>&1 | tail -1)
[ "$n" = "$B_EXPECT_N" ] || { echo "FATAL: $BENCH loaded $n records, expected $B_EXPECT_N — wrong data root or missing audio" >&2; exit 1; }
echo "records OK: $n"

LORA=()
if [ "${M_SERVE:-vllm}" = "vllm_lora" ]; then
  [ -d "$SPEECH_LORA_DIR" ] || { echo "FATAL: speech-LoRA missing -> $SPEECH_LORA_DIR" >&2; exit 1; }
  LORA=(--enable-lora --max-lora-rank "$SPEECH_LORA_RANK" --max-loras 1
        --lora-modules "${M_REQNAME}=${SPEECH_LORA_DIR}")
  echo "LoRA: serving $M_NAME + adapter '$M_REQNAME' from $SPEECH_LORA_DIR"
fi

pids=()
for g in $(seq 0 $((NGPU - 1))); do
  port=$((BASE_PORT + g)); log="$OUT_ROOT/servers/${M_NAME}_$(hostname -s)_gpu${g}.log"
  CUDA_VISIBLE_DEVICES=$g HF_HOME="$HF_HOME" HF_HUB_OFFLINE=1 \
  VLLM_USE_FLASHINFER_SAMPLER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  VLLM_CACHE_ROOT="$TMPDIR/vllm_$g" TRITON_CACHE_DIR="$TMPDIR/triton_$g" \
  TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor_$g" \
  nohup setsid "$EPF_VLLM" serve "$M_ID" --revision "$M_REV" \
    --served-model-name "$M_NAME" --port "$port" --trust-remote-code --dtype bfloat16 \
    --max-model-len "$MAXLEN" --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --allowed-local-media-path "$MEDIA_ROOT" --limit-mm-per-prompt '{"audio":3}' \
    --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens 8192 \
    "${LORA[@]}" > "$log" 2>&1 &
  pids+=($!); echo "$!" > "$OUT_ROOT/servers/${M_NAME}_$(hostname -s)_gpu${g}.pid"
  echo "  serving $M_NAME on :$port (GPU $g) -> $log"
done

# Kill by RECORDED pid — `pkill -f "vllm serve"` also matches this shell.
cleanup () { for p in "${pids[@]}"; do kill -- -"$p" 2>/dev/null || kill "$p" 2>/dev/null; done; sleep 8; }
trap cleanup EXIT

EPS=""
for g in $(seq 0 $((NGPU - 1))); do
  port=$((BASE_PORT + g)); t=0
  while [ "$t" -lt "$SERVE_TIMEOUT" ]; do
    curl -sf "http://localhost:$port/v1/models" >/dev/null 2>&1 && break
    if ! kill -0 "${pids[$g]}" 2>/dev/null; then
      echo "FATAL: engine on :$port died — see $OUT_ROOT/servers/${M_NAME}_$(hostname -s)_gpu${g}.log" >&2
      tail -25 "$OUT_ROOT/servers/${M_NAME}_$(hostname -s)_gpu${g}.log" >&2; exit 1
    fi
    sleep 5; t=$((t + 5))
  done
  [ "$t" -lt "$SERVE_TIMEOUT" ] || { echo "FATAL: :$port not healthy after ${SERVE_TIMEOUT}s" >&2; exit 1; }
  echo "  :$port healthy after ${t}s"
  EPS="${EPS:+$EPS,}http://localhost:$port/v1"
done

AUDIOARG=(); [ -n "${AUDIO_ROOT:-}" ] && AUDIOARG=(--audio-root "$AUDIO_ROOT")
STOREARG=()
[ "${STORE_TEXT:-0}" = 1 ] && STOREARG+=(--store-text)
[ "${STORE_STEPS:-0}" = 1 ] && STOREARG+=(--store-steps)
[ "${SAVE_TRACES:-0}" = 1 ] && STOREARG+=(--save-traces)
echo "--- probe: [$BENCH] $TAG budgets=$BUDGETS signals=$SIGNALS prompts=$PROMPTS limit=$LIMIT ---"
"$EPF_PY" -m "$B_MODULE" \
  --endpoints "$EPS" --model-name "$M_REQNAME" \
  --data-root "$DATA_ROOT" --subset "$SUBSET" \
  --select "$SELECT" --limit "$LIMIT" --shard "${SHARD_I}/${SHARD_N}" --seed "$SEED" \
  --min-choices "$MIN_CHOICES" "${AUDIOARG[@]}" \
  --prompts "$PROMPTS" --signals "$SIGNALS" --budgets "$BUDGETS" \
  --temp "$TEMP" --max-steps "$MAX_STEPS" --max-tokens-per-step "$MAX_TOKENS_PER_STEP" \
  --max-inflight "$MAX_INFLIGHT" \
  "${STOREARG[@]}" \
  --jsonl "$OUT_ROOT/samples/${TAG}.jsonl" \
  --csv   "$OUT_ROOT/samples/${TAG}.csv" \
  --log   "$OUT_ROOT/logs/${TAG}.report.txt" 2>&1 | tee -a "$OUT_ROOT/logs/${TAG}.log"
rc=${PIPESTATUS[0]}
echo "=== $(date '+%F %T') $TAG finished rc=$rc ==="
exit $rc
