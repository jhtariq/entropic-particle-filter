#!/bin/bash
# The 8 MMAR cells: {qwen2_audio, phi4mm} x {greedy_nocot, greedy_p4, sc_nocot, sc_p4}.
#
# One vLLM replica per GPU; the driver round-robins ITEMS across all replicas (rather
# than putting one model per GPU, which would need both models resident at once).
# Models are done sequentially: serve -> gate -> run its 4 arms -> kill by PID -> next.
#
# Every arm is resumable: sample_runner keys resume on unique_id and only treats an item
# as done when all N of its samples are present and error-free, so re-running the exact
# same command after a timeout/preemption is the whole recovery procedure.
set -uo pipefail

MMAR_SC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$MMAR_SC_DIR/config.sh"
source "$MMAR_SC_DIR/lib.sh"
cd "$REPO_ROOT"

mkdir -p "$OUT_ROOT/servers" "$OUT_ROOT/samples" "$OUT_ROOT/logs"

# Single-writer lock. We deliberately queue the same run on several partitions to beat the
# scheduler; if two of them ever start together they would interleave appends to the SAME
# JSONL (rows exceed PIPE_BUF, so concurrent O_APPEND is not atomic) and corrupt lines.
# Whoever gets the lock runs; the loser exits 0 immediately so its allocation is released.
exec 9> "$OUT_ROOT/.run.lock"
if ! flock -n 9; then
  echo "another mmar_sc run holds $OUT_ROOT/.run.lock — exiting so it can proceed"
  exit 0
fi
echo "acquired run lock (job ${SLURM_JOB_ID:-local} on $(hostname))"
EPS="$(endpoints_csv)"
echo "GPUs=$NUM_GPUS endpoints=$EPS out=$OUT_ROOT"
echo "data=$DATA_MMAR subset=$SUBSET"
echo

# hard preflight: load_mmar_mcq silently DROPS items whose wav is missing, which would
# shrink the eval set with no error at all. Assert the pinned counts before any GPU work.
"$EPF_PY" - "$DATA_MMAR" "$SUBSET" <<'PYEOF' || exit 1
import sys
from benchmarking.mmar.loader import load_mmar_mcq
root, subset = sys.argv[1], sys.argv[2]
recs = load_mmar_mcq(root, subset=subset)
gradeable = sum(1 for r in recs if r.answer_index is not None)
want = {"full": (1000, 996), "le30s": (983, 979)}[subset]
got = (len(recs), gradeable)
print(f"PREFLIGHT mmar {subset}: {got[0]} records, {got[1]} gradeable (want {want})")
if got != want:
    print("FATAL: MMAR record count mismatch — audio missing or wrong revision", file=sys.stderr)
    sys.exit(1)
PYEOF
echo

# Fail fast on missing checkpoints. Without this a bad HF_HOME costs SERVE_TIMEOUT (40 min)
# per model in wait_healthy before the job gives up — two hours to learn one env var is wrong.
echo "PREFLIGHT checkpoints under HF_HOME=$HF_HOME"
miss=0
for MODEL in $MMAR_MODELS; do
  model_cfg "$MODEL" || { miss=1; continue; }
  SNAP="$HF_HOME/hub/models--${M_ID//\//--}/snapshots/$M_REV"
  if [ -d "$SNAP" ] && [ -n "$(ls -A "$SNAP" 2>/dev/null)" ]; then
    echo "  OK   $M_NAME -> $SNAP"
  else
    echo "  MISSING $M_NAME -> $SNAP" >&2; miss=1
  fi
  if [ "$M_SERVE" = "vllm_lora" ] && [ ! -d "$SPEECH_LORA_DIR" ]; then
    echo "  MISSING $M_NAME speech-LoRA -> $SPEECH_LORA_DIR" >&2; miss=1
  fi
done
[ "$miss" = "0" ] || { echo "FATAL: checkpoints missing — fix HF_HOME before burning the allocation" >&2; exit 1; }
echo

for MODEL in $MMAR_MODELS; do
  model_cfg "$MODEL" || continue
  echo "================ $M_NAME ($M_ID @ ${M_REV:0:12}) ================"

  # size the servers for the LARGEST arm this model will run, so one set of replicas
  # serves all four arms without a restart
  kv_plan "$M_KV_KIB" "$M_WEIGHTS_GIB" "$COT_MAX_TOKENS" "$SC_N"
  echo "  kv_plan: $KVP_NOTE -> max_num_seqs=$KVP_MAXSEQS n_chunk=$KVP_NCHUNK inflight=$KVP_INFLIGHT"

  if [ "$DRY_RUN" = "1" ]; then
    for ARM in $MMAR_ARMS; do arm_cfg "$ARM"; echo "  would run $M_NAME/$A_ARM (method=$A_METHOD n=$A_N temp=$A_TEMP maxtok=$A_MAXTOK)"; done
    continue
  fi

  for gpu in $(seq 0 $((NUM_GPUS - 1))); do serve_one "$gpu" "$KVP_MAXSEQS"; done
  ok=1
  for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    wait_healthy $((BASE_PORT + gpu)) || ok=0
  done
  if [ "$ok" != "1" ]; then
    echo "  SKIPPING $M_NAME — a replica never became healthy" >&2
    kill_servers; continue
  fi

  for ARM in $MMAR_ARMS; do
    arm_cfg "$ARM" || continue
    OUT="$OUT_ROOT/samples/${M_NAME}__${A_ARM}.jsonl"
    LOG="$OUT_ROOT/logs/${M_NAME}__${A_ARM}.log"
    # greedy arms are n=1: no chunking, and more items in flight since KV per item is tiny
    if [ "$A_N" = "1" ]; then NCHUNK=1; INFLIGHT=$(( KVP_MAXSEQS < 32 ? KVP_MAXSEQS : 32 ))
    else NCHUNK="$KVP_NCHUNK"; INFLIGHT="$KVP_INFLIGHT"; fi
    echo "---- $M_NAME / $A_ARM  (method=$A_METHOD n=$A_N temp=$A_TEMP maxtok=$A_MAXTOK "
    echo "     n_chunk=$NCHUNK inflight=$INFLIGHT/endpoint) -> $OUT"
    STOPARG=(); [ -n "${M_STOP:-}" ] && STOPARG=(--stop "$M_STOP")
    "$EPF_PY" -m benchmarking.mmar.sample_runner \
      --endpoints "$EPS" --model-name "$M_REQNAME" "${STOPARG[@]}" \
      --data-root "$DATA_MMAR" --subset "$SUBSET" --audio-mode local-path \
      --method "$A_METHOD" --n "$A_N" --n-chunk "$NCHUNK" \
      --temperature "$A_TEMP" --max-tokens "$A_MAXTOK" --seed "$SC_SEED" \
      --max-inflight "$INFLIGHT" \
      --jsonl "$OUT" 2>&1 | tee -a "$LOG"
  done

  kill_servers
  echo
done

[ "$DRY_RUN" = "1" ] && { echo "(dry run — no report)"; exit 0; }

echo "================ report ================"
"$EPF_PY" -m benchmarking.mmar.sc_report \
  --jsonl "$OUT_ROOT/samples/*.jsonl" \
  --out "$OUT_ROOT/RESULTS.md" 2>&1 | tee "$OUT_ROOT/logs/report.log"
echo "run finished: $(date)"

# --- hold the allocation ------------------------------------------------------------
# The work needs ~1.2 h of an 8 h wall. Exiting here would hand 4 GPUs back to a queue
# that is 300+ deep, so by default we keep the allocation alive for follow-up work.
# Release it by deleting the HOLD file (or scancel). HOLD_AFTER_RUN=0 disables.
if [ "${HOLD_AFTER_RUN:-1}" = "1" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  HOLDFILE="$OUT_ROOT/HOLD"
  : > "$HOLDFILE"
  cat <<EOM

================ HOLDING ALLOCATION ================
job $SLURM_JOB_ID on $(hostname) — $NUM_GPUS GPUs reserved until you release them.

  use them:   srun --jobid=$SLURM_JOB_ID --overlap <command>
  release:    rm $HOLDFILE
  or:         scancel $SLURM_JOB_ID

Auto-releases at the 8 h wall regardless. Holding since $(date '+%H:%M:%S').
====================================================
EOM
  while [ -e "$HOLDFILE" ]; do sleep 30; done
  echo "HOLD released at $(date) — exiting, GPUs returned"
fi
echo "done: $(date)"
