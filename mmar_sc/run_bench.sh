#!/bin/bash
# Bench-parameterised driver: the same 12 cells (3 models x 4 arms) against either
# MMAR or MMAU test-mini. Select with BENCH=mmar|mmau (default mmar).
#
#   BENCH=mmau bash mmar_sc/run_bench.sh
#   BENCH=mmau DRY_RUN=1 bash mmar_sc/run_bench.sh
#
# Supersedes run_all.sh (which is MMAR-only). Output, logs and the single-writer lock
# all live under $OUT_ROOT = /u/awaheed/<bench>_sc_out, so an MMAU run and an MMAR run
# never collide even when both are alive.
set -uo pipefail

# cu130 wheels vs Delta's 570.x/CUDA-12.8 driver: without this every engine dies with
# "NVIDIA driver is too old (found version 12080)". The sbatch wrapper also loads it,
# but this script may be launched directly via srun, so load it here too (idempotent).
module load cuda-compat/13.0 2>/dev/null || true

MMAR_SC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$MMAR_SC_DIR/config.sh"
source "$MMAR_SC_DIR/lib.sh"
cd "$REPO_ROOT"

mkdir -p "$OUT_ROOT/servers" "$OUT_ROOT/samples" "$OUT_ROOT/logs"
EPS="$(endpoints_csv)"
echo "BENCH=$B_NAME  data=$B_DATA  runner=$B_MODULE"
echo "models: $B_MODELS"
echo "GPUs=$NUM_GPUS endpoints=$EPS out=$OUT_ROOT subset=$SUBSET"
echo

exec 9> "$OUT_ROOT/.run.lock"
if ! flock -n 9; then
  echo "another $B_NAME run holds $OUT_ROOT/.run.lock — exiting so it can proceed"
  exit 0
fi
echo "acquired run lock (job ${SLURM_JOB_ID:-local} on $(hostname))"

# Record-count preflight. The loaders have require_audio_exists=True and SILENTLY drop
# items whose audio is missing, so a wrong data root shrinks the eval set with no error.
"$EPF_PY" - "$B_DATA" "$SUBSET" "$B_EXPECT_N" "$B_EXPECT_GRADEABLE" <<PYEOF || exit 1
import sys
$B_LOADER
root, subset, want_n, want_g = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
recs = load(root, subset=subset)
g = sum(1 for r in recs if r.answer_index is not None)
print(f"PREFLIGHT $B_NAME {subset}: {len(recs)} records, {g} gradeable (want {want_n}, {want_g})")
if (len(recs), g) != (want_n, want_g):
    print("FATAL: record count mismatch — wrong data root, or audio missing", file=sys.stderr)
    sys.exit(1)
PYEOF

# Fail fast on missing checkpoints rather than burning SERVE_TIMEOUT per model.
echo "PREFLIGHT checkpoints under HF_HOME=$HF_HOME"
miss=0
for MODEL in $B_MODELS; do
  model_cfg "$MODEL" || { miss=1; continue; }
  SNAP="${M_HF_HOME:-$HF_HOME}/hub/models--${M_ID//\//--}/snapshots/$M_REV"
  if [ -d "$SNAP" ] && [ -n "$(ls -A "$SNAP" 2>/dev/null)" ]; then echo "  OK   $M_NAME"
  else echo "  MISSING $M_NAME -> $SNAP" >&2; miss=1; fi
  if [ "$M_SERVE" = "vllm_lora" ] && [ ! -d "$SPEECH_LORA_DIR" ]; then
    echo "  MISSING $M_NAME speech-LoRA -> $SPEECH_LORA_DIR" >&2; miss=1; fi
done
[ "$miss" = "0" ] || { echo "FATAL: checkpoints missing" >&2; exit 1; }
echo

for MODEL in $B_MODELS; do
  model_cfg "$MODEL" || continue
  echo "================ $M_NAME ($M_ID @ ${M_REV:0:12}) on $B_NAME ================"
  kv_plan "$M_KV_KIB" "$M_WEIGHTS_GIB" "$COT_MAX_TOKENS" "$SC_N"
  echo "  kv_plan: $KVP_NOTE -> max_num_seqs=$KVP_MAXSEQS n_chunk=$KVP_NCHUNK inflight=$KVP_INFLIGHT"

  if [ "$DRY_RUN" = "1" ]; then
    for ARM in $MMAR_ARMS; do arm_cfg "$ARM"
      echo "  would run $B_NAME/$M_NAME/$A_ARM (method=$A_METHOD n=$A_N temp=$A_TEMP maxtok=$A_MAXTOK)"; done
    continue
  fi

  for gpu in $(seq 0 $((NUM_GPUS - 1))); do serve_one "$gpu" "$KVP_MAXSEQS"; done
  ok=1
  for gpu in $(seq 0 $((NUM_GPUS - 1))); do wait_healthy $((BASE_PORT + gpu)) || ok=0; done
  if [ "$ok" != "1" ]; then
    echo "  SKIPPING $M_NAME — a replica never became healthy" >&2; kill_servers; continue
  fi

  for ARM in $MMAR_ARMS; do
    arm_cfg "$ARM" || continue
    OUT="$OUT_ROOT/samples/${M_NAME}__${A_ARM}.jsonl"
    LOG="$OUT_ROOT/logs/${M_NAME}__${A_ARM}.log"
    if [ "$A_N" = "1" ]; then NCHUNK=1; INFLIGHT=$(( KVP_MAXSEQS < 32 ? KVP_MAXSEQS : 32 ))
    else NCHUNK="$KVP_NCHUNK"; INFLIGHT="$KVP_INFLIGHT"; fi
    # a per-model cap overrides the kv_plan value (Kimi must stay at 1 — see config.sh)
    [ "${M_MAX_INFLIGHT:-0}" -gt 0 ] && INFLIGHT="$M_MAX_INFLIGHT"
    echo "---- $B_NAME / $M_NAME / $A_ARM  (method=$A_METHOD n=$A_N temp=$A_TEMP maxtok=$A_MAXTOK"
    echo "     n_chunk=$NCHUNK inflight=$INFLIGHT/endpoint) -> $OUT"
    STOPARG=(); [ -n "${M_STOP:-}" ] && STOPARG=(--stop "$M_STOP")
    "$EPF_PY" -m "$B_MODULE" \
      --endpoints "$EPS" --model-name "$M_REQNAME" "${STOPARG[@]}" \
      --data-root "$B_DATA" --subset "$SUBSET" --audio-mode local-path \
      --method "$A_METHOD" --n "$A_N" --n-chunk "$NCHUNK" \
      --temperature "$A_TEMP" --max-tokens "$A_MAXTOK" --seed "$SC_SEED" \
      --max-inflight "$INFLIGHT" \
      --jsonl "$OUT" 2>&1 | tee -a "$LOG"
  done

  kill_servers
  echo
done

[ "$DRY_RUN" = "1" ] && { echo "(dry run — no report)"; exit 0; }

echo "================ report ($B_NAME) ================"
"$EPF_PY" -m benchmarking.mmar.sc_report \
  --jsonl "$OUT_ROOT/samples/*.jsonl" \
  --out "$OUT_ROOT/RESULTS.md" 2>&1 | tee "$OUT_ROOT/logs/report.log"
echo "$B_NAME run finished: $(date)"

# --- hold the allocation ------------------------------------------------------------
# Exiting would hand the GPUs back to a 300+ deep queue. Keep them for follow-up work;
# release by deleting the HOLD file (or scancel). HOLD_AFTER_RUN=0 disables.
if [ "${HOLD_AFTER_RUN:-1}" = "1" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  HOLDFILE="$OUT_ROOT/HOLD"
  : > "$HOLDFILE"
  cat <<EOM

================ HOLDING ALLOCATION ================
job $SLURM_JOB_ID on $(hostname) — $NUM_GPUS GPUs reserved until you release them.

  use them:   srun --jobid=$SLURM_JOB_ID --overlap <command>
  release:    rm $HOLDFILE
  or:         scancel $SLURM_JOB_ID

Auto-releases at the wall regardless. Holding since $(date '+%H:%M:%S').
====================================================
EOM
  while [ -e "$HOLDFILE" ]; do sleep 30; done
  echo "HOLD released at $(date) — exiting, GPUs returned"
fi
