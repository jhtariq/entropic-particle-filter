#!/bin/bash
#SBATCH --account=bcey-delta-gpu
#SBATCH --partition=gpuA40x4,gpuA100x4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=06:00:00
#SBATCH --output=/u/awaheed/slurm_logs/epfdd-%x-%j.out
# Full-1000 noise/TTA promotion (env: MODEL=3b|7b).
#   3b job: phase-0 builds all full-996 views (16-way pool) + snr sweep views,
#           writes marker .done996; then runs its suite incl. SNR sweep @400.
#   7b job: waits for the marker, then runs the same core suite (no SNR sweep).
# All runs resume-safe; existing 400-item rows are skipped automatically.
set -uo pipefail
D=/work/hdd/bcey/awaheed/its-for-audio-reasoning/epf_deepdive
PY=/u/awaheed/envs/epf/bin/python
VIEWS=/work/hdd/bcey/awaheed/its-for-audio-reasoning/epf_data/deepdive_views
# job-unique port: two jobs on the SAME node both binding 8500 get kernel
# load-balanced via SO_REUSEPORT — requests randomly hit the wrong model
PORT=$((8300 + SLURM_JOB_ID % 700))
: "${MODEL:?3b|7b}"
case "$MODEL" in
  3b) MN=qwen-omni-3b; PROMPT=4 ;;
  7b) MN=qwen-omni;    PROMPT=5 ;;
  *) echo "FATAL bad MODEL"; exit 1 ;;
esac

if [ "$MODEL" = "3b" ]; then
  echo "=== $(date '+%F %T') phase-0: build full-996 views ==="
  "$PY" "$D/make_audio_views.py" --uids "$D/full996_uids.txt" \
    --views noise,noise2,eq,shift --workers 16 || { echo "FATAL augment"; exit 1; }
  "$PY" "$D/make_audio_views.py" --uids "$D/matched400_uids.txt" \
    --views snr33,snr21 --workers 16 || { echo "FATAL augment snr"; exit 1; }
  touch "$VIEWS/.done996"
  echo "views ready"
else
  echo "=== $(date '+%F %T') waiting for views marker ==="
  t=0; while [ $t -lt 5400 ]; do
    [ -f "$VIEWS/.done996" ] && break; sleep 30; t=$((t+30)); done
  [ -f "$VIEWS/.done996" ] || { echo "FATAL views never ready"; exit 1; }
fi

export SERVE_TAG="nz_${MODEL}_${SLURM_JOB_ID}"
bash "$D/serve_model.sh" "$MODEL" $PORT
t=0; while [ $t -lt 1800 ]; do
  curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && break; sleep 10; t=$((t+10)); done
[ $t -lt 1800 ] || { echo "FATAL server not healthy"; exit 1; }
echo "server healthy after ${t}s"

run_view () {  # view limit budget jsonl [inflight]
  "$PY" "$D/probe_epf.py" --endpoint http://localhost:$PORT/v1 --model-name "$MN" \
    --bench mmar --limit "$2" --budgets "$3" --prompt-method "$PROMPT" --max-steps 6 \
    --audio-root "$VIEWS/$1" --arms probe_adaptive_t3 --max-inflight "${5:-96}" \
    --jsonl "$D/probe_out/$4"
}

echo "=== $(date '+%F %T') SMOKE noise2 (limit 4, b4) ==="
run_view noise2 4 4 "smoke_noise2_${MODEL}.jsonl" 48
n=$("$PY" -c "import json;print(sum(1 for L in open('$D/probe_out/smoke_noise2_${MODEL}.jsonl') if not json.loads(L).get('error')))")
[ "$n" -ge 4 ] || { echo "FATAL noise2 smoke ($n/4)"; kill "$(cat "$D/serve_${SERVE_TAG}.pid")"; exit 1; }
echo "smoke pass ($n/4)"

echo "=== $(date '+%F %T') noise b8 -> 996 (resume) ==="
run_view noise 1000 8 "view_${MODEL}_noise_b8.jsonl"
echo "=== $(date '+%F %T') noise2 b8 @996 ==="
run_view noise2 1000 8 "view_${MODEL}_noise2_b8.jsonl"
echo "=== $(date '+%F %T') eq b8 -> 996 (resume) ==="
run_view eq 1000 8 "view_${MODEL}_eq_b8.jsonl"
echo "=== $(date '+%F %T') shift b8 -> 996 (resume) ==="
run_view shift 1000 8 "view_${MODEL}_shift_b8.jsonl"
if [ "$MODEL" = "3b" ]; then
  echo "=== $(date '+%F %T') SNR sweep @400 ==="
  run_view snr33 400 8 "view_3b_snr33_b8.jsonl"
  run_view snr21 400 8 "view_3b_snr21_b8.jsonl"
fi
echo "=== $(date '+%F %T') noise b32 @996 ==="
run_view noise 1000 32 "view_${MODEL}_noise_b32.jsonl" 384

kill "$(cat "$D/serve_${SERVE_TAG}.pid")" 2>/dev/null
echo "=== $(date '+%F %T') NOISE PROMOTION $MODEL DONE ==="
