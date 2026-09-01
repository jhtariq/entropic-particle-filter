#!/bin/bash
#SBATCH --account=bcey-delta-gpu
#SBATCH --partition=gpuA40x4,gpuA100x4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --output=/u/awaheed/slurm_logs/epfdd-%x-%j.out
# Parameterized probe-EPF experiment job. Env vars:
#   MODEL=3b|7b  BENCH=mmar|mmau_pro_d1k  ARMS=...  BUDGETS=...  PROMPT=4|5|9
#   LIMIT=400  TAG=outfile-tag  [STEP_TOKEN='\n'  PROBE_SUFFIX='Final Answer: \boxed{'  PMT=4]
set -uo pipefail
D=/work/hdd/bcey/awaheed/its-for-audio-reasoning/epf_deepdive
PY=/u/awaheed/envs/epf/bin/python
PORT=8500

: "${MODEL:?}" ; : "${BENCH:?}" ; : "${ARMS:?}" ; : "${BUDGETS:?}" ; : "${PROMPT:=4}"
# sbatch --export splits on commas: ARMS/BUDGETS are passed '+'-separated
ARMS="${ARMS//+/,}" ; BUDGETS="${BUDGETS//+/,}"
: "${LIMIT:=400}" ; : "${TAG:?}" ; : "${STEP_TOKEN:=\n\n}" ; : "${PROBE_SUFFIX:=Answer:}"
: "${PMT:=3}" ; : "${MAX_STEPS:=6}" ; : "${INFLIGHT:=96}"

export SERVE_TAG="${MODEL}_${SLURM_JOB_ID}"
bash "$D/serve_model.sh" "$MODEL" $PORT
t=0; while [ $t -lt 1800 ]; do
  curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && break; sleep 10; t=$((t+10)); done
[ $t -lt 1800 ] || { echo "FATAL server not healthy"; exit 1; }
echo "server healthy after ${t}s"
case "$MODEL" in 3b) MN=qwen-omni-3b ;; 7b) MN=qwen-omni ;; esac

"$PY" "$D/probe_epf.py" --endpoint http://localhost:$PORT/v1 --model-name "$MN" \
  --bench "$BENCH" --limit "$LIMIT" --budgets "$BUDGETS" --prompt-method "$PROMPT" \
  --step-token "$STEP_TOKEN" --max-steps "$MAX_STEPS" \
  --probe-suffix "$PROBE_SUFFIX" --probe-max-tokens "$PMT" \
  --arms "$ARMS" --max-inflight "$INFLIGHT" \
  --jsonl "$D/probe_out/${TAG}.jsonl"
rc=$?
kill "$(cat "$D/serve_${SERVE_TAG}.pid")" 2>/dev/null
echo "=== job done rc=$rc ==="
exit $rc
