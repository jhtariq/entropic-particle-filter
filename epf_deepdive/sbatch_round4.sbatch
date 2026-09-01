#!/bin/bash
#SBATCH --account=bcey-delta-gpu
#SBATCH --partition=gpuA40x4,gpuA100x4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=06:00:00
#SBATCH --output=/u/awaheed/slurm_logs/epfdd-%x-%j.out
# Round 4: raw weights + stopped-particle retirement + max_steps=3.
# Env: MODEL=3b|7b (required). Phases: smoke (limit 4, aborts on failure) ->
# 6-arm b8 main run -> 2-arm b8 max-steps-3 run (separate jsonl).
set -uo pipefail
D=/work/hdd/bcey/awaheed/its-for-audio-reasoning/epf_deepdive
PY=/u/awaheed/envs/epf/bin/python
PORT=8500
: "${MODEL:?}"

case "$MODEL" in
  3b) MN=qwen-omni-3b; PROMPT=4 ;;
  7b) MN=qwen-omni;    PROMPT=5 ;;
  *) echo "FATAL bad MODEL=$MODEL"; exit 1 ;;
esac
ARMS_MAIN="raw_probe,retire_t3,retire_t1,retire_every_t3,retire_every_t1,retire_every_raw"
ARMS_MS3="probe_adaptive_t3,retire_t3"

export SERVE_TAG="${MODEL}r4_${SLURM_JOB_ID}"
bash "$D/serve_model.sh" "$MODEL" $PORT
t=0; while [ $t -lt 1800 ]; do
  curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && break; sleep 10; t=$((t+10)); done
[ $t -lt 1800 ] || { echo "FATAL server not healthy"; exit 1; }
echo "server healthy after ${t}s"

run_probe () {  # arms, limit, max_steps, jsonl
  "$PY" "$D/probe_epf.py" --endpoint http://localhost:$PORT/v1 --model-name "$MN" \
    --bench mmar --limit "$2" --budgets 8 --prompt-method "$PROMPT" \
    --max-steps "$3" --arms "$1" --max-inflight 96 --jsonl "$4"
}

check_smoke () {  # jsonl, expected_rows
  "$PY" - "$1" "$2" <<'EOF'
import json, sys
ok = sum(1 for L in open(sys.argv[1]) if not json.loads(L).get("error"))
exp = int(sys.argv[2])
print(f"smoke {sys.argv[1]}: {ok}/{exp} clean rows")
sys.exit(0 if ok >= exp else 1)
EOF
}

echo "=== $(date '+%F %T') SMOKE (all new arms, limit 4, budget 4) ==="
"$PY" "$D/probe_epf.py" --endpoint http://localhost:$PORT/v1 --model-name "$MN" \
  --bench mmar --limit 4 --budgets 4 --prompt-method "$PROMPT" \
  --max-steps 6 --arms "$ARMS_MAIN" --max-inflight 48 \
  --jsonl "$D/probe_out/smoke_round4_${MODEL}.jsonl" \
  && check_smoke "$D/probe_out/smoke_round4_${MODEL}.jsonl" 24 \
  || { echo "FATAL main-arm smoke failed"; kill "$(cat "$D/serve_${SERVE_TAG}.pid")"; exit 1; }
"$PY" "$D/probe_epf.py" --endpoint http://localhost:$PORT/v1 --model-name "$MN" \
  --bench mmar --limit 4 --budgets 4 --prompt-method "$PROMPT" \
  --max-steps 3 --arms "$ARMS_MS3" --max-inflight 48 \
  --jsonl "$D/probe_out/smoke_round4ms3_${MODEL}.jsonl" \
  && check_smoke "$D/probe_out/smoke_round4ms3_${MODEL}.jsonl" 8 \
  || { echo "FATAL ms3 smoke failed"; kill "$(cat "$D/serve_${SERVE_TAG}.pid")"; exit 1; }

echo "=== $(date '+%F %T') MAIN round4 arms b8 @400 ==="
run_probe "$ARMS_MAIN" 400 6 "$D/probe_out/mmar_${MODEL}_b8_round4.jsonl"
rc1=$?
echo "=== $(date '+%F %T') MS3 arms b8 @400 (max_steps=3) ==="
run_probe "$ARMS_MS3" 400 3 "$D/probe_out/mmar_${MODEL}_b8_ms3.jsonl"
rc2=$?
kill "$(cat "$D/serve_${SERVE_TAG}.pid")" 2>/dev/null
echo "=== $(date '+%F %T') round4 done rc_main=$rc1 rc_ms3=$rc2 ==="
exit $(( rc1 || rc2 ))
