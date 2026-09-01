#!/bin/bash
#SBATCH --account=bcey-delta-gpu
#SBATCH --partition=gpuA40x4,gpuA100x4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=01:00:00
#SBATCH --output=/u/awaheed/slurm_logs/epfdd-%x-%j.out
# Dedicated round-4 pipeline smoke: short walltime so it backfills before the
# 6h main jobs. Serves 3B then 7B on port 8510 (no clash with main jobs' 8500)
# and runs every new arm at limit 4 / budget 4. Own output files — never shares
# a writer with the main jobs.
set -uo pipefail
D=/work/hdd/bcey/awaheed/its-for-audio-reasoning/epf_deepdive
PY=/u/awaheed/envs/epf/bin/python
PORT=8510
ARMS_MAIN="raw_probe,retire_t3,retire_t1,retire_every_t3,retire_every_t1,retire_every_raw"
ARMS_MS3="probe_adaptive_t3,retire_t3"
FAIL=0

check_rows () {  # jsonl expected
  "$PY" - "$1" "$2" <<'EOF'
import json, sys
ok = sum(1 for L in open(sys.argv[1]) if not json.loads(L).get("error"))
exp = int(sys.argv[2])
print(f"  {sys.argv[1].split('/')[-1]}: {ok}/{exp} clean rows -> {'PASS' if ok >= exp else 'FAIL'}")
sys.exit(0 if ok >= exp else 1)
EOF
}

for MODEL in 3b 7b; do
  case "$MODEL" in 3b) MN=qwen-omni-3b; PROMPT=4 ;; 7b) MN=qwen-omni; PROMPT=5 ;; esac
  export SERVE_TAG="smk_${MODEL}_${SLURM_JOB_ID}"
  echo "=== $(date '+%F %T') serving $MODEL ==="
  bash "$D/serve_model.sh" "$MODEL" $PORT
  t=0; while [ $t -lt 1500 ]; do
    curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && break; sleep 10; t=$((t+10)); done
  if [ $t -ge 1500 ]; then echo "FATAL $MODEL server not healthy"; FAIL=1
  else
    echo "$MODEL server healthy after ${t}s"
    "$PY" "$D/probe_epf.py" --endpoint http://localhost:$PORT/v1 --model-name "$MN" \
      --bench mmar --limit 4 --budgets 4 --prompt-method "$PROMPT" \
      --max-steps 6 --arms "$ARMS_MAIN" --max-inflight 48 \
      --jsonl "$D/probe_out/smoke_pre_${MODEL}.jsonl" \
      && check_rows "$D/probe_out/smoke_pre_${MODEL}.jsonl" 24 || FAIL=1
    "$PY" "$D/probe_epf.py" --endpoint http://localhost:$PORT/v1 --model-name "$MN" \
      --bench mmar --limit 4 --budgets 4 --prompt-method "$PROMPT" \
      --max-steps 3 --arms "$ARMS_MS3" --max-inflight 48 \
      --jsonl "$D/probe_out/smoke_pre_ms3_${MODEL}.jsonl" \
      && check_rows "$D/probe_out/smoke_pre_ms3_${MODEL}.jsonl" 8 || FAIL=1
  fi
  kill "$(cat "$D/serve_${SERVE_TAG}.pid")" 2>/dev/null
  sleep 20  # let the GPU free before the next serve
done
echo "=== $(date '+%F %T') SMOKE $( [ $FAIL -eq 0 ] && echo ALL-PASS || echo FAILED ) ==="
exit $FAIL
