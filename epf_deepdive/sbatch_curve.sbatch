#!/bin/bash
#SBATCH --account=bcey-delta-gpu
#SBATCH --partition=gpuA40x4,gpuA100x4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=06:00:00
#SBATCH --output=/u/awaheed/slurm_logs/epfdd-%x-%j.out
# Champion budget curve with full text/trace storage (env: MODEL=3b|7b).
# All 8 budgets re-run fresh into mmar_${MODEL}_curve.jsonl (old files untouched).
# Budgets descend so walltime overrun leaves only cheap tails to resume.
set -uo pipefail
D=/work/hdd/bcey/awaheed/its-for-audio-reasoning/epf_deepdive
PY=/u/awaheed/envs/epf/bin/python
PORT=$((8300 + SLURM_JOB_ID % 700))  # job-unique: avoids SO_REUSEPORT collisions
: "${MODEL:?3b|7b}"
case "$MODEL" in
  3b) MN=qwen-omni-3b; PROMPT=4 ;;
  7b) MN=qwen-omni;    PROMPT=5 ;;
  *) echo "FATAL bad MODEL"; exit 1 ;;
esac

export SERVE_TAG="curve_${MODEL}_${SLURM_JOB_ID}"
bash "$D/serve_model.sh" "$MODEL" $PORT
t=0; while [ $t -lt 1800 ]; do
  curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && break; sleep 10; t=$((t+10)); done
[ $t -lt 1800 ] || { echo "FATAL server not healthy"; exit 1; }
echo "server healthy after ${t}s (port $PORT)"

echo "=== $(date '+%F %T') SMOKE (limit 2, b2, store-texts) ==="
rm -f "$D/probe_out/smoke_curve_${MODEL}.jsonl"
"$PY" "$D/probe_epf.py" --endpoint http://localhost:$PORT/v1 --model-name "$MN" \
  --bench mmar --limit 2 --budgets 2 --prompt-method "$PROMPT" --max-steps 6 \
  --store-texts --arms probe_adaptive_t3 --max-inflight 16 \
  --jsonl "$D/probe_out/smoke_curve_${MODEL}.jsonl"
"$PY" - "$D/probe_out/smoke_curve_${MODEL}.jsonl" <<'EOF'
import json, sys
rows = [json.loads(L) for L in open(sys.argv[1])]
ok = [r for r in rows if not r.get("error")]
wt = sum(1 for r in ok if r.get("texts") and any(t.strip() for t in r["texts"]))
tr = sum(1 for r in ok if (r.get("trace") or {}).get("iterations"))
print(f"smoke: {len(ok)}/2 clean, {wt} with texts, {tr} with trace")
sys.exit(0 if (len(ok) >= 2 and wt >= 2 and tr >= 2) else 1)
EOF
[ $? -eq 0 ] || { echo "FATAL smoke"; kill "$(cat "$D/serve_${SERVE_TAG}.pid")"; exit 1; }
echo "=== $(date '+%F %T') SMOKE PASS ==="

echo "=== $(date '+%F %T') CURVE $MODEL budgets 128..1 ==="
"$PY" "$D/probe_epf.py" --endpoint http://localhost:$PORT/v1 --model-name "$MN" \
  --bench mmar --limit 1000 --budgets 128,64,32,16,8,4,2,1 --prompt-method "$PROMPT" \
  --max-steps 6 --store-texts --arms probe_adaptive_t3 --max-inflight 384 \
  --jsonl "$D/probe_out/mmar_${MODEL}_curve.jsonl"
rc=$?
kill "$(cat "$D/serve_${SERVE_TAG}.pid")" 2>/dev/null
echo "=== $(date '+%F %T') CURVE $MODEL DONE rc=$rc ==="
exit $rc
