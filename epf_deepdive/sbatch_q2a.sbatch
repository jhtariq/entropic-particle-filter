#!/bin/bash
#SBATCH --account=bcey-delta-gpu
#SBATCH --partition=gpuA40x4,gpuA100x4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --output=/u/awaheed/slurm_logs/epfdd-%x-%j.out
# Q2A MMAR transfer: initial EPF (base_epf) + champion (probe_adaptive_t3),
# P4 CoT, full-1000 (996 gradeable). Env: BUDGET=8|32 — one job per budget.
# Smoke gate first: Q2A chat template + 'Answer:' prefill probe is untested.
set -uo pipefail
D=/work/hdd/bcey/awaheed/its-for-audio-reasoning/epf_deepdive
PY=/u/awaheed/envs/epf/bin/python
PORT=8500
: "${BUDGET:?8|32}"
INFLIGHT=$(( BUDGET >= 32 ? 384 : 96 ))
ARMS="base_epf,probe_adaptive_t3"

export SERVE_TAG="q2a_${SLURM_JOB_ID}"
bash "$D/serve_model.sh" q2a $PORT
t=0; while [ $t -lt 1800 ]; do
  curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && break; sleep 10; t=$((t+10)); done
[ $t -lt 1800 ] || { echo "FATAL q2a server not healthy"; cat "$D/serve_${SERVE_TAG}.log" | tail -30; exit 1; }
echo "q2a server healthy after ${t}s"

echo "=== $(date '+%F %T') SMOKE q2a (limit 4, budget 4) ==="
"$PY" "$D/probe_epf.py" --endpoint http://localhost:$PORT/v1 --model-name qwen2-audio \
  --bench mmar --limit 4 --budgets 4 --prompt-method 4 --max-steps 6 \
  --arms "$ARMS" --max-inflight 48 \
  --jsonl "$D/probe_out/smoke_q2a_b${BUDGET}.jsonl"
"$PY" - "$D/probe_out/smoke_q2a_b${BUDGET}.jsonl" <<'EOF'
import json, sys
rows = [json.loads(L) for L in open(sys.argv[1])]
ok = [r for r in rows if not r.get("error")]
# the probe must actually see letter mass on this model, not just not-crash
probed = sum(1 for r in ok if any(d for d in (r.get("probe_dists") or []) if d))
print(f"smoke: {len(ok)}/8 clean rows; {probed} rows with live probe letter-mass")
sys.exit(0 if len(ok) >= 8 and probed >= 3 else 1)
EOF
[ $? -eq 0 ] || { echo "FATAL q2a smoke failed"; kill "$(cat "$D/serve_${SERVE_TAG}.pid")"; exit 1; }
echo "=== $(date '+%F %T') SMOKE PASS ==="

echo "=== $(date '+%F %T') MAIN q2a b${BUDGET} @full-1000 ==="
"$PY" "$D/probe_epf.py" --endpoint http://localhost:$PORT/v1 --model-name qwen2-audio \
  --bench mmar --limit 1000 --budgets "$BUDGET" --prompt-method 4 --max-steps 6 \
  --arms "$ARMS" --max-inflight "$INFLIGHT" \
  --jsonl "$D/probe_out/mmar_q2a_b${BUDGET}.jsonl"
rc=$?
kill "$(cat "$D/serve_${SERVE_TAG}.pid")" 2>/dev/null
echo "=== $(date '+%F %T') q2a b${BUDGET} done rc=$rc ==="
exit $rc
