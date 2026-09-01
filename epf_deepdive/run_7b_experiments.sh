#!/bin/bash
# 7B chain on one GPU: P5/P9 greedy baselines, probe-EPF arms on P5 and P9, SC refs.
set -uo pipefail
D=/work/hdd/bcey/awaheed/its-for-audio-reasoning/epf_deepdive
REPO=/work/hdd/bcey/awaheed/its-for-audio-reasoning/entropic-particle-filter
PY=/u/awaheed/envs/epf/bin/python
PORT=8500

export SERVE_TAG="7bchain_${SLURM_JOB_ID:-manual}"
bash "$D/serve_model.sh" 7b $PORT
t=0; while [ $t -lt 1800 ]; do
  curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && break; sleep 10; t=$((t+10)); done
[ $t -lt 1800 ] || { echo "FATAL server"; exit 1; }
echo "server healthy after ${t}s"

cd "$REPO"
for M in 5 9; do
  echo "=== $(date '+%F %T') greedy_p$M ==="
  "$PY" -m benchmarking.mmar.sample_runner \
    --endpoints http://localhost:$PORT/v1 --model-name qwen-omni \
    --data-root /u/awaheed/epf_data/mmar --subset full \
    --method $M --n 1 --temperature 0.0 --max-tokens 700 --max-inflight 16 \
    --jsonl "$D/mmar_omni_baseline/qwen-omni__greedy_p$M.jsonl"
done

echo "=== $(date '+%F %T') probe arms P5 ==="
"$PY" "$D/probe_epf.py" --endpoint http://localhost:$PORT/v1 --model-name qwen-omni \
  --bench mmar --limit 400 --budgets 8 --prompt-method 5 \
  --arms base_epf,probe_adaptive_t3,sc_probe --max-inflight 48 \
  --jsonl "$D/probe_out/mmar_7b_b8_p5.jsonl"

echo "=== $(date '+%F %T') probe arms P9 ==="
"$PY" "$D/probe_epf.py" --endpoint http://localhost:$PORT/v1 --model-name qwen-omni \
  --bench mmar --limit 400 --budgets 8 --prompt-method 9 \
  --step-token "\n" --max-steps 12 --probe-suffix "Final Answer: \boxed{" --probe-max-tokens 4 \
  --arms base_epf,probe_adaptive_t3,sc_probe --max-inflight 48 \
  --jsonl "$D/probe_out/mmar_7b_b8_p9.jsonl"

for M in 5 9; do
  echo "=== $(date '+%F %T') sc32_p$M ==="
  "$PY" -m benchmarking.mmar.sample_runner \
    --endpoints http://localhost:$PORT/v1 --model-name qwen-omni \
    --data-root /u/awaheed/epf_data/mmar --subset full \
    --method $M --n 32 --n-chunk 8 --temperature 0.8 --seed 1234 --max-tokens 700 --max-inflight 8 \
    --jsonl "$D/mmar_omni_baseline/qwen-omni__sc_p$M.jsonl"
done

kill $(cat "$D/serve_${SERVE_TAG}.pid") 2>/dev/null
echo "=== $(date '+%F %T') 7B chain done ==="
