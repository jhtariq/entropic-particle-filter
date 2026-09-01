#!/bin/bash
# 3B P9 chain on gpub028 (3B server already up on :8500): probe arms, greedy-P9, SC-P9.
set -uo pipefail
D=/work/hdd/bcey/awaheed/its-for-audio-reasoning/epf_deepdive
REPO=/work/hdd/bcey/awaheed/its-for-audio-reasoning/entropic-particle-filter
PY=/u/awaheed/envs/epf/bin/python

echo "=== $(date '+%F %T') probe arms P9 (3B) ==="
"$PY" "$D/probe_epf.py" --endpoint http://localhost:8500/v1 --model-name qwen-omni-3b \
  --bench mmar --limit 400 --budgets 8 --prompt-method 9 \
  --step-token "\n" --max-steps 12 --probe-suffix "Final Answer: \boxed{" --probe-max-tokens 4 \
  --arms base_epf,probe_adaptive_t6,sc_probe --max-inflight 64 \
  --jsonl "$D/probe_out/mmar_3b_b8_p9.jsonl"

cd "$REPO"
echo "=== $(date '+%F %T') greedy_p9 (3B) ==="
"$PY" -m benchmarking.mmar.sample_runner \
  --endpoints http://localhost:8500/v1 --model-name qwen-omni-3b \
  --data-root /u/awaheed/epf_data/mmar --subset full \
  --method 9 --n 1 --temperature 0.0 --max-tokens 700 --max-inflight 16 \
  --jsonl "$D/mmar_omni_baseline/qwen-omni-3b__greedy_p9.jsonl"

echo "=== $(date '+%F %T') sc32_p9 (3B) ==="
"$PY" -m benchmarking.mmar.sample_runner \
  --endpoints http://localhost:8500/v1 --model-name qwen-omni-3b \
  --data-root /u/awaheed/epf_data/mmar --subset full \
  --method 9 --n 32 --n-chunk 8 --temperature 0.8 --seed 1234 --max-tokens 700 --max-inflight 8 \
  --jsonl "$D/mmar_omni_baseline/qwen-omni-3b__sc_p9.jsonl"
echo "=== $(date '+%F %T') 3B P9 chain done ==="
