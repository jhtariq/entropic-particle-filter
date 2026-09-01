#!/bin/bash
# Full-1000 verification of retire_every_t3 b32 on the A100 node (resumes the
# 400 already done), then completes the leftover partial arm for that model.
#   bash run_r4full_node.sh 3b 0 8520
#   bash run_r4full_node.sh 7b 1 8521
set -uo pipefail
MODEL="${1:?3b|7b}"; GPU="${2:?gpu}"; PORT="${3:?port}"
D=/work/hdd/bcey/awaheed/its-for-audio-reasoning/epf_deepdive
PY=/u/awaheed/envs/epf/bin/python
case "$MODEL" in
  3b) MN=qwen-omni-3b; PROMPT=4 ;;
  7b) MN=qwen-omni;    PROMPT=5 ;;
esac
export CUDA_VISIBLE_DEVICES="$GPU"
export SERVE_TAG="full_${MODEL}"
bash "$D/serve_model.sh" "$MODEL" "$PORT"
t=0; while [ $t -lt 1500 ]; do
  curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && break; sleep 10; t=$((t+10)); done
[ $t -lt 1500 ] || { echo "FATAL $MODEL server not healthy"; exit 1; }
echo "$MODEL server healthy after ${t}s"

echo "=== $(date '+%F %T') FULL-1000 retire_every_t3 b32 $MODEL ==="
"$PY" "$D/probe_epf.py" --endpoint http://localhost:$PORT/v1 --model-name "$MN" \
  --bench mmar --limit 1000 --budgets 32 --prompt-method "$PROMPT" --max-steps 6 \
  --arms retire_every_t3 --max-inflight 384 \
  --jsonl "$D/probe_out/mmar_${MODEL}_b32_round4.jsonl"

if [ "$MODEL" = "3b" ]; then
  echo "=== $(date '+%F %T') COMPLETE 3b ms3-b32 @400 ==="
  "$PY" "$D/probe_epf.py" --endpoint http://localhost:$PORT/v1 --model-name "$MN" \
    --bench mmar --limit 400 --budgets 32 --prompt-method 4 --max-steps 3 \
    --arms probe_adaptive_t3 --max-inflight 384 \
    --jsonl "$D/probe_out/mmar_3b_b32_ms3.jsonl"
else
  echo "=== $(date '+%F %T') COMPLETE 7b retire_t3 b32 @400 ==="
  "$PY" "$D/probe_epf.py" --endpoint http://localhost:$PORT/v1 --model-name "$MN" \
    --bench mmar --limit 400 --budgets 32 --prompt-method 5 --max-steps 6 \
    --arms retire_t3 --max-inflight 384 \
    --jsonl "$D/probe_out/mmar_7b_b32_round4.jsonl"
fi
kill "$(cat "$D/serve_${SERVE_TAG}.pid")" 2>/dev/null
echo "=== $(date '+%F %T') FULL RUN $MODEL DONE ==="
