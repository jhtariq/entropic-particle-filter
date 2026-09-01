#!/bin/bash
# Round-4 EARLY b32 promotion on the interactive 2-GPU node (bonus compute).
#   bash run_r4b32_node.sh 3b 0 8520
#   bash run_r4b32_node.sh 7b 1 8521
# Smoke gate first (resumes smoke_pre_* — the dedicated smoke job must be
# cancelled first: single-writer rule), then b32@400 arms in priority order:
#   1. retire_every_t3   (flagship: every-step resampling among active only)
#   2. probe_adaptive_t3 with max_steps=3  (independent axis, cheap, best offline)
#   3. retire_t3         (gated + retirement; runs if node time remains)
# Writes ONLY b32 files — the running sbatch b8 jobs never touch these.
set -uo pipefail
MODEL="${1:?3b|7b}"; GPU="${2:?gpu index}"; PORT="${3:?port}"
D=/work/hdd/bcey/awaheed/its-for-audio-reasoning/epf_deepdive
PY=/u/awaheed/envs/epf/bin/python
case "$MODEL" in
  3b) MN=qwen-omni-3b; PROMPT=4 ;;
  7b) MN=qwen-omni;    PROMPT=5 ;;
  *) echo "FATAL bad MODEL=$MODEL"; exit 1 ;;
esac

export CUDA_VISIBLE_DEVICES="$GPU"
export SERVE_TAG="node_${MODEL}"
bash "$D/serve_model.sh" "$MODEL" "$PORT"
t=0; while [ $t -lt 1500 ]; do
  curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && break; sleep 10; t=$((t+10)); done
[ $t -lt 1500 ] || { echo "FATAL $MODEL server not healthy"; exit 1; }
echo "$MODEL server healthy after ${t}s (gpu $GPU, port $PORT)"

check_rows () {
  "$PY" - "$1" "$2" <<'EOF'
import json, sys
ok = sum(1 for L in open(sys.argv[1]) if not json.loads(L).get("error"))
print(f"  smoke {sys.argv[1].split('/')[-1]}: {ok}/{sys.argv[2]} clean rows")
sys.exit(0 if ok >= int(sys.argv[2]) else 1)
EOF
}

echo "=== $(date '+%F %T') SMOKE $MODEL (limit 4, budget 4) ==="
"$PY" "$D/probe_epf.py" --endpoint http://localhost:$PORT/v1 --model-name "$MN" \
  --bench mmar --limit 4 --budgets 4 --prompt-method "$PROMPT" --max-steps 6 \
  --arms retire_every_t3,retire_t3 --max-inflight 48 \
  --jsonl "$D/probe_out/smoke_pre_${MODEL}.jsonl" \
  && check_rows "$D/probe_out/smoke_pre_${MODEL}.jsonl" 8 \
  || { echo "FATAL smoke failed"; exit 1; }
"$PY" "$D/probe_epf.py" --endpoint http://localhost:$PORT/v1 --model-name "$MN" \
  --bench mmar --limit 4 --budgets 4 --prompt-method "$PROMPT" --max-steps 3 \
  --arms probe_adaptive_t3 --max-inflight 48 \
  --jsonl "$D/probe_out/smoke_pre_ms3_${MODEL}.jsonl" \
  && check_rows "$D/probe_out/smoke_pre_ms3_${MODEL}.jsonl" 4 \
  || { echo "FATAL ms3 smoke failed"; exit 1; }
echo "=== $(date '+%F %T') SMOKE $MODEL PASS ==="

echo "=== $(date '+%F %T') B32 retire_every_t3 $MODEL @400 ==="
"$PY" "$D/probe_epf.py" --endpoint http://localhost:$PORT/v1 --model-name "$MN" \
  --bench mmar --limit 400 --budgets 32 --prompt-method "$PROMPT" --max-steps 6 \
  --arms retire_every_t3 --max-inflight 384 \
  --jsonl "$D/probe_out/mmar_${MODEL}_b32_round4.jsonl"
echo "=== $(date '+%F %T') B32 ms3 probe_adaptive_t3 $MODEL @400 ==="
"$PY" "$D/probe_epf.py" --endpoint http://localhost:$PORT/v1 --model-name "$MN" \
  --bench mmar --limit 400 --budgets 32 --prompt-method "$PROMPT" --max-steps 3 \
  --arms probe_adaptive_t3 --max-inflight 384 \
  --jsonl "$D/probe_out/mmar_${MODEL}_b32_ms3.jsonl"
echo "=== $(date '+%F %T') B32 retire_t3 $MODEL @400 (time permitting) ==="
"$PY" "$D/probe_epf.py" --endpoint http://localhost:$PORT/v1 --model-name "$MN" \
  --bench mmar --limit 400 --budgets 32 --prompt-method "$PROMPT" --max-steps 6 \
  --arms retire_t3 --max-inflight 384 \
  --jsonl "$D/probe_out/mmar_${MODEL}_b32_round4.jsonl"
kill "$(cat "$D/serve_${SERVE_TAG}.pid")" 2>/dev/null
echo "=== $(date '+%F %T') NODE B32 $MODEL DONE ==="
