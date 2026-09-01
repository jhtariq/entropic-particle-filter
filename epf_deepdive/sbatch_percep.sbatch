#!/bin/bash
#SBATCH --account=bcey-delta-gpu
#SBATCH --partition=gpuA40x4,gpuA100x4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=06:00:00
#SBATCH --output=/u/awaheed/slurm_logs/epfdd-%x-%j.out
# Perception pilots on matched-400 (env: MODEL=3b|7b):
#   B  yes/no verification probes (cheap, first)
#   C  lens-seeded particles, b32
#   A  audio-view TTA: probe_adaptive_t3 b8 on noise/eq/shift views
#      (waits for the views-ready marker; orig view = existing champion rows)
set -uo pipefail
D=/work/hdd/bcey/awaheed/its-for-audio-reasoning/epf_deepdive
PY=/u/awaheed/envs/epf/bin/python
VIEWS=/work/hdd/bcey/awaheed/its-for-audio-reasoning/epf_data/deepdive_views
PORT=8500
: "${MODEL:?3b|7b}"
case "$MODEL" in
  3b) MN=qwen-omni-3b; PROMPT=4 ;;
  7b) MN=qwen-omni;    PROMPT=5 ;;
  *) echo "FATAL bad MODEL"; exit 1 ;;
esac

export SERVE_TAG="pcp_${MODEL}_${SLURM_JOB_ID}"
bash "$D/serve_model.sh" "$MODEL" $PORT
t=0; while [ $t -lt 1800 ]; do
  curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && break; sleep 10; t=$((t+10)); done
[ $t -lt 1800 ] || { echo "FATAL server not healthy"; exit 1; }
echo "server healthy after ${t}s"

check_rows () {
  "$PY" - "$1" "$2" <<'EOF'
import json, sys
ok = sum(1 for L in open(sys.argv[1]) if not json.loads(L).get("error"))
print(f"  smoke {sys.argv[1].split('/')[-1]}: {ok}/{sys.argv[2]} clean rows")
sys.exit(0 if ok >= int(sys.argv[2]) else 1)
EOF
}

echo "=== $(date '+%F %T') SMOKE yesno + lens ==="
"$PY" "$D/yesno_probe.py" --endpoint http://localhost:$PORT/v1 --model-name "$MN" \
  --uids "$D/matched400_uids.txt" --limit 4 --max-inflight 24 \
  --jsonl "$D/probe_out/smoke_yesno_${MODEL}.jsonl" \
  && check_rows "$D/probe_out/smoke_yesno_${MODEL}.jsonl" 4 \
  || { echo "FATAL yesno smoke"; exit 1; }
"$PY" "$D/probe_epf.py" --endpoint http://localhost:$PORT/v1 --model-name "$MN" \
  --bench mmar --limit 4 --budgets 4 --prompt-method "$PROMPT" --max-steps 6 \
  --arms lens_probe_adaptive_t3 --max-inflight 48 \
  --jsonl "$D/probe_out/smoke_lens_${MODEL}.jsonl" \
  && check_rows "$D/probe_out/smoke_lens_${MODEL}.jsonl" 4 \
  || { echo "FATAL lens smoke"; exit 1; }
echo "=== $(date '+%F %T') SMOKE PASS ==="

echo "=== $(date '+%F %T') B: yesno @400 ==="
"$PY" "$D/yesno_probe.py" --endpoint http://localhost:$PORT/v1 --model-name "$MN" \
  --uids "$D/matched400_uids.txt" --max-inflight 64 \
  --jsonl "$D/probe_out/yesno_${MODEL}.jsonl"

echo "=== $(date '+%F %T') C: lens b32 @400 ==="
"$PY" "$D/probe_epf.py" --endpoint http://localhost:$PORT/v1 --model-name "$MN" \
  --bench mmar --limit 400 --budgets 32 --prompt-method "$PROMPT" --max-steps 6 \
  --arms lens_probe_adaptive_t3 --max-inflight 384 \
  --jsonl "$D/probe_out/lens_${MODEL}_b32.jsonl"

echo "=== $(date '+%F %T') A: waiting for views marker ==="
t=0; while [ $t -lt 2400 ]; do
  [ -f "$VIEWS/.done400" ] && break; sleep 30; t=$((t+30)); done
[ -f "$VIEWS/.done400" ] || { echo "FATAL views never became ready"; exit 1; }
"$PY" "$D/probe_epf.py" --endpoint http://localhost:$PORT/v1 --model-name "$MN" \
  --bench mmar --limit 4 --budgets 4 --prompt-method "$PROMPT" --max-steps 6 \
  --audio-root "$VIEWS/noise" --arms probe_adaptive_t3 --max-inflight 48 \
  --jsonl "$D/probe_out/smoke_view_${MODEL}.jsonl" \
  && check_rows "$D/probe_out/smoke_view_${MODEL}.jsonl" 4 \
  || { echo "FATAL view smoke (file:// serving of deepdive_views?)"; exit 1; }
for V in noise eq shift; do
  echo "=== $(date '+%F %T') A: view=$V b8 @400 ==="
  "$PY" "$D/probe_epf.py" --endpoint http://localhost:$PORT/v1 --model-name "$MN" \
    --bench mmar --limit 400 --budgets 8 --prompt-method "$PROMPT" --max-steps 6 \
    --audio-root "$VIEWS/$V" --arms probe_adaptive_t3 --max-inflight 96 \
    --jsonl "$D/probe_out/view_${MODEL}_${V}_b8.jsonl"
done
kill "$(cat "$D/serve_${SERVE_TAG}.pid")" 2>/dev/null
echo "=== $(date '+%F %T') PERCEPTION PILOTS $MODEL DONE ==="
