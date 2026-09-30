#!/usr/bin/env bash
# One-command starbench smoke for a 2-GPU box (e.g. 2x RTX PRO 6000 96GB):
# judge on GPU 0, YOUR model on GPU 1, then a 4-item end-to-end sanity pass.
# Each workstation deploys one generator - download only judge + that model:
#   python sanity_check_models.py judge <MODEL>
# Usage: launch_starbench_smoke.sh MODEL
#   MODEL: qwen-omni | qwen-omni-3b | qwen2-audio | phi4mm | kimi-audio
set -euo pipefail

model="${1:?usage: launch_starbench_smoke.sh MODEL (qwen-omni|qwen-omni-3b|qwen2-audio|phi4mm|kimi-audio)}"

case "$model" in
  qwen-omni)    req_name=qwen-omni;    health=qwen-omni;    port=8812 ;;
  qwen-omni-3b) req_name=qwen-omni-3b; health=qwen-omni-3b; port=8813 ;;
  qwen2-audio)  req_name=qwen2-audio;  health=qwen2-audio;  port=8815 ;;
  phi4mm)       req_name=speech;       health=phi4mm;       port=8814 ;;
  kimi-audio)   req_name=kimi-audio;   health=kimi-audio;   port=8811 ;;
  *) echo "unknown MODEL=$model" >&2; exit 1 ;;
esac

cd "$(dirname "$0")"
mkdir -p out

if [[ -z "${BABEL_STARBENCH_ROOT:-}" ]]; then
  echo "set BABEL_STARBENCH_ROOT to the starbench dataset directory" >&2
  exit 1
fi
export BABEL_MEDIA_ROOT="$BABEL_STARBENCH_ROOT"

nvidia-smi > out/nvidia-smi.txt

nohup bash serve_judge_96g.sh 0 > out/serve_judge.log 2>&1 &
judge_pid=$!
echo "judge starting on GPU 0 (pid $judge_pid, port ${JUDGE_PORT:-8801})"
nohup bash serve_gen.sh "$model" 1 "$port" > "out/serve_${model}.log" 2>&1 &
gen_pid=$!
echo "$model starting on GPU 1 (pid $gen_pid, port $port)"

gen_url="http://localhost:$port/v1"
judge_url="http://localhost:${JUDGE_PORT:-8801}/v1"
until curl -s -m 3 "$gen_url/models" 2>/dev/null | grep -q "$health"; do sleep 20; done
echo "GEN_READY $(date +%H:%M)"
until curl -s -m 3 "$judge_url/models" 2>/dev/null | grep -q Omni-30B; do sleep 20; done
echo "JUDGE_READY $(date +%H:%M)"

rm -f out/smoke_starbench.jsonl
python probe_epf.py \
  --endpoint "$gen_url" \
  --model-name "$req_name" \
  --bench starbench \
  --limit 4 \
  --arms judge_prm_ess \
  --budgets 4 \
  --prompt-method 12 \
  --step-token '\n' \
  --max-steps 11 \
  --store-texts \
  --max-inflight 8 \
  --judge-endpoint "$judge_url" \
  --judge-concurrency 8 \
  --jsonl out/smoke_starbench.jsonl

python - <<'EOF'
import json
from collections import Counter

rows = [json.loads(l) for l in open("out/smoke_starbench.jsonl")]
print(f"\nrows: {len(rows)}  errors: {sum(bool(r['error']) for r in rows)}")
steps = Counter()
for r in rows:
    steps.update(r.get("steps_used") or [])
print(f"steps_used distribution: {dict(sorted(steps.items()))}")
ok_scores = all(
    any(any(s is not None for s in p) for p in (r.get("judge_scores_all") or []))
    for r in rows if not r["error"])
print(f"per-step judge scores present: {ok_scores}")
t = next((t for r in rows for t in r.get("texts", []) if "Step 1:" in t), None)
print(f"P12 'Step N:' format followed: {t is not None}")
if t:
    print("--- sample candidate ---")
    print(t[:900])
gate = (len(rows) >= 4 and not any(r["error"] for r in rows)
        and ok_scores and t is not None and (not steps or max(steps) <= 11))
print("STARBENCH_SMOKE_" + ("PASS" if gate else "FAIL"))
EOF

kill "$judge_pid" "$gen_pid" 2>/dev/null || true
echo "smoke done - servers stopped; full sweep: serve judge+gen, then bash run_starbench_prm.sh $model"
