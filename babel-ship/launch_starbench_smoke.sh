#!/usr/bin/env bash
# One-command starbench smoke for a 2-GPU box (e.g. 2x RTX PRO 6000 96GB):
# judge on GPU 0, Omni-3B generator on GPU 1, then the smoke-gated runner
# stopped right after its 2-item gate (limit stays tiny via SMOKE_ONLY).
# Requires: env installed, BABEL_STARBENCH_ROOT set, models downloaded
# (sanity_check_models.py judge qwen-omni-3b).
set -euo pipefail

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
nohup bash serve_gen.sh qwen-omni-3b 1 8813 > out/serve_qwen-omni-3b.log 2>&1 &
gen_pid=$!
echo "3B gen starting on GPU 1 (pid $gen_pid, port 8813)"

# The runner's built-in gate does the actual smoke; running with a 4-item
# limit keeps this launcher a pure sanity pass.
gen_url="http://localhost:8813/v1"
judge_url="http://localhost:${JUDGE_PORT:-8801}/v1"
until curl -s -m 3 "$gen_url/models" 2>/dev/null | grep -q qwen-omni-3b; do sleep 20; done
echo "GEN_READY $(date +%H:%M)"
until curl -s -m 3 "$judge_url/models" 2>/dev/null | grep -q Omni-30B; do sleep 20; done
echo "JUDGE_READY $(date +%H:%M)"

rm -f out/smoke_starbench.jsonl
python probe_epf.py \
  --endpoint "$gen_url" \
  --model-name qwen-omni-3b \
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
echo "smoke done - servers stopped; full sweep: bash run_starbench_prm.sh MODEL (serve first)"
