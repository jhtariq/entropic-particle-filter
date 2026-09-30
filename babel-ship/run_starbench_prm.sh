#!/usr/bin/env bash
# STAR-Bench-Perception judge sweep for one model: P12 numbered-step prompt,
# "\n" step delimiter, max 11 steps, judge_prm_ess arm (step-level judging,
# ESS-0.5-GATED resampling - NOT every-step). Resume-safe.
# Usage: run_starbench_prm.sh MODEL [BUDGETS]
#   MODEL: qwen-omni | qwen-omni-3b | qwen2-audio | phi4mm | kimi-audio
#   BUDGETS: default 128,64,32,16,8,4,2,1
set -euo pipefail

model="${1:?usage: run_starbench_prm.sh MODEL [BUDGETS]}"
budgets="${2:-128,64,32,16,8,4,2,1}"
JUDGE_PORT="${JUDGE_PORT:-8801}"

case "$model" in
  qwen-omni)    req_name=qwen-omni;    health=qwen-omni;    default_port=8812 ;;
  qwen-omni-3b) req_name=qwen-omni-3b; health=qwen-omni-3b; default_port=8813 ;;
  qwen2-audio)  req_name=qwen2-audio;  health=qwen2-audio;  default_port=8815 ;;
  phi4mm)       req_name=speech;       health=phi4mm;       default_port=8814 ;;
  kimi-audio)   req_name=kimi-audio;   health=kimi-audio;   default_port=8811 ;;
  *) echo "unknown MODEL=$model" >&2; exit 1 ;;
esac
gen_port="${GEN_PORT:-$default_port}"
gen_url="http://localhost:$gen_port/v1"
judge_url="http://localhost:$JUDGE_PORT/v1"

cd "$(dirname "$0")"
mkdir -p out
out_jsonl="out/starbench_${model}_prmjudge.jsonl"

until curl -s -m 3 "$gen_url/models" 2>/dev/null | grep -q "$health"; do sleep 20; done
echo "GEN_READY $model $(date +%H:%M)"
until curl -s -m 3 "$judge_url/models" 2>/dev/null | grep -q Omni-30B; do sleep 20; done
echo "JUDGE_READY $(date +%H:%M)"

# Smoke gate: 2 items end-to-end; require non-null per-step judge scores AND
# that the model actually followed the "Step N:" line format.
rm -f "out/smoke_starbench_${model}.jsonl"
python probe_epf.py \
  --endpoint "$gen_url" \
  --model-name "$req_name" \
  --bench starbench \
  --limit 2 \
  --arms judge_prm_ess \
  --budgets 2 \
  --prompt-method 12 \
  --step-token '\n' \
  --max-steps 11 \
  --store-texts \
  --max-inflight 8 \
  --judge-endpoint "$judge_url" \
  --judge-concurrency 8 \
  --jsonl "out/smoke_starbench_${model}.jsonl" > "out/smoke_starbench_${model}.log" 2>&1
gate="$(SMOKE="out/smoke_starbench_${model}.jsonl" python - <<'EOF'
import json, os
rows = [json.loads(l) for l in open(os.environ["SMOKE"])]
ok = len(rows) >= 2 and all(
    not r["error"] and r.get("texts")
    and any(any(s is not None for s in p) for p in (r.get("judge_scores_all") or []))
    for r in rows)
stepfmt = any("Step 1:" in t for r in rows for t in r.get("texts", []))
print("PASS" if (ok and stepfmt) else "FAIL")
EOF
)"
echo "STARBENCH_SMOKE_${model}_${gate}"
if [[ "$gate" != PASS ]]; then
  echo "smoke failed - see out/smoke_starbench_${model}.log" >&2
  exit 1
fi

exec python probe_epf.py \
  --endpoint "$gen_url" \
  --model-name "$req_name" \
  --bench starbench \
  --limit 626 \
  --arms judge_prm_ess \
  --budgets "$budgets" \
  --prompt-method 12 \
  --step-token '\n' \
  --max-steps 11 \
  --store-texts \
  --max-inflight 256 \
  --judge-endpoint "$judge_url" \
  --judge-concurrency 48 \
  --jsonl "$out_jsonl"
