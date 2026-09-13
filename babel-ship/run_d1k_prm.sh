#!/usr/bin/env bash
# Run the judge_epf_prm (process-reward) full budget curve 1-128 on MMAU-Pro
# d1k for one model. Waits for the generator and judge servers, runs a 2-item
# smoke gate, then the full sweep. Resume-safe: rerunning skips completed
# (item, budget) pairs already in the output jsonl.
# Usage: run_d1k_prm.sh MODEL [GEN_PORT]
#   MODEL: qwen-omni | qwen-omni-3b | qwen2-audio | phi4mm | kimi-audio
set -euo pipefail

model="${1:?usage: run_d1k_prm.sh MODEL [GEN_PORT]}"
JUDGE_PORT="${JUDGE_PORT:-8801}"

case "$model" in
  qwen-omni)    req_name=qwen-omni;    health=qwen-omni;    default_port=8812 ;;
  qwen-omni-3b) req_name=qwen-omni-3b; health=qwen-omni-3b; default_port=8813 ;;
  qwen2-audio)  req_name=qwen2-audio;  health=qwen2-audio;  default_port=8815 ;;
  phi4mm)       req_name=speech;       health=phi4mm;       default_port=8814 ;;
  kimi-audio)   req_name=kimi-audio;   health=kimi-audio;   default_port=8811 ;;
  *) echo "unknown MODEL=$model" >&2; exit 1 ;;
esac
gen_port="${2:-$default_port}"
gen_url="http://localhost:$gen_port/v1"
judge_url="http://localhost:$JUDGE_PORT/v1"

cd "$(dirname "$0")"
mkdir -p out
out_jsonl="out/d1k_${model}_prmjudge.jsonl"

until curl -s -m 3 "$gen_url/models" 2>/dev/null | grep -q "$health"; do sleep 20; done
echo "GEN_READY $model $(date +%H:%M)"
until curl -s -m 3 "$judge_url/models" 2>/dev/null | grep -q Omni-30B; do sleep 20; done
echo "JUDGE_READY $(date +%H:%M)"

# Smoke gate: 2 real items through the full gen+judge loop; require non-null
# per-step judge scores before committing to the sweep.
rm -f "out/smoke_${model}.jsonl"
python probe_epf.py \
  --endpoint "$gen_url" \
  --model-name "$req_name" \
  --bench mmau_pro_d1k \
  --limit 2 \
  --arms judge_epf_prm \
  --budgets 2 \
  --store-texts \
  --max-inflight 8 \
  --judge-endpoint "$judge_url" \
  --judge-concurrency 8 \
  --jsonl "out/smoke_${model}.jsonl" > "out/smoke_${model}.log" 2>&1
gate="$(SMOKE="out/smoke_${model}.jsonl" python - <<'EOF'
import json, os
rows = [json.loads(l) for l in open(os.environ["SMOKE"])]
ok = len(rows) >= 2 and all(
    not r["error"] and r.get("texts")
    and any(any(s is not None for s in p) for p in (r.get("judge_scores_all") or []))
    for r in rows)
print("PASS" if ok else "FAIL")
EOF
)"
echo "SMOKE_${model}_${gate}"
if [[ "$gate" != PASS ]]; then
  echo "smoke failed - see out/smoke_${model}.log" >&2
  exit 1
fi

exec python probe_epf.py \
  --endpoint "$gen_url" \
  --model-name "$req_name" \
  --bench mmau_pro_d1k \
  --limit 1000 \
  --arms judge_epf_prm \
  --budgets 128,64,32,16,8,4,2,1 \
  --store-texts \
  --max-inflight 384 \
  --judge-endpoint "$judge_url" \
  --judge-concurrency 48 \
  --jsonl "$out_jsonl"
