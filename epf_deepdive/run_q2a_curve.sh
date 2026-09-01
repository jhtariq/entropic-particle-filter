#!/bin/bash
# Q2A MMAR champion budget-curve, one vLLM instance per GPU.
#   bash run_q2a_curve.sh 0 8531 128 g0 384
#   bash run_q2a_curve.sh 1 8532 64,32,16,8,4,2,1 g1 256
# Smoke first (limit 2, budget 2, --store-texts): verifies the server answers as
# qwen2-audio AND that rows carry texts + pruned-lineage trace. Full texts stored
# for every budget (user requirement).
set -uo pipefail
GPU="${1:?gpu}"; PORT="${2:?port}"; BUDGETS="${3:?budgets}"; TAG="${4:?tag}"; INFLIGHT="${5:-256}"
D=/work/hdd/bcey/awaheed/its-for-audio-reasoning/epf_deepdive
PY=/u/awaheed/envs/epf/bin/python

export CUDA_VISIBLE_DEVICES="$GPU"
export SERVE_TAG="q2acurve_${TAG}"
bash "$D/serve_model.sh" q2a "$PORT"
t=0; while [ $t -lt 1500 ]; do
  curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && break; sleep 10; t=$((t+10)); done
[ $t -lt 1500 ] || { echo "FATAL server not healthy"; exit 1; }
echo "q2a server healthy on gpu $GPU port $PORT after ${t}s"

echo "=== $(date '+%F %T') SMOKE ($TAG) ==="
rm -f "$D/probe_out/smoke_q2acurve_${TAG}.jsonl"
"$PY" "$D/probe_epf.py" --endpoint "http://localhost:$PORT/v1" --model-name qwen2-audio \
  --bench mmar --limit 2 --budgets 2 --prompt-method 4 --max-steps 6 --store-texts \
  --arms probe_adaptive_t3 --max-inflight 16 \
  --jsonl "$D/probe_out/smoke_q2acurve_${TAG}.jsonl"
"$PY" - "$D/probe_out/smoke_q2acurve_${TAG}.jsonl" <<'EOF'
import json, sys
rows = [json.loads(L) for L in open(sys.argv[1])]
ok = [r for r in rows if not r.get("error")]
with_texts = sum(1 for r in ok if r.get("texts") and any(t.strip() for t in r["texts"]))
with_trace = sum(1 for r in ok if (r.get("trace") or {}).get("iterations"))
print(f"smoke: {len(ok)}/2 clean, {with_texts} with texts, {with_trace} with trace")
sys.exit(0 if (len(ok) >= 2 and with_texts >= 2 and with_trace >= 2) else 1)
EOF
[ $? -eq 0 ] || { echo "FATAL smoke ($TAG)"; kill "$(cat "$D/serve_${SERVE_TAG}.pid")"; exit 1; }
echo "=== $(date '+%F %T') SMOKE PASS ($TAG) ==="

echo "=== $(date '+%F %T') CURVE $TAG budgets=$BUDGETS ==="
"$PY" "$D/probe_epf.py" --endpoint "http://localhost:$PORT/v1" --model-name qwen2-audio \
  --bench mmar --limit 1000 --budgets "$BUDGETS" --prompt-method 4 --max-steps 6 \
  --store-texts --arms probe_adaptive_t3 --max-inflight "$INFLIGHT" \
  --jsonl "$D/probe_out/mmar_q2a_curve_${TAG}.jsonl"
rc=$?
kill "$(cat "$D/serve_${SERVE_TAG}.pid")" 2>/dev/null
echo "=== $(date '+%F %T') CURVE $TAG DONE rc=$rc ==="
