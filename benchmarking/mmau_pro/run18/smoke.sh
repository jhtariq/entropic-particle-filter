#!/bin/bash
# Sanity smoke test. Phase A is offline and exact on any machine; Phase B serves one
# replica (GPU 0), asserts the speech adapter is exposed + both phase0 gates, and
# runs a 4-item budget-8 probe cell end-to-end (~10-20 min, mostly the model load).
# SMOKE_PHASE=A runs only the offline half (no GPU needed); default runs both.
set -euo pipefail
: "${SMOKE_PHASE:=AB}"
RUN18_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUN18_DIR/config.sh"
source "$RUN18_DIR/lib.sh"
cd "$REPO_ROOT"
mkdir -p "$OUT_ROOT/smoke" "$OUT_ROOT/servers"

echo "=== SMOKE PHASE A (offline: tests, data, scoring, artifact identity) ==="
"$EPF_PY" -m pytest tests/ -q --ignore=tests/e2e     # expect: 122+ passed
"$EPF_PY" - "$DATA_TESTMINI" "$DATA_AUDIO" <<'PYEOF'
import sys
from benchmarking.mmau_pro.loader import load_mmau_mcq
from benchmarking.mmau_pro.scoring import extract_letter, match_answer_index, predicted_index
tm, au = sys.argv[1], sys.argv[2]
t = load_mmau_mcq(tm, subset="test", audio_root=au)
assert (len(t), sum(1 for r in t if r.answer_index is None)) == (5090, 24)
assert len(load_mmau_mcq(tm, subset="full")) == 957
assert extract_letter('The answer is clear.\n\nAnswer: B', 4) == 1
assert predicted_index('b) dog barking', ['cat', 'dog barking', 'dog']) == 1
assert match_answer_index('The Dog barking!', ['cat', 'dog barking', 'dog']) == 1
print("PHASE A data + scoring: OK")
PYEOF
check_snapshot "$RUN18_MODEL_ID" "$RUN18_REV"
check_merged
echo "=== PHASE A PASS ==="
[ "$SMOKE_PHASE" = "A" ] && { echo "(SMOKE_PHASE=A — skipping the served Phase B)"; exit 0; }

echo "=== SMOKE PHASE B (serve, adapter + gates, tiny b8 cell) ==="
serve_one 0
wait_healthy "$BASE_PORT" 1800
gate_endpoint "$BASE_PORT"
SM="$OUT_ROOT/smoke/run18_smoke"
rm -f "$SM.jsonl"
extra=(); [ -n "$STOP_REGEX" ] && extra+=(--stop-regex "$STOP_REGEX")
"$EPF_PY" -m benchmarking.mmau_pro.diversity_probe \
  --endpoints "http://localhost:$BASE_PORT/v1" --model-name "$(probe_model_name)" \
  --data-root "$DATA_TESTMINI" --subset "$RUN18_SUBSET" --audio-root "$DATA_AUDIO" \
  --prompts "${PROMPTS_RUN18%%,*}" --signals mean_logprob --budgets 8 \
  --select all --limit 4 --temp 0.8 --ess-threshold 0.6 --early-phase 0.7 \
  --max-inflight "$MAX_INFLIGHT" "${extra[@]}" \
  --jsonl "$SM.jsonl" --csv "$SM.csv" --log "$SM.log"
"$EPF_PY" "$RUN18_DIR/summarize_errors.py" "$SM.jsonl"
kill_servers
echo "=== SMOKE PASS — ready for: nohup bash $RUN18_DIR/run_all.sh > run18.log 2>&1 & ==="
