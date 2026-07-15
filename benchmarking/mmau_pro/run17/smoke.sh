#!/bin/bash
# Sanity smoke test. Phase A is offline and exact on any machine; Phase B optionally
# proves shim-vs-reference parity, then serves one shim (GPU 0), asserts both phase0
# gates, and runs a 4-item budget-8 probe cell end-to-end. ~10-15 min total (Mellow
# is tiny — most of it is the reference-parity model load).
# SMOKE_PHASE=A runs only the offline half (no GPU needed); default runs both.
# SMOKE_PARITY=0 skips the reference-parity step (it loads the reference wrapper,
# whose own downloader is unpinned — the shim itself always uses the pins).
set -euo pipefail
: "${SMOKE_PHASE:=AB}"
: "${SMOKE_PARITY:=1}"
RUN17_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUN17_DIR/config.sh"
source "$RUN17_DIR/lib.sh"
cd "$REPO_ROOT"
mkdir -p "$OUT_ROOT/smoke" "$OUT_ROOT/servers"

echo "=== SMOKE PHASE A (offline: tests, data, scoring, artifact identity) ==="
"$EPF_PY" -m pytest tests/ -q --ignore=tests/e2e     # expect: 122+ passed (104 + shim tests)
"$EPF_PY" - "$DATA_TESTMINI" "$DATA_AUDIO" <<'PYEOF'
import sys
from benchmarking.mmau_pro.loader import load_mmau_mcq
from benchmarking.mmau_pro.scoring import extract_letter, match_answer_index, predicted_index
tm, au = sys.argv[1], sys.argv[2]
t = load_mmau_mcq(tm, subset="test", audio_root=au)
assert (len(t), sum(1 for r in t if r.answer_index is None)) == (5090, 24)
n3a = load_mmau_mcq(tm, subset="test_no3a", audio_root=au)
assert (len(n3a), sum(1 for r in n3a if r.answer_index is None)) == (5073, 24)
assert len(load_mmau_mcq(tm, subset="test_le10s_flat", audio_root=au)) == 326
assert extract_letter('The answer is clear.\n\nAnswer: B', 4) == 1
assert predicted_index('b) dog barking', ['cat', 'dog barking', 'dog']) == 1
assert match_answer_index('The Dog barking!', ['cat', 'dog barking', 'dog']) == 1
print("PHASE A data + scoring: OK")
PYEOF
check_mellow_snapshot
check_smollm2_snapshot
check_mellow_code
echo "=== PHASE A PASS ==="
[ "$SMOKE_PHASE" = "A" ] && { echo "(SMOKE_PHASE=A — skipping the served Phase B)"; exit 0; }

echo "=== SMOKE PHASE B (parity, serve, gate, tiny b8 cell) ==="
if [ "$SMOKE_PARITY" = "1" ]; then
  echo "-- shim vs reference greedy parity (GPU 0, no server)"
  CUDA_VISIBLE_DEVICES=0 "$EPF_PY" "$RUN17_DIR/validate_shim.py" \
    --code-dir "$MELLOW_CODE_DIR" --data-root "$DATA_TESTMINI" \
    --audio-root "$DATA_AUDIO" --variant "$MELLOW_VARIANT" \
    --n-parity 3 --n-fill 2 --max-new 40
fi

serve_one 0
wait_healthy "$BASE_PORT" 600
gate_endpoint "$BASE_PORT"
SM="$OUT_ROOT/smoke/run17_smoke"
rm -f "$SM.jsonl"
"$EPF_PY" -m benchmarking.mmau_pro.diversity_probe \
  --endpoints "http://localhost:$BASE_PORT/v1" --model-name "$RUN17_NAME" \
  --data-root "$DATA_TESTMINI" --subset "$RUN17_SUBSET" --audio-root "$DATA_AUDIO" \
  --prompts "${PROMPTS_RUN17%%,*}" --signals mean_logprob --budgets 8 \
  --select all --limit 4 --temp 0.8 --ess-threshold 0.6 --early-phase 0.7 \
  --stop-regex "$STOP_REGEX" --max-inflight "$MAX_INFLIGHT" \
  --jsonl "$SM.jsonl" --csv "$SM.csv" --log "$SM.log"
"$EPF_PY" "$RUN17_DIR/summarize_errors.py" "$SM.jsonl"
kill_servers
echo "=== SMOKE PASS — ready for: nohup bash $RUN17_DIR/run_all.sh > run17.log 2>&1 & ==="
