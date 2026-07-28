#!/usr/bin/env bash
# run_smoke.sh <configs/model_X.sh> [configs/config_local.sh] [N]
# Stratified N-item smoke (default $SMOKE_N=100): BoN over $BUDGETS_BON and
# Beam over $BUDGETS_BEAM. Servers must already be up (serve_judge.sh +
# serve_policy.sh with the same OUT_DIR). Resumable: rerun to retry error rows.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../configs/common.sh"
MODEL_CONFIG="${1:?usage: run_smoke.sh <configs/model_X.sh> [run config] [N]}"
source "$MODEL_CONFIG"
source "${2:-$SCRIPT_DIR/../configs/config_local.sh}"
N="${3:-$SMOKE_N}"
RUN_NAME="${RUN_NAME:-smoke${N}_${SERVED_NAME}}"
OUT_DIR="${OUT_DIR:-$RESULTS_ROOT/$RUN_NAME}"
mkdir -p "$OUT_DIR"

for ep in "$POLICY_ENDPOINT" "$JUDGE_ENDPOINT"; do
    curl -s -m 5 "$ep/models" > /dev/null || { echo "ERROR: $ep not healthy" >&2; exit 1; }
done

cd "$REPO_ROOT"
COMMON_ARGS=(
    --endpoints "$POLICY_ENDPOINT" --model-name "$SERVED_NAME"
    --judge-endpoint "$JUDGE_ENDPOINT" --judge-model-name "${JUDGE_SERVED_NAME:-qwen-omni-judge}"
    --prompt-method "${PROMPT_METHOD:-4}" --data-root "$DATA_ROOT" --subset "${SUBSET:-full}"
    --select stratified --limit "$N" --seed "${SEED:-0}"
    --max-inflight "$MAX_INFLIGHT" --policy-inflight "$POLICY_INFLIGHT"
    --judge-inflight "$JUDGE_INFLIGHT"
)

"$EPF_PY" -m benchmarking.mmar_judge.run_judge "${COMMON_ARGS[@]}" \
    --alg bon --budgets "$(echo "$BUDGETS_BON" | tr ' ' ',')" \
    --jsonl "$OUT_DIR/bon.jsonl" --csv "$OUT_DIR/bon.csv" --log "$OUT_DIR/bon.log"

"$EPF_PY" -m benchmarking.mmar_judge.run_judge "${COMMON_ARGS[@]}" \
    --alg beam --budgets "$(echo "$BUDGETS_BEAM" | tr ' ' ',')" --beam-width "$BEAM_WIDTH" \
    --jsonl "$OUT_DIR/beam.jsonl" --csv "$OUT_DIR/beam.csv" --log "$OUT_DIR/beam.log"

echo "smoke done -> $OUT_DIR"
