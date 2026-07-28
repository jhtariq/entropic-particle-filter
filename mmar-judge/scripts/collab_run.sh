#!/usr/bin/env bash
# collab_run.sh <configs/model_X.sh> [configs/config_collab.sh]
# One-command collaborator driver: serve judge (GPU1:8110) + policy (GPU0:8100),
# run the full-set grid at the collab budgets, stop the servers. Safe to rerun —
# the runner resumes from the JSONL. Examples:
#   bash mmar-judge/scripts/collab_run.sh mmar-judge/configs/model_omni.sh
#   bash mmar-judge/scripts/collab_run.sh mmar-judge/configs/model_kimi.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../configs/common.sh"
MODEL_CONFIG="${1:?usage: collab_run.sh <configs/model_X.sh> [run config]}"
RUN_CONFIG="${2:-$SCRIPT_DIR/../configs/config_collab.sh}"
source "$MODEL_CONFIG"

export RUN_NAME="${RUN_NAME:-collab_${SERVED_NAME}}"
export OUT_DIR="${OUT_DIR:-$RESULTS_ROOT/$RUN_NAME}"
mkdir -p "$OUT_DIR"
echo "run: $RUN_NAME -> $OUT_DIR"

cleanup() { bash "$SCRIPT_DIR/stop_servers.sh" "$OUT_DIR" || true; }
trap cleanup EXIT

bash "$SCRIPT_DIR/serve_judge.sh"
bash "$SCRIPT_DIR/serve_policy.sh" "$MODEL_CONFIG"
bash "$SCRIPT_DIR/run_full.sh" "$MODEL_CONFIG" "$RUN_CONFIG"
echo "collab run complete -> $OUT_DIR"
