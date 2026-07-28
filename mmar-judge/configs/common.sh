# Shared constants for mmar-judge runs. Sources the proven MMAR constants
# (EPF_PY/EPF_VLLM/HF_HOME_DIR/DATA_ROOT/MEDIA_ROOT) then adds the judge/policy
# split: policy model on GPU0 :8100, judge on GPU1 :8110.
MJ_CONF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$MJ_CONF_DIR/../.." && pwd)}"
source "$REPO_ROOT/benchmarking/mmar/scripts/common.sh"

POLICY_GPU="${POLICY_GPU:-0}"     # serve.sh/serve_kimi.sh port formula: 8100 + gpu
JUDGE_GPU="${JUDGE_GPU:-1}"
POLICY_PORT=$((8100 + POLICY_GPU))
JUDGE_PORT="${JUDGE_PORT:-8110}"
POLICY_ENDPOINT="http://localhost:$POLICY_PORT/v1"
JUDGE_ENDPOINT="http://localhost:$JUDGE_PORT/v1"
RESULTS_ROOT="$REPO_ROOT/mmar-judge/results"
