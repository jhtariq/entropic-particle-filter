#!/usr/bin/env bash
# serve_judge.sh [configs/judge_omni.sh] — serve the audio judge on GPU $JUDGE_GPU
# at :$JUDGE_PORT (default 8110, deliberately off the 810<gpu> policy convention).
# Mirrors benchmarking/mmar/scripts/serve.sh: detached, PID file + log under
# $OUT_DIR/servers/, per-port conflict check, block until healthy.
# Requires OUT_DIR exported by the caller (run_smoke.sh/run_full.sh do this).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../configs/common.sh"
source "${1:-$SCRIPT_DIR/../configs/judge_omni.sh}"
: "${OUT_DIR:?export OUT_DIR before serving (e.g. \$RESULTS_ROOT/<run_name>)}"

if curl -s -m 2 "http://localhost:$JUDGE_PORT/v1/models" > /dev/null 2>&1; then
    echo "ERROR: :$JUDGE_PORT already serving — kill that server by PID first (never bare pkill -f)" >&2
    exit 1
fi

mkdir -p "$OUT_DIR/servers"
log="$OUT_DIR/servers/${JUDGE_SERVED_NAME}_gpu${JUDGE_GPU}.log"
HF_HOME="$HF_HOME_DIR" CUDA_VISIBLE_DEVICES=$JUDGE_GPU \
VLLM_USE_FLASHINFER_SAMPLER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup setsid "$EPF_VLLM" serve "$JUDGE_MODEL_ID" \
    --revision "$JUDGE_MODEL_REV" \
    --served-model-name "$JUDGE_SERVED_NAME" --port "$JUDGE_PORT" \
    --trust-remote-code --dtype bfloat16 \
    --max-model-len "$JUDGE_MAX_MODEL_LEN" --enforce-eager --gpu-memory-utilization 0.85 \
    --allowed-local-media-path "$MEDIA_ROOT" \
    --limit-mm-per-prompt '{"audio":3}' > "$log" 2>&1 &
echo $! > "$OUT_DIR/servers/${JUDGE_SERVED_NAME}_gpu${JUDGE_GPU}.pid"
echo "judge gpu$JUDGE_GPU: $JUDGE_MODEL_ID@$JUDGE_MODEL_REV -> :$JUDGE_PORT (pid $!, log $log)"

for i in $(seq 1 360); do
    if curl -s -m 5 "http://localhost:$JUDGE_PORT/v1/models" | grep -q "\"$JUDGE_SERVED_NAME\""; then
        echo ":$JUDGE_PORT healthy ($JUDGE_SERVED_NAME)"
        exit 0
    fi
    if [ "$i" -eq 360 ]; then
        echo "ERROR: :$JUDGE_PORT not healthy after 1800 s — see $log" >&2
        exit 1
    fi
    sleep 5
done
