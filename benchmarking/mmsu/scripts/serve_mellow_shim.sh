#!/usr/bin/env bash
# serve_mellow_shim.sh <config_run04_mellow.sh> — one serve_mellow.py shim per GPU
# (Mellow is NOT vLLM-servable; the shim is the OpenAI-compatible wrapper proven in
# mmau_pro Run 17). Mirrors run17/lib.sh serve_one: detached, PID files + logs under
# $OUT_DIR/servers/, then blocks until both endpoints are healthy.
# Config may override: GPUS (default "0 1").
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$SCRIPT_DIR/common.sh"
source "${1:?usage: serve_mellow_shim.sh <config_run04_mellow.sh>}"
GPUS="${GPUS:-0 1}"
: "${MELLOW_CODE_DIR:?set MELLOW_CODE_DIR in scripts/local.sh — the pinned clone of \
github.com/soham97/mellow @ \$MELLOW_CODE_COMMIT (reuse the one mmau_pro/run17/fetch_models.sh made)}"

for g in $GPUS; do
    port=$((8100 + g))
    if curl -s -m 2 "http://localhost:$port/v1/models" > /dev/null 2>&1; then
        echo "ERROR: :$port already serving — kill that server by PID first (never bare pkill -f)" >&2
        exit 1
    fi
done

mkdir -p "$OUT_DIR/servers"
cd "$REPO_ROOT"
for g in $GPUS; do
    port=$((8100 + g))
    log="$OUT_DIR/servers/${SERVED_NAME}_gpu${g}.log"
    pidf="$OUT_DIR/servers/${SERVED_NAME}_gpu${g}.pid"
    # --pid-file: the shim records its own PID (setsid re-forks, so $! would be stale)
    CUDA_VISIBLE_DEVICES=$g HF_HOME="$HF_HOME_DIR" nohup setsid "$EPF_PY" \
        -m benchmarking.mmau_pro.serve_mellow \
        --port "$port" --model-name "$SERVED_NAME" --variant "$MELLOW_VARIANT" \
        --revision "$MODEL_REV" --smollm2-revision "$SMOLLM2_REV" \
        --code-dir "$MELLOW_CODE_DIR" \
        --max-text-tokens "$MELLOW_MAX_TEXT_TOKENS" --single-audio-fill "$SINGLE_AUDIO_FILL" \
        --allowed-media-root "$MEDIA_ROOT" --waveform-cache "${WAVEFORM_CACHE:-6000}" \
        --pid-file "$pidf" > "$log" 2>&1 &
    echo "serving $SERVED_NAME on :$port (GPU $g) — log: $log"
done

for g in $GPUS; do
    port=$((8100 + g))
    waited=0
    until curl -s --max-time 3 "http://localhost:$port/v1/models" | grep -q '"id"'; do
        sleep 5; waited=$((waited + 5))
        if [ "$waited" -ge 600 ]; then
            echo "FATAL: :$port not healthy after 600s — check $OUT_DIR/servers/*.log" >&2
            exit 1
        fi
    done
    echo "endpoint :$port healthy after ${waited}s"
done
