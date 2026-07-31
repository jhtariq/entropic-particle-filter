#!/usr/bin/env bash
# serve.sh <config_runNN.sh> — one vLLM replica per GPU in $GPUS (port 810<g>),
# detached, PID files + logs under $OUT_DIR/servers/, then block until healthy.
# SETUP_GUIDE §3 template: util 0.85 (audio-encoder spikes), expandable_segments,
# flashinfer sampler off (Blackwell), media path = common parent.
# Config may override: GPUS (default "0 1"), MAX_MODEL_LEN (default 32768).
# Conflict check is PER-PORT (a server on another GPU/port is left alone).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$SCRIPT_DIR/common.sh"
source "${1:?usage: serve.sh <config_runNN.sh>}"
GPUS="${GPUS:-0 1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
# Optional speech/vision-LoRA serving (Phi-4-MM, Run 5). Config sets ENABLE_LORA=1
# + LORA_MODULES="name=/path"; requests then target that adapter name, not the base.
# SERVE_BASE_NAME is the vLLM --served-model-name; SERVED_NAME stays the REQUEST name
# (adapter) so the probe/watchdog/health-check all enforce the adapter is loaded.
SERVE_BASE_NAME="${SERVE_BASE_NAME:-$SERVED_NAME}"
LORA_ARGS=()
if [ "${ENABLE_LORA:-0}" = "1" ]; then
    LORA_ARGS=(--enable-lora --max-lora-rank "${MAX_LORA_RANK:-320}" \
               --max-loras "${MAX_LORAS:-1}" --lora-modules "$LORA_MODULES")
fi

for g in $GPUS; do
    port=$((8100 + g))
    if curl -s -m 2 "http://localhost:$port/v1/models" > /dev/null 2>&1; then
        echo "ERROR: :$port already serving — kill that server by PID first (never bare pkill -f)" >&2
        exit 1
    fi
done

mkdir -p "$OUT_DIR/servers"
for g in $GPUS; do
    port=$((8100 + g))
    log="$OUT_DIR/servers/${SERVED_NAME}_gpu${g}.log"
    HF_HOME="$HF_HOME_DIR" CUDA_VISIBLE_DEVICES=$g \
    VLLM_USE_FLASHINFER_SAMPLER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup setsid "$EPF_VLLM" serve "$MODEL_ID" \
        --revision "$MODEL_REV" \
        --served-model-name "$SERVE_BASE_NAME" --port "$port" --trust-remote-code --dtype bfloat16 \
        --max-model-len "$MAX_MODEL_LEN" --enforce-eager --gpu-memory-utilization 0.85 \
        --allowed-local-media-path "$MEDIA_ROOT" \
        --limit-mm-per-prompt '{"audio":3}' \
        ${LORA_ARGS[@]+"${LORA_ARGS[@]}"} > "$log" 2>&1 &
    echo $! > "$OUT_DIR/servers/${SERVED_NAME}_gpu${g}.pid"
    echo "gpu$g: $MODEL_ID@$MODEL_REV -> :$port (pid $!, log $log)"
done

for g in $GPUS; do
    port=$((8100 + g))
    for i in $(seq 1 360); do
        if curl -s -m 5 "http://localhost:$port/v1/models" | grep -q "\"$SERVED_NAME\""; then
            echo ":$port healthy ($SERVED_NAME)"
            break
        fi
        if [ "$i" -eq 360 ]; then
            echo "ERROR: :$port not healthy after 1800 s — see $OUT_DIR/servers/${SERVED_NAME}_gpu${g}.log" >&2
            exit 1
        fi
        sleep 5
    done
done
echo "all configured endpoints healthy"
