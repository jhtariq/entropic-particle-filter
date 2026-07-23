#!/usr/bin/env bash
# serve_kimi.sh <config_run06_kimi.sh> — serve Kimi-Audio through the REQUIRED run19
# wrapper. vLLM 0.22.1 registers MoonshotKimiaForCausalLM but its chat path is broken
# 6 ways (empty render, continue_final_message KeyError, concurrent-audio engine death,
# [EOS] re-encode 400, top_logprobs server-kill, no auto chat template);
# benchmarking/mmau_pro/run19/serve_kimi.py monkey-patches all of them, then delegates
# to the stock `vllm serve` CLI. Plain vllm serve / serve.sh will NOT work for Kimi.
#
# Mirrors serve.sh (per-port conflict check, PID file + log under $OUT_DIR/servers/,
# health-poll on /v1/models), but launches the wrapper with Run 19's serve_one flags
# verbatim: --limit-mm-per-prompt '{"audio":1}', --chat-template, util 0.85,
# max-model-len 8192 (= config.json max_position_embeddings; do NOT raise). One replica
# per GPU in $GPUS (port 810<g>). --pid-file is consumed by serve_kimi.py itself (it
# writes its OWN pid — run17 lesson — so kill-by-PID hits the real server, not a stale
# launcher). Per-port conflict check leaves a server on another port (phi4mm :8100) alone.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$SCRIPT_DIR/common.sh"
source "${1:?usage: serve_kimi.sh <config_run06_kimi.sh>}"
GPUS="${GPUS:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
RUN19_DIR="$REPO_ROOT/benchmarking/mmau_pro/run19"
SERVE_KIMI="$RUN19_DIR/serve_kimi.py"
KIMI_CHAT_TEMPLATE="$RUN19_DIR/template_kimi_audio_epf.jinja"

for f in "$SERVE_KIMI" "$KIMI_CHAT_TEMPLATE"; do
    [ -f "$f" ] || { echo "ERROR: missing required run19 asset: $f" >&2; exit 1; }
done

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
    pidf="$OUT_DIR/servers/${SERVED_NAME}_gpu${g}.pid"
    HF_HOME="$HF_HOME_DIR" CUDA_VISIBLE_DEVICES=$g \
    VLLM_USE_FLASHINFER_SAMPLER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup setsid "$EPF_PY" "$SERVE_KIMI" "$MODEL_ID" \
        --revision "$MODEL_REV" --pid-file "$pidf" \
        --served-model-name "$SERVED_NAME" --port "$port" --trust-remote-code --dtype bfloat16 \
        --max-model-len "$MAX_MODEL_LEN" --enforce-eager --gpu-memory-utilization 0.85 \
        --allowed-local-media-path "$MEDIA_ROOT" \
        --limit-mm-per-prompt '{"audio":1}' \
        --chat-template "$KIMI_CHAT_TEMPLATE" > "$log" 2>&1 &
    echo "gpu$g: $MODEL_ID@$MODEL_REV -> :$port (log $log; pid-file $pidf)"
done

for g in $GPUS; do
    port=$((8100 + g))
    log="$OUT_DIR/servers/${SERVED_NAME}_gpu${g}.log"
    for i in $(seq 1 360); do
        if curl -s -m 5 "http://localhost:$port/v1/models" | grep -q "\"$SERVED_NAME\""; then
            echo ":$port healthy ($SERVED_NAME)"
            break
        fi
        if [ "$i" -eq 360 ]; then
            echo "ERROR: :$port not healthy after 1800 s — see $log" >&2
            exit 1
        fi
        sleep 5
    done
done
echo "all configured endpoints healthy"
