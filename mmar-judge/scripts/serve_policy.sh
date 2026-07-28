#!/usr/bin/env bash
# serve_policy.sh <configs/model_X.sh> — serve the policy model on GPU0 :8100.
# Dispatches on SERVE_KIND: vllm (stock serve.sh), kimi (run19 wrapper via
# serve_kimi.sh), mellow (bespoke shim). The model config doubles as the run
# config the underlying serve script sources. Requires OUT_DIR exported.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../configs/common.sh"
CONFIG="${1:?usage: serve_policy.sh <configs/model_X.sh>}"
CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
source "$CONFIG"
: "${OUT_DIR:?export OUT_DIR before serving (e.g. \$RESULTS_ROOT/<run_name>)}"

case "$SERVE_KIND" in
    vllm)
        exec "$REPO_ROOT/benchmarking/mmar/scripts/serve.sh" "$CONFIG"
        ;;
    kimi)
        exec "$REPO_ROOT/benchmarking/mmar/scripts/serve_kimi.sh" "$CONFIG"
        ;;
    mellow)
        if curl -s -m 2 "http://localhost:$POLICY_PORT/v1/models" > /dev/null 2>&1; then
            echo "ERROR: :$POLICY_PORT already serving — kill that server by PID first" >&2
            exit 1
        fi
        mkdir -p "$OUT_DIR/servers"
        log="$OUT_DIR/servers/${SERVED_NAME}_gpu${POLICY_GPU}.log"
        pidfile="$OUT_DIR/servers/${SERVED_NAME}_gpu${POLICY_GPU}.pid"
        cd "$REPO_ROOT"
        HF_HOME="$HF_HOME_DIR" nohup setsid "$EPF_PY" -m benchmarking.mmau_pro.serve_mellow \
            --host 0.0.0.0 --port "$POLICY_PORT" --model-name "$SERVED_NAME" \
            --variant "$MELLOW_VARIANT" --revision "$MODEL_REV" \
            --code-dir "$MELLOW_CODE_DIR" --device "cuda:$POLICY_GPU" \
            --allowed-media-root "$MEDIA_ROOT" \
            --pid-file "$pidfile" > "$log" 2>&1 &
        echo "mellow shim -> :$POLICY_PORT (log $log)"
        for i in $(seq 1 120); do
            if curl -s -m 5 "http://localhost:$POLICY_PORT/v1/models" | grep -q "\"$SERVED_NAME\""; then
                echo ":$POLICY_PORT healthy ($SERVED_NAME)"
                exit 0
            fi
            if [ "$i" -eq 120 ]; then
                echo "ERROR: :$POLICY_PORT not healthy after 600 s — see $log" >&2
                exit 1
            fi
            sleep 5
        done
        ;;
    *)
        echo "ERROR: unknown SERVE_KIND '$SERVE_KIND' in $CONFIG" >&2
        exit 1
        ;;
esac
