#!/usr/bin/env bash
# watchdog.sh <config_runNN.sh> — 60 s health poll on :8100/:8101 keyed to the
# served model name; logs UP/DOWN transitions with timestamps. NEVER kills anything.
# Run detached (literal example for Run 1):
#   nohup setsid bash benchmarking/mmsu/scripts/watchdog.sh \
#     benchmarking/mmsu/scripts/config_run01.sh \
#     > benchmarking/mmsu/results/run01_omni7b/servers/watchdog.log 2>&1 &
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$SCRIPT_DIR/common.sh"
source "${1:?usage: watchdog.sh <config_runNN.sh>}"
GPUS="${GPUS:-0 1}"
PORTS=$(for g in $GPUS; do echo $((8100 + g)); done)

declare -A state
echo "$(date '+%F %T') watchdog armed for $SERVED_NAME on port(s) $(echo $PORTS | tr '\n' ' ')"
while true; do
    for port in $PORTS; do
        if curl -s -m 10 "http://localhost:$port/v1/models" | grep -q "\"$SERVED_NAME\""; then
            now=UP
        else
            now=DOWN
        fi
        prev="${state[$port]:-INIT}"
        if [ "$now" != "$prev" ]; then
            echo "$(date '+%F %T') :$port $prev -> $now"
            state[$port]=$now
        fi
    done
    sleep 60
done
