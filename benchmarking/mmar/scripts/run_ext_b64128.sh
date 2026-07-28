#!/usr/bin/env bash
# run_ext_b64128.sh <ext-config> — one-command b64/b128 extension for a single MMAR
# run: serve (via $SERVE_SCRIPT, default serve.sh; Kimi configs point at
# serve_kimi.sh) -> arm watchdog -> extend the run's existing grid JSONL with the
# config's BUDGETS (resume-safe; existing rows untouched) -> tear down by PID.
# On failure: servers torn down, JSONL resume-safe, rerun the same command.
#
# Usage (from repo root; no conda activation needed):
#   nohup setsid bash benchmarking/mmar/scripts/run_ext_b64128.sh \
#     benchmarking/mmar/scripts/config_run03_b64128.sh \
#     > benchmarking/mmar/results/run03_qwen2audio_le30s/ext_b64128.log 2>&1 &
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
CONFIG="${1:?usage: run_ext_b64128.sh <ext-config.sh>}"
# shellcheck disable=SC1090
source "$SCRIPT_DIR/common.sh"
source "$CONFIG"
SERVE_SCRIPT="$SCRIPT_DIR/${SERVE_SCRIPT_NAME:-serve.sh}"   # Kimi config sets serve_kimi.sh
GPUS="${GPUS:-0 1}"

echo "=== [$(date '+%F %T')] $SERVED_NAME: serving (gpus: $GPUS) ==="
bash "$SERVE_SCRIPT" "$CONFIG" || { echo "$SERVED_NAME: serve FAILED"; exit 1; }
nohup setsid bash "$SCRIPT_DIR/watchdog.sh" "$CONFIG" >> "$OUT_DIR/servers/watchdog.log" 2>&1 &
WD=$!

echo "=== [$(date '+%F %T')] $SERVED_NAME: grid budgets [$BUDGETS] (progress -> $OUT_DIR/stages.log) ==="
bash "$SCRIPT_DIR/run_grid.sh" "$CONFIG" >> "$OUT_DIR/stages.log" 2>&1
RC=$?

kill "$WD" 2>/dev/null
for g in $GPUS; do
    p="$OUT_DIR/servers/${SERVED_NAME}_gpu${g}.pid"
    [ -f "$p" ] && kill "$(cat "$p")" 2>/dev/null
done
sleep 20
if [ "$RC" -eq 0 ]; then
    echo "=== [$(date '+%F %T')] $SERVED_NAME: extension complete ==="
    echo "reminder: regenerate the combined bootstrap with all completed runs' CSVs"
else
    echo "=== [$(date '+%F %T')] $SERVED_NAME: grid FAILED rc=$RC (servers down; JSONL resume-safe) ==="
fi
exit "$RC"
