#!/usr/bin/env bash
# stop_servers.sh <run_out_dir> — kill servers by their PID files ONLY
# (house rule: never bare pkill -f; other servers on the box must survive).
set -euo pipefail
OUT_DIR="${1:?usage: stop_servers.sh <run_out_dir with servers/ inside>}"
shopt -s nullglob
found=0
for pidfile in "$OUT_DIR"/servers/*.pid; do
    found=1
    pid="$(cat "$pidfile")"
    if kill -0 "$pid" 2>/dev/null; then
        # setsid'd servers: kill the whole process group, fall back to the pid
        kill -- -"$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
        echo "killed $pid ($(basename "$pidfile"))"
    else
        echo "not running: $pid ($(basename "$pidfile"))"
    fi
    rm -f "$pidfile"
done
[ "$found" = 1 ] || echo "no PID files under $OUT_DIR/servers/"
