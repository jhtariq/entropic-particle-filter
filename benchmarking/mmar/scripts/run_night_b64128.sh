#!/usr/bin/env bash
# One-command overnight b64+b128 extension of MMAR Runs 1 (7B) and 2 (3B).
# For each model in turn: serve both GPUs -> arm watchdog -> extend the existing
# grid JSONL with budgets 64,128 (resume-safe; b1-32 rows untouched) -> tear down
# by PID. Finally regenerates the combined bootstrap HTML. ~15-16 h total
# (7B ~8.5 h, 3B ~7 h). If a phase fails, its servers are torn down, the JSONL
# stays resume-safe, and the script exits (rerunning it resumes where it died).
#
# Usage (from repo root; NO conda activation needed — absolute paths inside):
#   nohup setsid bash benchmarking/mmar/scripts/run_night_b64128.sh \
#     > benchmarking/mmar/results/night_b64128.log 2>&1 &
#   tail -f benchmarking/mmar/results/night_b64128.log
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
EPF_PY=/home/exx/miniconda3/envs/epf/bin/python
cd "$REPO_ROOT"
R1="benchmarking/mmar/results/run01_epf_grid"
R2="benchmarking/mmar/results/run02_omni3b"

phase() {  # phase <config> <outdir> <served-name>
    local cfg=$1 out=$2 name=$3 wd rc g p
    echo "=== [$(date '+%F %T')] $name: serving both GPUs ==="
    bash "$SCRIPT_DIR/serve.sh" "$cfg" || { echo "$name: serve FAILED"; return 1; }
    nohup setsid bash "$SCRIPT_DIR/watchdog.sh" "$cfg" >> "$out/servers/watchdog.log" 2>&1 &
    wd=$!
    echo "=== [$(date '+%F %T')] $name: grid b64+b128 (progress -> $out/stages.log) ==="
    bash "$SCRIPT_DIR/run_grid.sh" "$cfg" >> "$out/stages.log" 2>&1
    rc=$?
    kill "$wd" 2>/dev/null
    for g in 0 1; do
        p="$out/servers/${name}_gpu${g}.pid"
        [ -f "$p" ] && kill "$(cat "$p")" 2>/dev/null
    done
    sleep 20
    if [ "$rc" -eq 0 ]; then
        echo "=== [$(date '+%F %T')] $name: phase complete ==="
    else
        echo "=== [$(date '+%F %T')] $name: grid FAILED rc=$rc (servers down; JSONL resume-safe) ==="
    fi
    return "$rc"
}

phase "$SCRIPT_DIR/config_run01_b64128.sh" "$R1" qwen-omni || exit 1
phase "$SCRIPT_DIR/config_run02_b64128.sh" "$R2" qwen-omni-3b || exit 1

echo "=== [$(date '+%F %T')] regenerating combined bootstrap ==="
"$EPF_PY" -m benchmarking.mmau_pro.epf_bootstrap --n 10000 \
    --in "Qwen2.5-Omni-7B — MMAR Run 1=$R1/mmar_run01.csv" \
    --in "Qwen2.5-Omni-3B — MMAR Run 2=$R2/mmar_run02.csv" \
    --in "Qwen2-Audio-7B-Instruct — MMAR Run 3 (le30s)=benchmarking/mmar/results/run03_qwen2audio_le30s/mmar_run03.csv" \
    --out "$R1/mmar_bootstrap.html"
echo "=== [$(date '+%F %T')] night extension complete ==="
