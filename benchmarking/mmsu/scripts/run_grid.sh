#!/usr/bin/env bash
# run_grid.sh <config_runNN.sh> — budget-staged EPF grid over the full MMSU set,
# resumable JSONL, up to 3 attempts per stage. A stage passes only if (a) no error
# rows remain (summarize_errors) AND (b) every cell at this budget has all 5,000
# clean rows — so a probe process dying without recording errors cannot silently
# pass a stage. b1 runs --max-inflight 24 (audio-encoder OOM lesson, SETUP_GUIDE
# §10.1), else 64.
# Run detached (literal example for Run 1):
#   nohup setsid bash benchmarking/mmsu/scripts/run_grid.sh \
#     benchmarking/mmsu/scripts/config_run01.sh \
#     > benchmarking/mmsu/results/run01_omni7b/stages.log 2>&1 &
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$SCRIPT_DIR/common.sh"
source "${1:?usage: run_grid.sh <config_runNN.sh>}"

J="$OUT_DIR/$STEM.jsonl"
C="$OUT_DIR/$STEM.csv"
L="$OUT_DIR/$STEM.log"
BUDGETS="${BUDGETS:-1 8 16 32}"   # configs may override (e.g. "64 128" for extensions)
SUBSET="${SUBSET:-full}"          # configs may override
N_ITEMS="${N_ITEMS:-5000}"        # items in $SUBSET (drives the per-stage count gate)
SIGNALS="${SIGNALS:-mean_logprob,entropy}"  # configs may override (e.g. "random" for the ablation)
NSIG=$(echo "$SIGNALS" | tr ',' '\n' | grep -c .)
EXPECTED=$(( $(echo "$PROMPTS" | tr ',' '\n' | grep -c .) * NSIG * N_ITEMS ))  # prompts x signals x items
mkdir -p "$OUT_DIR"
cd "$REPO_ROOT"

for B in $BUDGETS; do
    INFLIGHT=64
    [ "$B" -eq 1 ] && INFLIGHT=24
    ok=0
    for attempt in 1 2 3; do
        echo "=== stage b$B attempt $attempt ($(date '+%F %T')) prompts=$PROMPTS inflight=$INFLIGHT ==="
        "$EPF_PY" -m benchmarking.mmsu.diversity_probe \
            --endpoints "$ENDPOINTS" --model-name "$SERVED_NAME" \
            --data-root "$DATA_ROOT" --subset "$SUBSET" \
            --prompts "$PROMPTS" --signals "$SIGNALS" --budgets "$B" \
            --temp 0.8 --ess-threshold 0.6 --early-phase 0.7 \
            --max-steps 6 --max-tokens-per-step 300 \
            --select all --limit 5100 --max-inflight "$INFLIGHT" \
            --jsonl "$J" --csv "$C" --log "$L"
        probe_rc=$?
        if [ "$probe_rc" -ne 0 ]; then
            echo "stage b$B attempt $attempt: probe exited rc=$probe_rc -> resume sweep"
            continue
        fi
        if ! "$EPF_PY" -m benchmarking.mmsu.summarize_errors "$J"; then
            echo "stage b$B attempt $attempt: error rows remain -> resume sweep"
            continue
        fi
        got=$("$EPF_PY" -c '
import json, sys
path, b = sys.argv[1], int(sys.argv[2])
methods = {int(m) for m in sys.argv[3].split(",")}
keys = set()
for line in open(path):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    if int(r["budget"]) == b and int(r["method"]) in methods and not r.get("error"):
        keys.add((r["unique_id"], int(r["method"]), r["signal"]))
print(len(keys))
' "$J" "$B" "$PROMPTS")
        if [ "$got" -ne "$EXPECTED" ]; then
            echo "stage b$B attempt $attempt: only $got/$EXPECTED clean (item,prompt,signal) keys -> resume sweep"
            continue
        fi
        echo "stage b$B: $got/$EXPECTED clean keys, 0 errors"
        ok=1
        break
    done
    if [ "$ok" -ne 1 ]; then
        echo "FAILED: stage b$B incomplete after 3 attempts" >&2
        exit 1
    fi
    echo "=== stage b$B complete ($(date '+%F %T')) ==="
done
echo "=== grid complete ($(date '+%F %T')) ==="
