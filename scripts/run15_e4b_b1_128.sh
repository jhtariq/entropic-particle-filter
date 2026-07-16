#!/usr/bin/env bash
# Run 15 (COLLABORATOR) — the EPF grid on Gemma 4 **E4B** IT, budgets 1 -> 128,
# **prompt P7 only** (P4 was dropped for E4B: at the shared tok30 x 12-step config
# its verbose plan-and-solve trajectories truncate and score a -10-13 pp artifact —
# RESULTS.md §21). Signals {mean_logprob, entropy}, temp 0.8, ess 0.6 / early 0.7,
# the same committed 1,947-item <=30 s single-audio ids (E4B has the same 30 s/clip cap).
#
# The JSONL below is COMMITTED with P7 budgets {1,8,16} already complete (11,682
# rows, run on the original box) — the b1/8/16 stages resume instantly and you
# effectively run only b32/64/128 (~12 h on 2 GPUs). Do not delete/rename it.
#
# Separate file from run14_collab_b64_128.sh ON PURPOSE: point each script's
# ENDPOINTS at a disjoint set of GPUs (E2B servers on ports 810x, E4B on 820x —
# scripts/serve_gemma4_e4b.sh) and run both concurrently.
#
# Setup + smoke: RUN14_COLLAB.md (§7 for the E4B specifics). Run detached:
#   nohup setsid bash scripts/run15_e4b_b1_128.sh \
#     > benchmarking/mmau_pro/results/run15_gemma4e4b/sweep_e4b.log 2>&1 &
# Crash/interruption recovery: rerun the same command (resumable JSONL).
set -euo pipefail

# ============ ADAPT THESE TO YOUR MACHINE ============
PY="${PY:-$HOME/miniconda3/envs/epf/bin/python}"    # client env python
# one E4B vLLM server per GPU in THIS run's GPU set (see serve_gemma4_e4b.sh):
ENDPOINTS="${ENDPOINTS:-http://localhost:8200/v1,http://localhost:8201/v1}"
# =====================================================

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_NAME=gemma4-e4b
DATA_ROOT=$REPO/data/mmau_pro
IDS=$REPO/benchmarking/mmau_pro/results/run14_gemma4e2b/le30s_single_ids.txt  # committed
OUT=$REPO/benchmarking/mmau_pro/results/run15_gemma4e4b
JSONL=$OUT/epf_gemma4e4b_max12.jsonl
CSV=$OUT/epf_gemma4e4b_max12.csv
LOG=$OUT/epf_gemma4e4b_max12.log

[ -f "$IDS" ] || { echo "missing $IDS — pull the branch (the ids file is committed)"; exit 1; }
[ -f "$JSONL" ] || { echo "missing $JSONL — pull the branch (the b1-16 seed is committed)"; exit 1; }
mkdir -p "$OUT"
cd "$REPO"

# Budget-staged over one resumable JSONL (b1/8/16 arrive pre-completed from git —
# expect "0 to do (1947 resumed)" for those stages). Inflight rule (SETUP_GUIDE
# §7/§9): low at b1 (many DISTINCT audios encode at once), 64 mid, 128 for the
# big budgets (particles of one item share the cached audio prefix).
for B in 1 8 16 32 64 128; do
  INFLIGHT=64
  [ "$B" -eq 1 ] && INFLIGHT=24
  [ "$B" -ge 64 ] && INFLIGHT=128
  $PY -m benchmarking.mmau_pro.diversity_probe \
    --endpoints "$ENDPOINTS" --model-name "$MODEL_NAME" \
    --data-root "$DATA_ROOT" --subset test \
    --prompts 7 --signals mean_logprob,entropy --budgets "$B" \
    --temp 0.8 --ess-threshold 0.6 --early-phase 0.7 \
    --max-steps 12 --max-tokens-per-step 300 --tokens-per-step 30 \
    --ids-file "$IDS" --max-inflight "$INFLIGHT" \
    --jsonl "$JSONL" --csv "$CSV" --log "$LOG"
done
# (no bootstrap here — epf_bootstrap's budget axis is hardcoded to {1,8,16,32};
#  analysis happens after the JSONLs come back)

echo "DONE — send back: $JSONL  $CSV  $LOG"
