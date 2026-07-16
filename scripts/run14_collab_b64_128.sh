#!/usr/bin/env bash
# Run 14c (COLLABORATOR) — extend the Gemma 4 E2B EPF grid to budgets 64 and 128.
# Config is fixed to match the local max_steps=12 run (epf_gemma_le30s_max12.*):
# prompts {4,7}, signals {mean_logprob, entropy}, tok30 x 12 steps, temp 0.8,
# ess 0.6 / early 0.7, the committed 1,947-item <=30s single-audio ids file.
# Fresh JSONL — no overlap with the local b1-32 keys; send the three output files back.
#
# Setup + smoke: see RUN14_COLLAB.md at the repo root. Run detached:
#   nohup setsid bash scripts/run14_collab_b64_128.sh \
#     > benchmarking/mmau_pro/results/run14_gemma4e2b/sweep_b64128.log 2>&1 &
# Crash/interruption recovery: rerun the same command (resumable JSONL).
set -euo pipefail

# ============ ADAPT THESE THREE TO YOUR MACHINE ============
PY="${PY:-$HOME/miniconda3/envs/epf/bin/python}"   # client env python (see RUN14_COLLAB.md §1)
# one vLLM server per GPU, comma-listed (see RUN14_COLLAB.md §4):
ENDPOINTS="${ENDPOINTS:-http://localhost:8100/v1,http://localhost:8101/v1}"
MAX_INFLIGHT="${MAX_INFLIGHT:-128}"                 # per-endpoint request target
# ===========================================================

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_NAME=gemma4-e2b
DATA_ROOT=$REPO/data/mmau_pro
OUT=$REPO/benchmarking/mmau_pro/results/run14_gemma4e2b
IDS=$OUT/le30s_single_ids.txt                       # committed to git — 1,947 ids
JSONL=$OUT/epf_gemma_le30s_max12_b64128.jsonl
CSV=$OUT/epf_gemma_le30s_max12_b64128.csv
LOG=$OUT/epf_gemma_le30s_max12_b64128.log

[ -f "$IDS" ] || { echo "missing $IDS — pull the branch (the ids file is committed)"; exit 1; }
mkdir -p "$OUT"
cd "$REPO"

for B in 64 128; do
  $PY -m benchmarking.mmau_pro.diversity_probe \
    --endpoints "$ENDPOINTS" --model-name "$MODEL_NAME" \
    --data-root "$DATA_ROOT" --subset test \
    --prompts 4,7 --signals mean_logprob,entropy --budgets "$B" \
    --temp 0.8 --ess-threshold 0.6 --early-phase 0.7 \
    --max-steps 12 --max-tokens-per-step 300 --tokens-per-step 30 \
    --ids-file "$IDS" --max-inflight "$MAX_INFLIGHT" \
    --jsonl "$JSONL" --csv "$CSV" --log "$LOG"
done
# (no bootstrap here — epf_bootstrap's budget axis is hardcoded to {1,8,16,32};
#  analysis happens after merging with the local b1-32 JSONL)

echo "DONE — send back: $JSONL  $CSV  $LOG"
