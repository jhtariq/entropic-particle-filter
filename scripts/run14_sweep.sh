#!/usr/bin/env bash
# Run 14 — the Gemma 4 E2B EPF grid on the <=30 s MMAU-Pro subset (2,190 items).
#
# Prerequisites (all verified by the preflight, RESULTS.md §20):
#   * both servers up:  scripts/serve_gemma4_e2b.sh 0  and  ... 1   (ports 8100/8101)
#   * ids files exist:  python -m benchmarking.mmau_pro.make_le30s_ids --data-root data/mmau_pro
#   * PROMPT / STEP_ARGS / MAX_STEPS below filled in from the probe results
#
# Run detached:
#   nohup setsid bash scripts/run14_sweep.sh \
#     > benchmarking/mmau_pro/results/run14_gemma4e2b/sweep_stages.log 2>&1 &
# Monitor: watch the log for "done in ... errors" cell lines; a block of identical
# ClientConnectorError rows = an endpoint died -> restart the server, rerun this same
# script (the JSONL is resumable; errored rows are retried automatically).
set -euo pipefail

# ============ CONFIG (finalize after the probes) ============
REPO=/home/tariqvrh4/entropic-particle-filter
PY=/home/tariqvrh4/miniconda3/envs/epf/bin/python
MODEL_NAME=gemma4-e2b
ENDPOINTS=http://localhost:8100/v1,http://localhost:8101/v1
DATA_ROOT=$REPO/data/mmau_pro
OUT=$REPO/benchmarking/mmau_pro/results/run14_gemma4e2b

# Decided from the Run-14 probes (RESULTS.md §20; max_steps revised 2026-07-14):
#   PROMPT 4,7      user pick from probe100_t00/t08 (P4 best greedy + most chunkable;
#                   P7 best t=0.8) — both swept
#   tok30 stepping  '\n\n' stepping is DEGENERATE on Gemma (1-token steps, 67-78%
#                   unparsed — step_probe_dnl_max6.log); tok30 is healthy
#   MAX_STEPS 6     user decision: exact Run-13 step-config parity (tok30 x 6 =
#                   180-token budget). Known tradeoff (step_probe_tok30_max12):
#                   P7 fits 83% of trajectories in 6 steps; P4 only 14% — P4 cells
#                   measure a truncated-CoT regime. (A max12 partial run of 1,514
#                   b1 rows was archived to epf_gemma_le30s.jsonl.max12_partial.)
PROMPT=4,7
STEP_ARGS="--tokens-per-step 30"
MAX_STEPS=6

# Single-audio only: the 243 multi-audio items were DROPPED after the stage-3b
# preflight showed Gemma 4 E2B multi-clip comprehension is unreliable (miscounts
# clips, confabulates descriptions, order-insensitive) despite correct plumbing.
IDS=$OUT/le30s_single_ids.txt       # 1,947 single-audio items, every clip <= 30 s
JSONL=$OUT/epf_gemma_le30s.jsonl
CSV=$OUT/epf_gemma_le30s.csv
LOG=$OUT/epf_gemma_le30s.log
# ============================================================

[ -f "$IDS" ] || { echo "missing $IDS — run make_le30s_ids first"; exit 1; }
cd "$REPO"

# Budget-staged (b1 -> b8 -> b16 -> b32) over one resumable JSONL; parity with
# Runs 11/13: temp 0.8, ess 0.6 / early 0.7, systematic + logit (hardcoded in
# diversity_probe.build_epf), both signals. Inflight 24 at b1 (many distinct
# audios encode at once), 64 otherwise (particles share the cached audio prefix).
for B in 1 8 16 32; do
  INFLIGHT=64; [ "$B" -eq 1 ] && INFLIGHT=24
  $PY -m benchmarking.mmau_pro.diversity_probe \
    --endpoints "$ENDPOINTS" --model-name "$MODEL_NAME" \
    --data-root "$DATA_ROOT" --subset test \
    --prompts "$PROMPT" --signals mean_logprob,entropy --budgets "$B" \
    --temp 0.8 --ess-threshold 0.6 --early-phase 0.7 \
    --max-steps "$MAX_STEPS" --max-tokens-per-step 300 $STEP_ARGS \
    --ids-file "$IDS" --max-inflight "$INFLIGHT" \
    --jsonl "$JSONL" --csv "$CSV" --log "$LOG"
done

# Error bars (std100 + SE_full). Note: epf_bootstrap's HTML only renders prompts
# {4,5,7,9} — fine here since PROMPT is one of them.
$PY -m benchmarking.mmau_pro.epf_bootstrap --n 10000 --subsample 100 --seed 0 \
  --in "$CSV" --out "$OUT/epf_gemma_le30s_bootstrap.html"

echo "DONE — artifacts in $OUT (single/multi slices: filter CSV on num_audio;"
echo "by-duration slices: join durations_test5090.csv on unique_id)"
