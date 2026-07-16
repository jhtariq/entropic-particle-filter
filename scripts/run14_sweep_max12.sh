#!/usr/bin/env bash
# Run 14b — max_steps=12 ABLATION of the Run-14 sweep (same grid, longer step budget).
# Answers: "are the results better with 12 thinking steps?" — esp. for P4, which
# truncates 86% of trajectories at max6 but only ~11% at 12.
#
# Run detached:
#   nohup setsid bash scripts/run14_sweep_max12.sh \
#     > benchmarking/mmau_pro/results/run14_gemma4e2b/sweep_stages_max12.log 2>&1 &
# JSONL seeded from epf_gemma_le30s.jsonl.max12_partial (1,514 bit-identical b1 rows
# from the aborted 2026-07-14 max12 launch). Launched CONCURRENTLY with the max6
# sweep's b32 stage (user decision) — per-cell timings of both runs are
# contention-polluted from that point on; do not use them as cost anchors.
set -euo pipefail

# ============ CONFIG (finalize after the probes) ============
REPO=/home/tariqvrh4/entropic-particle-filter
PY=/home/tariqvrh4/miniconda3/envs/epf/bin/python
MODEL_NAME=gemma4-e2b
ENDPOINTS=http://localhost:8100/v1,http://localhost:8101/v1
DATA_ROOT=$REPO/data/mmau_pro
OUT=$REPO/benchmarking/mmau_pro/results/run14_gemma4e2b

# Same config as run14_sweep.sh except MAX_STEPS=12 (coverage-based value:
# P4 ends naturally 89% by 12 / p95=13; P7 99% by 9 — step_probe_tok30_max12.log)
# and the _max12 output basenames.
PROMPT=4,7
STEP_ARGS="--tokens-per-step 30"
MAX_STEPS=12

# Single-audio only: the 243 multi-audio items were DROPPED after the stage-3b
# preflight showed Gemma 4 E2B multi-clip comprehension is unreliable (miscounts
# clips, confabulates descriptions, order-insensitive) despite correct plumbing.
IDS=$OUT/le30s_single_ids.txt       # 1,947 single-audio items, every clip <= 30 s
JSONL=$OUT/epf_gemma_le30s_max12.jsonl
CSV=$OUT/epf_gemma_le30s_max12.csv
LOG=$OUT/epf_gemma_le30s_max12.log
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
  --in "$CSV" --out "$OUT/epf_gemma_le30s_max12_bootstrap.html"

echo "DONE — artifacts in $OUT (single/multi slices: filter CSV on num_audio;"
echo "by-duration slices: join durations_test5090.csv on unique_id)"
