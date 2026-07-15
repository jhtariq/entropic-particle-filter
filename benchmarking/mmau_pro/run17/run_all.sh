#!/bin/bash
# THE single command: run the Mellow-v0 EPF grid (prompts × signals × budgets on
# the test_no3a subset) and build the bootstrap HTML. Fully resumable — on
# crash/interrupt, re-run this same script; completed cells are skipped via the
# probe's JSONL resume.
#
#   nohup bash benchmarking/mmau_pro/run17/run_all.sh > run17.log 2>&1 &
#
# Output: $OUT_ROOT/run17/$STEM.{jsonl,csv,log} + $OUT_ROOT/epf_mellow_bootstrap.html
# (a NEW file). DRY_RUN=1 prints the plan without executing.
set -euo pipefail
RUN17_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUN17_DIR/config.sh"
source "$RUN17_DIR/lib.sh"
cd "$REPO_ROOT"
mkdir -p "$OUT_ROOT/servers"
SUMMARY="$OUT_ROOT/summary.txt"
echo "=== RUN17 start $(date '+%F %T') | GPUs=$NUM_GPUS | budgets=$BUDGETS | prompts=$PROMPTS_RUN17 | variant=$MELLOW_VARIANT text_tokens=$MELLOW_MAX_TEXT_TOKENS fill=$SINGLE_AUDIO_FILL ===" | tee -a "$SUMMARY"

probe_stage() {  # probe_stage <budget> <max_inflight> — uses the run_cfg vars
  "$EPF_PY" -m benchmarking.mmau_pro.diversity_probe \
    --endpoints "$(endpoints_csv)" --model-name "$NAME" \
    --data-root "$DATA_TESTMINI" --subset "$SUBSET" --audio-root "$DATA_AUDIO" \
    --prompts "$PROMPTS" --signals mean_logprob,entropy --budgets "$1" \
    --select all --limit "$PROBE_LIMIT" --temp 0.8 --ess-threshold 0.6 --early-phase 0.7 \
    --stop-regex "$STOP_REGEX" --max-inflight "$2" \
    --jsonl "$RUN_DIR/$STEM.jsonl" --csv "$RUN_DIR/$STEM.csv" --log "$RUN_DIR/$STEM.log"
}

run_cfg run17
mkdir -p "$RUN_DIR"
# the local b1-16 rows ship as a gzipped seed: budget-extension boxes resume from
# them so only the new budget cells run (no seed file -> fresh start, e.g. locally)
[ -e "$RUN17_DIR/seeds/$STEM.seed.jsonl.gz" ] \
  && unpack_seed "$RUN17_DIR/seeds/$STEM.seed.jsonl.gz" "$RUN_DIR/$STEM.jsonl"
if [ "$DRY_RUN" = "1" ]; then
  echo "DRY: run17 -> model=$MODEL_ID@${REV:0:12} ($MELLOW_VARIANT) name=$NAME subset=$SUBSET" \
       "prompts=$PROMPTS budgets=[$BUDGETS] endpoints=$(endpoints_csv) out=$RUN_DIR/$STEM.*"
  echo "DRY RUN complete — no commands executed"
  exit 0
fi

ensure_mellow

for i in $(seq 0 $((NUM_GPUS - 1))); do serve_one "$i"; done
for i in $(seq 0 $((NUM_GPUS - 1))); do wait_healthy $((BASE_PORT + i)) 600; done
for i in $(seq 0 $((NUM_GPUS - 1))); do
  gate_endpoint $((BASE_PORT + i)) || { kill_servers; exit 1; }
done

for B in $BUDGETS; do
  MI="$MAX_INFLIGHT"; [ "$B" -eq 1 ] && MI="$MAX_INFLIGHT_B1"
  t0=$(date +%s)
  probe_stage "$B" "$MI"
  if ! "$EPF_PY" "$RUN17_DIR/summarize_errors.py" "$RUN_DIR/$STEM.jsonl" --min-budget "$B"; then
    echo "run17 b$B: errors detected — one automatic resume retry" | tee -a "$SUMMARY"
    probe_stage "$B" "$MI"
    "$EPF_PY" "$RUN17_DIR/summarize_errors.py" "$RUN_DIR/$STEM.jsonl" --min-budget "$B" \
      || echo "WARNING: run17 b$B still has errors after retry — check $RUN_DIR/$STEM.log and server logs" | tee -a "$SUMMARY"
  fi
  echo "run17 b$B stage done in $(( ($(date +%s) - t0) / 60 )) min" | tee -a "$SUMMARY"
done
kill_servers

[ -e "$RUN_DIR/$STEM.csv" ] || { echo "FATAL: no CSV found for the report" >&2; exit 1; }
# two-section report: FULL (5,073 test_no3a) + the 326-item fully-heard LE10S slice
# (a plain row filter of the full CSV). Each section = one interactive plot with the
# selected / oracle / majority curves for all 4 prompt×signal lines.
"$EPF_PY" "$RUN17_DIR/filter_csv_le10s.py" \
  --in "$RUN_DIR/$STEM.csv" --out "$RUN_DIR/epf_mellow_le10s.csv" \
  --le10s-parquet "$DATA_TESTMINI/data/test_le10s_flat-00000-of-00001.parquet"
"$EPF_PY" -m benchmarking.mmau_pro.epf_bootstrap --n "$N_BOOT" \
  --in "Mellow-${MELLOW_VARIANT} (167M) — Run 17 FULL (test_no3a, 5,073 MCQ)=$RUN_DIR/$STEM.csv" \
  --in "Mellow-${MELLOW_VARIANT} (167M) — Run 17 LE10S (326 fully-heard)=$RUN_DIR/epf_mellow_le10s.csv" \
  --out "$OUT_ROOT/epf_mellow_bootstrap.html"

echo "=== RUN17 COMPLETE $(date '+%F %T') -> $OUT_ROOT/epf_mellow_bootstrap.html ===" | tee -a "$SUMMARY"
