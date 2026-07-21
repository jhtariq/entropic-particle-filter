#!/bin/bash
# THE single command: run the Kimi-Audio EPF grid (prompts × signals × budgets on
# the 1,947-MCQ single-audio ≤30 s test subset) and build the bootstrap HTML.
# Fully resumable — on crash/interrupt, re-run this same script; completed cells
# are skipped via the probe's JSONL resume.
#
#   nohup bash benchmarking/mmau_pro/run19/run_all.sh > run19.log 2>&1 &
#
# Output: $OUT_ROOT/run19/$STEM.{jsonl,csv,log} + $OUT_ROOT/epf_kimi_bootstrap.html
# (a NEW file). DRY_RUN=1 prints the plan without executing.
set -euo pipefail
RUN19_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUN19_DIR/config.sh"
source "$RUN19_DIR/lib.sh"
cd "$REPO_ROOT"
mkdir -p "$OUT_ROOT/servers"
SUMMARY="$OUT_ROOT/summary.txt"
echo "=== RUN19 start $(date '+%F %T') | GPUs=$NUM_GPUS | budgets=$BUDGETS | prompts=$PROMPTS_RUN19 | subset=$RUN19_SUBSET maxlen=$RUN19_MAXLEN ===" | tee -a "$SUMMARY"

probe_stage() {  # probe_stage <budget> <max_inflight> — uses the run_cfg vars
  local extra=()
  [ -n "$STOP_REGEX" ] && extra+=(--stop-regex "$STOP_REGEX")
  "$EPF_PY" -m benchmarking.mmau_pro.diversity_probe \
    --endpoints "$(endpoints_csv)" --model-name "$NAME" \
    --data-root "$DATA_TESTMINI" --subset "$SUBSET" --audio-root "$DATA_AUDIO" \
    --prompts "$PROMPTS" --signals mean_logprob,entropy --budgets "$1" \
    --select all --limit "$PROBE_LIMIT" --temp 0.8 --ess-threshold 0.6 --early-phase 0.7 \
    --max-inflight "$2" "${extra[@]}" \
    --jsonl "$RUN_DIR/$STEM.jsonl" --csv "$RUN_DIR/$STEM.csv" --log "$RUN_DIR/$STEM.log"
}

run_cfg run19
mkdir -p "$RUN_DIR"
# the local b1-16 rows ship as a gzipped seed: budget-extension boxes resume from
# them so only the new budget cells run (no seed file -> fresh start)
[ -e "$RUN19_DIR/seeds/$STEM.seed.jsonl.gz" ] \
  && unpack_seed "$RUN19_DIR/seeds/$STEM.seed.jsonl.gz" "$RUN_DIR/$STEM.jsonl"
if [ "$DRY_RUN" = "1" ]; then
  echo "DRY: run19 -> model=$MODEL_ID@${REV:0:12} name=$NAME subset=$SUBSET" \
       "prompts=$PROMPTS budgets=[$BUDGETS] endpoints=$(endpoints_csv) out=$RUN_DIR/$STEM.*"
  echo "DRY RUN complete — no commands executed"
  exit 0
fi

ensure_model "$RUN19_MODEL_ID" "$RUN19_REV"

for i in $(seq 0 $((NUM_GPUS - 1))); do serve_one "$i"; done
for i in $(seq 0 $((NUM_GPUS - 1))); do wait_healthy $((BASE_PORT + i)) 1800; done
for i in $(seq 0 $((NUM_GPUS - 1))); do
  sanity_prompt $((BASE_PORT + i)) || { kill_servers; exit 1; }
  gate_endpoint $((BASE_PORT + i)) || { kill_servers; exit 1; }
done

for B in $BUDGETS; do
  MI="$MAX_INFLIGHT"; [ "$B" -eq 1 ] && MI="$MAX_INFLIGHT_B1"
  t0=$(date +%s)
  probe_stage "$B" "$MI"
  if ! "$EPF_PY" "$RUN19_DIR/summarize_errors.py" "$RUN_DIR/$STEM.jsonl" --min-budget "$B"; then
    echo "run19 b$B: errors detected — one automatic resume retry" | tee -a "$SUMMARY"
    probe_stage "$B" "$MI"
    "$EPF_PY" "$RUN19_DIR/summarize_errors.py" "$RUN_DIR/$STEM.jsonl" --min-budget "$B" \
      || echo "WARNING: run19 b$B still has errors after retry — check $RUN_DIR/$STEM.log and server logs" | tee -a "$SUMMARY"
  fi
  echo "run19 b$B stage done in $(( ($(date +%s) - t0) / 60 )) min" | tee -a "$SUMMARY"
done
kill_servers

[ -e "$RUN_DIR/$STEM.csv" ] || { echo "FATAL: no CSV found for the report" >&2; exit 1; }
"$EPF_PY" -m benchmarking.mmau_pro.epf_bootstrap --n "$N_BOOT" \
  --in "Kimi-Audio-7B-Instruct — Run 19 ($RUN19_SUBSET, 1,947 MCQ)=$RUN_DIR/$STEM.csv" \
  --out "$OUT_ROOT/epf_kimi_bootstrap.html"

echo "=== RUN19 COMPLETE $(date '+%F %T') -> $OUT_ROOT/epf_kimi_bootstrap.html ===" | tee -a "$SUMMARY"
