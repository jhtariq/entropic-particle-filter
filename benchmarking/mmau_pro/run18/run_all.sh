#!/bin/bash
# THE single command: run the Phi-4-multimodal EPF grid (prompts × signals × budgets
# on the FULL 5,090-MCQ test set) and build the bootstrap HTML. Fully resumable — on
# crash/interrupt, re-run this same script; completed cells are skipped via the
# probe's JSONL resume.
#
#   nohup bash benchmarking/mmau_pro/run18/run_all.sh > run18.log 2>&1 &
#
# Output: $OUT_ROOT/run18/$STEM.{jsonl,csv,log} + $OUT_ROOT/epf_phi4mm_bootstrap.html
# (a NEW file). DRY_RUN=1 prints the plan without executing.
set -euo pipefail
RUN18_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUN18_DIR/config.sh"
source "$RUN18_DIR/lib.sh"
cd "$REPO_ROOT"
mkdir -p "$OUT_ROOT/servers"
SUMMARY="$OUT_ROOT/summary.txt"
echo "=== RUN18 start $(date '+%F %T') | GPUs=$NUM_GPUS | budgets=$BUDGETS | prompts=$PROMPTS_RUN18 | mode=$SERVE_MODE maxlen=$RUN18_MAXLEN ===" | tee -a "$SUMMARY"

probe_stage() {  # probe_stage <budget> <max_inflight> — uses the run_cfg vars
  # model-name MUST be probe_model_name (the adapter in lora mode) — naming the base
  # against a lora-mode server is valid but silently skips the speech adapter
  local extra=()
  [ -n "$STOP_REGEX" ] && extra+=(--stop-regex "$STOP_REGEX")
  "$EPF_PY" -m benchmarking.mmau_pro.diversity_probe \
    --endpoints "$(endpoints_csv)" --model-name "$(probe_model_name)" \
    --data-root "$DATA_TESTMINI" --subset "$SUBSET" --audio-root "$DATA_AUDIO" \
    --prompts "$PROMPTS" --signals mean_logprob,entropy --budgets "$1" \
    --select all --limit "$PROBE_LIMIT" --temp 0.8 --ess-threshold 0.6 --early-phase 0.7 \
    --max-inflight "$2" "${extra[@]}" \
    --jsonl "$RUN_DIR/$STEM.jsonl" --csv "$RUN_DIR/$STEM.csv" --log "$RUN_DIR/$STEM.log"
}

run_cfg run18
mkdir -p "$RUN_DIR"
# the local b1-16 rows ship as a gzipped seed: budget-extension boxes resume from
# them so only the new budget cells run (no seed file -> fresh start)
[ -e "$RUN18_DIR/seeds/$STEM.seed.jsonl.gz" ] \
  && unpack_seed "$RUN18_DIR/seeds/$STEM.seed.jsonl.gz" "$RUN_DIR/$STEM.jsonl"
if [ "$DRY_RUN" = "1" ]; then
  echo "DRY: run18 -> model=$MODEL_ID@${REV:0:12} (mode=$SERVE_MODE) name=$NAME probe-as=$(probe_model_name)" \
       "subset=$SUBSET prompts=$PROMPTS budgets=[$BUDGETS] endpoints=$(endpoints_csv) out=$RUN_DIR/$STEM.*"
  echo "DRY RUN complete — no commands executed"
  exit 0
fi

ensure_model "$RUN18_MODEL_ID" "$RUN18_REV"
check_merged

for i in $(seq 0 $((NUM_GPUS - 1))); do serve_one "$i"; done
for i in $(seq 0 $((NUM_GPUS - 1))); do wait_healthy $((BASE_PORT + i)) 1800; done
for i in $(seq 0 $((NUM_GPUS - 1))); do
  gate_endpoint $((BASE_PORT + i)) || { kill_servers; exit 1; }
done

for B in $BUDGETS; do
  MI="$MAX_INFLIGHT"; [ "$B" -eq 1 ] && MI="$MAX_INFLIGHT_B1"
  t0=$(date +%s)
  probe_stage "$B" "$MI"
  if ! "$EPF_PY" "$RUN18_DIR/summarize_errors.py" "$RUN_DIR/$STEM.jsonl" --min-budget "$B"; then
    echo "run18 b$B: errors detected — one automatic resume retry" | tee -a "$SUMMARY"
    probe_stage "$B" "$MI"
    "$EPF_PY" "$RUN18_DIR/summarize_errors.py" "$RUN_DIR/$STEM.jsonl" --min-budget "$B" \
      || echo "WARNING: run18 b$B still has errors after retry — check $RUN_DIR/$STEM.log and server logs" | tee -a "$SUMMARY"
  fi
  echo "run18 b$B stage done in $(( ($(date +%s) - t0) / 60 )) min" | tee -a "$SUMMARY"
done
kill_servers

[ -e "$RUN_DIR/$STEM.csv" ] || { echo "FATAL: no CSV found for the report" >&2; exit 1; }
case "$SERVE_MODE" in
  lora)   MODE_LABEL="speech-LoRA" ;;
  merged) MODE_LABEL="speech-LoRA merged" ;;
  base)   MODE_LABEL="base, NO speech-LoRA" ;;
esac
"$EPF_PY" -m benchmarking.mmau_pro.epf_bootstrap --n "$N_BOOT" \
  --in "Phi-4-multimodal-instruct (5.6B, $MODE_LABEL) — Run 18 (test, 5,090 MCQ)=$RUN_DIR/$STEM.csv" \
  --out "$OUT_ROOT/epf_phi4mm_bootstrap.html"

echo "=== RUN18 COMPLETE $(date '+%F %T') -> $OUT_ROOT/epf_phi4mm_bootstrap.html ===" | tee -a "$SUMMARY"
