#!/bin/bash
# THE single command for the greedy baseline: provision everything on a fresh
# machine (env + 6 pinned checkpoints + 3 byte-verified datasets), smoke ALL 18
# cells, then run the 18 full cells one by one — greedy (temp 0, no CoT) inference
# for every model on every benchmark. Aborts IMMEDIATELY on the first failure.
# Fully resumable — on crash/interrupt, re-run this same script; provisioning is
# verify-first and completed items are skipped via each cell's CSV resume.
#
#   nohup bash greedy/run_all.sh > greedy_run.log 2>&1 &
#
# Output: greedy/results/<model>/<bench>/greedy.{csv,log} + results/summary.txt
# + results/summary_table.txt. DRY_RUN=1 prints the 18-cell plan; SMOKE=0 skips
# the smoke phase (not recommended).
set -euo pipefail
GREEDY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$GREEDY_DIR/config.sh"
source "$GREEDY_DIR/lib.sh"
cd "$REPO_ROOT"
mkdir -p "$OUT_ROOT/servers"
SUMMARY="$OUT_ROOT/summary.txt"

if [ "$DRY_RUN" = "1" ]; then
  echo "DRY: greedy plan — GPUs=$NUM_GPUS endpoints=$(endpoints_csv) out=$OUT_ROOT"
  for m in $GREEDY_MODELS; do
    for b in $GREEDY_BENCHES; do
      run_cell "$m" "$b"     # DRY_RUN short-circuits inside run_cell
    done
  done
  echo "DRY RUN complete — no commands executed"
  exit 0
fi

echo "=== GREEDY start $(date '+%F %T') | GPUs=$NUM_GPUS | models=[$GREEDY_MODELS] | benches=[$GREEDY_BENCHES] ===" | tee -a "$SUMMARY"
trap 'kill_servers' EXIT   # safety net: tear down whatever a dying phase left behind

echo "=== Phase 1a: environment ===" | tee -a "$SUMMARY"
bash "$GREEDY_DIR/setup_env.sh"
echo "=== Phase 1b: model checkpoints ===" | tee -a "$SUMMARY"
bash "$GREEDY_DIR/fetch_models.sh"
echo "=== Phase 1c: datasets ===" | tee -a "$SUMMARY"
bash "$GREEDY_DIR/fetch_data.sh"

if [ "$SMOKE" = "1" ]; then
  echo "=== Phase 1.5: smoke (all 18 cells, 3 items each) ===" | tee -a "$SUMMARY"
  bash "$GREEDY_DIR/smoke.sh"
else
  echo "=== Phase 1.5: smoke SKIPPED (SMOKE=0) ===" | tee -a "$SUMMARY"
fi

echo "=== Phase 2: the 18 full cells ===" | tee -a "$SUMMARY"
for m in $GREEDY_MODELS; do
  for b in $GREEDY_BENCHES; do
    t0=$(date +%s)
    echo "--- cell $m x $b start $(date '+%F %T')" | tee -a "$SUMMARY"
    bash "$GREEDY_DIR/cells/run_${m}_${b}.sh"
    echo "--- cell $m x $b done in $(( ($(date +%s) - t0) / 60 )) min" | tee -a "$SUMMARY"
  done
done

echo "=== Phase 3: summary table ===" | tee -a "$SUMMARY"
"$EPF_PY" -m greedy.summarize --results "$OUT_ROOT" | tee "$OUT_ROOT/summary_table.txt"

echo "=== GREEDY COMPLETE $(date '+%F %T') -> $OUT_ROOT ===" | tee -a "$SUMMARY"
