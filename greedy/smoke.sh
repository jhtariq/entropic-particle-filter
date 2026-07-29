#!/bin/bash
# Smoke ALL 18 cells before burning GPU-days: Phase A offline sanity, then for
# each of the six models serve once (one GPU) and run a 3-item greedy mini-run on
# EACH of the three benchmarks (18 smoke cells, each gated on 3 clean rows), plus
# targeted stress items for the three capability-limit assumptions:
#   qwen2_audio × >30 s clips  -> the encoder must TRUNCATE, not 400
#   mellow      × 3-audio item -> the runner clamp to 2 slots must hold
#   kimi_audio  × 2-audio item -> the runner clamp to 1 audio must hold
# Any failure aborts (set -e) — run_all.sh will not start the full cells.
set -euo pipefail
GREEDY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$GREEDY_DIR/config.sh"
source "$GREEDY_DIR/lib.sh"
cd "$REPO_ROOT"

# one server is enough to smoke; SMOKE_ALL_GPUS=1 to exercise every GPU instead
[ "${SMOKE_ALL_GPUS:-0}" = "1" ] || NUM_GPUS=1

echo "=== SMOKE Phase A: offline checks ==="
"$EPF_PY" -m pytest tests/ -q --ignore=tests/e2e

"$EPF_PY" - "$DATA_TESTMINI" "$DATA_AUDIO" "$DATA_MMAR" "$DATA_MMSU" <<'PYEOF'
import sys
from benchmarking.mmau_pro.loader import load_mmau_mcq
from benchmarking.mmar.loader import load_mmar_mcq
from benchmarking.mmsu.loader import load_mmsu_mcq
tm, au, mmar, mmsu = sys.argv[1:5]
assert len(load_mmau_mcq(tm, subset="test", audio_root=au)) == 5090
assert len(load_mmar_mcq(mmar, subset="full")) == 1000
assert len(load_mmsu_mcq(mmsu, subset="full")) == 5000
print("loaders OK: 5090 / 1000 / 5000")
PYEOF

for m in $GREEDY_MODELS; do check_model "$m"; done
echo "=== SMOKE Phase A OK ==="

smoke_cell() {  # smoke_cell <model> <bench> <label> [extra runner args...] — fresh mini-run each time
  local model="$1" bench="$2" label="$3"; shift 3
  local dir="$OUT_ROOT/smoke/$model/$label"
  mkdir -p "$dir"
  rm -f "$dir/greedy.csv" "$dir/greedy.log"   # no stale resume — every smoke is a REAL test
  bench_cfg "$bench"
  run_greedy "$bench" "$dir/greedy.csv" "$dir/greedy.log" "$@"
}

trap 'kill_servers' EXIT
echo "=== SMOKE Phase B: 18 served mini-cells (1 GPU) ==="
for m in $GREEDY_MODELS; do
  echo "--- smoke: $m"
  model_cfg "$m"
  ports_must_be_free
  serve_model "$m"
  wait_healthy "$BASE_PORT" "$M_REQNAME"
  sanity_prompt "$BASE_PORT" "$M_REQNAME"
  wav="$(first_item_wav mmau)"
  sanity_audio "$BASE_PORT" "$M_REQNAME" "$wav"

  for b in $GREEDY_BENCHES; do
    model_cfg "$m"           # re-set M_* (bench_cfg/first_item_wav ran in between)
    smoke_cell "$m" "$b" "$b" --limit 3
    echo "smoke $m x $b OK (3 items clean)"
  done

  # capability-limit stress items (all on the mmau bench)
  model_cfg "$m"
  case "$m" in
    qwen2_audio)
      smoke_cell "$m" mmau mmau_long --ids "$SMOKE_Q2A_LONG_IDS"
      echo "smoke $m stress OK (>30 s clips truncate, no 400)" ;;
    mellow)
      smoke_cell "$m" mmau mmau_3a   --ids "$SMOKE_3AUDIO_ID"
      echo "smoke $m stress OK (3-audio item clamped to 2 slots)" ;;
    kimi_audio)
      smoke_cell "$m" mmau mmau_2a   --ids "$SMOKE_2AUDIO_ID"
      echo "smoke $m stress OK (2-audio item clamped to 1)" ;;
  esac

  kill_servers
done
trap - EXIT
echo "=== SMOKE COMPLETE: all 18 cells + stress items clean ==="
