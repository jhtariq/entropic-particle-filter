#!/bin/bash
# Smoke the 4 target models x star_bench before burning GPU-time on the real 1,453-item
# run. This mirrors greedy/smoke.sh's pattern (offline checks, then one served mini-run
# per model) but is scoped to ONLY qwen_omni_3b/7b, qwen2_audio, phi4mm x star_bench —
# the greedy/ package has never been executed on this box (unlike mmar_sc/), so this is
# the first real end-to-end proof it works here, not just an add-on sanity pass.
set -euo pipefail
export MEDIA_ROOT="/work/hdd/bcey/awaheed/its-for-audio-reasoning"
module load cuda-compat/13.0 2>/dev/null || true
GREEDY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$GREEDY_DIR/config.sh"
source "$GREEDY_DIR/lib.sh"
cd "$REPO_ROOT"

STAR_BENCH_MODELS="qwen_omni_3b qwen_omni_7b qwen2_audio phi4mm"
NUM_GPUS=1   # one server is enough to smoke

echo "=== SMOKE Phase A: offline checks ==="
"$EPF_PY" -m pytest tests/test_star_bench.py -q
for m in $STAR_BENCH_MODELS; do check_model "$m"; done
echo "=== SMOKE Phase A OK ==="

smoke_cell() {  # smoke_cell <model> <label> [extra runner args...] — fresh mini-run each time
  local model="$1" label="$2"; shift 2
  local dir="$OUT_ROOT/smoke/$model/$label"
  mkdir -p "$dir"
  rm -f "$dir/greedy.csv" "$dir/greedy.log"   # no stale resume — every smoke is a REAL test
  bench_cfg star_bench
  run_greedy star_bench "$dir/greedy.csv" "$dir/greedy.log" "$@"
}

trap 'kill_servers' EXIT
echo "=== SMOKE Phase B: 4 served mini-cells (1 GPU, star_bench, limit=3/file -> 6 items) ==="
for m in $STAR_BENCH_MODELS; do
  echo "--- smoke: $m"
  model_cfg "$m"
  ports_must_be_free
  serve_model "$m"
  wait_healthy "$BASE_PORT" "$M_REQNAME"
  sanity_prompt "$BASE_PORT" "$M_REQNAME"
  wav="$(first_item_wav star_bench)"
  sanity_audio "$BASE_PORT" "$M_REQNAME" "$wav"

  model_cfg "$m"   # re-set M_* (bench_cfg/first_item_wav ran in between)
  smoke_cell "$m" star_bench --limit 3
  echo "smoke $m x star_bench OK (6 items clean: 3 perception + 3 spatial)"

  kill_servers
done
trap - EXIT
echo "=== SMOKE COMPLETE: all 4 model x star_bench cells clean ==="
