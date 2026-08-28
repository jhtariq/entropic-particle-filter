#!/bin/bash
# Full (non-smoke) STAR-Bench run: 4 models over the 1,453-item gradeable subset
# (Foundational Perception + Spatial Reasoning), 2 GPUs, 2 CONCURRENT lanes (2
# models running at a time), each lane pinned to one explicit GPU index + port.
#
# Must run inside a >=2-GPU allocation (this box: job 21528800, gpuA100x4 x2).
#
# Why pinned, not run_cell's auto NUM_GPUS loop: serve_one_vllm always computes
# `CUDA_VISIBLE_DEVICES=$gpu` from a loop starting at 0 — calling run_cell twice
# concurrently (once per lane) would send BOTH servers to GPU 0. run_cell_pinned
# below calls serve_one_<M_SERVE> directly with an explicit gpu index instead, and
# does its OWN scoped teardown (kill_servers globs *_gpu*.pid — shared across
# lanes — so it would tear down the OTHER lane's still-running server).
set -euo pipefail
export MEDIA_ROOT="/work/hdd/bcey/awaheed/its-for-audio-reasoning"
module load cuda-compat/13.0 2>/dev/null || true
GREEDY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$GREEDY_DIR/config.sh"
source "$GREEDY_DIR/lib.sh"
cd "$REPO_ROOT"

N_VISIBLE="$(nvidia-smi -L 2>/dev/null | wc -l)"
[ "$N_VISIBLE" -ge 2 ] || { echo "FATAL: run_star_bench_full.sh needs >=2 visible GPUs, got $N_VISIBLE" >&2; exit 1; }

run_cell_pinned() {  # run_cell_pinned <model_key> <bench_key> <gpu_index> — one explicit GPU/port
  local model="$1" bench="$2" gpu="$3"
  model_cfg "$model"
  bench_cfg "$bench"
  local port=$((BASE_PORT + gpu))
  local cell_dir="$OUT_ROOT/$model/$bench"
  mkdir -p "$cell_dir" "$OUT_ROOT/servers"

  if curl -s --max-time 2 "http://localhost:$port/v1/models" | grep -q '"id"'; then
    echo "FATAL: gpu$gpu: something already serving on :$port" >&2; return 1
  fi

  local runner_pid=0
  # shellcheck disable=SC2064 — intentional early expansion of $gpu/$port/model vars
  trap "[ \"\$runner_pid\" -gt 0 ] 2>/dev/null && kill -9 \$runner_pid 2>/dev/null
        [ -f '$OUT_ROOT/servers/${M_NAME}_gpu${gpu}.pid' ] && kill -9 \$(cat '$OUT_ROOT/servers/${M_NAME}_gpu${gpu}.pid') 2>/dev/null
        true" EXIT

  echo "gpu$gpu: serving $model ($M_ID) on :$port"
  "serve_one_${M_SERVE}" "$gpu"
  wait_healthy "$port" "$M_REQNAME"
  sanity_prompt "$port" "$M_REQNAME"
  local wav; wav="$(first_item_wav "$bench")"
  sanity_audio "$port" "$M_REQNAME" "$wav"

  bench_cfg "$bench"   # first_item_wav ran bench_cfg in between — keep vars coherent (mirrors run_cell)
  local endpoints="http://localhost:$port/v1"
  (cd "$REPO_ROOT" && exec "$EPF_PY" -m greedy.greedy_runner \
     --bench "$bench" --endpoints "$endpoints" --model-name "$M_REQNAME" \
     --data-root "$B_DATA_ROOT" --subset "$B_SUBSET" \
     --max-audios "$M_MAXAUDIOS" --max-tokens "$MAX_TOKENS" --max-inflight "$MAX_INFLIGHT" \
     --csv "$cell_dir/greedy.csv" --log "$cell_dir/greedy.log" --expected "$B_EXPECTED") &
  runner_pid=$!
  if ! wait "$runner_pid"; then
    runner_pid=0
    echo "gpu$gpu: cell $model x $bench: errors — one automatic resume retry"
    (cd "$REPO_ROOT" && exec "$EPF_PY" -m greedy.greedy_runner \
       --bench "$bench" --endpoints "$endpoints" --model-name "$M_REQNAME" \
       --data-root "$B_DATA_ROOT" --subset "$B_SUBSET" \
       --max-audios "$M_MAXAUDIOS" --max-tokens "$MAX_TOKENS" --max-inflight "$MAX_INFLIGHT" \
       --csv "$cell_dir/greedy.csv" --log "$cell_dir/greedy.log" --expected "$B_EXPECTED") &
    runner_pid=$!
    wait "$runner_pid" \
      || { runner_pid=0; echo "FATAL: gpu$gpu cell $model x $bench still failing after retry — see $cell_dir/greedy.log" >&2; return 1; }
  fi
  runner_pid=0

  # scoped teardown: THIS gpu's server only (kill_servers in lib.sh globs every
  # *_gpu*.pid in OUT_ROOT/servers, which would kill the OTHER concurrent lane)
  local pidf="$OUT_ROOT/servers/${M_NAME}_gpu${gpu}.pid"
  if [ -f "$pidf" ]; then
    local pid; pid="$(cat "$pidf")"
    kill "$pid" 2>/dev/null || true
    rm -f "$pidf"
    local waited=0
    while kill -0 "$pid" 2>/dev/null && [ "$waited" -lt 30 ]; do sleep 2; waited=$((waited + 2)); done
    kill -9 "$pid" 2>/dev/null || true
  fi
  sleep 10  # let GPU memory drain before this lane's next model loads
  trap - EXIT
  echo "gpu$gpu: cell $model x $bench COMPLETE -> $cell_dir/greedy.csv"
}

lane() {  # lane <gpu_index> <model1> <model2> — sequential within a lane, lanes run concurrently
  local gpu="$1" m1="$2" m2="$3"
  echo "LANE $gpu starting: $m1 -> $m2"
  run_cell_pinned "$m1" star_bench "$gpu" || { echo "LANE $gpu FAILED at $m1" >&2; return 1; }
  run_cell_pinned "$m2" star_bench "$gpu" || { echo "LANE $gpu FAILED at $m2" >&2; return 1; }
  echo "LANE $gpu COMPLETE ($m1, $m2)"
}

mkdir -p "$OUT_ROOT"
echo "=== STAR-BENCH FULL RUN starting: $(date -u +%FT%TZ) ==="
lane 0 qwen_omni_3b qwen2_audio > "$OUT_ROOT/lane0.log" 2>&1 &
PID0=$!
lane 1 qwen_omni_7b phi4mm      > "$OUT_ROOT/lane1.log" 2>&1 &
PID1=$!

status=0
wait "$PID0" || { echo "lane 0 exited non-zero — see $OUT_ROOT/lane0.log" >&2; status=1; }
wait "$PID1" || { echo "lane 1 exited non-zero — see $OUT_ROOT/lane1.log" >&2; status=1; }

echo "=== STAR-BENCH FULL RUN finished: $(date -u +%FT%TZ) (status=$status) ==="
{
  echo "STAR-Bench greedy no-CoT (temp 0) — 4 models x 1,453 items, generated $(date -u +%FT%TZ)"
  for m in qwen_omni_3b qwen_omni_7b qwen2_audio phi4mm; do
    echo ""
    echo "--- $m ---"
    cat "$OUT_ROOT/$m/star_bench/greedy.log" 2>/dev/null || echo "(no log — cell did not complete)"
  done
} | tee "$OUT_ROOT/star_bench_summary.txt"

exit $status
