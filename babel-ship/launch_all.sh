#!/usr/bin/env bash
# Run all five MMAU-Pro d1k process-reward sweeps on one 4x A100-80GB node.
#
# Layout (judge is never the bottleneck - one judge feeds all three slots):
#   GPU 0: judge (Qwen3-Omni-30B, single GPU)
#   GPU 1: kimi-audio                 (longest single sweep, ~13h)
#   GPU 2: qwen-omni  -> qwen-omni-3b
#   GPU 3: phi4mm     -> qwen2-audio
# Wall clock is bounded by the kimi sweep; everything lands in roughly a day.
#
# Requires: env activated, BABEL_D1K_ROOT set, sanity_check_models.py and
# sanity_check_data.py both green. Rerunning after a crash is safe (resume).
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p out

if [[ -z "${BABEL_D1K_ROOT:-}" ]]; then
  echo "set BABEL_D1K_ROOT to the local mmau_pro_d1k directory" >&2
  exit 1
fi

nvidia-smi > out/nvidia-smi.txt
python -m pip freeze > out/pip-freeze.txt

nohup bash serve_judge.sh 0 > out/serve_judge.log 2>&1 &
judge_pid=$!
echo "judge starting on GPU 0 (pid $judge_pid, port ${JUDGE_PORT:-8801})"

# One slot = one GPU running its models sequentially: serve, sweep, tear down.
slot() {
  local gpu="$1"; shift
  local model port
  for model in "$@"; do
    case "$model" in
      kimi-audio)   port=8811 ;;
      qwen-omni)    port=8812 ;;
      qwen-omni-3b) port=8813 ;;
      phi4mm)       port=8814 ;;
      qwen2-audio)  port=8815 ;;
    esac
    nohup bash serve_gen.sh "$model" "$gpu" "$port" > "out/serve_${model}.log" 2>&1 &
    local gen_pid=$!
    echo "[slot gpu$gpu] $model serving (pid $gen_pid, port $port)"
    bash run_d1k_prm.sh "$model" "$port" > "out/sweep_${model}.log" 2>&1 \
      || echo "[slot gpu$gpu] $model sweep FAILED - see out/sweep_${model}.log" >&2
    kill "$gen_pid" 2>/dev/null || true
    wait "$gen_pid" 2>/dev/null || true
    echo "[slot gpu$gpu] $model done $(date +%H:%M)"
  done
}

slot 1 kimi-audio &
slot 2 qwen-omni qwen-omni-3b &
slot 3 phi4mm qwen2-audio &
wait

kill "$judge_pid" 2>/dev/null || true
echo "ALL_SWEEPS_DONE $(date)"
echo "score with: bash score_all.sh"
