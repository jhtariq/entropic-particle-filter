#!/usr/bin/env bash
# Run 14c, end-to-end: one E2B vLLM server per GPU (port PORT_BASE+gpu), then the
# budget-{64,128} sweep across them. See RUN14_COLLAB.md for the full spec.
# usage: scripts/run14c_launch.sh <gpu> [gpu...]     e.g. scripts/run14c_launch.sh 0 1
# PORT_BASE defaults to 8110 (NOT 8100 — this node can run another job's vLLM
# servers on 8100-8103 too, and ports are a shared, non-namespaced resource
# across SLURM jobs on the same node even when GPUs are cleanly separate).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ $# -ge 1 ] || { echo "usage: $0 <gpu> [gpu...]"; exit 1; }

export GEMMA_VLLM="${GEMMA_VLLM:-/data/user_data/abdulw/envs/gemmaserve/bin/vllm}"
export HF_HOME="${HF_HOME:-/data/hf_cache}"
export ALLOWED_MEDIA_PATH="${ALLOWED_MEDIA_PATH:-/data/group_data/mlsp/abdulw/epf_data}"
export PORT_BASE="${PORT_BASE:-8110}"
PY="${PY:-/data/user_data/abdulw/envs/epf/bin/python}"
DATA_ROOT="${DATA_ROOT:-/data/group_data/mlsp/abdulw/epf_data/mmau_pro_testmini}"
AUDIO_ROOT="${AUDIO_ROOT:-/data/group_data/mlsp/abdulw/epf_data/mmau_pro_audio}"

OUT="$REPO/benchmarking/mmau_pro/results/run14_gemma4e2b"
mkdir -p "$OUT"

PIDS=()
ENDPOINTS=""
for GPU in "$@"; do
  "$REPO/scripts/serve_gemma4_e2b.sh" "$GPU" > "$OUT/serve_gpu$GPU.log" 2>&1 &
  PIDS+=($!)
  ENDPOINTS="${ENDPOINTS:+$ENDPOINTS,}http://localhost:$((PORT_BASE + GPU))/v1"
done
trap 'kill "${PIDS[@]}" 2>/dev/null || true' EXIT

for GPU in "$@"; do
  PORT=$((PORT_BASE + GPU))
  until curl -sf "http://localhost:$PORT/v1/models" > /dev/null; do sleep 5; done
  echo "gpu$GPU (port $PORT) healthy"
done

PY="$PY" ENDPOINTS="$ENDPOINTS" DATA_ROOT="$DATA_ROOT" AUDIO_ROOT="$AUDIO_ROOT" \
  bash "$REPO/scripts/run14_collab_b64_128.sh"
