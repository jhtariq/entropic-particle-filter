#!/bin/bash
# Fill the missing MMAR baselines for Qwen2.5-Omni on a single interactive GPU node.
# Arms: greedy nocot, greedy P4, SC-nocot n=128, SC-P4 n=32 (t=0.8, seed 1234).
# Usage: bash run_mmar_omni_baseline.sh <3b|7b>
# Everything is written under epf_deepdive/mmar_omni_baseline/ — no repo files touched.
set -uo pipefail

MODEL="${1:?want 3b|7b}"
REPO=/work/hdd/bcey/awaheed/its-for-audio-reasoning/entropic-particle-filter
OUT=/work/hdd/bcey/awaheed/its-for-audio-reasoning/epf_deepdive/mmar_omni_baseline
PY=/u/awaheed/envs/epf/bin/python
VLLM=/u/awaheed/envs/epf/bin/vllm
PORT=8500
mkdir -p "$OUT"

module load cuda-compat/13.0 2>/dev/null || true
export HF_HOME=/work/hdd/bcey/awaheed/hf_cache
export HF_HUB_OFFLINE=1
export TMPDIR="/tmp/$USER/baseline_$$"; mkdir -p "$TMPDIR"

case "$MODEL" in
  3b) M_ID=Qwen/Qwen2.5-Omni-3B; M_REV=f75b40e3da2003cdd6e1829b1f420ca70797c34e; M_NAME=qwen-omni-3b ;;
  7b) M_ID=Qwen/Qwen2.5-Omni-7B; M_REV=ae9e1690543ffd5c0221dc27f79834d0294cba00; M_NAME=qwen-omni ;;
  *) echo "unknown model $MODEL" >&2; exit 1 ;;
esac

echo "=== $(date '+%F %T') serving $M_NAME on :$PORT ==="
VLLM_USE_FLASHINFER_SAMPLER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
VLLM_CACHE_ROOT="$TMPDIR/vllm" TRITON_CACHE_DIR="$TMPDIR/triton" \
TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor" \
nohup setsid "$VLLM" serve "$M_ID" --revision "$M_REV" \
  --served-model-name "$M_NAME" --port "$PORT" --trust-remote-code --dtype bfloat16 \
  --max-model-len 8192 --gpu-memory-utilization 0.85 \
  --allowed-local-media-path /work/hdd/bcey/awaheed/its-for-audio-reasoning/epf_data \
  --limit-mm-per-prompt '{"audio":3}' \
  --max-num-seqs 128 --max-num-batched-tokens 8192 \
  --enable-prefix-caching > "$OUT/server_${M_NAME}.log" 2>&1 &
SRV=$!
echo "$SRV" > "$OUT/server_${M_NAME}.pid"
cleanup () { kill -- -"$SRV" 2>/dev/null || kill "$SRV" 2>/dev/null; sleep 5; }
trap cleanup EXIT

t=0
while [ "$t" -lt 1800 ]; do
  curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && break
  kill -0 "$SRV" 2>/dev/null || { echo "FATAL: server died"; tail -30 "$OUT/server_${M_NAME}.log"; exit 1; }
  sleep 10; t=$((t + 10))
done
[ "$t" -lt 1800 ] || { echo "FATAL: server not healthy after 1800s"; exit 1; }
echo "server healthy after ${t}s"

cd "$REPO"
run_arm () {  # run_arm <tag> <extra args...>
  local tag="$1"; shift
  echo "=== $(date '+%F %T') arm $tag ==="
  "$PY" -m benchmarking.mmar.sample_runner \
    --endpoints "http://localhost:$PORT/v1" --model-name "$M_NAME" \
    --data-root /u/awaheed/epf_data/mmar --subset full \
    --jsonl "$OUT/${M_NAME}__${tag}.jsonl" "$@" \
    2>&1 | tee -a "$OUT/${M_NAME}__${tag}.log"
}

run_arm greedy_nocot --method 0 --n 1 --temperature 0.0 --max-tokens 64  --max-inflight 16
run_arm greedy_p4    --method 4 --n 1 --temperature 0.0 --max-tokens 700 --max-inflight 16
run_arm sc_nocot     --method 0 --n 128 --n-chunk 32 --temperature 0.8 --seed 1234 --max-tokens 64  --max-inflight 8
run_arm sc_p4        --method 4 --n 32  --n-chunk 8  --temperature 0.8 --seed 1234 --max-tokens 700 --max-inflight 8

echo "=== $(date '+%F %T') all arms done for $M_NAME ==="
