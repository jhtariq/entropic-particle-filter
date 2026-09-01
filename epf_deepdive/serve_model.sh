#!/bin/bash
# Serve one Omni model on this node for interactive experiments (no auto-kill).
# Usage: bash serve_model.sh <3b|7b> [port]   — writes PID to serve_<model>.pid
set -uo pipefail
MODEL="${1:?want 3b|7b}"; PORT="${2:-8500}"
OUT=/work/hdd/bcey/awaheed/its-for-audio-reasoning/epf_deepdive
VLLM=/u/awaheed/envs/epf/bin/vllm
module load cuda-compat/13.0 2>/dev/null || true
export HF_HOME=/work/hdd/bcey/awaheed/hf_cache HF_HUB_OFFLINE=1
export TMPDIR="/tmp/$USER/serve_$$"; mkdir -p "$TMPDIR"
case "$MODEL" in
  3b) M_ID=Qwen/Qwen2.5-Omni-3B; M_REV=f75b40e3da2003cdd6e1829b1f420ca70797c34e; M_NAME=qwen-omni-3b ;;
  7b) M_ID=Qwen/Qwen2.5-Omni-7B; M_REV=ae9e1690543ffd5c0221dc27f79834d0294cba00; M_NAME=qwen-omni ;;
  q2a) M_ID=Qwen/Qwen2-Audio-7B-Instruct; M_REV=0a095220c30b7b31434169c3086508ef3ea5bf0a; M_NAME=qwen2-audio
       export HF_HOME=/u/awaheed/epf_data/hf_cache ;;  # Q2A weights live in the epf_data cache
  *) echo "unknown model $MODEL" >&2; exit 1 ;;
esac
# SERVE_TAG lets concurrent jobs (same model, different nodes) keep separate pid/log files
OUTTAG="${SERVE_TAG:-$M_NAME}"
VLLM_USE_FLASHINFER_SAMPLER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
VLLM_CACHE_ROOT="$TMPDIR/vllm" TRITON_CACHE_DIR="$TMPDIR/triton" \
TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor" \
nohup setsid "$VLLM" serve "$M_ID" --revision "$M_REV" \
  --served-model-name "$M_NAME" --port "$PORT" --trust-remote-code --dtype bfloat16 \
  --max-model-len 8192 --gpu-memory-utilization 0.85 \
  --allowed-local-media-path /work/hdd/bcey/awaheed/its-for-audio-reasoning/epf_data \
  --limit-mm-per-prompt '{"audio":3}' \
  --max-num-seqs 128 --max-num-batched-tokens 8192 \
  --enable-prefix-caching > "$OUT/serve_${OUTTAG}.log" 2>&1 &
echo $! > "$OUT/serve_${OUTTAG}.pid"
echo "serving $M_NAME on :$PORT (pid $(cat "$OUT/serve_${OUTTAG}.pid")) -> $OUT/serve_${OUTTAG}.log"
