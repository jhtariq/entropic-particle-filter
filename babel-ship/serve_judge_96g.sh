#!/usr/bin/env bash
# Serve the Qwen3-Omni-30B judge on ONE >=80GB GPU (RTX PRO 6000 Blackwell,
# A100-80, H100/H200...). 66GB bf16 weights fit single-GPU: no TP, none of the
# multi-GPU workarounds. Usage: serve_judge_96g.sh [GPU_ID]
set -euo pipefail

gpu="${1:-0}"
JUDGE_PORT="${JUDGE_PORT:-8801}"

media="${BABEL_MEDIA_ROOT:-${BABEL_STARBENCH_ROOT:-${BABEL_D1K_ROOT:-}}}"
if [[ -z "$media" ]]; then
  echo "set BABEL_MEDIA_ROOT (or BABEL_STARBENCH_ROOT / BABEL_D1K_ROOT) to the benchmark data root" >&2
  exit 1
fi
media="$(realpath "$media")"

# TORCH_SDPA encoder backend is mandatory in vLLM 0.22.1 on every GPU
# generation tested so far; first start on a new arch pays a one-time
# FlashInfer MoE JIT (~25 min, cached in ~/.cache/flashinfer).
CUDA_VISIBLE_DEVICES="$gpu" exec vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --revision 26291f793822fb6be9555850f06dfe95f2d7e695 \
  --host 0.0.0.0 \
  --port "$JUDGE_PORT" \
  --max-model-len 8192 \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.90 \
  --mm-encoder-attn-backend TORCH_SDPA \
  --allowed-local-media-path "$media" \
  --limit-mm-per-prompt '{"audio":1}'
