#!/usr/bin/env bash
# Serve the Qwen3-Omni-30B judge on one A100-80GB.
# Usage: serve_judge.sh [GPU_ID]   (default 0; port from JUDGE_PORT, default 8801)
set -euo pipefail

gpu="${1:-0}"
JUDGE_PORT="${JUDGE_PORT:-8801}"

if [[ -z "${BABEL_D1K_ROOT:-}" ]]; then
  echo "set BABEL_D1K_ROOT to the local mmau_pro_d1k directory" >&2
  exit 1
fi
media="$(realpath "$BABEL_D1K_ROOT")"

# vLLM 0.22.1: the Qwen3-Omni audio encoder passes CPU cu_seqlens into the
# flash-attn varlen kernel and crashes on EVERY GPU generation without the
# TORCH_SDPA encoder backend. First run per GPU arch pays ~25 min of
# FlashInfer CUTLASS MoE JIT (cached in ~/.cache/flashinfer).
CUDA_VISIBLE_DEVICES="$gpu" exec vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --revision 26291f793822fb6be9555850f06dfe95f2d7e695 \
  --host 0.0.0.0 \
  --port "$JUDGE_PORT" \
  --max-model-len 8192 \
  --max-num-seqs 48 \
  --gpu-memory-utilization 0.92 \
  --mm-encoder-attn-backend TORCH_SDPA \
  --allowed-local-media-path "$media" \
  --limit-mm-per-prompt '{"audio":1}'
