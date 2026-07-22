#!/usr/bin/env bash
# Serve google/gemma-4-E4B-it with vLLM on one GPU (Run 15).
# Identical to serve_gemma4_e2b.sh except the model and the PORT BASE: E4B uses
# 8200+gpu so both models can be served simultaneously on disjoint GPUs.
#
# usage:    scripts/serve_gemma4_e4b.sh <gpu 0|1|...>       # port = 8200 + gpu
# detached: nohup setsid scripts/serve_gemma4_e4b.sh 0 > serve_e4b_gpu0.log 2>&1 &
# health:   curl -s http://localhost:820<gpu>/v1/models
# stop:     kill by PID (pgrep -af '[v]llm serve' first) — never pkill -f.
#
# The serving env MUST carry the gemma4 batched-audio patch (architecture-level,
# same one as E2B):  <serving-env>/bin/python scripts/patch_vllm_gemma4.py
# E4B has the same 30 s/clip audio cap (750 tokens x 40 ms) — item filtering is
# client-side (the committed <=30 s ids), nothing to configure here.
set -euo pipefail
GPU="${1:?usage: serve_gemma4_e4b.sh <gpu-index>}"
PORT=$((8200 + GPU))
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GEMMA_VLLM="${GEMMA_VLLM:-/home/tariqvrh4/miniconda3/envs/gemmaserve/bin/vllm}"
HF_HOME="${HF_HOME:-/home/tariqvrh4/hf_cache}"
ALLOWED_MEDIA_PATH="${ALLOWED_MEDIA_PATH:-$REPO/data}"

HF_HOME="$HF_HOME" \
CUDA_VISIBLE_DEVICES="$GPU" \
VLLM_USE_FLASHINFER_SAMPLER=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
exec "$GEMMA_VLLM" serve google/gemma-4-E4B-it \
  --served-model-name gemma4-e4b --port "$PORT" \
  --dtype bfloat16 --max-model-len 16384 --enforce-eager \
  --gpu-memory-utilization 0.85 \
  --allowed-local-media-path "$ALLOWED_MEDIA_PATH" \
  --limit-mm-per-prompt '{"audio":3}'
