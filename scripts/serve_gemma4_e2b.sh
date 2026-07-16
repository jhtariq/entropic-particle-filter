#!/usr/bin/env bash
# Serve google/gemma-4-E2B-it with vLLM on one GPU (Run 14).
#
# usage:    scripts/serve_gemma4_e2b.sh <gpu 0|1>          # port = 8100 + gpu
# detached: mkdir -p benchmarking/mmau_pro/results/run14_gemma4e2b
#           nohup setsid scripts/serve_gemma4_e2b.sh 0 \
#             > benchmarking/mmau_pro/results/run14_gemma4e2b/serve_gpu0.log 2>&1 &
# health:   curl -s http://localhost:810<gpu>/v1/models
# stop:     kill by PID (pgrep -af '[v]llm serve' first) — NEVER `pkill -f "vllm serve"`,
#           it matches your own shell (SETUP_GUIDE §10.3).
#
# Flags (see SETUP_GUIDE §3 for the lessons):
#   VLLM_USE_FLASHINFER_SAMPLER=0            mandatory on Blackwell (sm_120)
#   --gpu-memory-utilization 0.85            headroom for mm-encoder spikes + orphans
#   --max-model-len 16384                    worst prompt ~3x750 audio tok + ~1k text;
#                                            worst gen ~2.4k — 16k is generous, and well
#                                            under Gemma 4's 131k (no long-len override)
#   --allowed-local-media-path <repo>/data   covers data/mmau_pro AND data/testwavs
#   no --chat-template override              native gemma4 template handles the system
#                                            role and one <|audio|> per clip
# Gemma 4 E2B hears max 30 s/clip (750 audio tokens x 40 ms); longer clips are
# TRUNCATED with a server-side warning — item filtering happens client-side
# (make_le30s_ids.py), not here.
#
# The serving env MUST carry the gemma4 batched-audio patch or the engine dies
# under any concurrency ('list' has no 'squeeze'; also broken in upstream vLLM
# main as of 2026-07-10) — apply with:
#   <serving-env>/bin/python scripts/patch_vllm_gemma4.py
# On the original box this env is `gemmaserve` (a patched clone of `af3serve`,
# vllm 0.22.1 + transformers 5.13); see RESULTS.md §20 and RUN14_COLLAB.md.
set -euo pipefail
GPU="${1:?usage: serve_gemma4_e2b.sh <gpu-index>}"
PORT=$((8100 + GPU))
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Override these two for your machine (defaults = the original Run-14 box):
GEMMA_VLLM="${GEMMA_VLLM:-/home/tariqvrh4/miniconda3/envs/gemmaserve/bin/vllm}"
HF_HOME="${HF_HOME:-/home/tariqvrh4/hf_cache}"

HF_HOME="$HF_HOME" \
CUDA_VISIBLE_DEVICES="$GPU" \
VLLM_USE_FLASHINFER_SAMPLER=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
exec "$GEMMA_VLLM" serve google/gemma-4-E2B-it \
  --served-model-name gemma4-e2b --port "$PORT" \
  --dtype bfloat16 --max-model-len 16384 --enforce-eager \
  --gpu-memory-utilization 0.85 \
  --allowed-local-media-path "$REPO/data" \
  --limit-mm-per-prompt '{"audio":3}'
