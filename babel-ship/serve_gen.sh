#!/usr/bin/env bash
# Serve one generator model on one A100-80GB.
# Usage: serve_gen.sh MODEL GPU_ID PORT
#   MODEL: qwen-omni | qwen-omni-3b | qwen2-audio | phi4mm | kimi-audio
# Revisions are pinned to the original campaign (see models_manifest.json);
# run sanity_check_models.py first so the snapshots are present and verified.
set -euo pipefail

model="${1:?usage: serve_gen.sh MODEL GPU_ID PORT}"
gpu="${2:?usage: serve_gen.sh MODEL GPU_ID PORT}"
port="${3:?usage: serve_gen.sh MODEL GPU_ID PORT}"

media="${BABEL_MEDIA_ROOT:-${BABEL_D1K_ROOT:-${BABEL_STARBENCH_ROOT:-}}}"
if [[ -z "$media" ]]; then
  echo "set BABEL_MEDIA_ROOT (or BABEL_D1K_ROOT / BABEL_STARBENCH_ROOT) to the benchmark data root" >&2
  exit 1
fi
media="$(realpath "$media")"
repo_root="${BABEL_REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"

export CUDA_VISIBLE_DEVICES="$gpu"

case "$model" in
  qwen-omni)
    exec vllm serve Qwen/Qwen2.5-Omni-7B \
      --revision ae9e1690543ffd5c0221dc27f79834d0294cba00 \
      --served-model-name qwen-omni \
      --host 0.0.0.0 \
      --port "$port" \
      --trust-remote-code \
      --dtype bfloat16 \
      --max-model-len 8192 \
      --gpu-memory-utilization 0.85 \
      --allowed-local-media-path "$media" \
      --limit-mm-per-prompt '{"audio":3}' \
      --max-num-seqs 128 \
      --max-num-batched-tokens 8192 \
      --enable-prefix-caching ;;
  qwen-omni-3b)
    exec vllm serve Qwen/Qwen2.5-Omni-3B \
      --revision f75b40e3da2003cdd6e1829b1f420ca70797c34e \
      --served-model-name qwen-omni-3b \
      --host 0.0.0.0 \
      --port "$port" \
      --trust-remote-code \
      --dtype bfloat16 \
      --max-model-len 8192 \
      --gpu-memory-utilization 0.85 \
      --allowed-local-media-path "$media" \
      --limit-mm-per-prompt '{"audio":3}' \
      --max-num-seqs 128 \
      --max-num-batched-tokens 8192 \
      --enable-prefix-caching ;;
  qwen2-audio)
    exec vllm serve Qwen/Qwen2-Audio-7B-Instruct \
      --revision 0a095220c30b7b31434169c3086508ef3ea5bf0a \
      --served-model-name qwen2-audio \
      --host 0.0.0.0 \
      --port "$port" \
      --trust-remote-code \
      --dtype bfloat16 \
      --max-model-len 8192 \
      --gpu-memory-utilization 0.85 \
      --allowed-local-media-path "$media" \
      --limit-mm-per-prompt '{"audio":3}' \
      --max-num-seqs 128 \
      --max-num-batched-tokens 8192 \
      --enable-prefix-caching ;;
  phi4mm)
    # Requests MUST target the LoRA adapter name "speech" - the base name is
    # accepted but silently skips the speech LoRA.
    m_rev=93f923e1a7727d1c4f446756212d9d3e8fcc5d81
    hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
    lora="$hf_home/hub/models--microsoft--Phi-4-multimodal-instruct/snapshots/$m_rev/speech-lora"
    if [[ ! -d "$lora" ]]; then
      echo "speech-lora dir not found at $lora - run sanity_check_models.py phi4mm" >&2
      exit 1
    fi
    exec vllm serve microsoft/Phi-4-multimodal-instruct \
      --revision "$m_rev" \
      --served-model-name phi4mm \
      --host 0.0.0.0 \
      --port "$port" \
      --trust-remote-code \
      --dtype bfloat16 \
      --max-model-len 32768 \
      --enforce-eager \
      --gpu-memory-utilization 0.85 \
      --allowed-local-media-path "$media" \
      --limit-mm-per-prompt '{"audio":3}' \
      --enable-lora \
      --max-lora-rank 320 \
      --max-loras 1 \
      --lora-modules "speech=$lora" \
      --enable-prefix-caching ;;
  kimi-audio)
    # NEVER plain `vllm serve` for Kimi - the wrapper + chat template are load-bearing.
    exec python "$repo_root/benchmarking/mmau_pro/run19/serve_kimi.py" \
      moonshotai/Kimi-Audio-7B-Instruct \
      --revision 9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b \
      --served-model-name kimi-audio \
      --host 0.0.0.0 \
      --port "$port" \
      --trust-remote-code \
      --dtype bfloat16 \
      --max-model-len 8192 \
      --enforce-eager \
      --gpu-memory-utilization 0.85 \
      --allowed-local-media-path "$media" \
      --limit-mm-per-prompt '{"audio":1}' \
      --chat-template "$repo_root/benchmarking/mmau_pro/run19/template_kimi_audio_epf.jinja" \
      --enable-prefix-caching ;;
  *)
    echo "unknown MODEL=$model" >&2
    exit 1 ;;
esac
