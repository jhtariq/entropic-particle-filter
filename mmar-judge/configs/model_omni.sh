# Policy: Qwen2.5-Omni-7B (validation model; NOTE judge is the same weights —
# self-judging bias is disclosed in results.md). Servable directly by
# benchmarking/mmar/scripts/serve.sh (sourced by it as the run config).
MODEL_ID="Qwen/Qwen2.5-Omni-7B"
MODEL_REV="ae9e1690543ffd5c0221dc27f79834d0294cba00"
SERVED_NAME="qwen-omni"
SERVE_KIND=vllm
SUBSET=full
GPUS="0"                                  # policy GPU -> :8100
