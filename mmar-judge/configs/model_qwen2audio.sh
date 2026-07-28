# Policy: Qwen2-Audio-7B-Instruct. Hard 30 s encoder window -> le30s subset
# (983 items), max_model_len 8192 (matches MMAR Run 3).
MODEL_ID="Qwen/Qwen2-Audio-7B-Instruct"
MODEL_REV="0a095220c30b7b31434169c3086508ef3ea5bf0a"
SERVED_NAME="qwen2-audio"
SERVE_KIND=vllm
SUBSET=le30s
MAX_MODEL_LEN=8192
GPUS="0"
