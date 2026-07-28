# MMAR Run 3 EXTENSION — budgets 64+128 on the existing Qwen2-Audio grid JSONL.
# DUAL-GPU (vs the original single-GPU run) — halves wall time; set GPUS="0" and
# ENDPOINTS to one port if a GPU is busy. Est. ~7 h dual / ~14.5 h single.
MODEL_ID="Qwen/Qwen2-Audio-7B-Instruct"
MODEL_REV="0a095220c30b7b31434169c3086508ef3ea5bf0a"
SERVED_NAME="qwen2-audio"
PROMPTS="4,9"
SUBSET="le30s"
N_ITEMS=983
GPUS="0 1"
ENDPOINTS="http://localhost:8100/v1,http://localhost:8101/v1"
MAX_MODEL_LEN=8192
BUDGETS="64 128"
STEM="mmar_run03"
OUT_DIR="$REPO_ROOT/benchmarking/mmar/results/run03_qwen2audio_le30s"
