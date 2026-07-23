# MMAR Run 3 — Qwen2-Audio-7B-Instruct EPF grid (mmau_pro Run 15 replica).
# Single GPU (gpu 0 / :8100) — GPU 1 is reserved for other work. le30s subset
# (983 items): Qwen2-Audio truncates clips at its hard 30 s encoder window.
# P4+P9 per Run 15's prompt screens (P5 unusable, P7 dominated on this model).
MODEL_ID="Qwen/Qwen2-Audio-7B-Instruct"
MODEL_REV="0a095220c30b7b31434169c3086508ef3ea5bf0a"
SERVED_NAME="qwen2-audio"
PROMPTS="4,9"
SUBSET="le30s"
N_ITEMS=983
GPUS="0"
ENDPOINTS="http://localhost:8100/v1"
MAX_MODEL_LEN=8192
STEM="mmar_run03"
OUT_DIR="$REPO_ROOT/benchmarking/mmar/results/run03_qwen2audio_le30s"
