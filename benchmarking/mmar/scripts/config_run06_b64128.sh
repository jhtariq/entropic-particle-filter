# MMAR Run 6 EXTENSION — budgets 64+128 on the existing Kimi-Audio grid JSONL.
# MUST serve via serve_kimi.sh (run19 wrapper; plain serve.sh cannot serve Kimi) —
# SERVE_SCRIPT below wires that into run_ext_b64128.sh. DUAL-GPU (vs the original
# single-GPU run). Kimi is the slowest model: est. ~48 h dual / ~97 h single for
# both budgets — consider BUDGETS="64" first (~15 h dual) and decide on 128 later.
MODEL_ID="moonshotai/Kimi-Audio-7B-Instruct"
MODEL_REV="9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b"
SERVED_NAME="kimi-audio"
SERVE_SCRIPT_NAME="serve_kimi.sh"
PROMPTS="2,4"
SUBSET="le30s"
N_ITEMS=983
GPUS="0 1"
ENDPOINTS="http://localhost:8100/v1,http://localhost:8101/v1"
MAX_MODEL_LEN=8192   # = config.json max_position_embeddings; do NOT raise
BUDGETS="64 128"
STEM="mmar_run06"
OUT_DIR="$REPO_ROOT/benchmarking/mmar/results/run06_kimiaudio_le30s"
