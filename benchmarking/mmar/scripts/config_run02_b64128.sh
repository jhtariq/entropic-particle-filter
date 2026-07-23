# MMAR Run 2 EXTENSION — budgets 64 + 128 on the existing 3B grid JSONL
# (resume-safe: b1–32 rows are kept, only the new budgets run).
MODEL_ID="Qwen/Qwen2.5-Omni-3B"
MODEL_REV="f75b40e3da2003cdd6e1829b1f420ca70797c34e"
SERVED_NAME="qwen-omni-3b"
PROMPTS="4,5"
BUDGETS="64 128"
STEM="mmar_run02"
OUT_DIR="$REPO_ROOT/benchmarking/mmar/results/run02_omni3b"
