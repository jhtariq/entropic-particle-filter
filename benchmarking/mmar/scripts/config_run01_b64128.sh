# MMAR Run 1 EXTENSION — budgets 64 + 128 on the existing 7B grid JSONL
# (resume-safe: b1–32 rows are kept, only the new budgets run).
MODEL_ID="Qwen/Qwen2.5-Omni-7B"
MODEL_REV="ae9e1690543ffd5c0221dc27f79834d0294cba00"
SERVED_NAME="qwen-omni"
PROMPTS="4,5"
BUDGETS="64 128"
STEM="mmar_run01"
OUT_DIR="$REPO_ROOT/benchmarking/mmar/results/run01_epf_grid"
