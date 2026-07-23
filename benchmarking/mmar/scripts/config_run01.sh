# MMAR Run 1 — Qwen2.5-Omni-7B EPF grid (mmau_pro Run 11 replica, narrowed to
# P4+P5 by user decision 2026-07-21 to halve compute; P7/P9 can be added later by
# rerunning with --prompts 7,9 against the same JSONL).
# Sourced by scripts/{serve,watchdog,run_grid}.sh AFTER they define REPO_ROOT.
MODEL_ID="Qwen/Qwen2.5-Omni-7B"
MODEL_REV="ae9e1690543ffd5c0221dc27f79834d0294cba00"
SERVED_NAME="qwen-omni"
PROMPTS="4,5"
STEM="mmar_run01"
OUT_DIR="$REPO_ROOT/benchmarking/mmar/results/run01_epf_grid"
