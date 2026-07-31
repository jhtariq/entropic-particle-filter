# MMAU Run 1 — Qwen2.5-Omni-7B EPF grid (MMSU Runs 1-2 lineage; same pins, same
# canonical config). Prompt FIXED at P4 to match MMSU Runs 1-2 (screen40 is a sanity
# check, not a selection); deep budget ladder to 128 for the selector-wall /
# oracle-saturation question on a benchmark with PUBLISHED baselines (Omni-7B 65.9
# test-mini avg, Omni-R1 GRPO 71.3 — arXiv 2505.09439 Table I).
# P4 x {mean_logprob,entropy} x b{1,8,16,32,64,128} x 1,000 (MMAU-v05.15.25 test-mini).
# Sourced by scripts/{serve,watchdog,run_grid}.sh AFTER they define REPO_ROOT.
MODEL_ID="Qwen/Qwen2.5-Omni-7B"
MODEL_REV="ae9e1690543ffd5c0221dc27f79834d0294cba00"
SERVED_NAME="qwen-omni"
PROMPTS="4"
BUDGETS="1 8 16 32 64 128"
STEM="mmau_run01"
OUT_DIR="$REPO_ROOT/benchmarking/mmau/results/run01_omni7b"
GPUS="0"   # single-GPU session — GPU 1 is the live Kimi MMAR job
