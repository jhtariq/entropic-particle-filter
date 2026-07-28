# MMSU Run 3 — Qwen2.5-Omni-7B RANDOM-SURVIVAL ablation (control for Run 1).
# Same model/rev/served-name/prompt/budgets as Run 1, but the single arm is SIGNALS=random:
# uniform particle weights (ESS pinned at B, entropic annealing inert) + multinomial
# resampling (genuine random death/duplication) + SAMPLE final pick (random survivor).
# Tests whether the self-certainty selector does anything vs a coin-flip. See RESULTS.md §3.
# Sourced by scripts/{serve,watchdog,run_grid}.sh AFTER they define REPO_ROOT.
MODEL_ID="Qwen/Qwen2.5-Omni-7B"
MODEL_REV="ae9e1690543ffd5c0221dc27f79834d0294cba00"
SERVED_NAME="qwen-omni"
PROMPTS="4"
SIGNALS="random"
BUDGETS="1 8 16 32 64 128"
STEM="mmsu_run03_random"
OUT_DIR="$REPO_ROOT/benchmarking/mmsu/results/run03_omni7b_random"
