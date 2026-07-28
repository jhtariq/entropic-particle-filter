# MMSU Run 2 — Qwen2.5-Omni-3B EPF grid. MIRRORS Run 1 exactly (P4 x {mean_logprob,
# entropy} x b{1,8,16,32,64,128} x 5,000) for a clean 7B-vs-3B contrast on identical axes
# (is the oracle climb / selector wall size-invariant?). Sourced by scripts/*.sh AFTER
# REPO_ROOT is set.
MODEL_ID="Qwen/Qwen2.5-Omni-3B"
MODEL_REV="f75b40e3da2003cdd6e1829b1f420ca70797c34e"
SERVED_NAME="qwen-omni-3b"
PROMPTS="4"
BUDGETS="1 8 16 32 64 128"
STEM="mmsu_run02"
OUT_DIR="$REPO_ROOT/benchmarking/mmsu/results/run02_omni3b"
