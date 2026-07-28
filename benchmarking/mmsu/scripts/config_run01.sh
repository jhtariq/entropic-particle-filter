# MMSU Run 1 — Qwen2.5-Omni-7B EPF grid (mmau_pro Run 11 / MMAR Run 1 lineage).
# The run01 cot_compare screen (2026-07-23) showed ALL of P4,5,7,9 parse 100% and
# reason on MMSU (P7/P9 did NOT break as they did on MMAU-Pro/MMAR — MMSU's shorter
# spoken-language items chunk cleanly). User picked a SINGLE prompt (P4, the best
# chunker at 4.5) but a DEEP budget ladder to 128 to test whether the oracle
# saturates / the selector ever benefits from more particles (MMAR left b64/128 as a
# pending extension). P4 x {mean_logprob,entropy} x b{1,8,16,32,64,128} x 5,000.
# Sourced by scripts/{serve,watchdog,run_grid}.sh AFTER they define REPO_ROOT.
MODEL_ID="Qwen/Qwen2.5-Omni-7B"
MODEL_REV="ae9e1690543ffd5c0221dc27f79834d0294cba00"
SERVED_NAME="qwen-omni"
PROMPTS="4"
BUDGETS="1 8 16 32 64 128"
STEM="mmsu_run01"
OUT_DIR="$REPO_ROOT/benchmarking/mmsu/results/run01_omni7b"
