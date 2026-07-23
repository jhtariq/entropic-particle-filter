# MMAR Run 6 — Kimi-Audio-7B-Instruct EPF grid (mmau_pro Run 19 replica).
# Single GPU (gpu 1 / :8101) — GPU 0 holds an unrelated phi4mm server; leave it be.
# le30s subset (983 items): Kimi's Whisper front-end hard-truncates clips at its
# 30 s window (same window Qwen2-Audio used in Run 3), so le30s is the eligible set.
# P2+P4 per Run 19's prompt screens. Grid knobs (temp/ess/early/step/max_steps) are
# diversity_probe defaults = Run 19's canonical config, so run_grid.sh replicates it.
# SERVING: use scripts/serve_kimi.sh (run19/serve_kimi.py wrapper) — NOT serve.sh,
# which can't wire Kimi's chat template / audio:1 limit / 6 vLLM bug patches.
MODEL_ID="moonshotai/Kimi-Audio-7B-Instruct"
MODEL_REV="9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b"
SERVED_NAME="kimi-audio"
PROMPTS="2,4"
SUBSET="le30s"
N_ITEMS=983
GPUS="1"
ENDPOINTS="http://localhost:8101/v1"
MAX_MODEL_LEN=8192
STEM="mmar_run06"
OUT_DIR="$REPO_ROOT/benchmarking/mmar/results/run06_kimiaudio_le30s"
# BUDGETS left default (1 8 16 32); may be overridden after the smoke projection.
