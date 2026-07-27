# MMSU Run 4 — Mellow-v0 (167M, serve_mellow.py shim; mmau_pro Run 17 lineage).
# MMSU convention (Runs 1–2): ONE prompt × {mean_logprob, entropy} × DEEP ladder
# b{1,8,16,32,64,128} × 5,000. P10 (native inline MCQ) chosen on Run 17 evidence:
# it beats P11 on selected acc in both the screens (.230 vs .197) and the grid
# (b16 sel .295 vs .253, entropy). Serve with scripts/serve_mellow_shim.sh (NOT
# serve.sh — Mellow is not vLLM-servable); watchdog/run_grid work unchanged.
# Sourced by scripts/*.sh AFTER REPO_ROOT is set.
MODEL_ID="soham97/mellow"
MODEL_REV="83672db0dae28764e283210d5bb732621e903d8a"   # HF ckpt rev (v0.ckpt/v0.yaml)
SERVED_NAME="mellow"
PROMPTS="10"
BUDGETS="1 8 16 32 64 128"
STEM="mmsu_run04"
OUT_DIR="$REPO_ROOT/benchmarking/mmsu/results/run04_mellow"

# --- shim knobs (run17-proven; consumed by serve_mellow_shim.sh) ---------------
MELLOW_VARIANT="v0"
SMOLLM2_REV="93efa2f097d58c2a74874c7e644dbc9b0cee75a2"  # SmolLM2-135M base (arch+tokenizer)
MELLOW_CODE_COMMIT="349f9b2be84bec713ac71e54fcbcac9bf4d116e5"  # github.com/soham97/mellow pin
# MELLOW_CODE_DIR must point at a clone of that commit — set it in scripts/local.sh
# (reuse the clone mmau_pro/run17/fetch_models.sh made, e.g. $EPF_DATA_ROOT/mellow_code)
MELLOW_MAX_TEXT_TOKENS=0        # 0 = faithful 129-token window (Run 17: the only viable mode)
SINGLE_AUDIO_FILL="silence"     # slot-2 filler — ALL 5,000 MMSU items are single-audio
WAVEFORM_CACHE=6000             # per-shim waveform LRU; caches the whole set (~8 GB RAM/shim)

# --- probe deltas vs the vLLM models -------------------------------------------
# Mellow answers in one short burst as 'c) ...' and never emits 'Answer:'/'\n\n'
# steps — end trajectories at the native answer (Run 17 HOLD-2 decision)
EXTRA_PROBE_ARGS='--stop-regex [a-k]\)'
MAX_INFLIGHT=32                 # shim = one request at a time per GPU (Run 17 setting)
MAX_INFLIGHT_B1=24
