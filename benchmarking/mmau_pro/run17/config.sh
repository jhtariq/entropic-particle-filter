# Run 17 configuration (Mellow-v0 × MMAU-Pro EPF grid). Every knob is overridable
# from the environment, e.g.
#   NUM_GPUS=8 bash run_all.sh
# or via a config.local.sh (gitignored) next to this file — it is sourced first and wins.
# Source this file, then lib.sh, from every run17 script.

RUN17_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$RUN17_DIR/../../.." && pwd)"

# local overrides (plain VAR=value lines) — sourced FIRST so the := defaults
# below respect them, including everything derived from EPF_DATA_ROOT
[ -f "$RUN17_DIR/config.local.sh" ] && source "$RUN17_DIR/config.local.sh"

# --- machine / environment ----------------------------------------------------
: "${EPF_ENV_NAME:=epf}"
: "${CONDA_BASE:=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")}"
: "${EPF_PY:=$CONDA_BASE/envs/$EPF_ENV_NAME/bin/python}"  # ALWAYS absolute — `conda activate` may lose the PATH race (SETUP_GUIDE §2)
: "${EPF_DATA_ROOT:=$HOME/epf_data}"       # will hold mmau_pro_testmini/ + mmau_pro_audio/ (~75 GB)
: "${HF_HOME:=$EPF_DATA_ROOT/hf_cache}"; export HF_HOME   # Mellow ckpts + SmolLM2 land here (~1.7 GB)
: "${NUM_GPUS:=$(nvidia-smi -L 2>/dev/null | wc -l)}"
: "${BASE_PORT:=8100}"                     # one serve_mellow.py shim per GPU on BASE_PORT+i
# NOTE: no GPU_MEM_UTIL here — Mellow is 167M params (~1.5 GB with activations).
# The vLLM VRAM/concurrency notes in SETUP_GUIDE §3 do NOT apply to this run: the
# shim serves one request at a time per GPU, so MAX_INFLIGHT below is the only
# throughput lever, and any single >=8 GB GPU is enough.

# --- experiment ----------------------------------------------------------------
# THE GRID (what run_all.sh executes, in this order): budgets 1..128 on the FULL
# test_no3a set (5,073 MCQ), prompts P10+P11 × signals {mean_logprob, entropy}
# -> 4 curves per plot. The reference box already computed b1/8/16 for a 500-item
# pilot + the 326-item le10s slice; those rows ship in run17/seeds/ and are
# REUSED automatically (resume skips them) — your run completes b1/8/16 on the
# remaining items, then runs the new b32/64/128 stages. Just run:
#   nohup bash benchmarking/mmau_pro/run17/run_all.sh > run17.log 2>&1 &
# Crashed/interrupted? Re-run the same line — nothing is recomputed.
: "${BUDGETS:=1 8 16 32 64 128}"           # staged in this order
: "${MAX_INFLIGHT:=32}"                    # per-endpoint item concurrency = max(1, MAX_INFLIGHT // budget)
: "${MAX_INFLIGHT_B1:=24}"                 # budget-1 stages need the lower cap (SETUP_GUIDE §7)
: "${PROBE_LIMIT:=6000}"                   # >= 5,073 covers every item; set 4 to rehearse the driver
: "${N_BOOT:=10000}"                       # bootstrap resamples for the final HTML
: "${WAVEFORM_CACHE:=6000}"                # shim per-GPU waveform LRU (~1.3 MB/entry; 6000 caches
                                           # the whole set, ~8 GB RAM per shim — lower on small-RAM boxes)

# --- model / prompts (finalized at the Phase-A HOLD points) ---------------------
RUN17_MODEL_ID="soham97/mellow"
RUN17_REV="83672db0dae28764e283210d5bb732621e903d8a"        # HF ckpt revision (v0.ckpt/v0_s.ckpt/v0.yaml)
RUN17_NAME="mellow"
RUN17_SUBSET="test_no3a"                    # 5,073 MCQ = full test minus 17 three-audio items
RUN17_STEM="epf_mellow_no3a"
: "${PROMPTS_RUN17:=10,11}"                # HOLD-2 decision (Mellow-native methods)
# Mellow answers in ONE short burst and never emits 'Answer:' or '\n\n' steps; this
# regex ends a trajectory at the first step containing a native answer ('c) ...'),
# instead of forcing 6 rounds of junk continuation (HOLD-2 decision)
: "${STOP_REGEX:=[a-k]\)}"
: "${MELLOW_VARIANT:=v0}"                  # v0 | v0_s (both ckpts are fetched)
: "${MELLOW_MAX_TEXT_TOKENS:=0}"           # 0 = faithful 129-token window; N>0 = extended, no pads
: "${SINGLE_AUDIO_FILL:=silence}"          # slot-2 filler for single-audio items (silence | duplicate)

# Mellow model CODE (no PyPI package): pinned clone of github.com/soham97/mellow
MELLOW_CODE_REPO="https://github.com/soham97/mellow"
MELLOW_CODE_COMMIT="349f9b2be84bec713ac71e54fcbcac9bf4d116e5"
: "${MELLOW_CODE_DIR:=$EPF_DATA_ROOT/mellow_code}"
# SmolLM2-135M base (architecture + tokenizer; weights are overwritten by the ckpt)
SMOLLM2_ID="HuggingFaceTB/SmolLM2-135M"
SMOLLM2_REV="93efa2f097d58c2a74874c7e644dbc9b0cee75a2"

: "${OUT_ROOT:=$REPO_ROOT/benchmarking/mmau_pro/results/run17_mellow}"
: "${DRY_RUN:=0}"                          # 1 = print the plan instead of executing

# --- dataset pins (same sources as run16) ---------------------------------------
PARQUET_DATASET="macabdul9/MMAU_Pro_Testmini"                     # ships testmini AND full-test parquets
PARQUET_REV="81eb01fb86bfc183dcb88b376ab1ca149a9d9c4b"
AUDIO_DATASET="gamma-lab-umd/MMAU-Pro"                            # data.zip: 44 GiB -> 5,787 files / 53 GB
AUDIO_ZIP_SHA256="8fab3e820b27bf7f239ae74a45a1085f1364b6971a7b5ac53f220d091b9b111c"

: "${DATA_TESTMINI:=$EPF_DATA_ROOT/mmau_pro_testmini}"
: "${DATA_AUDIO:=$EPF_DATA_ROOT/mmau_pro_audio}"
