# Run 19 configuration (Kimi-Audio-7B-Instruct × MMAU-Pro EPF grid). Every knob
# is overridable from the environment, e.g.
#   NUM_GPUS=8 bash run_all.sh
# or via a config.local.sh (gitignored) next to this file — it is sourced first and wins.
# Source this file, then lib.sh, from every run19 script.

RUN19_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$RUN19_DIR/../../.." && pwd)"

# local overrides (plain VAR=value lines) — sourced FIRST so the := defaults
# below respect them, including everything derived from EPF_DATA_ROOT
[ -f "$RUN19_DIR/config.local.sh" ] && source "$RUN19_DIR/config.local.sh"

# --- machine / environment ----------------------------------------------------
: "${EPF_ENV_NAME:=epf}"
: "${CONDA_BASE:=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")}"
: "${EPF_PY:=$CONDA_BASE/envs/$EPF_ENV_NAME/bin/python}"  # ALWAYS absolute — `conda activate` may lose the PATH race (SETUP_GUIDE §2)
: "${EPF_DATA_ROOT:=$HOME/epf_data}"       # will hold mmau_pro_testmini/ + mmau_pro_audio/ (~75 GB)
: "${HF_HOME:=$EPF_DATA_ROOT/hf_cache}"; export HF_HOME   # Kimi-Audio lands here (~23 GB — see fetch_models.sh)
: "${NUM_GPUS:=$(nvidia-smi -L 2>/dev/null | wc -l)}"
: "${BASE_PORT:=8100}"                     # one vLLM replica per GPU on BASE_PORT+i
: "${GPU_MEM_UTIL:=0.85}"                  # NOT 0.9 — audio-encoder attention spikes OOM'd an engine at 0.9 (SETUP_GUIDE §10)

# --- experiment ----------------------------------------------------------------
# THE GRID (what run_all.sh executes, in this order): budgets 1..32 on the
# single-audio ≤30 s test subset (1,947 MCQ), 2 prompts × signals
# {mean_logprob, entropy} -> 4 curves. The reference box computed b1/8/16; those
# rows ship in run19/seeds/ and are REUSED automatically (resume skips them) —
# your run completes any b1/8/16 remainder, then runs the new stages. Just run:
#   nohup bash benchmarking/mmau_pro/run19/run_all.sh > run19.log 2>&1 &
# Crashed/interrupted? Re-run the same line — nothing is recomputed.
: "${BUDGETS:=1 8 16 32}"                  # staged in this order; extend "64 128" only by agreement
: "${MAX_INFLIGHT:=64}"                    # per-endpoint item concurrency = max(1, MAX_INFLIGHT // budget)
: "${MAX_INFLIGHT_B1:=24}"                 # budget-1 stages need the lower cap (SETUP_GUIDE §7)
: "${PROBE_LIMIT:=2500}"                   # >= 1,947 covers every test_le30s_1a item; set 4 to rehearse
: "${N_BOOT:=10000}"                       # bootstrap resamples for the final HTML

# --- model / prompts (finalized at the Phase-A HOLD points) ---------------------
RUN19_MODEL_ID="moonshotai/Kimi-Audio-7B-Instruct"
RUN19_REV="9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b"        # refs/main 2025-05-29 (single snapshot)
RUN19_NAME="kimi-audio"
: "${RUN19_MAXLEN:=8192}"                  # = config.json max_position_embeddings — do NOT raise
: "${RUN19_SUBSET:=test_le30s_1a}"         # 1,947 single-audio ≤30 s MCQ — Kimi's Whisper front-end
                                           # truncates ALL audio to the first 30 s and vLLM's kimi
                                           # path accepts at most ONE audio per prompt (README)
: "${RUN19_STEM:=epf_kimi_le30s1a}"
: "${PROMPTS_RUN19:=2,4}"                  # HOLD-2 decision: P2 zero-shot CoT (best acc .507, parse 3%)
                                           # + P4 plan-and-solve (acc .493, chunks 7.7, cross-run anchor);
                                           # see RESULTS.md §19 screens
: "${STOP_REGEX:=}"                        # empty = none needed; revisit if screens show non-terminating output

# --- Kimi serving specifics -----------------------------------------------------
# vLLM 0.22.1's kimi chat path needs BOTH of these (see serve_kimi.py docstring):
# the wrapper (tokenizer patch) and an explicit chat template. serve_one wires them.
KIMI_CHAT_TEMPLATE="$RUN19_DIR/template_kimi_audio_epf.jinja"

: "${OUT_ROOT:=$REPO_ROOT/benchmarking/mmau_pro/results/run19_kimiaudio}"
: "${DRY_RUN:=0}"                          # 1 = print the plan instead of executing

# --- dataset pins (same sources as run16/17/18) -----------------------------------
PARQUET_DATASET="macabdul9/MMAU_Pro_Testmini"                     # ships testmini AND full-test parquets
PARQUET_REV="81eb01fb86bfc183dcb88b376ab1ca149a9d9c4b"
AUDIO_DATASET="gamma-lab-umd/MMAU-Pro"                            # data.zip: 44 GiB -> 5,787 files / 53 GB
AUDIO_ZIP_SHA256="8fab3e820b27bf7f239ae74a45a1085f1364b6971a7b5ac53f220d091b9b111c"

: "${DATA_TESTMINI:=$EPF_DATA_ROOT/mmau_pro_testmini}"
: "${DATA_AUDIO:=$EPF_DATA_ROOT/mmau_pro_audio}"
