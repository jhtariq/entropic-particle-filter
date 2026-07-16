# Run 18 configuration (Phi-4-multimodal-instruct × MMAU-Pro EPF grid). Every knob
# is overridable from the environment, e.g.
#   NUM_GPUS=8 bash run_all.sh
# or via a config.local.sh (gitignored) next to this file — it is sourced first and wins.
# Source this file, then lib.sh, from every run18 script.

RUN18_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$RUN18_DIR/../../.." && pwd)"

# local overrides (plain VAR=value lines) — sourced FIRST so the := defaults
# below respect them, including everything derived from EPF_DATA_ROOT
[ -f "$RUN18_DIR/config.local.sh" ] && source "$RUN18_DIR/config.local.sh"

# --- machine / environment ----------------------------------------------------
: "${EPF_ENV_NAME:=epf}"
: "${CONDA_BASE:=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")}"
: "${EPF_PY:=$CONDA_BASE/envs/$EPF_ENV_NAME/bin/python}"  # ALWAYS absolute — `conda activate` may lose the PATH race (SETUP_GUIDE §2)
: "${EPF_DATA_ROOT:=$HOME/epf_data}"       # will hold mmau_pro_testmini/ + mmau_pro_audio/ (~75 GB)
: "${HF_HOME:=$EPF_DATA_ROOT/hf_cache}"; export HF_HOME   # Phi-4-MM lands here (~12 GB)
: "${NUM_GPUS:=$(nvidia-smi -L 2>/dev/null | wc -l)}"
: "${BASE_PORT:=8100}"                     # one vLLM replica per GPU on BASE_PORT+i
: "${GPU_MEM_UTIL:=0.85}"                  # NOT 0.9 — audio-encoder attention spikes OOM'd an engine at 0.9 (SETUP_GUIDE §10)

# --- experiment ----------------------------------------------------------------
# THE GRID (what run_all.sh executes, in this order): budgets 1..128 on the FULL
# test set (5,090 MCQ), 2 prompts × signals {mean_logprob, entropy} -> 4 curves.
# The reference box computed b1/8/16; those rows ship in run18/seeds/ and are
# REUSED automatically (resume skips them) — your run completes any b1/8/16
# remainder, then runs the new b32/64/128 stages. Just run:
#   nohup bash benchmarking/mmau_pro/run18/run_all.sh > run18.log 2>&1 &
# Crashed/interrupted? Re-run the same line — nothing is recomputed.
: "${BUDGETS:=1 8 16 32 64 128}"           # staged in this order
: "${MAX_INFLIGHT:=64}"                    # per-endpoint item concurrency = max(1, MAX_INFLIGHT // budget)
: "${MAX_INFLIGHT_B1:=24}"                 # budget-1 stages need the lower cap (SETUP_GUIDE §7)
: "${PROBE_LIMIT:=6000}"                   # >= 5,090 covers every item; set 4 to rehearse the driver
: "${N_BOOT:=10000}"                       # bootstrap resamples for the final HTML

# --- model / prompts (finalized at the Phase-A HOLD points) ---------------------
RUN18_MODEL_ID="microsoft/Phi-4-multimodal-instruct"
RUN18_REV="93f923e1a7727d1c4f446756212d9d3e8fcc5d81"        # single snapshot = refs/main (3 shards + custom code)
RUN18_NAME="phi4mm"
: "${RUN18_MAXLEN:=32768}"                 # 12.5 audio tok/s -> worst 3x600s item ~22.5k; audited in Phase A
RUN18_SUBSET="test"                        # full 5,090 MCQ
RUN18_STEM="epf_phi4mm_5090"
: "${PROMPTS_RUN18:=4,8}"                  # HOLD-2 decision: P4 plan-and-solve (chunker, cross-run comparable)
                                           # + P8 anti-shortcut (best acc); see RESULTS.md §18 screens
: "${STOP_REGEX:=}"                        # empty = none needed (P4/P8 end cleanly at 'Answer: <letter>')

# --- speech-LoRA serving ---------------------------------------------------------
# The checkpoint ships a speech-lora/ adapter that vLLM 0.22.1 does NOT auto-load
# (Phi4MMForCausalLM.load_weights skips all "lora" tensors). Microsoft's intended
# audio path is base + speech-lora, so SERVE_MODE=lora attaches it explicitly and
# every request targets the ADAPTER name (lib.sh probe_model_name), not the base.
: "${SERVE_MODE:=lora}"                    # lora | merged | base (fallback ladder, HOLD-1 decision)
SPEECH_LORA_NAME="speech"                  # the model= value requests use in lora mode
SPEECH_LORA_RANK=320                       # r=320 in the checkpoint's speech_lora config
: "${SPEECH_LORA_DIR:=$HF_HOME/hub/models--${RUN18_MODEL_ID//\//--}/snapshots/$RUN18_REV/speech-lora}"
: "${MERGED_DIR:=$EPF_DATA_ROOT/phi4mm_speech_merged}"   # SERVE_MODE=merged only: offline-merged weights

: "${OUT_ROOT:=$REPO_ROOT/benchmarking/mmau_pro/results/run18_phi4mm}"
: "${DRY_RUN:=0}"                          # 1 = print the plan instead of executing

# --- dataset pins (same sources as run16/17) --------------------------------------
PARQUET_DATASET="macabdul9/MMAU_Pro_Testmini"                     # ships testmini AND full-test parquets
PARQUET_REV="81eb01fb86bfc183dcb88b376ab1ca149a9d9c4b"
AUDIO_DATASET="gamma-lab-umd/MMAU-Pro"                            # data.zip: 44 GiB -> 5,787 files / 53 GB
AUDIO_ZIP_SHA256="8fab3e820b27bf7f239ae74a45a1085f1364b6971a7b5ac53f220d091b9b111c"

: "${DATA_TESTMINI:=$EPF_DATA_ROOT/mmau_pro_testmini}"
: "${DATA_AUDIO:=$EPF_DATA_ROOT/mmau_pro_audio}"
