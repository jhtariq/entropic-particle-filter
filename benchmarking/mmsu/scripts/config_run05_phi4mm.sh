# MMSU Run 5 — Phi-4-multimodal-instruct (5.6B, speech-LoRA vLLM serving; mmau_pro
# Run 18 / MMAR Run 5 lineage). MMSU convention (Runs 1–2): ONE prompt ×
# {mean_logprob, entropy} × DEEP ladder b{1,8,16,32,64,128} × 5,000. P4 chosen on
# the evidence (best b32 selected on MMAR Run 5: .464 vs P8's .437, mean_logprob)
# and for MMSU-internal uniformity (P4 is also Runs 1–2's prompt). Serve with
# scripts/serve.sh (it handles the LoRA block below). Sourced AFTER REPO_ROOT is set.
#
# SERVED_NAME (= probe --model-name, watchdog + health-check grep) is the ADAPTER
# name "speech"; the base model gets SERVE_BASE_NAME via --served-model-name.
MODEL_ID="microsoft/Phi-4-multimodal-instruct"
MODEL_REV="93f923e1a7727d1c4f446756212d9d3e8fcc5d81"
SERVED_NAME="speech"
SERVE_BASE_NAME="phi4mm"
PROMPTS="4"
BUDGETS="1 8 16 32 64 128"
STEM="mmsu_run05"
OUT_DIR="$REPO_ROOT/benchmarking/mmsu/results/run05_phi4mm"

# --- speech-LoRA serving (consumed by serve.sh) --------------------------------
ENABLE_LORA=1
MAX_LORA_RANK=320
MAX_LORAS=1
# derived from HF_HOME_DIR so it follows scripts/local.sh; override there if your
# Phi-4 snapshot lives in a different cache (on the reference box it is in
# ~/.cache/huggingface, NOT the big volume)
: "${SPEECH_LORA_DIR:=$HF_HOME_DIR/hub/models--microsoft--Phi-4-multimodal-instruct/snapshots/$MODEL_REV/speech-lora}"
LORA_MODULES="speech=$SPEECH_LORA_DIR"
MAX_MODEL_LEN=32768
