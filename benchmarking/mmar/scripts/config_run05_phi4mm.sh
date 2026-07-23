# MMAR Run 5 — Phi-4-multimodal-instruct EPF grid (mmau_pro Run 18 replica).
# Single GPU (gpu 0 / :8100). Full MMAR set (1000 items, 996 gradeable).
#
# Speech-LoRA is served as adapter "speech" (vLLM 0.22.1 does NOT auto-load it);
# EVERY request must use model="speech" — a base-named ("phi4mm") request is
# accepted and silently answers WITHOUT the adapter (Run 18's #1 gotcha). So
# SERVED_NAME (= probe --model-name, watchdog + health-check grep) is "speech",
# while the base gets its own --served-model-name via SERVE_BASE_NAME.
#
# P4+P8 per Run 18's screen (P8 best raw acc; P4 keeps cross-model comparability).
MODEL_ID="microsoft/Phi-4-multimodal-instruct"
MODEL_REV="93f923e1a7727d1c4f446756212d9d3e8fcc5d81"
SERVED_NAME="speech"           # REQUEST/adapter name — probe --model-name, watchdog & health check use this
SERVE_BASE_NAME="phi4mm"       # vLLM --served-model-name (the base model)
ENABLE_LORA=1
MAX_LORA_RANK=320
MAX_LORAS=1
SPEECH_LORA_DIR="$HOME/.cache/huggingface/hub/models--microsoft--Phi-4-multimodal-instruct/snapshots/$MODEL_REV/speech-lora"
LORA_MODULES="speech=$SPEECH_LORA_DIR"
HF_HOME_DIR="$HOME/.cache/huggingface"   # OVERRIDE common.sh big-volume cache — checkpoint is already in the default cache
PROMPTS="4,8"
SUBSET="full"
N_ITEMS=1000
GPUS="0"
ENDPOINTS="http://localhost:8100/v1"
MAX_MODEL_LEN=32768
BUDGETS="1 8 16 32"
STEM="mmar_run05"
OUT_DIR="$REPO_ROOT/benchmarking/mmar/results/run05_phi4mm"
