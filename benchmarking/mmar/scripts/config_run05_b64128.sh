# MMAR Run 5 EXTENSION — budgets 64+128 on the existing Phi-4-MM grid JSONL.
# Speech-LoRA contract (Run 18's #1 gotcha): requests MUST target adapter "speech";
# SERVED_NAME stays "speech", the base gets SERVE_BASE_NAME. DUAL-GPU (vs the
# original single-GPU run) — two LoRA replicas; if the second replica misbehaves,
# set GPUS="0" + single ENDPOINT. Est. ~20 h dual / ~40 h single.
MODEL_ID="microsoft/Phi-4-multimodal-instruct"
MODEL_REV="93f923e1a7727d1c4f446756212d9d3e8fcc5d81"
SERVED_NAME="speech"
SERVE_BASE_NAME="phi4mm"
ENABLE_LORA=1
MAX_LORA_RANK=320
MAX_LORAS=1
SPEECH_LORA_DIR="$HOME/.cache/huggingface/hub/models--microsoft--Phi-4-multimodal-instruct/snapshots/$MODEL_REV/speech-lora"
LORA_MODULES="speech=$SPEECH_LORA_DIR"
HF_HOME_DIR="$HOME/.cache/huggingface"   # checkpoint lives in the default cache, not the big volume
PROMPTS="4,8"
SUBSET="full"
N_ITEMS=1000
GPUS="0 1"
ENDPOINTS="http://localhost:8100/v1,http://localhost:8101/v1"
MAX_MODEL_LEN=32768
BUDGETS="64 128"
STEM="mmar_run05"
OUT_DIR="$REPO_ROOT/benchmarking/mmar/results/run05_phi4mm"
