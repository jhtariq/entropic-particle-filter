# Policy: Phi-4-multimodal with the speech LoRA (matches MMAR Run 5).
# REQUEST model name is the adapter ("speech"), not the base ("phi4mm").
MODEL_ID="microsoft/Phi-4-multimodal-instruct"
MODEL_REV="93f923e1a7727d1c4f446756212d9d3e8fcc5d81"
SERVED_NAME="speech"           # request/adapter name — runner --model-name uses this
SERVE_BASE_NAME="phi4mm"       # vLLM --served-model-name (the base model)
SERVE_KIND=vllm
ENABLE_LORA=1
MAX_LORA_RANK=320
MAX_LORAS=1
SPEECH_LORA_DIR="$HOME/.cache/huggingface/hub/models--microsoft--Phi-4-multimodal-instruct/snapshots/$MODEL_REV/speech-lora"
LORA_MODULES="speech=$SPEECH_LORA_DIR"
HF_HOME_DIR="$HOME/.cache/huggingface"   # checkpoint lives in the default cache, not the big volume
SUBSET=full
GPUS="0"
