#!/bin/bash
# Download the Kimi-Audio-7B-Instruct checkpoint at the PINNED revision and verify
# it. Download is ~22.6 GB: the TTS-only subfolders (audio_detokenizer/ 19 GB +
# vocoder/ 1 GB) are EXCLUDED — vLLM's audio-in/text-out path never touches them.
# The whisper-large-v3/ encoder subfolder and tiktoken.model ARE required and
# ship inside this repo (no separate tokenizer download).
# Idempotent — re-running only fills gaps.
set -euo pipefail
RUN19_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUN19_DIR/config.sh"
source "$RUN19_DIR/lib.sh"

echo "== run19: $RUN19_MODEL_ID @ $RUN19_REV (~22.6 GB; excludes TTS-only detokenizer/vocoder)"
ensure_model "$RUN19_MODEL_ID" "$RUN19_REV"
echo "MODEL OK (pinned revision present in $HF_HOME)"
