#!/bin/bash
# Download the Phi-4-multimodal-instruct checkpoint at the PINNED revision and
# verify it (base shards + the speech-lora adapter; vision-lora may come along in
# the full snapshot download — harmless and unused for this audio-only benchmark).
# SERVE_MODE=merged additionally builds the offline speech-lora merge.
# Idempotent — re-running only fills gaps.
set -euo pipefail
RUN18_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUN18_DIR/config.sh"
source "$RUN18_DIR/lib.sh"

echo "== run18: $RUN18_MODEL_ID @ $RUN18_REV (SERVE_MODE=$SERVE_MODE)"
ensure_model "$RUN18_MODEL_ID" "$RUN18_REV"

if [ "$SERVE_MODE" = "merged" ] && ! check_merged > /dev/null 2>&1; then
  echo "== merging speech-lora into base weights -> $MERGED_DIR"
  "$EPF_PY" "$RUN18_DIR/merge_speech_lora.py" \
    --snapshot "$HF_HOME/hub/models--${RUN18_MODEL_ID//\//--}/snapshots/$RUN18_REV" \
    --out "$MERGED_DIR"
  check_merged
fi
echo "MODEL OK (pinned revision present in $HF_HOME)"
