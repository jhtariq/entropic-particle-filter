#!/bin/bash
# Download all six model checkpoints at the EXACT pinned revisions; verify-first,
# idempotent. Sizes: Kimi ~22.6 GB (TTS subfolders excluded), Omni-7B ~22 GB,
# Omni-3B ~8 GB, Qwen2-Audio ~17 GB, Phi-4-MM ~12 GB (+speech-lora), Mellow+
# SmolLM2 ~1.7 GB + a small code clone -> ~85 GB total under HF_HOME.
set -euo pipefail
GREEDY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$GREEDY_DIR/config.sh"
source "$GREEDY_DIR/lib.sh"
cd "$REPO_ROOT"

for m in $GREEDY_MODELS; do
  echo "== $m"
  ensure_model "$m"
done

echo "== all six checkpoints OK; HF cache size:"
du -sh "$HF_HOME" 2>/dev/null || true
