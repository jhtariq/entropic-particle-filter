#!/bin/bash
# Download Mellow's PINNED artifacts and verify them: the raw checkpoints from HF
# (v0.ckpt / v0_s.ckpt / v0.yaml), the SmolLM2-135M base (architecture + tokenizer;
# its weights are overwritten by the ckpt), and the pinned GitHub code clone
# (no PyPI package exists for Mellow). Idempotent — re-running only fills gaps.
set -euo pipefail
RUN17_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUN17_DIR/config.sh"
source "$RUN17_DIR/lib.sh"

echo "== run17: $RUN17_MODEL_ID @ $RUN17_REV (+ $SMOLLM2_ID @ ${SMOLLM2_REV:0:12} + code @ ${MELLOW_CODE_COMMIT:0:12})"
ensure_mellow
echo "ALL MODEL ARTIFACTS OK (pinned revisions present in $HF_HOME, code in $MELLOW_CODE_DIR)"
