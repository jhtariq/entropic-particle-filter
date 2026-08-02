#!/usr/bin/env bash
# Download the original MMAU benchmark (arXiv 2410.19168, MMAU-v05.15.25) from
# Hugging Face. Both splits are fetched:
#   - test-mini : 1,000 items, WITH answers, audio embedded in the parquet   (the gradeable split)
#   - test      : 9,000 items, answers withheld; audio in test-audios.tar.gz  (leaderboard-only)
# Download only — no extraction / materialization / moving.
#
# Usage:  bash download_mmau.sh [TARGET_DIR]        # default TARGET_DIR: ./mmau
#
# Requires the Hugging Face CLI:  pip install -U 'huggingface_hub[cli]'
set -euo pipefail

TARGET="${1:-./mmau}"

TESTMINI_REPO="gamma-lab-umd/MMAU-test-mini"
TESTMINI_REV="ccd9696c0111ea7060827598f310558df0b71b0a"  # pinned to the version we used
TEST_REPO="gamma-lab-umd/MMAU-test"

# resolve a downloader (huggingface-cli, or the newer 'hf')
if command -v huggingface-cli >/dev/null 2>&1; then
    HF="huggingface-cli download"
elif command -v hf >/dev/null 2>&1; then
    HF="hf download"
else
    echo "ERROR: Hugging Face CLI not found. Install it with:" >&2
    echo "         pip install -U 'huggingface_hub[cli]'" >&2
    exit 1
fi

echo "==> test-mini (1,000 items, with answers) -> $TARGET/test-mini"
$HF "$TESTMINI_REPO" --repo-type dataset --revision "$TESTMINI_REV" --local-dir "$TARGET/test-mini"

echo "==> test (9,000 items, blind) -> $TARGET/test"
$HF "$TEST_REPO" --repo-type dataset --local-dir "$TARGET/test"

echo "Done. MMAU downloaded to: $TARGET"

