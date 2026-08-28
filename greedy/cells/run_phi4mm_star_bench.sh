#!/bin/bash
# Greedy cell: phi4mm x star_bench. Self-contained servant: serve -> health-gate ->
# greedy run (resumable) -> teardown. Safe to re-run individually.
# MEDIA_ROOT is overridden (exported BEFORE sourcing config.sh, so the ":=" default
# there does not clobber it) because star-bench audio lives outside EPF_DATA_ROOT —
# see greedy/config.sh's DATA_STAR_BENCH comment.
set -euo pipefail
export MEDIA_ROOT="/work/hdd/bcey/awaheed/its-for-audio-reasoning"
# Delta driver (570.148.08 / CUDA 12.8) predates the pinned cu130 wheels;
# greedy/ (built for a different reference box) never loads this — required here.
module load cuda-compat/13.0 2>/dev/null || true
source "$(dirname "${BASH_SOURCE[0]}")/../config.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
cd "$REPO_ROOT"
run_cell phi4mm star_bench
