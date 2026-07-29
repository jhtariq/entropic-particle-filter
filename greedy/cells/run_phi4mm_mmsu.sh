#!/bin/bash
# Greedy cell: phi4mm x mmsu. Self-contained servant: serve -> health-gate ->
# greedy run (resumable) -> teardown. Safe to re-run individually.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../config.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
cd "$REPO_ROOT"
run_cell phi4mm mmsu
