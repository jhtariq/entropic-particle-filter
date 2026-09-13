"""Verify the local MMAU-Pro d1k copy matches what the runner expects.

Usage:  BABEL_D1K_ROOT=/path/to/mmau_pro_d1k python sanity_check_data.py

Checks: the repo loader parses the benchmark, the gradeable count matches the
original campaign (991 of 1000), and every referenced audio file exists.
"""
import os
import sys

sys.path.insert(0, os.environ.get(
    "BABEL_REPO_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

root = os.environ.get("BABEL_D1K_ROOT")
if not root:
    sys.exit("set BABEL_D1K_ROOT to the local mmau_pro_d1k directory")

from benchmarking.mmau_pro.loader import load_mmau_mcq  # noqa: E402

recs = load_mmau_mcq(root, subset="d1k")
gradeable = [r for r in recs if r.answer_index is not None]
print(f"records: {len(recs)}  gradeable: {len(gradeable)}")

missing = []
for r in recs:
    for a in r.audio_paths:
        if not os.path.isfile(a):
            missing.append(a)
if missing:
    print(f"MISSING {len(missing)} audio files, first: {missing[:3]}")
    sys.exit(1)

if len(gradeable) != 991:
    print("WARNING: campaign reference is 991 gradeable items - counts differ, "
          "results may not be directly comparable")
    sys.exit(1)
print("DATA_OK")
