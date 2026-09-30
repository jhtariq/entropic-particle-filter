"""Download STAR-Bench-Perception (macabdul9/STAR-Bench-Perception, 626 MCQ)
into BABEL_STARBENCH_ROOT and print a schema sanity summary.

Usage:  BABEL_STARBENCH_ROOT=/data/starbench python starbench_fetch.py
"""
import glob
import os
import sys

os.environ.pop("HF_HUB_OFFLINE", None)
DEST = os.environ.get("BABEL_STARBENCH_ROOT")
if not DEST:
    sys.exit("set BABEL_STARBENCH_ROOT to the target dataset directory")

from huggingface_hub import snapshot_download

path = snapshot_download("macabdul9/STAR-Bench-Perception", repo_type="dataset",
                         local_dir=DEST)
print(f"snapshot -> {path}")

import pandas as pd

parquets = sorted(glob.glob(os.path.join(DEST, "data", "*.parquet")))
df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
print(f"rows: {len(df)}  columns: {list(df.columns)}")
assert len(df) == 626, f"expected 626 rows, got {len(df)}"
assert all(len(list(c)) == 4 for c in df["choices"]), "expected 4 choices everywhere"
assert all(str(a) in list(c) for a, c in zip(df["answer"], df["choices"])), \
    "answer string must appear in choices"
print("STARBENCH_FETCH_OK")
