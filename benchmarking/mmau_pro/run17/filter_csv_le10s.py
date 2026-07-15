"""Split the Run 17 grid CSV into the LE10S slice (for the two-section report).

The 326-item `test_le10s_flat` subset is a strict subset of the `test_no3a` grid set,
so its rows are filtered straight out of the full CSV — no extra compute.

Usage:
    python benchmarking/mmau_pro/run17/filter_csv_le10s.py \
        --in .../run17/epf_mellow_no3a.csv --out .../run17/epf_mellow_le10s.csv \
        --le10s-parquet .../mmau_pro_testmini/data/test_le10s_flat-00000-of-00001.parquet
"""

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--le10s-parquet", required=True)
    args = ap.parse_args()

    import pandas as pd

    ids = set(pd.read_parquet(args.le10s_parquet, columns=["id"])["id"].astype(str))
    df = pd.read_csv(args.src)
    out = df[df["unique_id"].astype(str).isin(ids)]
    out.to_csv(args.out, index=False)
    print(f"{len(df)} rows -> {len(out)} LE10S rows ({out['unique_id'].nunique()} items) -> {args.out}")


if __name__ == "__main__":
    main()
