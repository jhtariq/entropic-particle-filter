"""Per-budget accuracy report for Run 17, on BOTH evaluation sets.

Reads the grid JSONL and prints selected/oracle/majority accuracy per
(prompt, signal, budget) cell for:
  - FULL:   every gradeable item in the run subset (test_no3a, 5,073 MCQ)
  - LE10S:  the 326-item fully-heard slice (test_le10s_flat parquet ids) —
            a strict subset of FULL, so no extra compute is ever needed.

Usage:
    python benchmarking/mmau_pro/run17/report_budget_acc.py \
        --jsonl .../results/run17_mellow/run17/epf_mellow_no3a.jsonl \
        --le10s-parquet .../mmau_pro_testmini/data/test_le10s_flat-00000-of-00001.parquet \
        [--budget 8]
"""

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--le10s-parquet",
                    default="/home/exx/inference-time-scaling/mmau_pro_testmini/data/"
                            "test_le10s_flat-00000-of-00001.parquet")
    ap.add_argument("--budget", type=int, default=None, help="only this budget")
    ap.add_argument("--full-ids-file", default=None,
                    help="restrict the FULL slice to these ids (one per line) — e.g. a "
                         "500-item test selection; default = every row in the jsonl")
    args = ap.parse_args()

    import numpy as np
    import pandas as pd

    le10s_ids = set(pd.read_parquet(args.le10s_parquet, columns=["id"])["id"].astype(str))
    full_ids = None
    if args.full_ids_file:
        with open(args.full_ids_file) as f:
            full_ids = {line.strip() for line in f if line.strip()}

    # dedupe keep-last per cell key (resume may append replacements)
    rows = {}
    with open(args.jsonl) as f:
        for line in f:
            r = json.loads(line)
            rows[(r["unique_id"], r["method"], r["signal"], r["budget"])] = r
    rows = [r for r in rows.values() if not r.get("error")]

    cells = {}
    for r in rows:
        cells.setdefault((r["budget"], r["method"], r["signal"]), []).append(r)

    def acc(rs, metric):
        graded = [r for r in rs if r.get("gold_letter") is not None and r.get(metric) is not None]
        return (np.mean([bool(r[metric]) for r in graded]), len(graded)) if graded else (float("nan"), 0)

    print(f"{'set':6s} {'bud':>3s} {'prompt':>6s} {'signal':12s} {'n':>5s} "
          f"{'selected':>8s} {'oracle':>8s} {'majority':>8s} {'parsed':>7s}")
    for (b, m, sig) in sorted(cells):
        if args.budget is not None and b != args.budget:
            continue
        rs = cells[(b, m, sig)]
        full_rows = rs if full_ids is None else [r for r in rs if str(r["unique_id"]) in full_ids]
        for label, sel in (("FULL", full_rows), ("LE10S", [r for r in rs if str(r["unique_id"]) in le10s_ids])):
            s, n = acc(sel, "selected_correct")
            o, _ = acc(sel, "oracle_correct")
            mj, _ = acc(sel, "majority_correct")
            parsed = np.mean([r.get("parsed_ratio") or 0 for r in sel]) if sel else float("nan")
            print(f"{label:6s} {b:3d} {'P'+str(m):>6s} {sig:12s} {n:5d} "
                  f"{s:8.3f} {o:8.3f} {mj:8.3f} {parsed:7.2f}")


if __name__ == "__main__":
    main()
