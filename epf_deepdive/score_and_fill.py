"""Score a probe-EPF result JSONL and (optionally) file the results into the
consolidated CSVs — the standing 3-selector reporting convention in one command.

For every (arm, budget) cell in the input file(s) this prints the THREE
committed selectors side by side: probe_marg_mean, probe_marginal, ent_marg.
Never report a lone best-of-cell number.

With --fill-csv it appends 3 rows per (arm, budget) to
consolidated_results/<benchmark>.csv (selector name in `signal`, accuracy in
`majority_acc`, method=epf-probe). Append-only; refuses partial files and
duplicate rows unless overridden.

Usage (login node is fine — pure post-processing):
  /u/awaheed/envs/epf/bin/python score_and_fill.py probe_out/mmar_7b_curve.jsonl \
      --benchmark mmar --model qwen2.5-omni-7b --prompt "P5 CoT" \
      --expect 996 --fill-csv \
      --notes "champion probe_adaptive_t3 (probe logit(pmax)/3, ESS-0.5 gate); t=0.8"

Multiple JSONLs (e.g. a run split across two GPUs) can be listed; rows are
deduped by (unique_id, arm, budget), last write wins.
"""

import argparse
import csv
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_epf import selectors_from_row  # noqa: E402  (parity: same code scores everything)

COMMITTED = ["probe_marg_mean", "probe_marginal", "ent_marg"]
SCHEMA = ["benchmark", "model", "method", "prompt", "signal", "budget", "n_items",
          "n_gradeable", "selected_acc", "majority_acc", "oracle_acc",
          "run_folder", "date", "notes"]
CSV_DIR = "/work/hdd/bcey/awaheed/its-for-audio-reasoning/consolidated_results"


def load_rows(paths):
    by = {}
    n_err = 0
    for path in paths:
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("error"):
                    n_err += 1
                    continue
                by[(r["unique_id"], r["arm"], r["budget"])] = r
    return list(by.values()), n_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="+")
    ap.add_argument("--benchmark", required=True,
                    help="consolidated CSV stem AND the benchmark column value "
                         "(mmar, mmau-pro-d1k, ...)")
    ap.add_argument("--model", required=True,
                    help="model column, matching existing rows "
                         "(qwen2.5-omni-3b | qwen2.5-omni-7b | qwen2-audio | ...)")
    ap.add_argument("--prompt", required=True, help='e.g. "P4 CoT" or "P5 CoT"')
    ap.add_argument("--expect", type=int, default=None,
                    help="expected n_gradeable per cell; cells below this are "
                         "flagged PARTIAL and NOT filed (partial files inflate "
                         "accuracy — early loader uids are easier)")
    ap.add_argument("--n-items", type=int, default=1000,
                    help="items attempted (the --limit of the run)")
    ap.add_argument("--fill-csv", action="store_true",
                    help="append 3 selector rows per cell to the consolidated CSV")
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="append even if a matching row already exists")
    ap.add_argument("--notes", default="")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default: today)")
    ap.add_argument("--csv-dir", default=CSV_DIR)
    a = ap.parse_args()

    rows, n_err = load_rows(a.jsonl)
    if n_err:
        print(f"note: skipped {n_err} error rows (unresumed failures?)")
    cells = {}
    for r in rows:
        cells.setdefault((r["arm"], r["budget"]), []).append(r)

    print(f"\n{'arm':>20} {'bud':>4} {'n':>5}  " + "  ".join(f"{k:>15}" for k in COMMITTED))
    results = []
    for (arm, bud), rs in sorted(cells.items()):
        accs = {}
        for k in COMMITTED:
            hit = sum(1 for r in rs if selectors_from_row(r).get(k) == r["gold"])
            accs[k] = hit / len(rs)
        partial = a.expect is not None and len(rs) < a.expect
        flag = "  <-- PARTIAL, not filed" if partial else ""
        print(f"{arm:>20} {bud:>4} {len(rs):>5}  "
              + "  ".join(f"{accs[k]:>15.4f}" for k in COMMITTED) + flag)
        results.append((arm, bud, len(rs), accs, partial))

    if not a.fill_csv:
        return
    path = os.path.join(a.csv_dir, f"{a.benchmark}.csv")
    with open(path) as f:
        existing = list(csv.DictReader(f))
        assert existing and list(existing[0].keys()) == SCHEMA, f"schema mismatch in {path}"
    have = {(e["model"], e["method"], e["prompt"], e["signal"], e["budget"])
            for e in existing}
    date = a.date or datetime.date.today().isoformat()
    run_folder = os.path.dirname(os.path.abspath(a.jsonl[0]))
    files = "+".join(os.path.basename(p) for p in a.jsonl)
    notes = (a.notes + "; " if a.notes else "") + f"file(s): {files}"

    new = []
    for arm, bud, n, accs, partial in results:
        if partial and not a.allow_partial:
            continue
        for k in COMMITTED:
            key = (a.model, "epf-probe", a.prompt, k, str(bud))
            if key in have and not a.force:
                print(f"skip duplicate: {key} already in {os.path.basename(path)}")
                continue
            new.append({"benchmark": a.benchmark, "model": a.model,
                        "method": "epf-probe", "prompt": a.prompt, "signal": k,
                        "budget": bud, "n_items": a.n_items, "n_gradeable": n,
                        "selected_acc": "", "majority_acc": f"{accs[k]:.4f}",
                        "oracle_acc": "", "run_folder": run_folder,
                        "date": date, "notes": notes})
    if not new:
        print("nothing to file")
        return
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SCHEMA)
        w.writerows(new)
    print(f"appended {len(new)} rows to {path}")


if __name__ == "__main__":
    main()
