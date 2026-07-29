"""Final 6x3 accuracy table from the 18 greedy CSVs.

    python -m greedy.summarize --results greedy/results

Accuracy = correct / gradeable (unparsed counts as wrong), matching the per-cell
report in greedy_runner. Cells whose CSV is missing show as '-'.
"""

import csv
import os

import click

MODELS = ["qwen_omni_7b", "qwen_omni_3b", "qwen2_audio", "phi4mm", "kimi_audio", "mellow"]
BENCHES = ["mmau", "mmar", "mmsu"]


def cell_stats(path: str):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    graded = [r for r in rows if r["correct"] in ("True", "False")]
    corr = sum(1 for r in graded if r["correct"] == "True")
    parsed = sum(1 for r in rows if r["pred_letter"])
    errs = sum(1 for r in rows if r["error"])
    return {
        "n": len(graded), "acc": corr / max(len(graded), 1),
        "parse_rate": parsed / max(len(rows), 1), "errors": errs,
    }


@click.command()
@click.option("--results", default="greedy/results")
def main(results):
    header = f"{'model':14s} " + " ".join(f"{b:>22s}" for b in BENCHES)
    lines = ["greedy direct no-CoT (temp 0) — accuracy (n graded, parse rate)", header,
             "-" * len(header)]
    for m in MODELS:
        cells = []
        for b in BENCHES:
            s = cell_stats(os.path.join(results, m, b, "greedy.csv"))
            if s is None:
                cells.append(f"{'-':>22s}")
            else:
                cells.append(f"{s['acc']:.4f} (n={s['n']}, p={s['parse_rate']:.2f})".rjust(22))
        lines.append(f"{m:14s} " + " ".join(cells))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
