"""Cross-check our lettered-MCQ scoring against MMAU's official token-overlap scorer.

Offline. Takes a diversity_probe CSV, maps each row's `selected_letter` back to the
choice TEXT, and grades that text with `string_match` imported from the official
`code/evaluation.py` (github.com/Sakshi113/MMAU @ 7468292, fetched into the data
root). Reports, per (prompt, signal) cell at the chosen budget: our lettered
accuracy (over gradeable rows) vs official accuracy on the same denominator, plus
the official all-1000 number.

LOAD-BEARING vs the MMAR edition: the official scorer is fed the RAW prefixed
strings (`choices_raw`/`answer_raw`, e.g. "(A) Man"), NOT the stripped texts the
prompt shows. string_match requires every gold token in the prediction — stripped
"Man" lacks the gold letter token "a" of "(A) Man" and would grade wrong across
the board. Feeding raw pred vs raw gold/choices keeps both scorers letter-exact,
including on the 16 items whose gold TEXT is duplicated under a second letter
(both scorers grade the twin letter wrong).

    python -m benchmarking.mmau.official_crosscheck \
        --csv benchmarking/mmau/results/run01_omni7b/mmau_run01.csv --budget 1
"""

import csv
import importlib.util
import json
import os
from collections import defaultdict

import click

from benchmarking.mmau.loader import DEFAULT_DATA_ROOT, SUBSET_FILES
from benchmarking.mmau_pro.scoring import LETTERS


def _load_string_match(eval_script):
    spec = importlib.util.spec_from_file_location("mmau_official_eval", eval_script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.string_match


@click.command()
@click.option("--csv", "csv_path", required=True, help="diversity_probe output CSV")
@click.option("--budget", default=1, help="grade the cells at this budget")
@click.option("--data-root", default=DEFAULT_DATA_ROOT)
@click.option("--eval-script", default=None,
              help="path to the official evaluation.py (default <data-root>/code/evaluation.py)")
@click.option("--out", "out_path", default=None, help="write the JSON report here")
@click.option("--emit-json", default=None,
              help="also write one official-format input JSON per cell into this dir")
def main(csv_path, budget, data_root, eval_script, out_path, emit_json):
    string_match = _load_string_match(eval_script or os.path.join(data_root, "code", "evaluation.py"))
    with open(os.path.join(data_root, SUBSET_FILES["full"])) as f:
        meta = {str(r["id"]): r for r in json.load(f)}

    cells = defaultdict(list)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if int(row["budget"]) == budget:
                cells[(int(row["method"]), row["signal"])].append(row)

    report = []
    print(f"{'cell':>22} {'n':>5} {'ours(grad)':>10} {'official(grad)':>15} "
          f"{'official(all)':>14} {'delta':>7}")
    for (method, signal) in sorted(cells):
        rows = cells[(method, signal)]
        ours_n = ours_c = off_g_c = off_all_c = 0
        samples = []
        for row in rows:
            m = meta[row["unique_id"]]
            letter = row["selected_letter"]
            idx = LETTERS.index(letter) if letter and letter in LETTERS else None
            # RAW prefixed text — what the official scorer's token match needs
            pred_text = m["choices_raw"][idx] if idx is not None and idx < len(m["choices_raw"]) else ""
            official = string_match(m["answer_raw"], pred_text, m["choices_raw"])
            off_all_c += int(official)
            if row["selected_correct"] in ("True", "False"):  # gradeable under our scorer
                ours_n += 1
                ours_c += int(row["selected_correct"] == "True")
                off_g_c += int(official)
            if emit_json:
                # official-format sample: the code's output_key is 'model_output'
                # (the repo README says model_prediction — the code wins)
                samples.append({
                    "id": m["id"], "question": m["question"],
                    "choices": m["choices_raw"], "answer": m["answer_raw"],
                    "task": m["task"], "difficulty": m["difficulty"],
                    "sub-category": m["sub-category"], "category": m["category"],
                    "model_output": pred_text,
                })
        ours = ours_c / max(ours_n, 1)
        off_g = off_g_c / max(ours_n, 1)
        off_all = off_all_c / max(len(rows), 1)
        print(f"P{method} {signal:>15} b{budget:<3} {len(rows):>5} {ours:>10.4f} "
              f"{off_g:>15.4f} {off_all:>14.4f} {off_g - ours:>+7.4f}")
        report.append({"method": method, "signal": signal, "budget": budget, "rows": len(rows),
                       "ours_gradeable": ours, "official_gradeable": off_g,
                       "official_all": off_all, "delta": off_g - ours})
        if emit_json:
            os.makedirs(emit_json, exist_ok=True)
            dest = os.path.join(emit_json, f"crosscheck_p{method}_{signal}_b{budget}.json")
            with open(dest, "w") as f:
                json.dump(samples, f, ensure_ascii=False, indent=1)
            print(f"    official-format input -> {dest}")

    if out_path:
        with open(out_path, "w") as f:
            json.dump(report, f, indent=1)
        print(f"report -> {out_path}")


if __name__ == "__main__":
    main()
