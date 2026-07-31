"""Greedy (temp 0), NO-CoT, single-shot direct-answer baseline over MMAU test-mini.

Unlike the EPF/PF grid (step-by-step CoT + resampling) and cot_compare (CoT prompts),
this asks the model for the letter directly — no reasoning — at temperature 0, once per
item. It is the honest "greedy zero-shot without CoT" reference for the selected/oracle
numbers from the particle runs, and the row that compares against the PUBLISHED
MMAU test-mini baselines (Omni-7B 65.9 avg — arXiv 2505.09439 Table I).

The prompt reuses `build_messages` (the terse lettered-MCQ prompt) but with a NO-CoT
system prompt (the default MMAU_MCQ_SYSTEM_PROMPT elicits step-by-step reasoning, so it is
NOT used here). Parsing reuses the SAME `predicted_index` the grid uses, so the accuracy is
directly comparable. Concurrent across endpoints, resumable via the CSV.

    python -m benchmarking.mmau.greedy_baseline \
        --endpoints http://localhost:8100/v1 --model-name qwen-omni \
        --csv  benchmarking/mmau/results/run00_greedy_nocot/greedy.csv \
        --log  benchmarking/mmau/results/run00_greedy_nocot/greedy.log
"""

import asyncio
import csv
import os

import click

from benchmarking.mmau.loader import (
    DEFAULT_DATA_ROOT,
    SUBSET_FILES,
    load_metadata,
    load_mmau_mcq,
)
from benchmarking.mmau_pro.prompt import METHODS, build, build_messages
from benchmarking.mmau_pro.scoring import LETTERS, is_correct, predicted_index
from its_hub import OpenAICompatibleLanguageModel
from its_hub.core.utils import extract_content_from_lm_response

# Direct-answer system prompt — deliberately NO "reason step by step" (that is what the
# default MMAU_MCQ_SYSTEM_PROMPT does). This is the no-CoT baseline.
NO_COT_SYS = (
    "You are an expert audio analyst. Listen to the provided audio and answer the "
    "multiple-choice question by replying with ONLY the single letter of the correct "
    "option (e.g. 'B'). Do not explain or show any reasoning."
)

CSV_FIELDS = ["unique_id", "category", "gold_letter", "pred_letter", "correct",
              "n_choices", "content", "error"]


@click.command()
@click.option("--endpoints", default="http://localhost:8100/v1",
              help="comma list of vLLM endpoints; items round-robin across them")
@click.option("--model-name", required=True)
@click.option("--api-key", default="NO_API_KEY")
@click.option("--data-root", default=DEFAULT_DATA_ROOT)
@click.option("--subset", type=click.Choice(list(SUBSET_FILES)), default="full")
@click.option("--audio-root", default=None)
@click.option("--audio-mode", default="local-path")
@click.option("--method", default=0,
              help="0 = direct no-CoT (build_messages); else a build() CoT method id (e.g. 4 = plan-and-solve)")
@click.option("--limit", default=None, type=int, help="cap items (default: all)")
@click.option("--max-tokens", default=64, help="short for no-CoT; raise (e.g. 1024) for CoT")
@click.option("--max-inflight", default=8, help="concurrent requests PER endpoint (low, to co-exist with a running grid)")
@click.option("--csv", "csv_path", required=True)
@click.option("--log", "log_path", default=None)
def main(endpoints, model_name, api_key, data_root, subset, audio_root, audio_mode,
         method, limit, max_tokens, max_inflight, csv_path, log_path):
    records = load_mmau_mcq(data_root, subset=subset, limit=limit, audio_root=audio_root)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # resume: skip unique_ids already present (non-error) in the CSV
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                if not r.get("error"):
                    done.add(r["unique_id"])
    todo = [r for r in records if r.unique_id not in done]
    print(f"loaded {len(records)} records (subset={subset}); {len(todo)} to do "
          f"({len(done)} resumed) | endpoints={endpoints}", flush=True)

    eps = [e.strip() for e in endpoints.split(",") if e.strip()]
    lms = [OpenAICompatibleLanguageModel(endpoint=e, api_key=api_key, model_name=model_name)
           for e in eps]
    sems = [asyncio.Semaphore(max_inflight) for _ in lms]

    new_file = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    fcsv = open(csv_path, "a", newline="")  # noqa: SIM115 — long-lived append handle, closed in _run finally
    writer = csv.DictWriter(fcsv, fieldnames=CSV_FIELDS)
    if new_file:
        writer.writeheader()
    lock = asyncio.Lock()

    async def _one(rec, lm, sem):
        # method 0 = terse direct no-CoT (NO_COT_SYS); else a build() CoT prompt (e.g. P4)
        if method:
            msgs, _ = build(method, rec, audio_mode=audio_mode)
        else:
            msgs = build_messages(rec, audio_mode=audio_mode, system_prompt=NO_COT_SYS)
        try:
            async with sem:
                resp = await lm.agenerate_single(msgs, max_tokens=max_tokens, temperature=0.0)
            content = extract_content_from_lm_response(resp)
            err = None
        except Exception as e:  # record and continue
            content, err = "", f"{type(e).__name__}: {e}"
        pi = None if err else predicted_index(content, rec.choices)
        gi = rec.answer_index
        correct = None if err else is_correct(content, rec.choices, gi)
        row = {
            "unique_id": rec.unique_id, "category": rec.category,
            "gold_letter": LETTERS[gi] if gi is not None else "",
            "pred_letter": LETTERS[pi] if pi is not None else "",
            "correct": "" if correct is None else bool(correct),
            "n_choices": len(rec.choices),
            "content": (content or "").replace("\n", " ")[:200], "error": err,
        }
        async with lock:
            writer.writerow(row)
            fcsv.flush()
        return row

    async def _run():
        try:
            tasks = [_one(rec, lms[i % len(lms)], sems[i % len(lms)])
                     for i, rec in enumerate(todo)]
            for fut in asyncio.as_completed(tasks):
                await fut
        finally:
            fcsv.close()
            for lm in lms:
                await lm.close()

    asyncio.run(_run())

    # score everything on disk (resumed + new)
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    meta = load_metadata(data_root, subset=subset)
    n = sum(1 for r in rows if r["correct"] in ("True", "False"))
    corr = sum(1 for r in rows if r["correct"] == "True")
    parsed = sum(1 for r in rows if r["pred_letter"])
    errs = sum(1 for r in rows if r["error"])

    def _split(key_fn):
        acc = {}
        for r in rows:
            if r["correct"] not in ("True", "False"):
                continue
            a = acc.setdefault(key_fn(r), [0, 0])
            a[0] += r["correct"] == "True"
            a[1] += 1
        return acc

    # three joinable axes: task (CSV category = the official leaderboard axis),
    # coarse Reasoning/Information-Extraction, and difficulty (both via load_metadata)
    splits = [
        ("task", _split(lambda r: r["category"])),
        ("category", _split(lambda r: meta.get(r["unique_id"], {}).get("category") or "?")),
        ("difficulty", _split(lambda r: meta.get(r["unique_id"], {}).get("difficulty") or "?")),
    ]
    label = f"P{method} {METHODS.get(method, '?')} (CoT)" if method else "direct no-CoT"
    lines = [
        f"greedy {label} (temp 0), subset={subset}: n={n} errors={errs}",
        f"  overall accuracy : {corr}/{n} = {corr / max(n,1):.4f}   (unparsed counts as wrong)",
        f"  acc among parsed : {corr}/{parsed} = {corr / max(parsed,1):.4f}",
        f"  parse rate       : {parsed}/{len(rows)} = {parsed / max(len(rows),1):.4f}",
    ]
    for axis, split in splits:
        for c in sorted(split):
            s, m = split[c]
            lines.append(f"  {axis} {c:24s}: {s}/{m} = {s / max(m,1):.4f}")
    report = "\n".join(lines)
    print("\n" + report)
    if log_path:
        with open(log_path, "w") as f:
            f.write(report + "\n")


if __name__ == "__main__":
    main()
