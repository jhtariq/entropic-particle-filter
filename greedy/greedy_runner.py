"""Greedy (temp 0), NO-CoT, single-shot direct-answer baseline — generic over the
three benchmarks (MMAU-Pro / MMAR / MMSU) and all six served models.

One plain chat completion per item: no particle filtering, no StepGeneration, no
budget machinery. The prompt is the terse lettered-MCQ `build_messages` with the
NO-CoT system prompt from `benchmarking.mmsu.greedy_baseline` ("reply with ONLY
the single letter"); parsing reuses the SAME `predicted_index` the grids use, so
accuracy is directly comparable. Concurrent across endpoints, resumable via the CSV.

`--max-audios N` clamps how many clips are SENT per item (locked full-set decision:
capability limits are handled by clamping, not subsetting) — Kimi-Audio accepts one
audio per prompt, the Mellow shim has two slots; extra clips are dropped in order.

`--expected N` turns the final tally into a hard gate: the process exits non-zero
unless every loaded record has an error-free row and exactly N distinct items are
clean — this is what lets the bash servants (and the master) abort reliably.

    python -m greedy.greedy_runner \
        --bench mmar --endpoints http://localhost:8100/v1 --model-name qwen-omni \
        --data-root ~/epf_data/mmar --max-audios 3 --expected 1000 \
        --csv  greedy/results/qwen_omni_7b/mmar/greedy.csv \
        --log  greedy/results/qwen_omni_7b/mmar/greedy.log
"""

import asyncio
import csv
import os
import sys

import click

from benchmarking.mmar.loader import load_mmar_mcq
from benchmarking.mmau_pro.loader import load_mmau_mcq
from benchmarking.mmau_pro.prompt import build_messages
from benchmarking.mmau_pro.scoring import LETTERS, is_correct, predicted_index
from benchmarking.mmsu.greedy_baseline import NO_COT_SYS
from benchmarking.mmsu.loader import load_mmsu_mcq
from benchmarking.star_bench.loader import load_star_bench_mcq
from its_hub import OpenAICompatibleLanguageModel
from its_hub.core.utils import extract_content_from_lm_response

LOADERS = {
    "mmau": load_mmau_mcq,
    "mmar": load_mmar_mcq,
    "mmsu": load_mmsu_mcq,
    "star_bench": load_star_bench_mcq,
}
DEFAULT_SUBSET = {"mmau": "test", "mmar": "full", "mmsu": "full", "star_bench": "full"}

CSV_FIELDS = ["unique_id", "category", "gold_letter", "pred_letter", "correct",
              "n_choices", "n_audios_sent", "content", "error"]


@click.command()
@click.option("--bench", type=click.Choice(list(LOADERS)), required=True)
@click.option("--endpoints", default="http://localhost:8100/v1,http://localhost:8101/v1",
              help="comma list of endpoints; items round-robin across them")
@click.option("--model-name", required=True,
              help="the model= value for requests (for phi4mm: the LoRA name 'speech')")
@click.option("--api-key", default="NO_API_KEY")
@click.option("--data-root", required=True)
@click.option("--subset", default=None, help="default: test (mmau) / full (mmar, mmsu)")
@click.option("--audio-root", default=None,
              help="root for relative audio paths when it differs from data-root (mmau test)")
@click.option("--audio-mode", default="local-path")
@click.option("--max-audios", default=3, type=int,
              help="clamp: send at most N clips per item (kimi=1, mellow=2, else 3)")
@click.option("--limit", default=None, type=int, help="cap items (default: all)")
@click.option("--ids", default=None,
              help="comma list of unique_ids to run (smoke stress targeting)")
@click.option("--max-tokens", default=64, help="short — the prompt asks for ONLY the letter")
@click.option("--max-inflight", default=16, help="concurrent requests PER endpoint")
@click.option("--expected", default=None, type=int,
              help="hard gate: exit non-zero unless exactly N distinct items are clean")
@click.option("--csv", "csv_path", required=True)
@click.option("--log", "log_path", default=None)
def main(bench, endpoints, model_name, api_key, data_root, subset, audio_root,
         audio_mode, max_audios, limit, ids, max_tokens, max_inflight, expected,
         csv_path, log_path):
    subset = subset or DEFAULT_SUBSET[bench]
    records = LOADERS[bench](data_root, subset=subset, limit=limit, audio_root=audio_root)
    if ids:
        want = {i.strip() for i in ids.split(",") if i.strip()}
        records = [r for r in records if r.unique_id in want]
        missing_ids = want - {r.unique_id for r in records}
        if missing_ids:
            print(f"FATAL: --ids not found in {bench}/{subset}: {sorted(missing_ids)}", flush=True)
            sys.exit(2)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # resume: skip unique_ids already present (non-error) in the CSV
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                if not r.get("error"):
                    done.add(r["unique_id"])
    todo = [r for r in records if r.unique_id not in done]
    print(f"loaded {len(records)} records (bench={bench} subset={subset}); {len(todo)} to do "
          f"({len(done)} resumed) | max_audios={max_audios} | endpoints={endpoints}", flush=True)

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
        rec.audio_paths = rec.audio_paths[:max_audios]  # capability clamp, in clip order
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
            "n_audios_sent": len(rec.audio_paths),
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

    # score everything on disk (resumed + new); the CSV is append-only, so after a
    # crash + resume an item can have an old error row AND a fresh clean row —
    # keep only the LAST (latest) row per item for the report and the gate
    with open(csv_path) as f:
        latest = {r["unique_id"]: r for r in csv.DictReader(f)}
    rows = list(latest.values())
    n = sum(1 for r in rows if r["correct"] in ("True", "False"))
    corr = sum(1 for r in rows if r["correct"] == "True")
    parsed = sum(1 for r in rows if r["pred_letter"])
    errs = sum(1 for r in rows if r["error"])
    split = {}
    for r in rows:
        if r["correct"] not in ("True", "False"):
            continue
        a = split.setdefault(r["category"], [0, 0])
        a[0] += r["correct"] == "True"
        a[1] += 1
    lines = [
        f"greedy direct no-CoT (temp 0), bench={bench} subset={subset}: n={n} errors={errs}",
        f"  overall accuracy : {corr}/{n} = {corr / max(n,1):.4f}   (unparsed counts as wrong)",
        f"  acc among parsed : {corr}/{parsed} = {corr / max(parsed,1):.4f}",
        f"  parse rate       : {parsed}/{len(rows)} = {parsed / max(len(rows),1):.4f}",
    ]
    for c in sorted(split):
        s, m = split[c]
        lines.append(f"  {c:24s} : {s}/{m} = {s / max(m,1):.4f}")
    report = "\n".join(lines)
    print("\n" + report)
    if log_path:
        with open(log_path, "w") as f:
            f.write(report + "\n")

    # hard completion gate — makes servant/master failure reliable
    ok_ids = {r["unique_id"] for r in rows if not r.get("error")}
    missing = [r.unique_id for r in records if r.unique_id not in ok_ids]
    if missing:
        print(f"GATE FAIL: {len(missing)} items have no clean row "
              f"(first few: {missing[:5]})", flush=True)
        sys.exit(2)
    if expected is not None and len(ok_ids & {r.unique_id for r in records}) != expected:
        print(f"GATE FAIL: {len(ok_ids)} clean items, expected {expected}", flush=True)
        sys.exit(2)
    print("GATE OK: all loaded items have a clean row"
          + (f" (n={expected})" if expected is not None else ""))


if __name__ == "__main__":
    main()
