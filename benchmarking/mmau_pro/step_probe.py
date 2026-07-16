"""Budget-1 step-trajectory probe: how does a model behave under the EPF step machinery?

Reimplements the (scratchpad-only) Run-12 AF3 "param check". For each (prompt, item)
it drives ONE trajectory through StepGeneration exactly as EPF does — same
stop_token="Answer:", same step chunking (`--tokens-per-step N` or the default
"\\n\\n" delimiter), same `base_messages` audio carry, logprobs requested — and
records how the trajectory ends and how it is shaped:

  n_steps / tok_counts / total_tokens   step structure (tok_counts from the logprob summary)
  end in {stop, max_steps, eos}          stop = "Answer:" seen; eos = natural end-of-turn
                                         (vLLM stop_reason); max_steps = hit the cap
  parsed / pred / correct                answer extraction (report counts unparsed as wrong)
  n_chunks_dnl / n_sentences / n_step_markers   chunk-structure of the final response
  step_mean_logprobs                     per-step mean token logprob (weight-hijack check:
                                         tiny steps with near-0 logprob distorted AF3)

Use it to pick the step mode ("\\n\\n" vs tokens-per-step) and max_steps for a new
model before an EPF sweep, and to verify "Answer:"/EOS stopping actually fires.

NOTE: the resume key is (unique_id, method) — the step config is NOT part of it.
One JSONL per step config; encode it in the filename (step_probe_dnl_max6.jsonl,
step_probe_tok30_max8.jsonl, ...).

    python -m benchmarking.mmau_pro.step_probe \
        --endpoints http://localhost:8100/v1,http://localhost:8101/v1 --model-name gemma4-e2b \
        --data-root data/mmau_pro --subset test \
        --ids-file benchmarking/mmau_pro/results/run14_gemma4e2b/probe_ids_100.txt \
        --prompts 4,5,7,9 --temp 0.8 --max-steps 6 \
        --jsonl .../step_probe_dnl_max6.jsonl --csv .../step_probe_dnl_max6.csv
"""

import asyncio
import csv
import json
import os
import re
import time
from collections import Counter, defaultdict

import click

from benchmarking.mmau_pro.cot_compare import _select_all, _select_items_stratified
from benchmarking.mmau_pro.loader import load_mmau_mcq
from benchmarking.mmau_pro.prompt import METHODS, build
from benchmarking.mmau_pro.scoring import LETTERS, predicted_index
from its_hub import OpenAICompatibleLanguageModel, StepGeneration

CSV_FIELDS = [
    "unique_id", "category", "length_type", "num_audio", "n_choices",
    "method", "method_name", "n_steps", "tok_counts", "total_tokens", "end",
    "parsed", "pred", "gold", "correct",
    "n_chunks_dnl", "n_sentences", "n_step_markers", "step_mean_logprobs",
    "response", "error",
]

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_STEP_MARKER = re.compile(r"(?im)(?:^|\n)\s*(?:#+\s*)?step\s*\d+\s*[:.]")


def build_sg(temp, max_steps, tokens_per_step=None):
    """Same construction as diversity_probe.build_epf's StepGeneration."""
    if tokens_per_step:
        return StepGeneration(step_token=None, tokens_per_step=tokens_per_step,
                              stop_token="Answer:", max_steps=max_steps, temperature=temp)
    return StepGeneration(step_token="\n\n", stop_token="Answer:",
                          max_steps=max_steps, temperature=temp)


async def run_trajectory(sg, lm, msgs):
    """Drive one budget-1 trajectory to completion, exactly like the EPF loop
    (`while not stopped: sg.aforward(...)`; aforward itself enforces max_steps)."""
    steps, tok_counts, mean_lps = [], [], []
    while True:
        n_before = len(steps)
        step, stopped, summary = await sg.aforward(
            lm, "", steps_so_far=steps, base_messages=msgs, return_logprobs=True)
        steps.append(step)
        tok_counts.append(summary.get("num_tokens") or 0)
        mean_lps.append(round(summary.get("mean_logprob") or 0.0, 3))
        if stopped:
            if sg.stop_token and sg.stop_token in step:
                end = "stop"
            elif n_before >= sg.max_steps:
                end = "max_steps"
            else:
                end = "eos"
            break
        if len(steps) > sg.max_steps + 1:  # can't happen (aforward caps); belt & braces
            end = "max_steps"
            break
    response = sg._post_process(steps, stopped=True)
    return response, tok_counts, mean_lps, end


def _load_resume(path):
    rows, done = [], set()
    if path and os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                rows.append(r)
                if not r.get("error"):
                    done.add((r["unique_id"], int(r["method"])))
    return rows, done


def _dedupe(rows):
    by = {}
    for r in rows:
        by[(r["unique_id"], int(r["method"]))] = r
    return list(by.values())


def _pctile(vals, q):
    vals = sorted(vals)
    if not vals:
        return float("nan")
    return vals[min(int(q * (len(vals) - 1)), len(vals) - 1)]


def build_report(rows) -> str:
    """Per-prompt decision table: acc (unparsed=wrong), step shape, end reasons."""
    by_m = defaultdict(list)
    for r in rows:
        if not r.get("error"):
            by_m[int(r["method"])].append(r)
    errors = sum(1 for r in rows if r.get("error"))
    lines = [f"rows: {len(rows)} | errors: {errors}", ""]
    lines.append(f"{'#':>2} {'method':28s} {'acc':>6} {'unparsed':>9} {'steps':>6} "
                 f"{'p95':>4} {'tok/step':>9} {'tot_tok':>8} {'chunks':>7} {'n':>4}")
    for m in sorted(by_m):
        rs = by_m[m]
        graded = [r for r in rs if r["gold"]]
        acc = (sum(1 for r in graded if r["correct"] is True) / len(graded)) if graded else 0.0
        unparsed = sum(1 for r in rs if not r["parsed"]) / len(rs)
        nsteps = [r["n_steps"] for r in rs]
        toks = [t for r in rs for t in r["tok_counts"]]
        tot = [r["total_tokens"] for r in rs]
        chunks = [r["n_chunks_dnl"] for r in rs]
        lines.append(
            f"{m:>2} {METHODS[m][:28]:28s} {acc:6.3f} {unparsed:9.1%} "
            f"{sum(nsteps)/len(nsteps):6.2f} {_pctile(nsteps, 0.95):>4} "
            f"{(sum(toks)/len(toks) if toks else 0):9.1f} {sum(tot)/len(tot):8.1f} "
            f"{sum(chunks)/len(chunks):7.2f} {len(graded):>4}"
        )
        ends = Counter(r["end"] for r in rs)
        med_tok = _pctile(toks, 0.5) if toks else 0
        lines.append("     end: " + ", ".join(f"{k}={v}" for k, v in ends.most_common())
                     + f" | median tok/step: {med_tok}")
    return "\n".join(lines)


@click.command()
@click.option("--endpoints", default="http://localhost:8100/v1,http://localhost:8101/v1",
              help="comma list of vLLM endpoints; items round-robin across them")
@click.option("--model-name", required=True)
@click.option("--api-key", default="NO_API_KEY")
@click.option("--data-root", default="data/mmau_pro")
@click.option("--subset", type=click.Choice(["full", "le30s", "test"]), default="test")
@click.option("--audio-root", default=None, help="root for relative audio paths (if outside --data-root)")
@click.option("--ids-file", default=None,
              help="file with one unique_id per line; overrides --select/--limit")
@click.option("--select", "select_mode", type=click.Choice(["stratified", "all"]), default="stratified")
@click.option("--limit", default=100)
@click.option("--prompts", default="4,5,7,9", help="comma list of prompt methods")
@click.option("--temp", default=0.8)
@click.option("--max-steps", default=6)
@click.option("--tokens-per-step", default=None, type=int,
              help="chunk steps by token count instead of '\\n\\n' (prose models)")
@click.option("--max-tokens-per-step", default=300,
              help="LM max_tokens (the per-step cap in '\\n\\n' mode)")
@click.option("--max-inflight", default=24, help="concurrent items PER endpoint")
@click.option("--jsonl", "jsonl_path", default=None)
@click.option("--csv", "csv_path", default=None)
@click.option("--log", "log_path", default=None)
def main(endpoints, model_name, api_key, data_root, subset, audio_root, ids_file, select_mode,
         limit, prompts, temp, max_steps, tokens_per_step, max_tokens_per_step, max_inflight,
         jsonl_path, csv_path, log_path):
    eps = [e.strip() for e in endpoints.split(",") if e.strip()]
    methods = [int(m) for m in prompts.split(",")]

    recs = load_mmau_mcq(data_root, subset=subset, audio_root=audio_root)
    if ids_file:
        with open(ids_file) as f:
            wanted = [ln.strip() for ln in f if ln.strip()]
        by_id = {r.unique_id: r for r in recs}
        missing = [w for w in wanted if w not in by_id]
        if missing:
            raise SystemExit(f"--ids-file: {len(missing)} ids not in {subset} pool, e.g. {missing[:3]}")
        records = [by_id[w] for w in wanted]
    else:
        records = _select_all(recs, limit) if select_mode == "all" else _select_items_stratified(recs, limit)

    mode = f"tokens_per_step={tokens_per_step}" if tokens_per_step else "step_token='\\n\\n'"
    print(f"step probe: prompts={methods} items={len(records)} {mode} "
          f"max_steps={max_steps} temp={temp} endpoints={len(eps)}", flush=True)

    lms = [OpenAICompatibleLanguageModel(
        endpoint=ep, api_key=api_key, model_name=model_name,
        max_tokens=max_tokens_per_step, max_concurrency=-1) for ep in eps]

    resumed_rows, done = _load_resume(jsonl_path)
    if resumed_rows:
        print(f"resume: {len(done)} (item,method) already done", flush=True)

    sg = build_sg(temp, max_steps, tokens_per_step=tokens_per_step)

    async def _run_one(method, rec, lm, sem, out, lock):
        base = {
            "unique_id": rec.unique_id, "category": rec.category,
            "length_type": rec.length_type, "num_audio": len(rec.audio_paths),
            "n_choices": len(rec.choices), "method": method, "method_name": METHODS[method],
            "gold": (LETTERS[rec.answer_index] if rec.answer_index is not None else ""),
        }
        async with sem:
            try:
                msgs, _seed = build(method, rec, audio_mode="local-path")
                response, tok_counts, mean_lps, end = await run_trajectory(sg, lm, msgs)
                pi = predicted_index(response, rec.choices)
                row = {**base,
                       "n_steps": len(tok_counts), "tok_counts": tok_counts,
                       "total_tokens": sum(tok_counts), "end": end,
                       "parsed": pi is not None,
                       "pred": (LETTERS[pi] if pi is not None else ""),
                       "correct": (pi == rec.answer_index if rec.answer_index is not None else None),
                       "n_chunks_dnl": len([p for p in response.split("\n\n") if p.strip()]),
                       "n_sentences": len([s for s in _SENT_SPLIT.split(response) if s.strip()]),
                       "n_step_markers": len(_STEP_MARKER.findall(response)),
                       "step_mean_logprobs": mean_lps,
                       "response": response, "error": None}
            except Exception as e:
                row = {**base, "n_steps": 0, "tok_counts": [], "total_tokens": 0, "end": "",
                       "parsed": False, "pred": "", "correct": None, "n_chunks_dnl": 0,
                       "n_sentences": 0, "n_step_markers": 0, "step_mean_logprobs": [],
                       "response": "", "error": f"{type(e).__name__}: {e}"}
        if out is not None:
            async with lock:
                out.write(json.dumps(row) + "\n")
                out.flush()
        return row

    async def _run():
        new_rows = []
        lock = asyncio.Lock()
        out = open(jsonl_path, "a") if jsonl_path else None  # noqa: SIM115 (closed in finally; spans loop)
        try:
            for method in methods:
                sems = [asyncio.Semaphore(max_inflight) for _ in lms]
                todo = [r for r in records if (r.unique_id, method) not in done]
                print(f"[P{method}] {len(todo)} to do ({len(records)-len(todo)} resumed)", flush=True)
                if not todo:
                    continue
                t0 = time.time()
                tasks = [_run_one(method, rec, lms[i % len(lms)], sems[i % len(lms)], out, lock)
                         for i, rec in enumerate(todo)]
                rows = await asyncio.gather(*tasks)
                new_rows.extend(rows)
                errs = sum(1 for r in rows if r.get("error"))
                print(f"    done in {time.time()-t0:.0f}s "
                      f"({(time.time()-t0)/len(todo):.2f}s/item, {errs} errors)", flush=True)
        finally:
            if out is not None:
                out.close()
            for lm in lms:
                await lm.close()
        return new_rows

    new_rows = asyncio.run(_run())
    final_rows = _dedupe(_load_resume(jsonl_path)[0] if jsonl_path else (resumed_rows + new_rows))

    if csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            w.writeheader()
            for r in final_rows:
                w.writerow({**r,
                            "tok_counts": ";".join(str(t) for t in r["tok_counts"]),
                            "step_mean_logprobs": ";".join(str(x) for x in r["step_mean_logprobs"]),
                            "correct": ("" if r["correct"] is None else r["correct"])})
        print(f"\nwrote {len(final_rows)} rows -> {csv_path}", flush=True)

    report = build_report(final_rows)
    print("\n" + report)
    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        with open(log_path, "w") as f:
            f.write(f"step probe: prompts={methods} items={len(records)} {mode} "
                    f"max_steps={max_steps} temp={temp}\n\n")
            f.write(report + "\n")
        print(f"wrote report -> {log_path}", flush=True)


if __name__ == "__main__":
    main()
