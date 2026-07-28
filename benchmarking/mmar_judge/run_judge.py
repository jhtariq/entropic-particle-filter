"""Run BestOfN / BeamSearch on MMAR with an audio LLM judge as the reward model.

One invocation runs ONE algorithm over a budget list, e.g.:

  python -m benchmarking.mmar_judge.run_judge \
      --endpoints http://localhost:8100/v1 --model-name qwen-omni \
      --judge-endpoint http://localhost:8110/v1 --judge-model-name qwen-omni-judge \
      --alg bon --budgets 1,4,8 --prompt-method 4 \
      --select stratified --limit 100 --seed 0 \
      --jsonl OUT/bon.jsonl --csv OUT/bon.csv --log OUT/bon.log

Conventions follow the prior MMAR probe runners: the JSONL is the resumable
source of truth keyed (unique_id, alg, budget) with error rows retried on the
next pass; the CSV keeps the columns official_crosscheck.py expects
(method/signal/budget/unique_id/selected_letter/selected_correct); the .log is
a human-readable trend report grouped by (alg, budget).
"""

import asyncio
import csv
import json
import os
import random
import time
from collections import Counter, defaultdict

import click

from benchmarking.mmar.loader import DEFAULT_DATA_ROOT, SUBSET_FILES, load_mmar_mcq
from benchmarking.mmar_judge.judge import AudioJudgeORM, AudioJudgePRM, MMARAudioJudge
from benchmarking.mmau_pro.prompt import METHODS, build
from benchmarking.mmau_pro.scoring import LETTERS, predicted_index
from its_hub.core.algorithms.beam_search import BeamSearch
from its_hub.core.algorithms.bon import BestOfN
from its_hub.core.lms.openai_lm import OpenAICompatibleLanguageModel
from its_hub.core.lms.step_generation import StepGeneration
from its_hub.core.orchestrator import LMOrchestrator
from its_hub.core.reward_models.llm_judge import LLMJudge

CSV_FIELDS = [
    "unique_id", "category", "length_type", "n_choices", "method", "method_name",
    "signal", "alg", "budget", "beam_width", "gold_letter", "selected_letter",
    "majority_letter", "preds", "judge_scores", "judge_selected_score",
    "judge_parse_ok_ratio", "n_judge_calls", "steps_used", "distinct_ratio",
    "consensus", "parsed_ratio", "oracle_correct", "selected_correct",
    "majority_correct", "elapsed_s", "error",
]


def _letter(idx: int | None) -> str:
    return LETTERS[idx] if idx is not None and 0 <= idx < len(LETTERS) else ""


def select_records(records, select: str, limit: int | None, seed: int):
    """`all`: first `limit` records; `stratified`: proportional sample by category."""
    if select == "all" or limit is None or limit >= len(records):
        return records[:limit] if limit else records
    by_cat: dict[str, list] = defaultdict(list)
    for rec in records:
        by_cat[rec.category].append(rec)
    rng = random.Random(seed)
    cats = sorted(by_cat)
    # largest-remainder proportional allocation, at least 1 per category
    quotas = {c: len(by_cat[c]) * limit / len(records) for c in cats}
    counts = {c: max(1, int(quotas[c])) for c in cats}
    while sum(counts.values()) > limit:
        c = max(cats, key=lambda c: counts[c] - quotas[c])
        counts[c] = max(1, counts[c] - 1)
        if all(counts[c] == 1 for c in cats):
            break
    remainders = sorted(cats, key=lambda c: quotas[c] - int(quotas[c]), reverse=True)
    i = 0
    while sum(counts.values()) < limit:
        counts[remainders[i % len(remainders)]] += 1
        i += 1
    picked = []
    for c in cats:
        pool = sorted(by_cat[c], key=lambda r: r.unique_id)
        picked.extend(rng.sample(pool, min(counts[c], len(pool))))
    return sorted(picked, key=lambda r: r.unique_id)


def compute_metrics(result, rec) -> dict:
    """Selection/oracle/majority metrics from a BestOfNResult or BeamSearchResult."""
    texts = [(r.get("content") or "") if isinstance(r, dict) else str(r) for r in result.responses]
    preds = [predicted_index(t, rec.choices) for t in texts]
    sel = result.selected_index
    non_none = [p for p in preds if p is not None]
    majority = Counter(non_none).most_common(1)[0][0] if non_none else None

    def graded(pred):
        if rec.answer_index is None:
            return None
        return pred is not None and pred == rec.answer_index

    return {
        "gold_letter": _letter(rec.answer_index),
        "selected_letter": _letter(preds[sel]),
        "majority_letter": _letter(majority),
        "preds": [_letter(p) for p in preds],
        "judge_scores": [round(s, 4) for s in result.scores],
        "judge_selected_score": round(result.scores[sel], 4),
        "selected_correct": graded(preds[sel]),
        "oracle_correct": None if rec.answer_index is None
        else any(p == rec.answer_index for p in preds),
        "majority_correct": graded(majority),
        "parsed_ratio": round(sum(p is not None for p in preds) / len(preds), 4),
        "distinct_ratio": round(len(set(texts)) / len(texts), 4),
        "consensus": round(Counter(non_none).most_common(1)[0][1] / len(preds), 4)
        if non_none else 0.0,
        "steps_used": max(getattr(result, "steps_used", [0])),
    }


def load_resume(jsonl_path: str) -> tuple[dict, set]:
    """Return (last row per key, keys done without error)."""
    rows_by_key: dict[tuple, dict] = {}
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows_by_key[(row.get("unique_id"), row.get("alg"), row.get("budget"))] = row
    done = {k for k, row in rows_by_key.items() if not row.get("error")}
    return rows_by_key, done


def build_report(rows: list[dict]) -> str:
    """Trend report over gradeable, non-error rows grouped by (alg, budget)."""
    lines = []
    cells: dict[tuple, list[dict]] = defaultdict(list)
    n_errors = sum(1 for r in rows if r.get("error"))
    for r in rows:
        if not r.get("error"):
            cells[(r["alg"], r["budget"])].append(r)
    lines.append(f"rows: {len(rows)} | errors: {n_errors}")
    header = (f"{'alg':>5} {'bud':>4} {'sel_acc':>8} {'oracle':>7} {'major':>6} "
              f"{'j_sel':>6} {'j_parse':>8} {'distinct':>8} {'consen':>7} {'parsed':>7} {'n':>5}")
    lines.append(header)
    for (alg, budget), cell in sorted(cells.items()):
        gradeable = [r for r in cell if r.get("selected_correct") is not None]
        if not gradeable:
            continue

        def acc(field, rows=gradeable):
            return sum(bool(r[field]) for r in rows) / len(rows)
        j_sel = sum(r.get("judge_selected_score") or 0 for r in gradeable) / len(gradeable)
        j_parse = sum(r.get("judge_parse_ok_ratio") or 0 for r in gradeable) / len(gradeable)
        distinct = sum(r.get("distinct_ratio") or 0 for r in gradeable) / len(gradeable)
        consen = sum(r.get("consensus") or 0 for r in gradeable) / len(gradeable)
        parsed = sum(r.get("parsed_ratio") or 0 for r in gradeable) / len(gradeable)
        lines.append(
            f"{alg:>5} {budget:>4} {acc('selected_correct'):>8.4f} {acc('oracle_correct'):>7.4f} "
            f"{acc('majority_correct'):>6.4f} {j_sel:>6.3f} {j_parse:>8.3f} "
            f"{distinct:>8.3f} {consen:>7.3f} {parsed:>7.3f} {len(gradeable):>5}"
        )
        # per-modality breakdown
        by_cat: dict[str, list[dict]] = defaultdict(list)
        for r in gradeable:
            by_cat[r.get("category") or "?"].append(r)
        for cat in sorted(by_cat):
            sub = by_cat[cat]
            sel_acc = sum(bool(r["selected_correct"]) for r in sub) / len(sub)
            orc_acc = sum(bool(r["oracle_correct"]) for r in sub) / len(sub)
            lines.append(f"{'':>5} {'':>4}   {cat:<12} sel {sel_acc:.4f}  oracle {orc_acc:.4f}  n {len(sub)}")
    return "\n".join(lines)


def write_outputs(rows_by_key: dict, csv_path: str, log_path: str):
    rows = list(rows_by_key.values())
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["alg"], r["budget"], str(r["unique_id"]))):
            flat = dict(row)
            for key in ("preds", "judge_scores"):
                if isinstance(flat.get(key), list):
                    flat[key] = " ".join(str(x) for x in flat[key])
            writer.writerow(flat)
    report = build_report(rows)
    with open(log_path, "w") as f:
        f.write(report + "\n")
    return report


@click.command()
@click.option("--endpoints", default="http://localhost:8100/v1", help="comma-separated policy endpoints (round-robin)")
@click.option("--model-name", required=True, help="served policy model name")
@click.option("--judge-endpoint", default="http://localhost:8110/v1")
@click.option("--judge-model-name", default="qwen-omni-judge")
@click.option("--api-key", default="NO_API_KEY")
@click.option("--alg", type=click.Choice(["bon", "beam"]), required=True)
@click.option("--budgets", default="4", help="comma-separated budgets, e.g. 1,4,8")
@click.option("--beam-width", default=4, show_default=True)
@click.option("--prompt-method", default=4, show_default=True, help=f"CoT method: {METHODS}")
@click.option("--data-root", default=DEFAULT_DATA_ROOT)
@click.option("--subset", type=click.Choice(list(SUBSET_FILES)), default="full")
@click.option("--audio-root", default=None, help="audio root if different from data-root")
@click.option("--audio-mode", type=click.Choice(["local-path", "base64"]), default="local-path")
@click.option("--select", "select_mode", type=click.Choice(["all", "stratified"]), default="all")
@click.option("--limit", default=None, type=int, help="max items")
@click.option("--seed", default=0, show_default=True, help="stratified sampling seed")
@click.option("--max-inflight", default=8, show_default=True, help="concurrent items")
@click.option("--policy-inflight", default=32, show_default=True, help="concurrent policy requests per endpoint")
@click.option("--judge-inflight", default=16, show_default=True, help="concurrent judge requests (global)")
@click.option("--temp", default=0.8, show_default=True, help="policy sampling temperature")
@click.option("--max-tokens", default=None, type=int, help="policy max tokens (default: 1024 bon, 300 beam per-step)")
@click.option("--max-steps", default=6, show_default=True, help="beam: max reasoning steps")
@click.option("--step-token", default="\n\n", help="beam: step delimiter")
@click.option("--stop-token", default="Answer:", help="beam: stop marker within a step")
@click.option("--judge-max-tokens", default=256, show_default=True)
@click.option("--judge-max-candidate-chars", default=4000, show_default=True)
@click.option("--judge-structured/--no-judge-structured", default=True,
              help="use strict-JSON response_format for judge calls")
@click.option("--store-responses/--no-store-responses", default=True,
              help="store full candidate texts in the JSONL")
@click.option("--store-trace/--no-store-trace", default=True,
              help="store per-call judge trace (scores + reasoning) in the JSONL")
@click.option("--jsonl", "jsonl_path", required=True)
@click.option("--csv", "csv_path", required=True)
@click.option("--log", "log_path", required=True)
def main(endpoints, model_name, judge_endpoint, judge_model_name, api_key, alg,
         budgets, beam_width, prompt_method, data_root, subset, audio_root,
         audio_mode, select_mode, limit, seed, max_inflight, policy_inflight,
         judge_inflight, temp, max_tokens, max_steps, step_token, stop_token,
         judge_max_tokens, judge_max_candidate_chars, judge_structured,
         store_responses, store_trace, jsonl_path, csv_path, log_path):
    """Judge-scored BestOfN / BeamSearch over MMAR."""
    budget_list = [int(b) for b in str(budgets).split(",") if b.strip()]
    if alg == "beam":
        bad = [b for b in budget_list if b < beam_width or b % beam_width != 0]
        if bad:
            raise click.UsageError(
                f"beam budgets must be multiples of beam_width={beam_width}, got {bad}"
            )
    if max_tokens is None:
        max_tokens = 1024 if alg == "bon" else 300

    records = load_mmar_mcq(data_root, subset=subset, audio_root=audio_root)
    selected = select_records(records, select_mode, limit, seed)
    click.echo(f"items: {len(selected)}/{len(records)} ({select_mode}) | alg={alg} "
               f"budgets={budget_list} method=P{prompt_method}")

    rows_by_key, done = load_resume(jsonl_path)
    os.makedirs(os.path.dirname(os.path.abspath(jsonl_path)), exist_ok=True)

    endpoint_list = [e.strip() for e in endpoints.split(",") if e.strip()]
    policy_lms = [
        OpenAICompatibleLanguageModel(
            endpoint=ep, api_key=api_key, model_name=model_name,
            temperature=temp, max_tokens=max_tokens, max_concurrency=policy_inflight,
        )
        for ep in endpoint_list
    ]
    judge_lm = OpenAICompatibleLanguageModel(
        endpoint=judge_endpoint, api_key=api_key, model_name=judge_model_name,
        max_concurrency=judge_inflight,
    )
    judge_orch = LMOrchestrator(max_concurrency=judge_inflight)
    policy_orch = LMOrchestrator(max_concurrency=-1)  # bounded by LM max_concurrency

    async def run_all():
        sem = asyncio.Semaphore(max_inflight)
        jsonl_lock = asyncio.Lock()

        async def run_one(rec, budget, policy_lm):
            key = (rec.unique_id, alg, budget)
            async with sem:
                start = time.time()
                core = MMARAudioJudge(
                    judge_lm, rec, orchestrator=judge_orch, audio_mode=audio_mode,
                    response_format=(
                        LLMJudge.SCORE_RESPONSE_FORMAT if judge_structured else None
                    ),
                    max_candidate_chars=judge_max_candidate_chars,
                    judge_max_tokens=judge_max_tokens,
                )
                row = {
                    "unique_id": rec.unique_id, "category": rec.category,
                    "length_type": rec.length_type, "n_choices": len(rec.choices),
                    "method": prompt_method, "method_name": METHODS[prompt_method],
                    "signal": f"judge_{alg}", "alg": alg, "budget": budget,
                    "beam_width": beam_width if alg == "beam" else "",
                    "error": "",
                }
                try:
                    messages, _seed = build(prompt_method, rec, audio_mode=audio_mode)
                    if alg == "bon":
                        algo = BestOfN(AudioJudgeORM(core), orchestrator=policy_orch)
                        result = await algo.ainfer(
                            policy_lm, messages, budget, return_response_only=False
                        )
                    else:
                        sg = StepGeneration(
                            step_token=step_token, stop_token=stop_token,
                            max_steps=max_steps, temperature=temp,
                        )
                        algo = BeamSearch(sg, AudioJudgePRM(core), beam_width=beam_width)
                        result = await algo.ainfer(
                            policy_lm, messages, budget, return_response_only=False
                        )
                    row.update(compute_metrics(result, rec))
                    row["judge_parse_ok_ratio"] = (
                        round(sum(t["parse_ok"] for t in core.trace) / len(core.trace), 4)
                        if core.trace else None
                    )
                    row["n_judge_calls"] = len(core.trace)
                    if store_responses:
                        row["responses"] = [
                            (r.get("content") or "") if isinstance(r, dict) else str(r)
                            for r in result.responses
                        ]
                    if store_trace:
                        row["judge_trace"] = core.trace
                except Exception as e:  # error rows are retried on resume
                    row["error"] = f"{type(e).__name__}: {e}"[:500]
                row["elapsed_s"] = round(time.time() - start, 2)
                async with jsonl_lock:
                    with open(jsonl_path, "a") as f:
                        f.write(json.dumps(row) + "\n")
                rows_by_key[key] = row
                return row

        for budget in budget_list:
            todo = [r for r in selected if (r.unique_id, alg, budget) not in done]
            click.echo(f"budget {budget}: {len(todo)} items to run "
                       f"({len(selected) - len(todo)} resumed)")
            if not todo:
                continue
            tasks = [
                run_one(rec, budget, policy_lms[i % len(policy_lms)])
                for i, rec in enumerate(todo)
            ]
            results = await asyncio.gather(*tasks)
            n_err = sum(1 for r in results if r.get("error"))
            click.echo(f"budget {budget}: done, {n_err} errors")

        for lm in [*policy_lms, judge_lm]:
            close = getattr(lm, "close", None)
            if close is not None:
                await close()

    asyncio.run(run_all())
    report = write_outputs(rows_by_key, csv_path, log_path)
    click.echo(report)


if __name__ == "__main__":
    main()
