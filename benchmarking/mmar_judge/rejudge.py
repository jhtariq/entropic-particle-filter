"""Re-judge stored BoN candidates from a prior run under a different scoring mode.

Controlled scorer comparison: the candidates are the EXACT texts stored in a
prior run's JSONL (--store-responses), so generation noise is removed and only
the judge's scoring rule varies. Selection (argmax, first-tied wins) and all
metrics are recomputed exactly as the live runner does.

  python -m benchmarking.mmar_judge.rejudge \
      --in-jsonl mmar-judge/results/smoke100_qwen-omni/bon.jsonl \
      --scoring pyes --budgets 4,8 \
      --jsonl OUT/bon_pyes.jsonl --log OUT/bon_pyes.log
"""

import asyncio
import json
import os
import time
from collections import namedtuple

import click

from benchmarking.mmar.loader import DEFAULT_DATA_ROOT, SUBSET_FILES, load_mmar_mcq
from benchmarking.mmar_judge.judge import MMARAudioJudge
from benchmarking.mmar_judge.run_judge import build_report, compute_metrics
from its_hub.core.lms.openai_lm import OpenAICompatibleLanguageModel
from its_hub.core.orchestrator import LMOrchestrator

FakeResult = namedtuple("FakeResult", "responses scores selected_index steps_used")


@click.command()
@click.option("--in-jsonl", "in_jsonl", required=True, help="source run with stored responses")
@click.option("--scoring", type=click.Choice(list(MMARAudioJudge.SCORING_MODES)), required=True)
@click.option("--budgets", default="4,8", show_default=True)
@click.option("--judge-endpoint", default="http://localhost:8110/v1")
@click.option("--judge-model-name", default="qwen-omni-judge")
@click.option("--api-key", default="NO_API_KEY")
@click.option("--data-root", default=DEFAULT_DATA_ROOT)
@click.option("--subset", type=click.Choice(list(SUBSET_FILES)), default="full")
@click.option("--audio-root", default=None)
@click.option("--max-inflight", default=8, show_default=True, help="concurrent items")
@click.option("--judge-inflight", default=16, show_default=True)
@click.option("--jsonl", "jsonl_path", required=True)
@click.option("--log", "log_path", required=True)
def main(in_jsonl, scoring, budgets, judge_endpoint, judge_model_name, api_key,
         data_root, subset, audio_root, max_inflight, judge_inflight,
         jsonl_path, log_path):
    """Re-score stored candidates; recompute selection and accuracy."""
    budget_set = {int(b) for b in budgets.split(",") if b.strip()}
    src = []
    with open(in_jsonl) as f:
        for line in f:
            row = json.loads(line)
            if (row.get("alg") == "bon" and row.get("budget") in budget_set
                    and not row.get("error") and row.get("responses")):
                src.append(row)
    recs = {r.unique_id: r for r in load_mmar_mcq(data_root, subset=subset, audio_root=audio_root)}
    click.echo(f"re-judging {len(src)} rows x scoring={scoring} "
               f"({sum(len(r['responses']) for r in src)} candidates)")

    judge_lm = OpenAICompatibleLanguageModel(
        endpoint=judge_endpoint, api_key=api_key, model_name=judge_model_name,
        max_concurrency=judge_inflight,
    )
    judge_orch = LMOrchestrator(max_concurrency=judge_inflight)
    out_rows = []

    async def run_all():
        sem = asyncio.Semaphore(max_inflight)

        async def run_one(row):
            async with sem:
                start = time.time()
                rec = recs[row["unique_id"]]
                out = {
                    "unique_id": row["unique_id"], "category": row["category"],
                    "n_choices": row["n_choices"], "method": row["method"],
                    "signal": f"judge_bon_{scoring}", "scoring": scoring,
                    "alg": "bon", "budget": row["budget"], "error": "",
                }
                try:
                    core = MMARAudioJudge(
                        judge_lm, rec, orchestrator=judge_orch, scoring=scoring,
                    )
                    scores = await core.ascore_texts(row["responses"], partial=False)
                    result = FakeResult(
                        responses=[{"role": "assistant", "content": t} for t in row["responses"]],
                        scores=scores,
                        selected_index=scores.index(max(scores)),
                        steps_used=[0],
                    )
                    out.update(compute_metrics(result, rec))
                    out["judge_parse_ok_ratio"] = (
                        round(sum(t["parse_ok"] for t in core.trace) / len(core.trace), 4)
                        if core.trace else None
                    )
                    out["n_judge_calls"] = len(core.trace)
                    out["judge_trace"] = core.trace
                except Exception as e:
                    out["error"] = f"{type(e).__name__}: {e}"[:500]
                out["elapsed_s"] = round(time.time() - start, 2)
                out_rows.append(out)

        await asyncio.gather(*(run_one(r) for r in src))
        await judge_lm.close()

    asyncio.run(run_all())
    os.makedirs(os.path.dirname(os.path.abspath(jsonl_path)), exist_ok=True)
    with open(jsonl_path, "w") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\n")
    report = build_report(out_rows)
    with open(log_path, "w") as f:
        f.write(report + "\n")
    click.echo(report)


if __name__ == "__main__":
    main()
