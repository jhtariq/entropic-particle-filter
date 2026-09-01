"""Per-option Yes/No verification probes: a second evidence channel.

For each item and each option, ask the model to VERIFY the option against the
audio and read P(Yes) vs P(No) from temp-0 logprobs behind an 'Answer:' prefill.
Normalizing across options cancels yes-bias. No letters, no ordering — immune to
the letter-identity bias that killed the binary runoff.

  python yesno_probe.py --endpoint http://localhost:8500/v1 --model-name qwen-omni \
    --uids matched400_uids.txt --jsonl probe_out/yesno_7b.jsonl
"""
import argparse
import asyncio
import json
import math
import os
import sys
import time

sys.path.insert(0, "/work/hdd/bcey/awaheed/its-for-audio-reasoning/entropic-particle-filter")

from benchmarking.mmau_pro.audio import audio_content_parts  # noqa: E402
from benchmarking.mmau_pro.scoring import LETTERS  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:8500/v1")
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--uids", required=True)
    ap.add_argument("--limit", type=int, default=None, help="cap items (smoke)")
    ap.add_argument("--max-inflight", type=int, default=64)
    ap.add_argument("--jsonl", required=True)
    a = ap.parse_args()

    from benchmarking.mmar.loader import load_mmar_mcq
    uids = set(open(a.uids).read().split())
    recs = [r for r in load_mmar_mcq("/u/awaheed/epf_data/mmar", subset="full")
            if r.answer_index is not None and r.unique_id in uids]
    if a.limit:
        recs = recs[: a.limit]

    done = set()
    if os.path.exists(a.jsonl):
        for line in open(a.jsonl):
            try:
                rr = json.loads(line)
                if not rr.get("error"):
                    done.add(rr["unique_id"])
            except Exception:
                pass
    todo = [r for r in recs if r.unique_id not in done]
    print(f"{len(recs)} items; {len(todo)} to run ({len(done)} resumed)", flush=True)

    client = AsyncOpenAI(base_url=a.endpoint, api_key="NO_API_KEY", timeout=600, max_retries=2)
    sem = asyncio.Semaphore(a.max_inflight)
    out = open(a.jsonl, "a")
    lock = asyncio.Lock()

    async def yes_mass(rec, opt_text):
        parts = audio_content_parts(rec.audio_paths, mode="local-path")
        txt = (f"Question: {rec.question}\n\nProposed answer: {opt_text}\n\n"
               f"Listen to the audio carefully and judge whether this proposed answer "
               f"is correct. Reply with exactly one word: Yes or No.")
        msgs = [{"role": "system", "content": "You are an expert audio analyst."},
                {"role": "user", "content": [*parts, {"type": "text", "text": txt}]},
                {"role": "assistant", "content": "Answer:"}]
        async with sem:
            resp = await client.chat.completions.create(
                model=a.model_name, messages=msgs, max_tokens=2, temperature=0.0,
                logprobs=True, top_logprobs=20,
                extra_body={"add_generation_prompt": False, "continue_final_message": True})
        content = resp.choices[0].logprobs.content if resp.choices[0].logprobs else []
        for pos in content:
            y = n = 0.0
            for e in (pos.top_logprobs or []):
                tok = (e.token or "").strip(")>].,:;!?*'\"( ").lower()
                if tok == "yes":
                    y += math.exp(e.logprob)
                elif tok == "no":
                    n += math.exp(e.logprob)
            if y or n:
                return y, n
        return 0.0, 0.0

    async def one(rec):
        row = {"unique_id": rec.unique_id, "gold": LETTERS[rec.answer_index],
               "n_choices": len(rec.choices), "error": None}
        try:
            yn = await asyncio.gather(*(yes_mass(rec, c) for c in rec.choices))
            row["yes"] = [round(y, 5) for y, _ in yn]
            row["no"] = [round(n, 5) for _, n in yn]
            scores = [(y / (y + n)) if (y + n) > 0 else 0.5 for y, n in yn]
            row["scores"] = [round(s, 5) for s in scores]
            row["pred"] = LETTERS[max(range(len(scores)), key=lambda i: scores[i])]
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
        async with lock:
            out.write(json.dumps(row) + "\n")
            out.flush()

    t0 = time.time()
    await asyncio.gather(*(one(r) for r in todo))
    out.close()
    await client.close()

    ok = tot = 0
    for line in open(a.jsonl):
        try:
            rr = json.loads(line)
        except Exception:
            continue
        if not rr.get("error"):
            tot += 1
            ok += (rr["pred"] == rr["gold"])
    print(f"done in {time.time()-t0:.0f}s; standalone yes/no acc = {ok/max(tot,1):.4f} (n={tot})",
          flush=True)


if __name__ == "__main__":
    asyncio.run(main())
