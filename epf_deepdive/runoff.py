"""Runoff rejuvenation for probe-EPF: binary re-ask on vote-contested items.

From a finished probe-EPF run, find items whose top-2 vote margin is thin,
re-present ONLY the two candidate choices to the model (same audio, fresh
context), sample k binary decisions, and let the runoff winner override the vote.

  python runoff.py --endpoint http://localhost:8500/v1 --model-name qwen-omni-3b \
    --bench mmar --probe-jsonl probe_out/mmar_3b_b32.jsonl --arm probe_adaptive_t3 \
    --budget 128 --margin 0.34 --k 8 --jsonl probe_out/runoff_3b.jsonl
"""

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, "/work/hdd/bcey/awaheed/its-for-audio-reasoning/entropic-particle-filter")

from benchmarking.mmau_pro.audio import audio_content_parts  # noqa: E402
from benchmarking.mmau_pro.scoring import LETTERS  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402


def load_records(bench):
    if bench == "mmar":
        from benchmarking.mmar.loader import load_mmar_mcq
        return {r.unique_id: r for r in load_mmar_mcq("/u/awaheed/epf_data/mmar", subset="full")}
    from benchmarking.mmau_pro.loader import load_mmau_mcq
    return {r.unique_id: r for r in
            load_mmau_mcq("/u/awaheed/epf_data/mmau_pro_d1k", subset="d1k")}


def vote_tally(r):
    """Probe-marginal tally over unique texts (same as probe_marginal selector)."""
    n = r["n"]; th = r["texts_hash"]; ad = r.get("probe_dists_all") or [[]] * n
    letts = r["pred_letters"]
    seen = set(); t = defaultdict(float)
    for i in range(n):
        if th[i] in seen:
            continue
        seen.add(th[i])
        pd = [d for d in (ad[i] if i < len(ad) else []) if d]
        if pd:
            for L, mass in pd[-1].items():
                t[L] += mass
        elif letts[i]:
            t[letts[i]] += 1.0
    return dict(t)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:8500/v1")
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--bench", default="mmar")
    ap.add_argument("--probe-jsonl", required=True)
    ap.add_argument("--arm", default="probe_adaptive_t3")
    ap.add_argument("--budget", type=int, default=128)
    ap.add_argument("--margin", type=float, default=0.34,
                    help="runoff when (top1-top2)/total vote share < margin")
    ap.add_argument("--k", type=int, default=8, help="binary samples per contested item")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--max-inflight", type=int, default=24)
    ap.add_argument("--jsonl", required=True)
    a = ap.parse_args()

    recs = load_records(a.bench)
    items = []
    for line in open(a.probe_jsonl):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("error") or r["arm"] != a.arm or r["budget"] != a.budget:
            continue
        t = vote_tally(r)
        if not t:
            continue
        top = sorted(t.items(), key=lambda kv: -kv[1])
        vote_winner = top[0][0]
        tot = sum(t.values())
        marg = (top[0][1] - (top[1][1] if len(top) > 1 else 0.0)) / tot
        items.append({
            "unique_id": r["unique_id"], "gold": r["gold"], "vote": vote_winner,
            "margin": marg,
            "cands": [top[0][0], top[1][0]] if len(top) > 1 else [top[0][0]],
        })
    contested = [it for it in items if len(it["cands"]) == 2 and it["margin"] < a.margin]
    print(f"{len(items)} items; {len(contested)} contested (margin<{a.margin})", flush=True)

    done = set()
    if os.path.exists(a.jsonl):
        for line in open(a.jsonl):
            try:
                rr = json.loads(line)
                if not rr.get("error"):
                    done.add(rr["unique_id"])
            except Exception:
                pass
    todo = [it for it in contested if it["unique_id"] not in done]
    print(f"{len(todo)} to run ({len(done)} resumed)", flush=True)

    client = AsyncOpenAI(base_url=a.endpoint, api_key="NO_API_KEY", timeout=600, max_retries=2)
    sem = asyncio.Semaphore(a.max_inflight)
    out = open(a.jsonl, "a")
    lock = asyncio.Lock()

    def binary_txt(rec, L1, L2):
        # candidates are RELABELED to literal A/B: the model has a strong prior
        # for emitting early letters after 'Answer:', which order-swapping alone
        # cannot cancel (letter identity, not list position). Fresh labels +
        # assignment-swap across the two probes cancels it exactly.
        i1, i2 = LETTERS.index(L1), LETTERS.index(L2)
        return (f"Question: {rec.question}\n\nTwo candidate answers remain:\n"
                f"A. {rec.choices[i1]}\nB. {rec.choices[i2]}\n\n"
                f"Listen to the audio carefully and decide which candidate is correct. "
                f"Answer with exactly one letter: A or B.")

    async def _mass(rec, L1, L2):
        """One temp-0 logprob probe: candidate L1 labeled 'A', L2 labeled 'B'.
        Returns {original letter: prob mass} mapped back from the A/B labels."""
        parts = audio_content_parts(rec.audio_paths, mode="local-path")
        msgs = [{"role": "system", "content": "You are an expert audio analyst."},
                {"role": "user", "content": [*parts, {"type": "text", "text": binary_txt(rec, L1, L2)}]},
                {"role": "assistant", "content": "Answer:"}]
        resp = await client.chat.completions.create(
            model=a.model_name, messages=msgs, max_tokens=2, temperature=0.0,
            logprobs=True, top_logprobs=20,
            extra_body={"add_generation_prompt": False, "continue_final_message": True})
        ch = resp.choices[0]
        content = ch.logprobs.content if ch.logprobs else []
        import math as _math
        label_map = {"A": L1, "B": L2}  # map fresh labels back to original letters
        for pos in content:
            masses = {}
            for e in (pos.top_logprobs or []):
                tok = (e.token or "").strip(")>].,:;!?*'\"( ")
                if len(tok) == 1 and tok.upper() in label_map:
                    L = label_map[tok.upper()]
                    masses[L] = masses.get(L, 0.0) + _math.exp(e.logprob)
            if masses:
                return masses
        return {}

    async def one(it):
        rec = recs[it["unique_id"]]
        La, Lb = it["cands"]
        row = {"unique_id": it["unique_id"], "gold": it["gold"], "vote": it["vote"],
               "margin": round(it["margin"], 4), "cands": it["cands"], "error": None}
        try:
            async with sem:
                m1 = await _mass(rec, La, Lb)   # leader listed first
                m2 = await _mass(rec, Lb, La)   # runner-up listed first
            avg = {L: (m1.get(L, 0.0) + m2.get(L, 0.0)) / 2 for L in (La, Lb)}
            row["mass_order1"] = {k: round(v, 4) for k, v in m1.items()}
            row["mass_order2"] = {k: round(v, 4) for k, v in m2.items()}
            row["runoff"] = (max(avg, key=avg.get) if any(avg.values()) else it["vote"])
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
            row["runoff"] = it["vote"]
        async with lock:
            out.write(json.dumps(row) + "\n")
            out.flush()
        return row

    t0 = time.time()
    await asyncio.gather(*(one(it) for it in todo))
    out.close()
    await client.close()
    print(f"runoff done in {time.time()-t0:.0f}s", flush=True)

    # report: baseline vote acc vs runoff-overridden acc on the full item set
    runoff = {}
    for line in open(a.jsonl):
        try:
            rr = json.loads(line)
        except Exception:
            continue
        if not rr.get("error"):
            runoff[rr["unique_id"]] = rr
    base = sum(it["vote"] == it["gold"] for it in items)
    over = sum((runoff.get(it["unique_id"], {}).get("runoff", it["vote"])) == it["gold"]
               for it in items)
    n = len(items)
    cn = sum(1 for it in contested)
    cb = sum(it["vote"] == it["gold"] for it in contested)
    co = sum(runoff.get(it["unique_id"], {}).get("runoff", it["vote"]) == it["gold"]
             for it in contested)
    print(f"\noverall: vote={base/n:.4f} -> +runoff={over/n:.4f}  (n={n})")
    print(f"contested only: vote={cb/max(cn,1):.4f} -> runoff={co/max(cn,1):.4f}  (n={cn})")


if __name__ == "__main__":
    asyncio.run(main())
