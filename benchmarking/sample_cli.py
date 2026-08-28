"""Sample N completions per MCQ benchmark item at a chosen temperature — one JSONL row per sample.

A single code path serves all four arms of the MMAR greedy / self-consistency study, so
the ONLY thing that differs between arms is the sampling parameters:

    greedy, no CoT             --n 1   --temperature 0.0 --method 0 --max-tokens 64
    greedy, CoT (P4)           --n 1   --temperature 0.0 --method 4 --max-tokens 700
    self-consistency, CoT      --n 128 --temperature 0.8 --method 4 --max-tokens 700
    self-consistency, no CoT   --n 128 --temperature 0.8 --method 0 --max-tokens 64

Why this exists rather than reusing an existing runner:

  * `its_hub`'s OpenAICompatibleLanguageModel never emits OpenAI's `n` field, so N samples
    would become N separate requests and the audio encoder would run N times per item.
  * `greedy/greedy_runner.py` is no-CoT only and truncates `content` to 200 characters,
    which would destroy a reasoning trace.
  * `cot_compare.py` keeps full text but is hardwired to one greedy generation per item.

Throughput / memory model
-------------------------
Asking for n=N in ONE request makes vLLM encode the audio and prefill the prompt once,
then fork N decode sequences that share those KV blocks — far cheaper than N requests.
But the N decode sequences are NOT shared, and their KV cost is the binding constraint:

    KV per token   Qwen2-Audio-7B  512 KiB  (32 layers x 32 kv_heads x 128, full MHA)
                   Phi-4-MM        128 KiB  (32 layers x  8 kv_heads x 128, GQA)

At max_tokens=700 a single n=128 request therefore needs 43.8 GiB of KV for Qwen2-Audio
— more than an A40 has after weights (~22 GiB usable). So N is split into `--n-chunk`
sized requests issued sequentially per item, which bounds the KV a single in-flight item
can hold to `n_chunk * max_tokens * KV_per_token`. Concurrency across items is then
`--max-inflight`, and the two together must fit the server's KV budget:

    peak KV  ~=  max_inflight * n_chunk * max_tokens * KV_per_token

Enable `--enable-prefix-caching` on the server and the chunks of one item reuse the
prompt/audio prefill too, so chunking costs essentially nothing.

Prompts and answer parsing are the repo's own (`benchmarking.mmau_pro.prompt` /
`.scoring`), so these numbers are directly comparable to the MMAU-Pro and MMSU baselines.
Every sample's full response text is persisted, which is what `sc_report.py` needs to do
the budget-subsampled majority vote without re-running inference.
"""

import asyncio
import json
import math
import os
import time

import click
from openai import AsyncOpenAI

from benchmarking.mmau_pro.prompt import METHODS, build, build_messages
from benchmarking.mmau_pro.scoring import LETTERS, is_correct, predicted_index

# Single source of truth for the no-CoT system prompt — the same constant the MMAU and
# MMSU greedy baselines use, imported (not re-copied) so the arms cannot silently diverge.
from benchmarking.mmsu.greedy_baseline import NO_COT_SYS


def arm_label(method: int, temperature: float, n: int) -> str:
    """Human-readable cell name, e.g. 'sc128_p4_t0.8' or 'greedy_nocot'."""
    prompt = "nocot" if method == 0 else f"p{method}"
    if n == 1 and temperature == 0.0:
        return f"greedy_{prompt}"
    return f"sc{n}_{prompt}_t{temperature:g}"


def to_openai_messages(msgs) -> list[dict]:
    """ChatMessage -> plain dicts. Audio parts are already OpenAI-shaped (see audio.py)."""
    return [{"role": m.role, "content": m.content} for m in msgs]


def build_prompt(rec, method: int, audio_mode: str):
    """method 0 = terse direct no-CoT; otherwise a build() CoT prompt (4 = plan-and-solve)."""
    if method:
        msgs, _seed = build(method, rec, audio_mode=audio_mode)
    else:
        msgs = build_messages(rec, audio_mode=audio_mode, system_prompt=NO_COT_SYS)
    return to_openai_messages(msgs)


def chunk_sizes(n: int, n_chunk: int) -> list[int]:
    """Split N samples into request-sized pieces, e.g. (128, 32) -> [32, 32, 32, 32]."""
    if n_chunk <= 0 or n_chunk >= n:
        return [n]
    return [min(n_chunk, n - i * n_chunk) for i in range(math.ceil(n / n_chunk))]


def load_done(jsonl_path: str, n: int) -> set[str]:
    """unique_ids that already have all N error-free samples on disk.

    Item-level granularity: a partially written item is redone in full rather than topped
    up, so an item's N samples always come from one uninterrupted run of the same command.
    """
    if not os.path.exists(jsonl_path):
        return set()
    counts: dict[str, int] = {}
    bad: set[str] = set()
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:  # truncated tail from a hard kill
                continue
            uid = row.get("unique_id")
            if uid is None:
                continue
            if row.get("error"):
                bad.add(uid)
            else:
                counts[uid] = counts.get(uid, 0) + 1
    return {u for u, c in counts.items() if c >= n and u not in bad}



def make_cli(loader_fn, subset_files, default_root, bench_name="MCQ"):
    """Build the sampling CLI bound to one benchmark's loader.

    Same factory idiom the repo already uses for cot_compare / diversity_probe /
    phase0_gate / ab_causality, so a new benchmark is a ~6-line binding rather than a
    fork of this file. `loader_fn` must accept (data_root, subset=, audio_root=) and
    return MCQRecord objects — true for load_mmar_mcq, load_mmau_mcq and load_mmsu_mcq.
    """
    @click.command()
    @click.option("--endpoints", default="http://localhost:8100/v1",
                  help="comma list of vLLM endpoints; items round-robin across them")
    @click.option("--model-name", required=True, help="the `model=` value requests target (phi4mm: 'speech')")
    @click.option("--api-key", default="NO_API_KEY")
    @click.option("--data-root", default=default_root)
    @click.option("--subset", type=click.Choice(list(subset_files)), default="full")
    @click.option("--audio-root", default=None)
    @click.option("--audio-mode", default="local-path",
                  help="local-path avoids audio.py's unbounded base64 cache — keep it")
    @click.option("--method", default=0, type=int,
                  help="0 = direct no-CoT (build_messages + NO_COT_SYS); else a build() CoT id (4 = plan-and-solve)")
    @click.option("--n", "n_samples", default=1, type=int, help="total completions per item")
    @click.option("--n-chunk", default=0, type=int,
                  help="max samples per request (0 = all N in one). Bounds KV per in-flight item.")
    @click.option("--temperature", default=0.0, type=float)
    @click.option("--top-p", default=1.0, type=float)
    @click.option("--seed", default=None, type=int,
                  help="base RNG seed; chunk c of item i uses seed+c so chunks are not identical")
    @click.option("--max-tokens", default=64, type=int)
    @click.option("--stop", "stop_strs", default=None,
                  help="comma-separated stop strings. REQUIRED for Kimi-Audio ('[EOS]'), which "
                       "emits its end marker as literal TEXT rather than a real EOS token, so "
                       "without it every generation runs to --max-tokens and then loops.")
    @click.option("--limit", default=None, type=int)
    @click.option("--ids", default=None, help="comma list of unique_ids to restrict to")
    @click.option("--shard", default=None, help="i/N — take records[i::N]; each shard needs its OWN --jsonl")
    @click.option("--max-inflight", default=4, type=int,
                  help="concurrent items per endpoint; peak KV ~= max_inflight * n_chunk * max_tokens * KV/token")
    @click.option("--retries", default=3, type=int)
    @click.option("--request-timeout", default=1800.0, type=float)
    @click.option("--jsonl", "jsonl_path", required=True)
    def main(endpoints, model_name, api_key, data_root, subset, audio_root, audio_mode, method,
             n_samples, n_chunk, temperature, top_p, seed, max_tokens, stop_strs, limit, ids,
             shard, max_inflight, retries, request_timeout, jsonl_path):
        stop = [s for s in (stop_strs or "").split(",") if s] or None
        records = loader_fn(data_root, subset=subset, audio_root=audio_root)
        n_loaded = len(records)
        if ids:
            keep = {x.strip() for x in ids.split(",") if x.strip()}
            records = [r for r in records if r.unique_id in keep]
        if shard:
            i, total = (int(x) for x in shard.split("/"))
            records = records[i::total]
        if limit is not None:
            records = records[:limit]

        os.makedirs(os.path.dirname(os.path.abspath(jsonl_path)), exist_ok=True)
        done = load_done(jsonl_path, n_samples)
        todo = [r for r in records if r.unique_id not in done]

        arm = arm_label(method, temperature, n_samples)
        pname = "direct no-CoT" if method == 0 else f"P{method} {METHODS.get(method, '?')}"
        sizes = chunk_sizes(n_samples, n_chunk)
        eps = [e.strip() for e in endpoints.split(",") if e.strip()]
        print(f"{bench_name} {subset}: {n_loaded} loaded, {len(records)} in scope"
              f"{f' (shard {shard})' if shard else ''}, {len(todo)} to do ({len(done)} resumed)\n"
              f"arm={arm} | prompt={pname} | n={n_samples} as {len(sizes)} request(s) of {sizes[0]} "
              f"| temp={temperature} top_p={top_p} max_tokens={max_tokens} stop={stop}\n"
              f"model={model_name} | endpoints={eps} | max_inflight={max_inflight}/endpoint",
              flush=True)
        if not todo:
            print("nothing to do", flush=True)
            return

        clients = [AsyncOpenAI(base_url=e, api_key=api_key, timeout=request_timeout, max_retries=0)
                   for e in eps]  # retries handled here so they are logged
        sems = [asyncio.Semaphore(max_inflight) for _ in clients]
        fout = open(jsonl_path, "a")  # noqa: SIM115 — long-lived append handle, closed in _run finally
        lock = asyncio.Lock()
        stats = {"items": 0, "samples": 0, "errors": 0, "truncated": 0, "unparsed": 0,
                 "gen_tokens": 0}
        t_start = time.time()

        async def _one(rec, client, sem):
            msgs = build_prompt(rec, method, audio_mode)
            gi = rec.answer_index
            common = {
                "unique_id": rec.unique_id, "category": rec.category,
                "length_type": rec.length_type, "arm": arm, "method": method,
                "temperature": temperature, "n_requested": n_samples,
                "model": model_name, "gold_letter": LETTERS[gi] if gi is not None else "",
                "n_choices": len(rec.choices),
            }
            rows: list[dict] = []
            err = None
            t0 = time.time()
            base_idx = 0
            for c, size in enumerate(sizes):
                kw = dict(model=model_name, messages=msgs, n=size, temperature=temperature,
                          top_p=top_p, max_tokens=max_tokens)
                if stop:
                    kw["stop"] = stop
                if seed is not None:
                    kw["seed"] = seed + c
                resp = None
                async with sem:
                    for attempt in range(retries + 1):
                        try:
                            resp = await client.chat.completions.create(**kw)
                            err = None
                            break
                        except Exception as e:
                            err = f"{type(e).__name__}: {e}"
                            if attempt < retries:
                                await asyncio.sleep(2**attempt)
                if err is not None or resp is None:
                    break
                ptok = getattr(resp.usage, "prompt_tokens", None) if resp.usage else None
                ctok = getattr(resp.usage, "completion_tokens", None) if resp.usage else None
                stats["gen_tokens"] += ctok or 0
                for k, ch in enumerate(resp.choices):
                    text = (ch.message.content or "") if ch.message else ""
                    pi = predicted_index(text, rec.choices)
                    fr = ch.finish_reason or ""
                    if fr == "length":
                        stats["truncated"] += 1
                    if pi is None:
                        stats["unparsed"] += 1
                    first = (c == 0 and k == 0)
                    rows.append({**common, "sample_idx": base_idx + k, "chunk": c,
                                 "response": text,
                                 "pred_letter": LETTERS[pi] if pi is not None else "",
                                 "correct": is_correct(text, rec.choices, gi),
                                 "finish_reason": fr,
                                 "prompt_tokens": ptok if k == 0 else None,
                                 "chunk_completion_tokens": ctok if k == 0 else None,
                                 "latency_s": round(time.time() - t0, 3) if first else None,
                                 "error": None})
                base_idx += size

            if err is not None:
                # keep any completed chunks (they are valid samples) plus one error marker,
                # so load_done() re-runs the item rather than trusting a short sample set
                rows.append({**common, "sample_idx": -1, "chunk": -1, "response": "",
                             "pred_letter": "", "correct": None, "finish_reason": "",
                             "prompt_tokens": None, "chunk_completion_tokens": None,
                             "latency_s": round(time.time() - t0, 3), "error": err})
                stats["errors"] += 1
            else:
                stats["samples"] += len(rows)
                stats["items"] += 1

            async with lock:
                for row in rows:
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
                n_done = stats["items"] + stats["errors"]
                if n_done % 10 == 0 or n_done == len(todo):
                    el = time.time() - t_start
                    rate = el / max(n_done, 1)
                    print(f"  {n_done}/{len(todo)} items | {stats['samples']} samples | "
                          f"{stats['errors']} err | {rate:.1f}s/item | "
                          f"{stats['gen_tokens'] / max(el, 1e-9):.0f} tok/s | "
                          f"ETA {rate * (len(todo) - n_done) / 60:.0f}m", flush=True)

        async def _run():
            try:
                await asyncio.gather(*(
                    _one(r, clients[i % len(clients)], sems[i % len(clients)])
                    for i, r in enumerate(todo)))
            finally:
                fout.close()
                for c in clients:
                    await c.close()

        asyncio.run(_run())

        el = time.time() - t_start
        print(f"\ndone in {el:.0f}s: {stats['items']} items, {stats['samples']} samples, "
              f"{stats['errors']} errors, {stats['truncated']} truncated (finish_reason=length), "
              f"{stats['unparsed']} unparsed\n"
              f"throughput: {el / max(stats['items'], 1):.2f} s/item, "
              f"{stats['gen_tokens'] / max(el, 1e-9):.0f} generated tok/s\n-> {jsonl_path}",
              flush=True)
        if stats["errors"]:
            print(f"WARNING: {stats['errors']} items errored — re-run the SAME command to resume them",
                  flush=True)


    return main
