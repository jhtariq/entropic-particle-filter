"""Audio-capacity audit against a LIVE endpoint (MMAU test-mini edition of the
run18/19 tool).

vLLM audio models expand clips into prompt tokens SERVER-side, so offline token
estimates are unreliable — ask the served model instead:

  1) empirical audio-token rate: paired 1-token requests (audio vs text-only) on
     N duration-quantile items; audio tokens = delta usage.prompt_tokens
  2) 1-token preflight on the N items with the LONGEST total clip duration: each
     must return HTTP 200 and fit --max-model-len with --gen-headroom to spare
     (headroom = max_steps x max_tokens_per_step of the EPF grid, 6x300 by default)

Exit code 1 iff any preflight request fails or overflows the context budget.
This is a TOKEN-FIT gate, not a clip-length gate — MMAU test-mini's longest clip
is ~34.5 s (well within Qwen2.5-Omni's 32k window), but the preflight ALSO
exercises the format tail (46 MP3-in-.wav, 69 six-channel, float/PCM_32 payloads)
through the server's decoder. Do NOT trust a first-N smoke instead of this —
metadata order biases short audio (SETUP_GUIDE §10 anomaly 4).
"""

import json
import time

import click
import httpx
import soundfile as sf

from benchmarking.mmau.loader import DEFAULT_DATA_ROOT, load_mmau_mcq
from benchmarking.mmau_pro.audio import audio_content_parts

_SYS = "You are an expert audio analyst. Listen to the audio carefully."


def _one_token_request(client, endpoint, model_name, parts, timeout):
    """POST a max_tokens=1 chat completion; return (status, prompt_tokens, err, seconds)."""
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": _SYS},
            {"role": "user", "content": parts},
        ],
        "max_tokens": 1,
        "temperature": 0,
    }
    t0 = time.time()
    try:
        r = client.post(f"{endpoint}/chat/completions", json=payload, timeout=timeout)
    except httpx.HTTPError as e:
        return 0, None, f"{type(e).__name__}: {e}", time.time() - t0
    dt = time.time() - t0
    if r.status_code != 200:
        return r.status_code, None, r.text[:300], dt
    return 200, r.json().get("usage", {}).get("prompt_tokens"), None, dt


def _total_duration(rec) -> float:
    return sum(sf.info(p).duration for p in rec.audio_paths)


@click.command()
@click.option("--endpoint", default="http://localhost:8100/v1")
@click.option("--model-name", required=True)
@click.option("--data-root", default=DEFAULT_DATA_ROOT)
@click.option("--subset", default="full")
@click.option("--audio-root", default=None)
@click.option("--n-longest", default=30, help="preflight the N items with the longest total audio")
@click.option("--n-rate", default=10, help="duration-quantile single-audio items for the rate fit")
@click.option("--max-model-len", default=32768)
@click.option("--gen-headroom", default=1800, help="tokens reserved for generation (max_steps x max_tokens_per_step)")
@click.option("--timeout", default=600.0, help="per-request timeout (s) — long clips are slow through the encoder")
@click.option("--out", "out_path", default=None, help="write the full JSON report here")
def main(endpoint, model_name, data_root, subset, audio_root, n_longest, n_rate,
         max_model_len, gen_headroom, timeout, out_path):
    records = load_mmau_mcq(data_root, subset=subset, audio_root=audio_root)
    print(f"loaded {len(records)} records (subset={subset}); measuring clip durations …")
    durs = [(r, _total_duration(r)) for r in records]
    durs.sort(key=lambda x: x[1], reverse=True)

    client = httpx.Client()
    report = {"endpoint": endpoint, "model": model_name, "subset": subset,
              "max_model_len": max_model_len, "gen_headroom": gen_headroom,
              "rate": [], "preflight": []}

    # -- 1) audio-token rate: single-audio items at duration quantiles ------------
    singles = [(r, d) for r, d in durs if len(r.audio_paths) == 1]
    picks = [singles[int(q * (len(singles) - 1))] for q in
             (i / max(1, n_rate - 1) for i in range(n_rate))]
    rates = []
    for rec, dur in picks:
        parts = audio_content_parts(rec.audio_paths, mode="local-path")
        text = {"type": "text", "text": "Reply with the single word OK."}
        st_a, tok_a, err_a, _ = _one_token_request(client, endpoint, model_name, [*parts, text], timeout)
        st_t, tok_t, err_t, _ = _one_token_request(client, endpoint, model_name, [text], timeout)
        row = {"unique_id": rec.unique_id, "duration_s": round(dur, 1),
               "prompt_tokens_audio": tok_a, "prompt_tokens_text": tok_t,
               "error": err_a or err_t}
        if st_a == st_t == 200 and dur > 0:
            row["tokens_per_s"] = round((tok_a - tok_t) / dur, 3)
            rates.append(row["tokens_per_s"])
        report["rate"].append(row)
        print(f"  rate: {dur:7.1f}s -> audio {tok_a} vs text {tok_t} tokens"
              f" ({row.get('tokens_per_s', 'ERR')} tok/s) [{rec.unique_id}]")
    rates.sort()
    median_rate = rates[len(rates) // 2] if rates else None
    report["median_tokens_per_s"] = median_rate

    # -- 2) 1-token preflight on the longest items --------------------------------
    n_fail, max_prompt = 0, 0
    for rec, dur in durs[:n_longest]:
        parts = audio_content_parts(rec.audio_paths, mode="local-path")
        text = {"type": "text", "text": f"{rec.question}\nReply with the single word OK."}
        status, ptok, err, dt = _one_token_request(client, endpoint, model_name, [*parts, text], timeout)
        fits = ptok is not None and ptok + gen_headroom <= max_model_len
        if status != 200 or not fits:
            n_fail += 1
        max_prompt = max(max_prompt, ptok or 0)
        report["preflight"].append({
            "unique_id": rec.unique_id, "n_audio": len(rec.audio_paths),
            "total_s": round(dur, 1), "status": status, "prompt_tokens": ptok,
            "latency_s": round(dt, 1), "fits": fits, "error": err,
        })
        print(f"  preflight: {dur:7.1f}s x{len(rec.audio_paths)} -> HTTP {status}, "
              f"{ptok} prompt tokens, {dt:5.1f}s {'OK' if status == 200 and fits else 'FAIL'}"
              f" [{rec.unique_id}]")

    verdict = "PASS" if n_fail == 0 else "FAIL"
    report["verdict"] = verdict
    report["max_prompt_tokens"] = max_prompt
    print(f"\nmedian audio-token rate: {median_rate} tok/s")
    print(f"max prompt tokens (longest {n_longest}): {max_prompt} "
          f"(+{gen_headroom} headroom vs {max_model_len} max-model-len)")
    print(f"AUDIT {verdict}: {n_fail}/{n_longest} preflight failures")
    if out_path:
        with open(out_path, "w") as f:
            json.dump(report, f, indent=1)
        print(f"report -> {out_path}")
    raise SystemExit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
