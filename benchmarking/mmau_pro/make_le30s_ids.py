"""Build the Run-14 (Gemma 4 E2B) ids files: the <=30 s MMAU-Pro subset.

Gemma 4 E2B hears at most 30 s per clip (processor audio_seq_length=750 x
audio_ms_per_token=40 ms; vLLM warns and truncates beyond). This script measures
every clip's duration (soundfile header read; librosa fallback) and emits, into
--out-dir:

  durations_test5090.csv   per-item durations (source for by-length slices later;
                           multi-audio items have length_type "nan" in the parquet)
  le30s_single_ids.txt     single-audio items with the clip <= cap
  le30s_multi_ids.txt      multi-audio items with EVERY clip <= cap
  le30s_all_ids.txt        single + multi concatenated (the sweep --ids-file)
  probe_ids_100.txt        category round-robin over the <=cap single-audio pool,
                           seeded shuffle within category (NOT smallest-file-first:
                           the existing stratified selector's size bias is wrong
                           for an accuracy probe)
  smoke_ids_16.txt         12 single + 4 multi (seeded; multi included on purpose
                           so the smoke exercises the multi-clip path through EPF)

Offline, no GPU. Exits nonzero if any audio file is unreadable.

    python -m benchmarking.mmau_pro.make_le30s_ids --data-root data/mmau_pro
"""

import csv
import os
import random
from collections import Counter, defaultdict

import click

from benchmarking.mmau_pro.loader import load_mmau_mcq


def clip_duration(path: str, cache: dict) -> float | None:
    """Duration in seconds via a header read; None if unreadable."""
    if path in cache:
        return cache[path]
    dur = None
    try:
        import soundfile as sf

        dur = float(sf.info(path).duration)
    except Exception:
        try:
            import librosa

            dur = float(librosa.get_duration(path=path))
        except Exception:
            dur = None
    cache[path] = dur
    return dur


def _round_robin_probe(pool, n, seed):
    """Category round-robin (largest category first so it absorbs any remainder),
    seeded shuffle within category. Deterministic for a given pool and seed."""
    rng = random.Random(seed)
    by_cat = defaultdict(list)
    for r in pool:
        by_cat[r.category].append(r)
    for cat in by_cat:
        by_cat[cat].sort(key=lambda r: r.unique_id)  # stable base order
        rng.shuffle(by_cat[cat])
    cats = sorted(by_cat, key=lambda c: (-len(by_cat[c]), c))
    chosen, idx = [], dict.fromkeys(cats, 0)
    while len(chosen) < n:
        progressed = False
        for c in cats:
            if len(chosen) >= n:
                break
            if idx[c] < len(by_cat[c]):
                chosen.append(by_cat[c][idx[c]])
                idx[c] += 1
                progressed = True
        if not progressed:
            break  # pool exhausted
    return chosen


def _write_ids(path, recs):
    with open(path, "w") as f:
        for r in recs:
            f.write(r.unique_id + "\n")
    print(f"wrote {len(recs):>5} ids -> {path}", flush=True)


def _pctile(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    i = min(int(q * (len(sorted_vals) - 1)), len(sorted_vals) - 1)
    return sorted_vals[i]


@click.command()
@click.option("--data-root", default="data/mmau_pro")
@click.option("--subset", type=click.Choice(["full", "le30s", "test"]), default="test")
@click.option("--audio-root", default=None, help="root for relative audio paths (if outside --data-root)")
@click.option("--out-dir", default="benchmarking/mmau_pro/results/run14_gemma4e2b")
@click.option("--cap-seconds", default=30.0, help="max clip duration; items with ANY clip above are excluded")
@click.option("--probe-n", default=100, help="size of the prompt-probe id list (single-audio)")
@click.option("--smoke-n", default=16, help="size of the smoke id list")
@click.option("--smoke-multi", default=4,
              help="how many of the smoke items are multi-audio (0 if multi is out of scope)")
@click.option("--seed", default=14, help="RNG seed for the probe/smoke samples (14 = run number)")
def main(data_root, subset, audio_root, out_dir, cap_seconds, probe_n, smoke_n, smoke_multi, seed):
    recs = load_mmau_mcq(data_root, subset=subset, audio_root=audio_root)
    print(f"loaded {len(recs)} MCQ records (subset={subset})", flush=True)

    cache: dict = {}
    unreadable = []
    per_item = []  # (rec, durs)
    for rec in recs:
        durs = [clip_duration(p, cache) for p in rec.audio_paths]
        if any(d is None for d in durs):
            unreadable.append(rec.unique_id)
        per_item.append((rec, durs))
    if unreadable:
        print(f"ERROR: {len(unreadable)} items with unreadable audio, e.g. {unreadable[:3]}")
        raise SystemExit(1)

    os.makedirs(out_dir, exist_ok=True)

    # durations CSV — the later by-length slicing source (length_type is NaN for multi items)
    csv_path = os.path.join(out_dir, "durations_test5090.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["unique_id", "category", "length_type", "num_audio",
                    "dur_max", "dur_total", "durs", "le30s"])
        for rec, durs in per_item:
            w.writerow([rec.unique_id, rec.category, rec.length_type, len(durs),
                        round(max(durs), 2), round(sum(durs), 2),
                        ";".join(f"{d:.2f}" for d in durs),
                        all(d <= cap_seconds for d in durs)])
    print(f"wrote durations -> {csv_path}", flush=True)

    single = [rec for rec, durs in per_item
              if len(durs) == 1 and durs[0] <= cap_seconds]
    multi = [rec for rec, durs in per_item
             if len(durs) > 1 and all(d <= cap_seconds for d in durs)]
    print(f"<= {cap_seconds:.0f}s pool: {len(single)} single-audio + {len(multi)} multi-audio "
          f"= {len(single) + len(multi)} items "
          f"(of {len(recs)}; excluded {len(recs) - len(single) - len(multi)})", flush=True)
    cat_mix = Counter(r.category for r in single)
    print("single-audio category mix: "
          + ", ".join(f"{c}:{n}" for c, n in sorted(cat_mix.items())), flush=True)

    all_durs = sorted(d for _, durs in per_item for d in durs)
    print("clip duration percentiles (s): "
          + " ".join(f"p{int(q*100)}={_pctile(all_durs, q):.1f}"
                     for q in (0.1, 0.25, 0.5, 0.75, 0.9, 1.0)), flush=True)

    _write_ids(os.path.join(out_dir, "le30s_single_ids.txt"), single)
    _write_ids(os.path.join(out_dir, "le30s_multi_ids.txt"), multi)
    _write_ids(os.path.join(out_dir, "le30s_all_ids.txt"), single + multi)

    probe = _round_robin_probe(single, probe_n, seed)
    _write_ids(os.path.join(out_dir, f"probe_ids_{len(probe)}.txt"), probe)
    print("probe category mix: "
          + ", ".join(f"{c}:{n}" for c, n in sorted(Counter(r.category for r in probe).items())),
          flush=True)

    n_multi_smoke = min(smoke_multi, len(multi))
    probe_ids = {r.unique_id for r in probe}
    smoke_single_pool = sorted((r for r in single if r.unique_id not in probe_ids),
                               key=lambda r: r.unique_id)
    smoke_multi_pool = sorted(multi, key=lambda r: r.unique_id)
    rng = random.Random(seed)
    smoke = (rng.sample(smoke_single_pool, smoke_n - n_multi_smoke)
             + rng.sample(smoke_multi_pool, n_multi_smoke))
    _write_ids(os.path.join(out_dir, f"smoke_ids_{len(smoke)}.txt"), smoke)


if __name__ == "__main__":
    main()
