"""Verify the MMAU test-mini data on disk against the loader contract.

Prints record/gradeable counts (with every fuzzy-resolved and unmatched gold answer
shown verbatim), audio existence, task/category/difficulty/sub-category/dataset/
choice-count mixes, duration stats, and — new vs the mmsu template — the sample-rate
and channel mixes (RESULTS.md records them; they drive other models' subset
policies). Also counts the dataset quirks pinned at materialization: 27 choice
lists with duplicate stripped texts, 16 of them duplicating the GOLD text. Exit 1
if records != 1000, gradeable != 1000, or any referenced audio file is missing.

    python -m benchmarking.mmau.verify_data [--data-root /home/exx/inference-time-scaling/data/mmau]
"""

import json
import os

import click

from benchmarking.mmau.loader import (
    DEFAULT_DATA_ROOT,
    SUBSET_FILES,
    bucket_length_type,
    record_from_mmau_row,
)
from benchmarking.mmau_pro.scoring import normalize

# per-subset pins: (records, gradeable)
EXPECT = {"full": (1000, 1000)}


def _mix(counter: dict) -> str:
    return ", ".join(f"{k}:{n}" for k, n in sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0]))))


@click.command()
@click.option("--data-root", default=DEFAULT_DATA_ROOT)
@click.option("--subset", type=click.Choice(list(SUBSET_FILES)), default="full")
def main(data_root, subset):
    path = os.path.join(data_root, SUBSET_FILES[subset])
    with open(path) as f:
        rows = json.load(f)
    print(f"metadata: {path} -> {len(rows)} rows")

    expect_records, expect_gradeable = EXPECT[subset]
    recs = [record_from_mmau_row(r, data_root) for r in rows]
    recs = [r for r in recs if r is not None]
    print(f"MCQ records: {len(recs)} (expect {expect_records})")

    missing = [p for r in recs for p in r.audio_paths if not os.path.exists(p)]
    print(f"missing audio files: {len(missing)} (expect 0)")
    for p in missing[:10]:
        print(f"  MISSING {p}")

    # gold-answer match classification (verbatim / normalized / fuzzy / UNMATCHED)
    verbatim = norm_exact = fuzzy = 0
    for r in recs:
        if r.answer in r.choices:
            verbatim += 1
        elif r.answer_index is not None and normalize(r.answer) in [normalize(c) for c in r.choices]:
            norm_exact += 1
        elif r.answer_index is not None:
            fuzzy += 1
            print(f"  FUZZY   [{r.unique_id}] answer={r.answer!r} -> choice {r.answer_index}: "
                  f"{r.choices[r.answer_index]!r}")
    ungradeable = [r for r in recs if r.answer_index is None]
    for r in ungradeable:
        print(f"  UNMATCHED [{r.unique_id}] answer={r.answer!r} choices={r.choices!r}")
    gradeable = len(recs) - len(ungradeable)
    print(f"gradeable: {gradeable} (expect {expect_gradeable}) = "
          f"{verbatim} verbatim + {norm_exact} normalized + {fuzzy} fuzzy; "
          f"{len(ungradeable)} ungradeable")

    cat = {}
    for r in recs:
        cat[r.category] = cat.get(r.category, 0) + 1
    print(f"task mix ({len(cat)} tasks): {_mix(cat)}")

    nch = {}
    for r in recs:
        nch[len(r.choices)] = nch.get(len(r.choices), 0) + 1
    print(f"choice-count mix: {_mix(nch)}")

    # dataset quirks pinned at materialization
    dup = [r for r in rows if len(set(r["choices"])) != len(r["choices"])]
    gold_dup = [r for r in rows if r["choices"].count(r["answer"]) > 1]
    raw_verbatim = sum(1 for r in rows if r.get("answer_raw") in (r.get("choices_raw") or []))
    print(f"choice lists w/ duplicate stripped texts: {len(dup)} (expect 27); "
          f"gold text duplicated: {len(gold_dup)} (expect 16); "
          f"raw answer-in-choices: {raw_verbatim}/{len(rows)}")

    by_id = {str(r.get("id")): r for r in rows}
    for field, label in (("category", "category"), ("difficulty", "difficulty"),
                         ("sub-category", "sub-category"), ("dataset", "dataset")):
        c = {}
        for r in recs:
            v = by_id[r.unique_id].get(field)
            c[str(v)] = c.get(str(v), 0) + 1
        print(f"{label} mix: {_mix(c)}")

    # duration + format stats: soundfile if available (MMAU ships no duration fallback)
    try:
        import soundfile as sf

        infos = [sf.info(p) for r in recs for p in r.audio_paths]
        durs = sorted(i.duration for i in infos)
        src = "soundfile"
    except ImportError:
        infos, durs = [], []
        src = "soundfile unavailable"
    if durs:
        q = lambda p: durs[min(len(durs) - 1, int(p * len(durs)))]  # noqa: E731
        print(f"durations [{src}]: n={len(durs)} min={durs[0]:.1f}s p50={q(0.5):.1f}s "
              f"p90={q(0.9):.1f}s max={durs[-1]:.1f}s total={sum(durs) / 3600:.2f}h")
        buckets = {}
        for d in durs:
            b = bucket_length_type(d)
            buckets[b] = buckets.get(b, 0) + 1
        print(f"length_type buckets: {_mix(buckets)}")
        sr = {}
        ch = {}
        fmt = {}
        for i in infos:
            sr[i.samplerate] = sr.get(i.samplerate, 0) + 1
            ch[i.channels] = ch.get(i.channels, 0) + 1
            key = f"{i.format}/{i.subtype}"
            fmt[key] = fmt.get(key, 0) + 1
        print(f"sample-rate mix: {_mix(sr)}")
        print(f"channel mix: {_mix(ch)}")
        print(f"format mix: {_mix(fmt)}")
    else:
        print(f"durations: {src}")

    ok = len(recs) == expect_records and gradeable == expect_gradeable and not missing
    print(f"\nVERIFY {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
