"""Verify the MMSU data on disk against the loader contract (SETUP_GUIDE §11 step 2).

Prints record/gradeable counts (with every fuzzy-resolved and unmatched gold answer
shown verbatim), audio existence, task_name/category/sub-category/sub-sub-category/
linguistics_sub_discipline/choice-count mixes, and duration stats. Exit 1 if records
!= 5000, gradeable != 5000, or any referenced audio file is missing.

    python -m benchmarking.mmsu.verify_data [--data-root /home/exx/inference-time-scaling/data/mmsu]
"""

import json
import os

import click

from benchmarking.mmau_pro.scoring import normalize
from benchmarking.mmsu.loader import (
    DEFAULT_DATA_ROOT,
    SUBSET_FILES,
    bucket_length_type,
    record_from_mmsu_row,
)

# per-subset pins: (records, gradeable)
EXPECT = {"full": (5000, 5000)}


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
    recs = [record_from_mmsu_row(r, data_root) for r in rows]
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
    print(f"task_name mix ({len(cat)} tasks): {_mix(cat)}")

    nch = {}
    for r in recs:
        nch[len(r.choices)] = nch.get(len(r.choices), 0) + 1
    print(f"choice-count mix: {_mix(nch)}")

    by_id = {str(r.get("id")): r for r in rows}
    for field, label in (("category", "category"), ("sub-category", "sub-category"),
                         ("sub-sub-category", "sub-sub-category"),
                         ("linguistics_sub_discipline", "linguistics_sub_discipline")):
        c = {}
        for r in recs:
            v = by_id[r.unique_id].get(field)
            c[str(v)] = c.get(str(v), 0) + 1
        print(f"{label} mix: {_mix(c)}")

    # duration + channel stats: soundfile if available (MMSU ships no timestamp fallback)
    chans = {}
    try:
        import soundfile as sf

        infos = [[sf.info(p) for p in r.audio_paths] for r in recs] if not missing else []
        durs = sorted(sum(i.duration for i in per_rec) for per_rec in infos)
        for per_rec in infos:
            for i in per_rec:
                chans[i.channels] = chans.get(i.channels, 0) + 1
        src = "soundfile"
    except ImportError:
        durs = []
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
    else:
        print(f"durations: {src}")
    # channel mix (reference box: 4550 mono + 450 stereo) — the shim and vLLM both
    # downmix, so stereo is not an error, but a shifted count means a different copy
    print(f"channel mix [{src}]: {_mix(chans) if chans else src}")

    ok = len(recs) == expect_records and gradeable == expect_gradeable and not missing
    print(f"\nVERIFY {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
