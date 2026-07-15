"""Rebuild the Run 17 Mellow subset parquets (provenance for run17/data/*.parquet).

Generalizes run16/build_le30s_subset.py with the knobs Run 17 needs:

  --max-audios N       drop rows with more than N clips (Mellow has exactly 2 audio
                       slots; 17 of the 5,090 test MCQ rows have 3 clips)
  --max-seconds S      keep rows whose clips ALL fit S seconds
  --flatten-channels   measure duration the way Mellow ingests audio: channels are
                       concatenated end-to-end (reference wrapper reshape(-1)), so a
                       10 s stereo clip counts as 20 s

The two committed Run 17 parquets:

  test_no3a (5,090 MCQ -> 5,073; the grid subset):
      --src-parquet .../mmau_pro_testmini/data/test-00000-of-00001.parquet \
      --audio-root  .../mmau_pro_audio --max-audios 2 \
      --excluded-ids-out excluded_3audio_ids.txt
  test_le10s_flat (326 fully-heard MCQ; sanity-slice subset):
      --src-parquet .../mmau_pro_testmini/data/test-00000-of-00001.parquet \
      --audio-root  .../mmau_pro_audio --max-audios 2 --max-seconds 10 --flatten-channels

--verify-against compares row count + id set with an existing parquet (content
equality; byte equality is not expected — parquet metadata differs per write).
"""

import argparse
import os

import pandas as pd
import soundfile as sf


def clip_seconds(path, flatten_channels, cache):
    if path not in cache:
        try:
            info = sf.info(path)
            secs = info.frames / info.samplerate
            cache[path] = secs * info.channels if flatten_channels else secs
        except Exception:
            cache[path] = None
    return cache[path]


def fits(row, audio_root, max_seconds, flatten_channels, cache):
    paths = [] if row is None else list(row)
    if not paths:
        return False
    for p in paths:
        dur = clip_seconds(os.path.join(audio_root, p), flatten_channels, cache)
        if dur is None or (max_seconds is not None and dur > max_seconds):
            return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-parquet", required=True)
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--out", help="write the filtered parquet here")
    ap.add_argument("--verify-against", help="compare row count + id set with this parquet")
    ap.add_argument("--mcq-only", action="store_true",
                    help="also drop open-ended rows (empty choices)")
    ap.add_argument("--max-audios", type=int, default=None)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--flatten-channels", action="store_true",
                    help="count channels concatenated (Mellow's reshape(-1) ingestion)")
    ap.add_argument("--excluded-ids-out",
                    help="write ids dropped by --max-audios (one per line)")
    args = ap.parse_args()

    df = pd.read_parquet(args.src_parquet)
    n0 = len(df)
    if args.mcq_only:
        df = df[df["choices"].map(lambda c: c is not None and len(list(c)) > 0)]

    if args.max_audios is not None:
        n_audio = df["audio_path"].map(lambda r: 0 if r is None else len(list(r)))
        dropped = df[n_audio > args.max_audios]
        if args.excluded_ids_out:
            with open(args.excluded_ids_out, "w") as f:
                f.writelines(f"{i}\n" for i in dropped["id"])
            print(f"wrote {len(dropped)} excluded ids -> {args.excluded_ids_out}")
        df = df[n_audio <= args.max_audios]

    cache = {}
    if args.max_seconds is not None:
        df = df[df["audio_path"].map(
            lambda r: fits(r, args.audio_root, args.max_seconds, args.flatten_channels, cache)
        )]
    out = df.reset_index(drop=True)
    print(f"{n0} rows -> {len(out)} "
          f"(max_audios={args.max_audios}, max_seconds={args.max_seconds}, "
          f"flatten={args.flatten_channels}, mcq_only={args.mcq_only})")

    if args.out:
        out.to_parquet(args.out, index=False)
        print(f"wrote {args.out}")
    if args.verify_against:
        ref = pd.read_parquet(args.verify_against)
        assert len(ref) == len(out), f"row count {len(out)} != committed {len(ref)}"
        assert set(ref["id"]) == set(out["id"]), "id sets differ"
        print(f"VERIFY OK vs {args.verify_against}")


if __name__ == "__main__":
    main()
