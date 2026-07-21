"""Build the Run-19 single-audio MMAU-Pro parquets (provenance for run19/data/).

vLLM's Kimi-Audio implementation hard-caps ONE audio per prompt (multi-audio
requests 400 at request time), so Run 19 filters the parent parquets to rows
with exactly one clip:

    test_le30s (2,190 MCQ)  ->  test_le30s_1a (1,947 MCQ, 13 ungradeable)
    test       (5,305 rows) ->  test_1a       (4,660 MCQ, 22 ungradeable)

    python benchmarking/mmau_pro/run19/build_1a_subset.py \
        --src-parquet <parent.parquet> --out <child_1a.parquet> \
        [--audio-root <dir> --data-root <mmau_pro_testmini>]

With --audio-root/--data-root it re-loads the output through the real loader and
prints the MCQ/ungradeable counts to pin in fetch_data.sh and smoke.sh.
"""

import argparse
import os

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", help="mmau_pro_testmini root for loader verification")
    ap.add_argument("--audio-root", help="audio root for loader verification")
    args = ap.parse_args()

    df = pd.read_parquet(args.src_parquet)
    keep = df["audio_path"].apply(len) == 1
    out = df[keep].reset_index(drop=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"{os.path.basename(args.src_parquet)}: {len(df)} rows -> "
          f"{len(out)} single-audio rows -> {args.out}")

    if args.data_root:
        from benchmarking.mmau_pro.loader import SUBSET_FILES, load_mmau_mcq

        name = os.path.basename(args.out)
        subset = next((k for k, v in SUBSET_FILES.items() if v == name), None)
        if subset is None:
            print(f"(no SUBSET_FILES entry for {name} — skipping loader verification)")
            return
        recs = load_mmau_mcq(args.data_root, subset=subset, audio_root=args.audio_root)
        n_ungr = sum(1 for r in recs if r.answer_index is None)
        assert all(len(r.audio_paths) == 1 for r in recs)
        print(f"loader check [{subset}]: {len(recs)} MCQ ({n_ungr} ungradeable), all single-audio")


if __name__ == "__main__":
    main()
