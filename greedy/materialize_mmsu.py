"""Materialize the MMSU dataset layout (MMSU-meta.json + audio/<id>.wav) from the
official parquets of ddwang2000/MMSU.

The reference copy on the original box was produced this way, so a collaborator
running this against the pinned parquet revision reproduces it byte-identically
(fetch_data.sh verifies via the committed greedy/manifests/mmsu.sha256):

- rows in parquet order (train-00000 -> train-00002)
- each row's `audio.bytes` written verbatim to audio/<id>.wav (the parquet `path`
  field names everything .wav even when the payload is an MP3 bitstream — that is
  faithful to the reference copy; soundfile/vLLM decode them fine)
- meta entries keep the choices DICT exactly as shipped, INCLUDING the None
  padding on the 453 two-choice items (the loader drops the padding at load time)
- serialized with json.dumps(items, indent=2, ensure_ascii=True), no trailing
  newline

    python -m greedy.materialize_mmsu --parquet-dir <download>/data --out-root <root>/mmsu
"""

import glob
import json
import os

import click


@click.command()
@click.option("--parquet-dir", required=True, help="dir holding train-*-of-*.parquet")
@click.option("--out-root", required=True, help="target dir for MMSU-meta.json + audio/")
def main(parquet_dir, out_root):
    import pandas as pd

    paths = sorted(glob.glob(os.path.join(parquet_dir, "train-*-of-*.parquet")))
    if not paths:
        raise SystemExit(f"FATAL: no train-*.parquet under {parquet_dir}")
    audio_dir = os.path.join(out_root, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    def clean(v):
        # parquet nulls (the None-padded choices) arrive as pandas NaN; the
        # reference JSON has them as null, and json.dumps would emit invalid NaN
        return None if pd.isna(v) else v

    items = []
    for p in paths:
        df = pd.read_parquet(p)
        for _, row in df.iterrows():
            rid = row["id"]
            wav_path = os.path.join(audio_dir, f"{rid}.wav")
            payload = row["audio"]["bytes"]
            if not (os.path.exists(wav_path) and os.path.getsize(wav_path) == len(payload)):
                with open(wav_path, "wb") as f:
                    f.write(payload)
            items.append({
                "id": rid,
                "task_name": clean(row["task_name"]),
                "audio_path": f"./audio/{rid}.wav",
                "question": clean(row["question"]),
                "choices": {
                    "A": clean(row["choice_a"]), "B": clean(row["choice_b"]),
                    "C": clean(row["choice_c"]), "D": clean(row["choice_d"]),
                },
                "answer": clean(row["answer_gt"]),
                "category": clean(row["category"]),
                "sub-category": clean(row["sub-category"]),
                "sub-sub-category": clean(row["sub-sub-category"]),
                "linguistics_sub_discipline": clean(row["linguistics_sub_discipline"]),
            })

    meta_path = os.path.join(out_root, "MMSU-meta.json")
    with open(meta_path, "w") as f:
        f.write(json.dumps(items, indent=2, ensure_ascii=True))
    print(f"materialized {len(items)} items -> {meta_path} + {audio_dir}/")


if __name__ == "__main__":
    main()
