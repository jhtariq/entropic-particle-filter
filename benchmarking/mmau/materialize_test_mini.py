"""Materialize the MMAU test-mini layout (MMAU-meta.json + audio/<id>.wav) from the
official parquet of gamma-lab-umd/MMAU-test-mini (HF sha ccd9696, MMAU-v05.15.25).

Mirrors greedy/materialize_mmsu.py, adapted to the MMAU test-mini schema:

- audio is EMBEDDED per row under `context.bytes` (`context.path` is null), written
  verbatim to audio/<id>.wav — idempotent size-check skip. Payloads are mixed as
  shipped: mostly PCM WAV (16/32-bit and float, 16-48 kHz, mono/stereo/6-channel)
  plus 46 MP3 bitstreams named .wav (soundfile/vLLM decode them fine, the MMSU
  precedent)
- all metadata (`id`, `dataset`, `task`, `split`, `category`, `sub-category`,
  `difficulty`) is packed in the `other_attributes` JSON string and unpacked here;
  13 rows ship `category` as a one-element list (e.g. ["Reasoning"]) — normalized
  to the plain string
- choices AND answer carry `(A) `-style letter prefixes; the prompt builder letters
  choices itself, so the prefixed text would render as "A. (A) Man". The prefixes
  are STRIPPED into `choices`/`answer` (what the model sees / what our scorer
  grades) while the shipped strings are kept as `choices_raw`/`answer_raw` — the
  official token-overlap scorer (code/evaluation.py `string_match`) must be fed
  the raw text, since stripped "Man" lacks the gold token "a" of "(A) Man"
- serialized with json.dumps(items, indent=2, ensure_ascii=True), no trailing
  newline

    python -m benchmarking.mmau.materialize_test_mini \
        --parquet /home/exx/inference-time-scaling/data/mmau/raw/test-mini/test_mini.parquet \
        --out-root /home/exx/inference-time-scaling/data/mmau
"""

import json
import os
import re

import click

PREFIX_RE = re.compile(r"^\([A-Z]\)\s*")


def strip_prefix(text: str) -> str:
    return PREFIX_RE.sub("", text)


@click.command()
@click.option("--parquet", required=True, help="path to test_mini.parquet")
@click.option("--out-root", required=True, help="target dir for MMAU-meta.json + audio/")
def main(parquet, out_root):
    import pandas as pd

    df = pd.read_parquet(parquet)
    audio_dir = os.path.join(out_root, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    items = []
    n_dup_after_strip = 0
    for _, row in df.iterrows():
        attrs = json.loads(row["other_attributes"])
        rid = attrs["id"]
        payload = row["context"]["bytes"]
        wav_path = os.path.join(audio_dir, f"{rid}.wav")
        if not (os.path.exists(wav_path) and os.path.getsize(wav_path) == len(payload)):
            with open(wav_path, "wb") as f:
                f.write(payload)
        choices_raw = [str(c) for c in row["choices"]]
        answer_raw = str(row["answer"])
        assert answer_raw in choices_raw, f"{rid}: answer not among choices"
        choices = [strip_prefix(c) for c in choices_raw]
        if len(set(choices)) != len(choices):
            n_dup_after_strip += 1
        category = attrs.get("category")
        if isinstance(category, list):  # 13 rows ship e.g. ["Reasoning"]
            (category,) = category
        items.append({
            "id": rid,
            "audio_path": f"./audio/{rid}.wav",
            "question": str(row["instruction"]),
            "choices": choices,
            "answer": strip_prefix(answer_raw),
            "choices_raw": choices_raw,
            "answer_raw": answer_raw,
            "task": attrs.get("task"),
            "dataset": attrs.get("dataset"),
            "category": category,
            "sub-category": attrs.get("sub-category"),
            "difficulty": attrs.get("difficulty"),
            "split": attrs.get("split"),
        })

    assert len(items) == 1000, f"expected 1000 rows, got {len(items)}"
    ids = [it["id"] for it in items]
    assert len(set(ids)) == len(ids), "duplicate ids"

    meta_path = os.path.join(out_root, "MMAU-meta.json")
    with open(meta_path, "w") as f:
        f.write(json.dumps(items, indent=2, ensure_ascii=True))
    print(f"materialized {len(items)} items -> {meta_path} + {audio_dir}/")
    print(f"choice lists with duplicates after prefix-strip: {n_dup_after_strip}")


if __name__ == "__main__":
    main()
