"""STAR-Bench-Perception loader (macabdul9/STAR-Bench-Perception, 626 MCQ).

The HF parquet embeds WAV bytes per row; vLLM and the judge need real files,
so on first load the audio is materialized once to <root>/audio/<id>.wav.
Records reuse benchmarking.mmau_pro.loader.MCQRecord so everything downstream
(prompt build, judge scorer, selectors, filing) works unchanged.

Set BABEL_STARBENCH_ROOT to the dataset snapshot directory (the one holding
data/*.parquet, produced by starbench_fetch.py).
"""
import glob
import os

from benchmarking.mmau_pro.loader import MCQRecord

ROOT = os.environ.get(
    "BABEL_STARBENCH_ROOT",
    "/work/hdd/bcey/awaheed/its-for-audio-reasoning/epf_data/starbench")


def _materialize_audio(df, audio_dir):
    os.makedirs(audio_dir, exist_ok=True)
    written = 0
    for _, row in df.iterrows():
        dest = os.path.join(audio_dir, f"{row['id']}.wav")
        if os.path.isfile(dest) and os.path.getsize(dest) > 0:
            continue
        blob = row["audio"]
        data = blob["bytes"] if isinstance(blob, dict) else blob
        tmp = dest + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, dest)
        written += 1
    return written


def load_starbench_mcq(root=None):
    import pandas as pd

    root = root or ROOT
    parquets = sorted(glob.glob(os.path.join(root, "data", "*.parquet")))
    if not parquets:
        raise FileNotFoundError(f"no parquet files under {root}/data - run starbench_fetch.py")
    df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)

    audio_dir = os.path.join(root, "audio")
    n = _materialize_audio(df, audio_dir)
    if n:
        print(f"[starbench] materialized {n} wav files -> {audio_dir}")

    recs = []
    for _, row in df.iterrows():
        choices = list(row["choices"])
        answer = str(row["answer"])
        answer_index = choices.index(answer) if answer in choices else None
        recs.append(MCQRecord(
            unique_id=str(row["id"]),
            question=str(row["question"]),
            choices=choices,
            answer=answer,
            audio_paths=[os.path.join(audio_dir, f"{row['id']}.wav")],
            category=str(row.get("category", "")),
            length_type=str(row.get("task", "")),
            answer_index=answer_index,
        ))
    return recs
