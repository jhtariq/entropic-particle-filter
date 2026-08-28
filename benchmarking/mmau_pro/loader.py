"""Load the MMAU-Pro test-mini MCQ subset (957 items) from the local HF parquet."""

import os
from dataclasses import dataclass

from benchmarking.mmau_pro.scoring import match_answer_index

SUBSET_FILES = {
    "full": "testmini-00000-of-00001.parquet",
    # MMAU-Pro-D1K (macabdul9/MMAU-Pro-D1K): curated 1,000-item single-audio subset of the
    # full test set; ships its own audio, so the parquet and clips share one data/ root
    "d1k": "test.parquet",
    "le30s": "testmini_le30s-00000-of-00001.parquet",
    "test": "test-00000-of-00001.parquet",  # the FULL MMAU-Pro test set (5,305 rows)
    # full test set filtered to items whose clips ALL fit Qwen2-Audio's 30 s encoder
    # window (built by scripts alongside Run 15; 2,190 MCQ)
    "test_le30s": "test_le30s-00000-of-00001.parquet",
    # Run 17 (Mellow, 2 audio slots): test minus the 17 three-audio rows (5,073 MCQ)
    "test_no3a": "test_no3a-00000-of-00001.parquet",
    # Run 17 sanity slice: items Mellow FULLY hears — all clips <= 10 s with channels
    # counted concatenated, the way its wrapper ingests audio (326 MCQ, single-audio)
    "test_le10s_flat": "test_le10s_flat-00000-of-00001.parquet",
    # Run 19 (Kimi-Audio: vLLM caps at ONE audio/prompt, Whisper window is 30 s):
    # single-audio filters of test_le30s / test (built by run19/build_1a_subset.py)
    "test_le30s_1a": "test_le30s_1a-00000-of-00001.parquet",  # 1,947 MCQ (13 ungradeable)
    "test_1a": "test_1a-00000-of-00001.parquet",  # 4,660 MCQ (22 ungradeable) — capacity audit
}


@dataclass
class MCQRecord:
    unique_id: str
    question: str
    choices: list[str]
    answer: str
    audio_paths: list[str]  # absolute paths
    category: str
    length_type: str
    answer_index: int | None  # precomputed (possibly fuzzy); None => ungradeable


def record_from_row(row: dict, data_root: str) -> MCQRecord | None:
    """Convert one parquet row (as a dict) to an MCQRecord; None if not MCQ.

    Pure (no I/O beyond path joining) so it is unit-testable with synthetic dicts.
    """
    # parquet list-columns arrive as numpy arrays; avoid truthiness on arrays
    choices_raw = row.get("choices")
    choices = [] if choices_raw is None else list(choices_raw)
    if len(choices) == 0:
        return None  # non-MCQ / open-ended → excluded
    audio_raw = row.get("audio_path")
    audio_rel = [] if audio_raw is None else list(audio_raw)
    audio_paths = [os.path.join(data_root, p) for p in audio_rel]
    answer = row.get("answer")
    return MCQRecord(
        unique_id=str(row.get("id")),
        question=str(row.get("question")),
        choices=choices,
        answer=str(answer),
        audio_paths=audio_paths,
        category=str(row.get("category")),
        length_type=str(row.get("length_type")),
        answer_index=match_answer_index(answer, choices),
    )


def load_mmau_mcq(
    data_root: str,
    subset: str = "full",
    limit: int | None = None,
    require_audio_exists: bool = True,
    audio_root: str | None = None,
) -> list[MCQRecord]:
    """Read the parquet and return the MCQ records (957 for `full`, 5,090 for `test`).

    `audio_root` resolves the relative audio paths when the audio files live under
    a different root than the parquet (e.g. the full `test` parquet ships in the
    testmini repo while its audio lives in `mmau_pro_audio/`). Defaults to
    `data_root` (the original single-root layout).
    """
    import pandas as pd

    if subset not in SUBSET_FILES:
        raise ValueError(f"subset must be one of {list(SUBSET_FILES)}, got {subset!r}")
    path = os.path.join(data_root, "data", SUBSET_FILES[subset])
    df = pd.read_parquet(path)

    records: list[MCQRecord] = []
    for _, row in df.iterrows():
        rec = record_from_row(row.to_dict(), audio_root or data_root)
        if rec is None:
            continue
        if require_audio_exists and not all(os.path.exists(p) for p in rec.audio_paths):
            continue
        records.append(rec)
        if limit is not None and len(records) >= limit:
            break
    return records
