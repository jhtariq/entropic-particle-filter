"""Load the MMSU benchmark (5,000 single-audio spoken-language-understanding MCQ)
from its JSON release.

MMSU ships as one metadata file (`MMSU-meta.json`, a JSON array) plus `audio/*.wav`
under a single root (default /home/exx/inference-time-scaling/data/mmsu). Records are
emitted as the SAME `MCQRecord` dataclass as MMAU-Pro (imported, not copied), so all
mmau_pro machinery (prompt/scoring/probe runners) consumes them unchanged:

  - `category`     <- MMSU `task_name` (47 fine-grained tasks): the stratification and
    reporting axis. MMSU's OWN `category` field is the coarse Perception/Reasoning axis,
    a strict 1-into-2 refinement of task_name — kept out of MCQRecord and exposed via
    `load_metadata()` so both cuts stay retrievable.
  - `length_type`  <- duration bucket (le10s/10to30s/gt30s/unknown), measured with
    soundfile when importable (MMSU ships no `timestamp` field, so without soundfile the
    bucket is 'unknown'). Full-set clip durations run 0.3-35.9 s.
  - `answer_index` <- verbatim choice index (MMSU gold `answer` is the full choice TEXT
    and matches a choice exactly in all 5,000 rows), fuzzy `match_answer_index` fallback.

MMSU ships no official scorer (no `code/evaluation.py`), so there is no crosscheck tool.

`choices` is a DICT keyed by letter ({"A": .., "B": .., "C": .., "D": ..}); the loader
flattens it to an ordered list by letter (A, B, C, D) so it matches the MCQRecord contract.
"""

import contextlib
import json
import os

from benchmarking.mmau_pro.loader import MCQRecord
from benchmarking.mmau_pro.scoring import LETTERS, match_answer_index

SUBSET_FILES = {
    "full": "MMSU-meta.json",  # all 5,000 single-audio MCQ
}

DEFAULT_DATA_ROOT = "/home/exx/inference-time-scaling/data/mmsu"


def bucket_length_type(duration_s) -> str:
    """Deterministic duration buckets for MCQRecord.length_type."""
    if duration_s is None:
        return "unknown"
    if duration_s <= 10:
        return "le10s"
    if duration_s <= 30:
        return "10to30s"
    return "gt30s"


def record_from_mmsu_row(row: dict, data_root: str, duration_s: float | None = None) -> MCQRecord | None:
    """Convert one MMSU-meta.json entry to an MCQRecord; None if not MCQ.

    Pure (no I/O beyond path joining) so it is unit-testable with synthetic dicts.
    `audio_path` is a single RELATIVE string ('./audio/<id>.wav'); abspath-joining
    against data_root also normalizes the './'. `choices` is a letter-keyed DICT
    flattened to a list in A,B,C,D order.
    """
    choices_raw = row.get("choices")
    if not choices_raw:
        return None  # non-MCQ / empty guard
    # DICT keyed by letter -> ordered list (LETTERS = "ABCDEFGHIJK"). 453/5,000 items are
    # 2-choice, PADDED to 4 letter-keys with Python None VALUES, and the padding can sit in
    # ANY position (e.g. {"A":None,"B":"male","C":None,"D":"female"}). Drop the None padding
    # so the MCQ presents only real options (mmar-style variable choice counts: 4547 four-
    # choice + 453 two-choice). The literal string "None" is a real option and is kept.
    choices = [choices_raw[k] for k in LETTERS if choices_raw.get(k) is not None]
    if not choices:
        return None
    audio_rel = row.get("audio_path")
    audio_paths = [] if not audio_rel else [os.path.abspath(os.path.join(data_root, str(audio_rel)))]
    answer = row.get("answer")
    # Verbatim index first (MMSU gold is the full choice text; 5,000/5,000 verbatim);
    # fuzzy match_answer_index kept as a parity fallback (dead code on MMSU today).
    if answer in choices:
        answer_index = choices.index(answer)
    else:
        answer_index = match_answer_index(answer, choices)
    return MCQRecord(
        unique_id=str(row.get("id")),
        question=str(row.get("question")),
        choices=choices,
        answer=str(answer),
        audio_paths=audio_paths,
        category=str(row.get("task_name")),  # NOT MMSU's Perception/Reasoning `category` field
        length_type=bucket_length_type(duration_s),
        answer_index=answer_index,
    )


def load_mmsu_mcq(
    data_root: str = DEFAULT_DATA_ROOT,
    subset: str = "full",
    limit: int | None = None,
    require_audio_exists: bool = True,
    audio_root: str | None = None,
) -> list[MCQRecord]:
    """Read MMSU-meta.json and return the MCQ records (5,000 for `full`).

    Call-signature-compatible with mmau_pro's `load_mmau_mcq` so the probe runners
    can be instantiated with either loader (`loader_fn(data_root, subset=..., audio_root=...)`).
    `audio_root` resolves the relative audio paths when the audio lives under a
    different root than the metadata; defaults to `data_root`.

    Durations for `length_type` are measured with soundfile when importable
    (~0.2 s for all 5,000 headers); otherwise the bucket stays 'unknown' (MMSU has
    no `timestamp` fallback).
    """
    if subset not in SUBSET_FILES:
        raise ValueError(f"subset must be one of {list(SUBSET_FILES)}, got {subset!r}")
    path = os.path.join(data_root, SUBSET_FILES[subset])
    with open(path) as f:
        rows = json.load(f)

    try:
        import soundfile as sf
    except ImportError:
        sf = None  # duration buckets stay 'unknown' (uv test env has no soundfile)

    records: list[MCQRecord] = []
    for row in rows:
        rec = record_from_mmsu_row(row, audio_root or data_root)
        if rec is None:
            continue
        if require_audio_exists and not all(os.path.exists(p) for p in rec.audio_paths):
            continue
        if sf is not None and all(os.path.exists(p) for p in rec.audio_paths):
            with contextlib.suppress(RuntimeError):  # unreadable header -> keep 'unknown'
                rec.length_type = bucket_length_type(sum(sf.info(p).duration for p in rec.audio_paths))
        records.append(rec)
        if limit is not None and len(records) >= limit:
            break
    return records


def load_metadata(data_root: str = DEFAULT_DATA_ROOT, subset: str = "full") -> dict[str, dict]:
    """unique_id -> the MMSU coarse-taxonomy fields that stay out of MCQRecord.

    `category` here is MMSU's own Perception/Reasoning axis (MCQRecord.category carries
    the fine-grained `task_name`); each task_name maps to exactly one of these, so the
    2-way axis is fully recoverable by joining a CSV's unique_id through this map.
    """
    path = os.path.join(data_root, SUBSET_FILES[subset])
    with open(path) as f:
        rows = json.load(f)
    return {
        str(r.get("id")): {
            "category": r.get("category"),  # Perception / Reasoning (coarse)
            "sub_category": r.get("sub-category"),  # hyphenated source keys
            "sub_sub_category": r.get("sub-sub-category"),
            "linguistics_sub_discipline": r.get("linguistics_sub_discipline"),
            "task_name": r.get("task_name"),
        }
        for r in rows
    }
