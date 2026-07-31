"""Load the ORIGINAL MMAU benchmark test-mini split (1,000 single-audio MCQ across
sound/music/speech) from its materialized JSON release. NOT MMAU-Pro.

Data: MMAU-v05.15.25, HF gamma-lab-umd/MMAU-test-mini @ sha ccd9696, materialized by
`materialize_test_mini.py` as one metadata file (`MMAU-meta.json`, a JSON array) plus
`audio/<id>.wav` (mixed as shipped: 16-48 kHz sample rates, mono/stereo/6-channel,
mostly PCM WAV plus 46 MP3 bitstreams named .wav, clips 1.5-34.5 s) under a single
root (default /home/exx/inference-time-scaling/data/mmau). Records are emitted as the SAME
`MCQRecord` dataclass as MMAU-Pro (imported, not copied), so all mmau_pro machinery
(prompt/scoring/probe runners) consumes them unchanged:

  - `category`     <- MMAU `task` (sound/music/speech): the official leaderboard axis,
    used for stratification and reporting. The Reasoning/Information-Extraction
    `category`, `difficulty`, `sub-category` (27), and source `dataset` axes are kept
    out of MCQRecord and exposed via `load_metadata()`.
  - `choices`/`answer` <- the PREFIX-STRIPPED texts ("(A) Man" -> "Man"): the shipped
    strings embed letter prefixes that would double-letter under the shared prompt
    builder. The shipped strings survive as `choices_raw`/`answer_raw` in the meta and
    are what `official_crosscheck.py` feeds the official token-overlap scorer.
  - `answer_index` <- raw-verbatim index first (`choices_raw.index(answer_raw)`,
    exact on 1,000/1,000 and immune to the 27 items whose stripped choices contain
    duplicates), stripped-verbatim second, fuzzy `match_answer_index` fallback.
  - `length_type`  <- duration bucket via soundfile (le10s:623, 10to30s:311,
    gt30s:66; max clip 34.5 s — comfortably inside the 32k window, no le30s subset).

Choice counts are NOT uniform (4:948, 5:24, 2:27, 8:1); LETTERS ("ABCDEFGHIJK")
covers the single 8-choice item. 16 items duplicate the GOLD text under a second
letter — the twin letter is graded wrong by both our scorer and the official one.

MMAU ships an official scorer (`code/evaluation.py`, token-overlap string_match);
see `official_crosscheck.py`.
"""

import contextlib
import json
import os

from benchmarking.mmau_pro.loader import MCQRecord
from benchmarking.mmau_pro.scoring import match_answer_index

SUBSET_FILES = {
    "full": "MMAU-meta.json",  # the 1,000-item test-mini split (the only graded public split)
}

DEFAULT_DATA_ROOT = "/home/exx/inference-time-scaling/data/mmau"


def bucket_length_type(duration_s) -> str:
    """Deterministic duration buckets for MCQRecord.length_type."""
    if duration_s is None:
        return "unknown"
    if duration_s <= 10:
        return "le10s"
    if duration_s <= 30:
        return "10to30s"
    return "gt30s"


def record_from_mmau_row(row: dict, data_root: str, duration_s: float | None = None) -> MCQRecord | None:
    """Convert one MMAU-meta.json entry to an MCQRecord; None if not MCQ.

    Pure (no I/O beyond path joining) so it is unit-testable with synthetic dicts.
    `audio_path` is a single RELATIVE string ('./audio/<id>.wav'); abspath-joining
    against data_root also normalizes the './'. `choices` is already the stripped
    ordered list (materialization strips the '(A) ' prefixes).
    """
    choices = row.get("choices")
    if not choices:
        return None  # non-MCQ / empty guard
    choices = [str(c) for c in choices]
    audio_rel = row.get("audio_path")
    audio_paths = [] if not audio_rel else [os.path.abspath(os.path.join(data_root, str(audio_rel)))]
    answer = row.get("answer")
    choices_raw = row.get("choices_raw")
    answer_raw = row.get("answer_raw")
    # Raw-verbatim first: the shipped prefixed strings are unique even for the 27
    # items whose STRIPPED choices contain duplicate texts, so this index is
    # deterministic (1,000/1,000 exact). Stripped-verbatim and fuzzy are parity
    # fallbacks (dead code on MMAU today).
    if choices_raw and answer_raw in choices_raw:
        answer_index = choices_raw.index(answer_raw)
    elif answer in choices:
        answer_index = choices.index(answer)
    else:
        answer_index = match_answer_index(answer, choices)
    return MCQRecord(
        unique_id=str(row.get("id")),
        question=str(row.get("question")),
        choices=choices,
        answer=str(answer),
        audio_paths=audio_paths,
        category=str(row.get("task")),  # sound/music/speech — the official leaderboard axis
        length_type=bucket_length_type(duration_s),
        answer_index=answer_index,
    )


def load_mmau_mcq(
    data_root: str = DEFAULT_DATA_ROOT,
    subset: str = "full",
    limit: int | None = None,
    require_audio_exists: bool = True,
    audio_root: str | None = None,
) -> list[MCQRecord]:
    """Read MMAU-meta.json and return the MCQ records (1,000 for `full`).

    Call-signature-compatible with mmau_pro's `load_mmau_mcq` so the probe runners
    can be instantiated with either loader (`loader_fn(data_root, subset=..., audio_root=...)`).
    `audio_root` resolves the relative audio paths when the audio lives under a
    different root than the metadata; defaults to `data_root`.

    Durations for `length_type` are measured with soundfile when importable
    (~0.2 s for all 1,000 headers); otherwise the bucket stays 'unknown' (MMAU has
    no duration field fallback).
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
        rec = record_from_mmau_row(row, audio_root or data_root)
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
    """unique_id -> the MMAU taxonomy fields that stay out of MCQRecord.

    `category` here is MMAU's own Reasoning/Information-Extraction axis
    (MCQRecord.category carries `task`); `difficulty` (easy/medium/hard),
    `sub_category` (27), source `dataset` (15), and `split` complete the joinable
    axes. Also carries `choices_raw`/`answer_raw` (the shipped prefixed strings)
    for the official-scorer crosscheck.
    """
    path = os.path.join(data_root, SUBSET_FILES[subset])
    with open(path) as f:
        rows = json.load(f)
    return {
        str(r.get("id")): {
            "category": r.get("category"),  # Reasoning / Information Extraction (coarse)
            "difficulty": r.get("difficulty"),  # easy / medium / hard
            "sub_category": r.get("sub-category"),  # 27 skills (hyphenated source key)
            "dataset": r.get("dataset"),  # 15 source datasets
            "task": r.get("task"),  # sound / music / speech (= MCQRecord.category)
            "split": r.get("split"),
            "choices_raw": r.get("choices_raw"),
            "answer_raw": r.get("answer_raw"),
        }
        for r in rows
    }
