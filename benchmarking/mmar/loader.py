"""Load the MMAR benchmark (1,000 single-audio MCQ reasoning items) from its JSON release.

MMAR ships as one metadata file (`MMAR-meta.json`, a JSON array) plus `audio/*.wav`
under a single root (default /home/exx/inference-time-scaling/mmar). Records are
emitted as the SAME `MCQRecord` dataclass as MMAU-Pro (imported, not copied), so all
mmau_pro machinery (prompt/scoring/probe runners) consumes them unchanged:

  - `category`     <- MMAR `modality` (sound/music/speech/mix-*): the stratification
    and reporting axis, matching the official scorer's primary breakdown
  - `length_type`  <- duration bucket (le10s/10to30s/gt30s/unknown), measured with
    soundfile when importable, else parsed from `timestamp` (45/1000 timestamps are
    malformed -> 'unknown' without soundfile)
  - `answer_index` <- mmau_pro's fuzzy `match_answer_index` (996/1000 gradeable)

MMAR's own `category` field (Semantic/Perception/Cultural/Signal Layer),
`sub-category` and `language` stay OUT of MCQRecord; use `load_metadata()` for
those reporting cuts.
"""

import contextlib
import json
import os

from benchmarking.mmau_pro.loader import MCQRecord
from benchmarking.mmau_pro.scoring import match_answer_index

SUBSET_FILES = {
    "full": "MMAR-meta.json",  # all 1,000 single-audio MCQ
    # clips that fit Qwen2-Audio's hard 30 s encoder window (983 items; built by
    # build_le30s_subset.py, soundfile-measured — mirrors mmau_pro test_le30s)
    "le30s": "MMAR-meta-le30s.json",
}

DEFAULT_DATA_ROOT = "/home/exx/inference-time-scaling/mmar"


def parse_timestamp_duration(ts) -> float | None:
    """'00:11:10,00:11:30' -> 20.0 seconds; None for malformed or non-positive spans."""
    if not isinstance(ts, str) or "," not in ts:
        return None

    def _sec(t: str) -> float:
        h, m, s = (float(x) for x in t.strip().split(":"))
        return h * 3600 + m * 60 + s

    try:
        start, end = ts.split(",", 1)
        dur = _sec(end) - _sec(start)
    except (TypeError, ValueError):
        return None
    return dur if dur > 0 else None


def bucket_length_type(duration_s) -> str:
    """Deterministic duration buckets for MCQRecord.length_type."""
    if duration_s is None:
        return "unknown"
    if duration_s <= 10:
        return "le10s"
    if duration_s <= 30:
        return "10to30s"
    return "gt30s"


def record_from_mmar_row(row: dict, data_root: str, duration_s: float | None = None) -> MCQRecord | None:
    """Convert one MMAR-meta.json entry to an MCQRecord; None if not MCQ.

    Pure (no I/O beyond path joining) so it is unit-testable with synthetic dicts.
    Unlike the MMAU-Pro parquet, `audio_path` is a single RELATIVE string
    ('./audio/<id>.wav'); abspath-joining against data_root also normalizes the './'.
    """
    choices_raw = row.get("choices")
    choices = [] if choices_raw is None else list(choices_raw)
    if len(choices) == 0:
        return None  # non-MCQ guard (vacuous for MMAR: all 1,000 items are MCQ)
    audio_rel = row.get("audio_path")
    audio_paths = [] if not audio_rel else [os.path.abspath(os.path.join(data_root, str(audio_rel)))]
    answer = row.get("answer")
    if duration_s is None:
        duration_s = parse_timestamp_duration(row.get("timestamp"))
    # Prefer the verbatim index: normalize() strips non-latin chars, so CJK choices can
    # collapse to the same normalized string and match_answer_index would return the
    # first of them (real case: KBn0lNwcPHA_00-00-00_00-00-30, gold = choices[3]).
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
        category=str(row.get("modality")),
        length_type=bucket_length_type(duration_s),
        answer_index=answer_index,
    )


def load_mmar_mcq(
    data_root: str = DEFAULT_DATA_ROOT,
    subset: str = "full",
    limit: int | None = None,
    require_audio_exists: bool = True,
    audio_root: str | None = None,
) -> list[MCQRecord]:
    """Read MMAR-meta.json and return the MCQ records (1,000 for `full`).

    Call-signature-compatible with mmau_pro's `load_mmau_mcq` so the probe runners
    can be instantiated with either loader (`loader_fn(data_root, subset=..., audio_root=...)`).
    `audio_root` resolves the relative audio paths when the audio lives under a
    different root than the metadata; defaults to `data_root`.

    Durations for `length_type` are measured with soundfile when importable
    (~0.13 s for all 1,000 headers); otherwise the `timestamp` fallback from
    `record_from_mmar_row` stands.
    """
    if subset not in SUBSET_FILES:
        raise ValueError(f"subset must be one of {list(SUBSET_FILES)}, got {subset!r}")
    path = os.path.join(data_root, SUBSET_FILES[subset])
    with open(path) as f:
        rows = json.load(f)

    try:
        import soundfile as sf
    except ImportError:
        sf = None  # timestamp-derived buckets stand (uv test env has no soundfile)

    records: list[MCQRecord] = []
    for row in rows:
        rec = record_from_mmar_row(row, audio_root or data_root)
        if rec is None:
            continue
        if require_audio_exists and not all(os.path.exists(p) for p in rec.audio_paths):
            continue
        if sf is not None and all(os.path.exists(p) for p in rec.audio_paths):
            with contextlib.suppress(RuntimeError):  # unreadable header -> keep timestamp bucket
                rec.length_type = bucket_length_type(sum(sf.info(p).duration for p in rec.audio_paths))
        records.append(rec)
        if limit is not None and len(records) >= limit:
            break
    return records


def load_metadata(data_root: str = DEFAULT_DATA_ROOT, subset: str = "full") -> dict[str, dict]:
    """unique_id -> the MMAR-only fields that stay out of MCQRecord.

    'layer' is MMAR's own `category` field (Semantic/Perception/Cultural/Signal
    Layer) — renamed here because MCQRecord.category carries `modality`.
    """
    path = os.path.join(data_root, SUBSET_FILES[subset])
    with open(path) as f:
        rows = json.load(f)
    return {
        str(r.get("id")): {
            "modality": r.get("modality"),
            "layer": r.get("category"),
            "sub_category": r.get("sub-category"),
            "language": r.get("language"),
            "source": r.get("source"),
            "url": r.get("url"),
            "timestamp": r.get("timestamp"),
        }
        for r in rows
    }
