"""Load the STAR-Bench benchmark (internlm/STAR-Bench v1.0) from its HF dataset
snapshot: two `meta_info/*.json` files (JSON arrays) plus `starbench_audios/*.wav`
under a single root. Records are emitted as the SAME `MCQRecord` dataclass as
MMAU-Pro/MMAR (imported, not copied), so all prompt/scoring/serving machinery
consumes them unchanged.

Only the two subsets that ship a gold `answer` are covered here:
  - "perception" <- meta_info/foundation_perception.json (951 items)
  - "spatial"     <- meta_info/holistic_reasoning_spatial.json (502 items)
  - "full"        <- both, concatenated (1,453 items)

STAR-Bench's third subset, Temporal Reasoning (meta_info/holistic_reasoning_temporal.json,
900 items), has NO `answer` field in the v1.0 release (labels withheld, presumably for
a held-out leaderboard) and is intentionally NOT loadable here.

Field mapping (mirrors benchmarking/mmar/loader.py):
  - `unique_id`    <- "<task_prefix>/<row id>" (task_prefix disambiguates the two
    source files; STAR-Bench ids are not guaranteed unique across them)
  - `category`     <- "<task_prefix>:<row category>" (STAR-Bench's own `category`
    field, e.g. "Absolute Perception Range" / "Dynamic Trajectory Tracking") — the
    stratification/reporting axis, prefixed so a merged "full" report keeps the two
    task types visually distinct
  - `audio_paths`  <- row `audio_paths` (list of paths RELATIVE to data_root),
    abspath-joined against data_root
  - `answer_index` <- verbatim-index match first (STAR-Bench answers are almost
    always a literal copy of one option), else mmau_pro's fuzzy `match_answer_index`
"""

import contextlib
import json
import os

from benchmarking.mmau_pro.loader import MCQRecord
from benchmarking.mmau_pro.scoring import match_answer_index

SUBSET_FILES = {
    "perception": ["foundation_perception"],
    "spatial": ["holistic_reasoning_spatial"],
    "full": ["foundation_perception", "holistic_reasoning_spatial"],
}
_FILE_NAME = {
    "foundation_perception": "meta_info/foundation_perception.json",
    "holistic_reasoning_spatial": "meta_info/holistic_reasoning_spatial.json",
}
_TASK_PREFIX = {
    "foundation_perception": "perception",
    "holistic_reasoning_spatial": "spatial",
}

DEFAULT_DATA_ROOT = "/work/hdd/bcey/awaheed/its-for-audio-reasoning/star-bench"


def record_from_star_bench_row(row: dict, data_root: str, task_prefix: str) -> MCQRecord | None:
    """Convert one meta_info/*.json entry to an MCQRecord; None if not MCQ.

    Pure (no I/O beyond path joining) so it is unit-testable with synthetic dicts.
    """
    choices_raw = row.get("options")
    choices = [] if choices_raw is None else list(choices_raw)
    if len(choices) == 0:
        return None  # non-MCQ guard (vacuous for these two files: all items are MCQ)
    audio_rel = row.get("audio_paths") or []
    audio_paths = [os.path.abspath(os.path.join(data_root, str(p))) for p in audio_rel]
    answer = row.get("answer")
    if answer in choices:
        answer_index = choices.index(answer)
    else:
        answer_index = match_answer_index(answer, choices)
    return MCQRecord(
        unique_id=f"{task_prefix}/{row.get('id')}",
        question=str(row.get("question")),
        choices=choices,
        answer=str(answer),
        audio_paths=audio_paths,
        category=f"{task_prefix}:{row.get('category')}",
        length_type="unknown",  # filled in below when soundfile is importable
        answer_index=answer_index,
    )


def load_star_bench_mcq(
    data_root: str = DEFAULT_DATA_ROOT,
    subset: str = "full",
    limit: int | None = None,
    require_audio_exists: bool = True,
    audio_root: str | None = None,
) -> list[MCQRecord]:
    """Read the STAR-Bench meta_info JSON(s) and return MCQ records.

    Call-signature-compatible with mmau_pro's `load_mmau_mcq` / mmar's
    `load_mmar_mcq` so the greedy runner can dispatch on `--bench star_bench`.
    `audio_root` resolves the relative audio paths when the audio lives under a
    different root than the metadata; defaults to `data_root`.

    For `subset="full"` (both source files), `limit` is applied PER FILE, not to
    the merged total — e.g. `limit=3` yields up to 3 perception + 3 spatial items,
    not 3 total. This is deliberate: it is what a smoke test wants (exercise both
    question shapes every run), and the real (non-smoke) runs never pass `limit`.
    """
    if subset not in SUBSET_FILES:
        raise ValueError(f"subset must be one of {list(SUBSET_FILES)}, got {subset!r}")

    try:
        import soundfile as sf
    except ImportError:
        sf = None

    records: list[MCQRecord] = []
    for file_key in SUBSET_FILES[subset]:
        task_prefix = _TASK_PREFIX[file_key]
        path = os.path.join(data_root, _FILE_NAME[file_key])
        with open(path) as f:
            rows = json.load(f)
        n_this_file = 0
        for row in rows:
            rec = record_from_star_bench_row(row, audio_root or data_root, task_prefix)
            if rec is None:
                continue
            if require_audio_exists and not all(os.path.exists(p) for p in rec.audio_paths):
                continue
            if sf is not None and rec.audio_paths and all(os.path.exists(p) for p in rec.audio_paths):
                with contextlib.suppress(RuntimeError):  # unreadable header -> keep 'unknown'
                    dur = sum(sf.info(p).duration for p in rec.audio_paths)
                    rec.length_type = "le10s" if dur <= 10 else "10to30s" if dur <= 30 else "gt30s"
            records.append(rec)
            n_this_file += 1
            if limit is not None and n_this_file >= limit:
                break
    return records
