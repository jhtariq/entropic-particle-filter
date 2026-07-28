"""Offline tests for the MMSU harness (benchmarking/mmsu): loader contract and
wrapper CLIs. MMSU is all-verbatim (gold answer is the full choice text, 5,000/5,000
gradeable, 0 fuzzy), single-audio, always 4 letter-keyed choices — so there are no
timestamp / CJK-collapse / multi-answer / le30s cases to pin (see test_mmar.py for
those). The mmau_pro CLI byte-identity pin lives in test_mmar.py and is not duplicated."""

import os

import pytest

from benchmarking.mmau_pro.loader import MCQRecord
from benchmarking.mmsu.loader import (
    DEFAULT_DATA_ROOT,
    SUBSET_FILES,
    bucket_length_type,
    load_metadata,
    load_mmsu_mcq,
    record_from_mmsu_row,
)

ROW = {
    "id": "accent_identification_demo0001",
    "task_name": "accent_identification",
    "audio_path": "./audio/accent_identification_demo0001.wav",
    "question": "What accent does the speaker most likely have?",
    "choices": {"A": "Owl", "B": "Robot", "C": "Rooster", "D": "Parrot"},
    "answer": "Parrot",
    "category": "Perception",
    "sub-category": "Linguistics",
    "sub-sub-category": "Phonology",
    "linguistics_sub_discipline": "Prosody",
}


# ---------------------------------------------------------------- loader unit


def test_record_from_mmsu_row_maps_contract():
    rec = record_from_mmsu_row(ROW, "/root")
    assert isinstance(rec, MCQRecord)  # the SAME dataclass as mmau_pro
    assert rec.unique_id == "accent_identification_demo0001"
    assert rec.question == "What accent does the speaker most likely have?"
    # letter-keyed DICT -> ordered list (A, B, C, D)
    assert rec.choices == ["Owl", "Robot", "Rooster", "Parrot"]
    assert rec.answer == "Parrot"
    # relative './audio/...' -> single ABSOLUTE, normalized path
    assert rec.audio_paths == ["/root/audio/accent_identification_demo0001.wav"]
    assert all(os.path.isabs(p) for p in rec.audio_paths)
    # category is the fine-grained task_name, NOT MMSU's Perception/Reasoning field
    assert rec.category == "accent_identification"
    assert rec.length_type == "unknown"  # no duration passed, no timestamp field
    assert rec.answer_index == 3
    assert all(c is not None for c in rec.choices)  # no None padding


def test_none_padded_choices_filtered():
    # 453/5000 MMSU items are 2-choice, PADDED to 4 letter-keys with Python None VALUES in
    # ANY position; the loader drops the padding so the MCQ shows only real options.
    padded = {**ROW, "choices": {"A": None, "B": "male", "C": None, "D": "female"},
              "answer": "female"}
    rec = record_from_mmsu_row(padded, "/root")
    assert rec.choices == ["male", "female"]  # None dropped, letter order preserved
    assert rec.answer_index == 1  # gold 'female' is option B (index 1), not padded-D
    assert all(c is not None for c in rec.choices)
    # a legitimate string "None" option is a real choice and is KEPT
    strnone = {**ROW, "choices": {"A": "yes", "B": "no", "C": "maybe", "D": "None"},
               "answer": "None"}
    rec2 = record_from_mmsu_row(strnone, "/root")
    assert rec2.choices == ["yes", "no", "maybe", "None"]
    assert rec2.answer_index == 3


def test_choices_dict_to_list_ordering():
    # a scrambled-key dict must still flatten to A, B, C, D order (by LETTERS, not
    # dict insertion order), and answer_index must track the gold TEXT
    scrambled = {"C": "Rooster", "A": "Owl", "D": "Parrot", "B": "Robot"}
    rec = record_from_mmsu_row({**ROW, "choices": scrambled, "answer": "Rooster"}, "/root")
    assert rec.choices == ["Owl", "Robot", "Rooster", "Parrot"]
    assert rec.answer_index == 2


def test_bucket_length_type_boundaries():
    assert bucket_length_type(None) == "unknown"
    assert bucket_length_type(3.0) == "le10s"
    assert bucket_length_type(10.0) == "le10s"
    assert bucket_length_type(10.5) == "10to30s"
    assert bucket_length_type(30.0) == "10to30s"
    assert bucket_length_type(35.9) == "gt30s"  # MMSU's real max is ~35.9 s


def test_duration_param_sets_bucket():
    assert record_from_mmsu_row(ROW, "/root", duration_s=5.0).length_type == "le10s"
    assert record_from_mmsu_row(ROW, "/root", duration_s=35.9).length_type == "gt30s"


def test_empty_choices_returns_none():
    assert record_from_mmsu_row({**ROW, "choices": {}}, "/root") is None
    assert record_from_mmsu_row({**ROW, "choices": None}, "/root") is None


def test_verbatim_and_normalized_index():
    # exact-verbatim gold -> its own index
    assert record_from_mmsu_row(ROW, "/root").answer_index == 3
    # normalized-exact (case/punctuation) exercises the match_answer_index fallback
    rec = record_from_mmsu_row({**ROW, "answer": "the parrot"}, "/root")
    assert rec.answer_index == 3


# ---------------------------------------------------------------- real data


requires_mmsu = pytest.mark.skipif(
    not os.path.isdir(DEFAULT_DATA_ROOT), reason="MMSU data not on this machine"
)


@requires_mmsu
def test_real_mmsu_5000_records_5000_gradeable():
    recs = load_mmsu_mcq()
    assert len(recs) == 5000
    # all-verbatim: 5,000/5,000 gradeable, and every gold maps to its own choice index
    assert {r.unique_id for r in recs if r.answer_index is None} == set()
    assert all(r.choices[r.answer_index] == r.answer for r in recs)
    assert all(os.path.isabs(p) and os.path.exists(p) for r in recs for p in r.audio_paths)
    assert all(len(r.audio_paths) == 1 for r in recs)  # MMSU is single-audio
    # 453/5000 items are 2-choice (None-padded to 4 letter-keys in the source); the loader
    # drops the padding, so choice counts are {4547 four-choice, 453 two-choice} and NO
    # choice value is None/blank (would otherwise show the model empty options).
    from collections import Counter
    assert dict(Counter(len(r.choices) for r in recs)) == {4: 4547, 2: 453}
    assert not any(c is None or str(c).strip() == "" for r in recs for c in r.choices)
    # category carries the 47 fine-grained task_names
    cats = {}
    for r in recs:
        cats[r.category] = cats.get(r.category, 0) + 1
    assert len(cats) == 47
    assert sum(cats.values()) == 5000
    assert cats["gender_prediction"] == 120
    assert cats["total_speaker_counting"] == 120
    # soundfile-measured in the epf env (incl. gt30s from the ~35.9 s tail);
    # 'unknown' in the soundfile-less uv env — superset-safe either way
    assert {r.length_type for r in recs} <= {"le10s", "10to30s", "gt30s", "unknown"}


@requires_mmsu
def test_real_mmsu_metadata_lookup():
    md = load_metadata()
    assert len(md) == 5000
    cat = {}
    sub = {}
    for v in md.values():
        cat[v["category"]] = cat.get(v["category"], 0) + 1
        sub[v["sub_category"]] = sub.get(v["sub_category"], 0) + 1
    assert cat == {"Perception": 2580, "Reasoning": 2420}
    assert sub == {"Linguistics": 3655, "Paralinguistics": 1345}
    # the literal string "None" is a real linguistics_sub_discipline value
    assert "None" in {v["linguistics_sub_discipline"] for v in md.values()}


@requires_mmsu
def test_real_mmsu_limit_and_missing_audio_flags():
    assert len(load_mmsu_mcq(limit=7)) == 7
    assert len(load_mmsu_mcq(require_audio_exists=False)) == 5000


# ---------------------------------------------------------------- wrapper CLIs


def _params(cmd):
    return {p.name: p for p in cmd.params}


def test_build_epf_random_arm():
    """The random-survival ablation arm (--signals random): uniform weights +
    multinomial resampling + SAMPLE final pick. The weighted arms are unchanged."""
    from benchmarking.mmau_pro.diversity_probe import build_epf
    from its_hub.core.algorithms.particle_filtering import (
        ResamplingMethod,
        SelectionMethod,
    )

    kw = {"temp": 0.8, "max_steps": 6, "ess_threshold": 0.6, "early_phase": 0.7}
    rnd = build_epf("random", **kw)
    assert rnd.uniform_weights is True
    assert rnd.resampling_method == ResamplingMethod.MULTINOMIAL
    assert rnd.final_response_selection == SelectionMethod.SAMPLE
    # weighted arms unchanged: systematic resampling, argmax pick, no uniform weights
    for sig in ("mean_logprob", "entropy"):
        wl = build_epf(sig, **kw)
        assert wl.uniform_weights is False
        assert wl.resampling_method == ResamplingMethod.SYSTEMATIC
        assert wl.final_response_selection == SelectionMethod.ARGMAX
        assert wl.self_certainty_signal == sig


def test_mmsu_wrappers_mirror_mmau_cli():
    import benchmarking.mmau_pro.ab_causality as mmau_ab
    import benchmarking.mmau_pro.cot_compare as mmau_cc
    import benchmarking.mmau_pro.diversity_probe as mmau_dp
    import benchmarking.mmau_pro.phase0_gate as mmau_pg
    import benchmarking.mmsu.ab_causality as mmsu_ab
    import benchmarking.mmsu.cot_compare as mmsu_cc
    import benchmarking.mmsu.diversity_probe as mmsu_dp
    import benchmarking.mmsu.phase0_gate as mmsu_pg

    for mmsu_mod, mmau_mod in [(mmsu_dp, mmau_dp), (mmsu_cc, mmau_cc),
                               (mmsu_pg, mmau_pg), (mmsu_ab, mmau_ab)]:
        assert [p.name for p in mmsu_mod.main.params] == [p.name for p in mmau_mod.main.params]
        assert _params(mmsu_mod.main)["data_root"].default == DEFAULT_DATA_ROOT

    for mod in (mmsu_dp, mmsu_cc):
        subset = _params(mod.main)["subset"]
        assert list(subset.type.choices) == list(SUBSET_FILES) == ["full"]
        assert subset.default == "full"
