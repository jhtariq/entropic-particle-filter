"""Offline tests for the MMAR harness (benchmarking/mmar): loader contract,
wrapper CLIs, and the byte-identity pin on the refactored mmau_pro CLIs."""

import os

import pytest

from benchmarking.mmar.loader import (
    DEFAULT_DATA_ROOT,
    SUBSET_FILES,
    bucket_length_type,
    load_metadata,
    load_mmar_mcq,
    parse_timestamp_duration,
    record_from_mmar_row,
)
from benchmarking.mmau_pro.loader import SUBSET_FILES as MMAU_SUBSET_FILES
from benchmarking.mmau_pro.loader import MCQRecord

ROW = {
    "id": "demo_0001",
    "audio_path": "./audio/demo_0001.wav",
    "question": "What is producing the sound?",
    "choices": ["Owl", "Robot", "Rooster", "Parrot"],
    "answer": "Parrot",
    "modality": "sound",
    "category": "Semantic Layer",
    "sub-category": "Speaker Analysis",
    "language": "en",
    "source": "youtube",
    "url": "https://example.com",
    "timestamp": "00:11:10,00:11:30",
}

UNGRADEABLE_IDS = {
    "BV1NtQuYTELJ_00-00-00_00-00-26",
    "BV16j411E7Z9_00-00-13_00-00-34",
    "BV1mifMY4Ebq_00-02-41_00-03-10",
    "BV1P4411677K_0-00_0-20",
}


# ---------------------------------------------------------------- loader unit


def test_record_from_mmar_row_maps_contract():
    rec = record_from_mmar_row(ROW, "/root")
    assert isinstance(rec, MCQRecord)  # the SAME dataclass as mmau_pro
    assert rec.unique_id == "demo_0001"
    assert rec.question == "What is producing the sound?"
    assert rec.choices == ["Owl", "Robot", "Rooster", "Parrot"]
    assert rec.answer == "Parrot"
    # relative './audio/...' -> single ABSOLUTE, normalized path
    assert rec.audio_paths == ["/root/audio/demo_0001.wav"]
    assert all(os.path.isabs(p) for p in rec.audio_paths)
    assert rec.category == "sound"  # modality, NOT MMAR's layer field
    assert rec.length_type == "10to30s"  # 20 s from the timestamp
    assert rec.answer_index == 3


def test_bucket_length_type_boundaries():
    assert bucket_length_type(None) == "unknown"
    assert bucket_length_type(3.0) == "le10s"
    assert bucket_length_type(10.0) == "le10s"
    assert bucket_length_type(10.5) == "10to30s"
    assert bucket_length_type(30.0) == "10to30s"
    assert bucket_length_type(31.0) == "gt30s"


def test_parse_timestamp_duration():
    assert parse_timestamp_duration("00:11:10,00:11:30") == pytest.approx(20.0)
    assert parse_timestamp_duration("00:00:12,00:00:31") == pytest.approx(19.0)
    # malformed forms present in MMAR-meta.json -> None
    assert parse_timestamp_duration("0-00,0-20") is None
    assert parse_timestamp_duration("00:00:33,00:00:00") is None  # non-positive span
    assert parse_timestamp_duration(None) is None
    assert parse_timestamp_duration("garbage") is None


def test_duration_param_overrides_timestamp():
    rec = record_from_mmar_row(ROW, "/root", duration_s=5.0)
    assert rec.length_type == "le10s"


def test_variable_choice_counts():
    two = record_from_mmar_row({**ROW, "choices": ["On a plane", "On a ship"],
                                "answer": "On a ship"}, "/root")
    assert two.answer_index == 1
    six = record_from_mmar_row({**ROW, "choices": [f"option {i}" for i in range(6)],
                                "answer": "option 5"}, "/root")
    assert six.answer_index == 5


def test_empty_choices_returns_none():
    assert record_from_mmar_row({**ROW, "choices": []}, "/root") is None
    assert record_from_mmar_row({**ROW, "choices": None}, "/root") is None


def test_answer_normalization_cases():
    # normalized-exact: articles/case/punctuation stripped
    rec = record_from_mmar_row({**ROW, "answer": "The Parrot!"}, "/root")
    assert rec.answer_index == 3
    # fuzzy >= 0.85 (the BV1NX4y1p7Xq pattern: answer is a near-copy of one choice)
    rec = record_from_mmar_row(
        {**ROW,
         "choices": ["Because verbal slip, 'terrierst' and 'terrorist' sound similar", "Other"],
         "answer": "Verbal slip, 'terrierst' and 'terrorist' sound similar"}, "/root")
    assert rec.answer_index == 0
    # multi-answer gold matching no single choice -> ungradeable (the BV1NtQuYTELJ pattern)
    rec = record_from_mmar_row(
        {**ROW, "choices": ["American", "Colombian"], "answer": "American\nColombian"}, "/root")
    assert rec.answer_index is None
    # verbatim index wins even when normalize() collapses CJK choices to the same
    # string (the KBn0lNwcPHA pattern: fuzzy matching alone would return index 0)
    cjk = ["奥巴马 Bond", "克林顿 Bond", "布什 Bond", "里根 Bond"]
    rec = record_from_mmar_row({**ROW, "choices": cjk, "answer": cjk[3]}, "/root")
    assert rec.answer_index == 3


# ---------------------------------------------------------------- real data


requires_mmar = pytest.mark.skipif(
    not os.path.isdir(DEFAULT_DATA_ROOT), reason="MMAR data not on this machine"
)


@requires_mmar
def test_real_mmar_1000_records_996_gradeable():
    recs = load_mmar_mcq()
    assert len(recs) == 1000
    ungradeable = {r.unique_id for r in recs if r.answer_index is None}
    assert ungradeable == UNGRADEABLE_IDS
    # every verbatim gold maps to ITS OWN index (guards the CJK normalize-collapse case)
    assert all(r.choices[r.answer_index] == r.answer for r in recs if r.answer in r.choices)
    assert all(os.path.isabs(p) and os.path.exists(p) for r in recs for p in r.audio_paths)
    assert all(len(r.audio_paths) == 1 for r in recs)  # MMAR is single-audio
    cats = {}
    for r in recs:
        cats[r.category] = cats.get(r.category, 0) + 1
    assert cats["speech"] == 294
    assert cats["sound"] == 165
    nch = {}
    for r in recs:
        nch[len(r.choices)] = nch.get(len(r.choices), 0) + 1
    assert nch[4] == 815
    assert nch[2] == 171
    # membership only: soundfile-less envs fall back to timestamps ('unknown' for malformed)
    assert {r.length_type for r in recs} <= {"le10s", "10to30s", "gt30s", "unknown"}


@requires_mmar
def test_real_mmar_metadata_lookup():
    md = load_metadata()
    assert len(md) == 1000
    layers = {v["layer"] for v in md.values()}
    assert layers == {"Semantic Layer", "Perception Layer", "Cultural Layer", "Signal Layer"}


@requires_mmar
def test_real_mmar_limit_and_missing_audio_flags():
    assert len(load_mmar_mcq(limit=7)) == 7
    assert len(load_mmar_mcq(require_audio_exists=False)) == 1000


@pytest.mark.skipif(
    not os.path.exists(os.path.join(DEFAULT_DATA_ROOT, "MMAR-meta-le30s.json")),
    reason="le30s subset not built on this machine",
)
def test_real_mmar_le30s_subset():
    recs = load_mmar_mcq(subset="le30s")
    assert len(recs) == 983  # 17 clips >30 s dropped (Qwen2-Audio window)
    ungradeable = {r.unique_id for r in recs if r.answer_index is None}
    assert ungradeable == UNGRADEABLE_IDS  # all 4 are <=30 s -> 979 gradeable
    full_ids = {r.unique_id for r in load_mmar_mcq()}
    assert {r.unique_id for r in recs} <= full_ids


# ---------------------------------------------------------------- wrapper CLIs


def _params(cmd):
    return {p.name: p for p in cmd.params}


def test_mmar_wrappers_mirror_mmau_cli():
    import benchmarking.mmar.ab_causality as mmar_ab
    import benchmarking.mmar.cot_compare as mmar_cc
    import benchmarking.mmar.diversity_probe as mmar_dp
    import benchmarking.mmar.phase0_gate as mmar_pg
    import benchmarking.mmau_pro.ab_causality as mmau_ab
    import benchmarking.mmau_pro.cot_compare as mmau_cc
    import benchmarking.mmau_pro.diversity_probe as mmau_dp
    import benchmarking.mmau_pro.phase0_gate as mmau_pg

    for mmar_mod, mmau_mod in [(mmar_dp, mmau_dp), (mmar_cc, mmau_cc),
                               (mmar_pg, mmau_pg), (mmar_ab, mmau_ab)]:
        assert [p.name for p in mmar_mod.main.params] == [p.name for p in mmau_mod.main.params]
        assert _params(mmar_mod.main)["data_root"].default == DEFAULT_DATA_ROOT

    for mod in (mmar_dp, mmar_cc):
        subset = _params(mod.main)["subset"]
        assert list(subset.type.choices) == list(SUBSET_FILES) == ["full", "le30s"]
        assert subset.default == "full"


def test_mmau_cli_pinned_byte_identical():
    """Enforce the refactor invariant: the mmau_pro CLIs kept their exact surface."""
    import benchmarking.mmau_pro.ab_causality as ab
    import benchmarking.mmau_pro.cot_compare as cc
    import benchmarking.mmau_pro.diversity_probe as dp
    import benchmarking.mmau_pro.phase0_gate as pg

    mmau_root = "/home/exx/inference-time-scaling/mmau_pro_testmini"

    p = _params(dp.main)
    assert p["model_name"].required
    assert p["endpoints"].default == "http://localhost:8100/v1,http://localhost:8101/v1"
    assert p["api_key"].default == "NO_API_KEY"
    assert p["data_root"].default == mmau_root
    assert list(p["subset"].type.choices) == list(MMAU_SUBSET_FILES)
    assert p["subset"].default == "full"
    assert p["prompts"].default == "4,5,7,9"
    assert p["signals"].default == "mean_logprob,entropy"
    assert p["budgets"].default == "1,8,16,32"
    assert p["temp"].default == 0.8
    assert p["ess_threshold"].default == 0.6
    assert p["early_phase"].default == 0.7
    assert p["max_steps"].default == 6
    assert p["step_token"].default == "\n\n"
    assert p["stop_regex"].default is None
    assert p["stop_on_repeat"].default is False
    assert p["max_tokens_per_step"].default == 300
    assert p["limit"].default == 100
    assert list(p["select_mode"].type.choices) == ["stratified", "catlen", "all"]
    assert p["select_mode"].default == "stratified"
    assert p["max_inflight"].default == 64
    assert p["store_text"].default is False
    assert p["jsonl_path"].default is None and p["csv_path"].default is None
    assert p["log_path"].default is None

    p = _params(cc.main)
    assert p["endpoint"].required and p["model_name"].required
    assert p["data_root"].default == mmau_root
    assert list(p["subset"].type.choices) == list(MMAU_SUBSET_FILES)
    assert p["subset"].default == "le30s"
    assert list(p["select_mode"].type.choices) == ["smallest", "stratified", "all"]
    assert p["select_mode"].default == "smallest"
    assert p["methods"].default is None and p["ids"].default is None
    assert p["limit"].default is None
    assert p["audio_mode"].default == "base64"
    assert p["max_tokens"].default == 700
    assert p["concurrency"].default == 6

    p = _params(pg.main)
    assert p["endpoint"].required and p["model_name"].required
    assert p["data_root"].default == mmau_root
    assert p["audio_mode"].default == "local-path"
    assert list(p["audio_mode"].type.choices) == ["local-path", "base64"]
    assert p["single_audio"].default is False

    p = _params(ab.main)
    assert p["endpoint"].required and p["model_name"].required
    assert p["data_root"].default == mmau_root
    assert p["limit"].default == 15
    assert p["max_tokens"].default == 512
    assert p["method"].default == 0
    assert p["single_audio"].default is False
