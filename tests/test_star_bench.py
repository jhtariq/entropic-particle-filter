"""Offline tests for the STAR-Bench harness (benchmarking/star_bench): loader
contract plus real-data sanity checks (skipped when the dataset isn't present)."""

import os

import pytest

from benchmarking.mmau_pro.loader import MCQRecord
from benchmarking.star_bench.loader import (
    DEFAULT_DATA_ROOT,
    SUBSET_FILES,
    load_star_bench_mcq,
    record_from_star_bench_row,
)

PERCEPTION_ROW = {
    "id": "range_azimuth_az120_001",
    "question": "Which azimuth range is the sound coming from?",
    "options": ["Front-Right", "Back-Right", "Back-Left", "Front-Left", "Unable to determine"],
    "answer": "Back-Right",
    "audio_paths": ["starbench_audios/foundation_perception/spatial/x/telephones.wav"],
    "task": "Foundational Acoustic Perception",
    "category": "Absolute Perception Range",
    "sub-category": "range_azimuth",
    "source": "Pyroomacoustics synthesis",
}

SPATIAL_ROW = {
    "id": "spatial_trajectory_000",
    "question": "What is the motion trajectory of a horse-drawn carriage?",
    "options": ["From left to right", "From right to left", "Remains unchanged"],
    "answer": "From right to left",
    "audio_paths": ["starbench_audios/holistic_reasoning/spatial/trajectory_000.wav"],
    "task": "Spatial Reasoning",
    "category": "Dynamic Trajectory Tracking",
    "source": "In-the-wild",
}


# ---------------------------------------------------------------- loader unit


def test_record_from_star_bench_row_maps_contract():
    rec = record_from_star_bench_row(PERCEPTION_ROW, "/root", "perception")
    assert isinstance(rec, MCQRecord)  # the SAME dataclass as mmau_pro/mmar
    assert rec.unique_id == "perception/range_azimuth_az120_001"
    assert rec.question == "Which azimuth range is the sound coming from?"
    assert rec.choices == ["Front-Right", "Back-Right", "Back-Left", "Front-Left", "Unable to determine"]
    assert rec.answer == "Back-Right"
    assert rec.audio_paths == [
        "/root/starbench_audios/foundation_perception/spatial/x/telephones.wav"
    ]
    assert all(os.path.isabs(p) for p in rec.audio_paths)
    assert rec.category == "perception:Absolute Perception Range"
    assert rec.answer_index == 1  # "Back-Right" is choices[1]


def test_record_from_star_bench_row_task_prefix_disambiguates_ids():
    p = record_from_star_bench_row(PERCEPTION_ROW, "/root", "perception")
    s = record_from_star_bench_row(SPATIAL_ROW, "/root", "spatial")
    assert s.unique_id == "spatial/spatial_trajectory_000"
    assert s.category == "spatial:Dynamic Trajectory Tracking"
    assert p.unique_id != s.unique_id


def test_multi_audio_item():
    row = {**PERCEPTION_ROW, "audio_paths": ["a/clip1.wav", "a/clip2.wav"]}
    rec = record_from_star_bench_row(row, "/root", "perception")
    assert rec.audio_paths == ["/root/a/clip1.wav", "/root/a/clip2.wav"]


def test_empty_options_returns_none():
    assert record_from_star_bench_row({**PERCEPTION_ROW, "options": []}, "/root", "perception") is None
    assert record_from_star_bench_row({**PERCEPTION_ROW, "options": None}, "/root", "perception") is None


def test_answer_normalization_fuzzy_fallback():
    # answer text is a near-copy of one option, not verbatim -> fuzzy match
    row = {**PERCEPTION_ROW,
           "options": ["Back-Right (90 degrees - 180 degrees)", "Other"],
           "answer": "Back-Right (90 degrees-180 degrees)"}
    rec = record_from_star_bench_row(row, "/root", "perception")
    assert rec.answer_index == 0


def test_unresolvable_answer_is_ungradeable():
    row = {**PERCEPTION_ROW, "options": ["A", "B"], "answer": "completely unrelated text"}
    rec = record_from_star_bench_row(row, "/root", "perception")
    assert rec.answer_index is None


# ---------------------------------------------------------------- real data


requires_star_bench = pytest.mark.skipif(
    not os.path.isdir(DEFAULT_DATA_ROOT), reason="STAR-Bench data not on this machine"
)


@requires_star_bench
def test_real_star_bench_full_1453_records():
    recs = load_star_bench_mcq(subset="full")
    assert len(recs) == 1453
    assert all(os.path.isabs(p) and os.path.exists(p) for r in recs for p in r.audio_paths)
    n_perception = sum(1 for r in recs if r.unique_id.startswith("perception/"))
    n_spatial = sum(1 for r in recs if r.unique_id.startswith("spatial/"))
    assert n_perception == 951
    assert n_spatial == 502
    # ids are unique across the merged subset
    assert len({r.unique_id for r in recs}) == 1453


@requires_star_bench
def test_real_star_bench_perception_and_spatial_subsets():
    assert len(load_star_bench_mcq(subset="perception")) == 951
    assert len(load_star_bench_mcq(subset="spatial")) == 502


@requires_star_bench
def test_real_star_bench_all_gradeable():
    # STAR-Bench answers are near-always a literal option copy; expect ~0 ungradeable
    recs = load_star_bench_mcq(subset="full")
    ungradeable = [r.unique_id for r in recs if r.answer_index is None]
    assert len(ungradeable) == 0, ungradeable


@requires_star_bench
def test_real_star_bench_full_limit_is_per_file():
    recs = load_star_bench_mcq(subset="full", limit=3)
    assert len(recs) == 6  # 3 perception + 3 spatial, not 3 total
    assert sum(1 for r in recs if r.unique_id.startswith("perception/")) == 3
    assert sum(1 for r in recs if r.unique_id.startswith("spatial/")) == 3


def test_subset_files_choices():
    assert set(SUBSET_FILES) == {"perception", "spatial", "full"}
