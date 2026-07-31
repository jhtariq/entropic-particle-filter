"""MMAU (original) audio-MCQ benchmark harness — the 1,000-item test-mini split of
MMAU-v05.15.25 (arXiv 2410.19168; NOT MMAU-Pro).

Loader-only package: `loader.py` emits the same `MCQRecord`s as mmau_pro, and the
runners (`diversity_probe`, `cot_compare`, `phase0_gate`, `ab_causality`) are the
mmau_pro scripts instantiated via their `make_cli(loader_fn=..., ...)` factories.
See README.md in this directory.
"""
