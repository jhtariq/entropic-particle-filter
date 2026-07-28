"""MMAR audio-MCQ benchmark harness (1,000 single-audio reasoning items).

Loader-only package: `loader.py` emits the same `MCQRecord`s as mmau_pro, and the
runners (`diversity_probe`, `cot_compare`, `phase0_gate`, `ab_causality`) are the
mmau_pro scripts instantiated via their `make_cli(loader_fn=..., ...)` factories.
See README.md in this directory.
"""
