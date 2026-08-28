"""STAR-Bench audio-MCQ benchmark harness (internlm/STAR-Bench).

Loader-only package: `loader.py` emits the same `MCQRecord`s as mmau_pro/mmar, so
the greedy no-CoT runner consumes them unchanged. Only the two subsets that ship a
gold `answer` are covered (Foundational Acoustic Perception, Spatial Reasoning);
the Temporal Reasoning subset has no answer field in the v1.0 release and is
excluded. See loader.py docstring for the JSON -> MCQRecord field mapping.
"""
