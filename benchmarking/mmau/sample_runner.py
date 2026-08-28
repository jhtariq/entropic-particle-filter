"""Sample N completions per MMAU test-mini item — the MMAU binding of the shared CLI.

Same four arms as the MMAR sweep, and the same code path, so the two benchmarks are
directly comparable:

    greedy, no CoT             --n 1   --temperature 0.0 --method 0 --max-tokens 64
    greedy, CoT (P4)           --n 1   --temperature 0.0 --method 4 --max-tokens 700
    self-consistency, CoT      --n 128 --temperature 0.8 --method 4 --max-tokens 700
    self-consistency, no CoT   --n 128 --temperature 0.8 --method 0 --max-tokens 64

MMAU test-mini is 1,000 items, **all 1,000 gradeable** (MMAR has 4 ungradeable), balanced
sound/speech/music, and must be materialized from the official parquet first:

    python -m benchmarking.mmau.materialize_test_mini --parquet <test_mini.parquet> \
        --out-root $EPF_DATA_ROOT/mmau

Note 66 of the 1,000 clips exceed 30 s, which Qwen2-Audio's encoder truncates; the
report breaks accuracy out by `length_type` so that effect stays visible rather than
silently folded into the headline number.
"""

from benchmarking.mmau.loader import DEFAULT_DATA_ROOT, SUBSET_FILES, load_mmau_mcq
from benchmarking.sample_cli import make_cli

main = make_cli(load_mmau_mcq, SUBSET_FILES, DEFAULT_DATA_ROOT, bench_name="MMAU")

if __name__ == "__main__":
    main()
