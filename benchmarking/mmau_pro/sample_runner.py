"""Sample N completions per MMAU-Pro item — the MMAU-Pro binding of the shared CLI.

Same code path as the MMAR/MMAU sweeps so the benchmarks stay directly comparable.
Primary use: the `d1k` subset (macabdul9/MMAU-Pro-D1K, 1,000 single-audio items whose
parquet and clips share one data/ root — see loader.SUBSET_FILES).
"""

from benchmarking.mmau_pro.loader import SUBSET_FILES, load_mmau_mcq
from benchmarking.sample_cli import make_cli

# the mmau_pro loader exports no DEFAULT_DATA_ROOT (unlike mmar/mmau); default to the
# same testmini root its diversity_probe binding uses — callers pass --data-root anyway
main = make_cli(load_mmau_mcq, SUBSET_FILES,
                "/home/exx/inference-time-scaling/mmau_pro_testmini", bench_name="MMAU-Pro")

if __name__ == "__main__":
    main()
