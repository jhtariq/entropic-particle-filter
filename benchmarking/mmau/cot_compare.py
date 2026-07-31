"""MMAU greedy prompt screen — mmau_pro's cot_compare bound to the MMAU loader.

    python -m benchmarking.mmau.cot_compare \
        --endpoint http://localhost:8100/v1 --model-name qwen-omni \
        --select stratified --limit 40 --methods 4 --audio-mode local-path
"""

from benchmarking.mmau.loader import DEFAULT_DATA_ROOT, SUBSET_FILES, load_mmau_mcq
from benchmarking.mmau_pro.cot_compare import make_cli

main = make_cli(loader_fn=load_mmau_mcq, subset_choices=SUBSET_FILES,
                default_data_root=DEFAULT_DATA_ROOT, default_subset="full")

if __name__ == "__main__":
    main()
