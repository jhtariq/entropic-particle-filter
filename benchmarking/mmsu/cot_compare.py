"""MMSU greedy prompt screen — mmau_pro's cot_compare bound to the MMSU loader.

    python -m benchmarking.mmsu.cot_compare \
        --endpoint http://localhost:8100/v1 --model-name qwen-omni \
        --select stratified --limit 40 --methods 4,5,7,9 --audio-mode local-path
"""

from benchmarking.mmau_pro.cot_compare import make_cli
from benchmarking.mmsu.loader import DEFAULT_DATA_ROOT, SUBSET_FILES, load_mmsu_mcq

main = make_cli(loader_fn=load_mmsu_mcq, subset_choices=SUBSET_FILES,
                default_data_root=DEFAULT_DATA_ROOT, default_subset="full")

if __name__ == "__main__":
    main()
