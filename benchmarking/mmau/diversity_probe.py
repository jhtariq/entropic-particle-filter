"""MMAU EPF grid runner — the mmau_pro diversity probe bound to the MMAU loader.

    python -m benchmarking.mmau.diversity_probe \
        --endpoints http://localhost:8100/v1 --model-name qwen-omni \
        --prompts 4 --signals mean_logprob,entropy --budgets 1,8,16,32,64,128 \
        --select all --limit 1100 --max-inflight 64 \
        --jsonl benchmarking/mmau/results/run01_omni7b/mmau_run01.jsonl \
        --csv   benchmarking/mmau/results/run01_omni7b/mmau_run01.csv \
        --log   benchmarking/mmau/results/run01_omni7b/mmau_run01.log
"""

from benchmarking.mmau.loader import DEFAULT_DATA_ROOT, SUBSET_FILES, load_mmau_mcq
from benchmarking.mmau_pro.diversity_probe import make_cli

main = make_cli(loader_fn=load_mmau_mcq, subset_choices=SUBSET_FILES,
                default_data_root=DEFAULT_DATA_ROOT, default_subset="full",
                subset_help="MMAU subset: full = MMAU-meta.json (1,000 test-mini "
                            "single-audio MCQ, MMAU-v05.15.25)")

if __name__ == "__main__":
    main()
