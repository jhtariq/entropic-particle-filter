"""MMSU EPF grid runner — the mmau_pro diversity probe bound to the MMSU loader.

    python -m benchmarking.mmsu.diversity_probe \
        --endpoints http://localhost:8100/v1,http://localhost:8101/v1 --model-name qwen-omni \
        --prompts 4,5,7,9 --signals mean_logprob,entropy --budgets 1,8,16,32 \
        --select all --limit 5100 --max-inflight 64 \
        --jsonl benchmarking/mmsu/results/run01_omni7b/mmsu_run01.jsonl \
        --csv   benchmarking/mmsu/results/run01_omni7b/mmsu_run01.csv \
        --log   benchmarking/mmsu/results/run01_omni7b/mmsu_run01.log
"""

from benchmarking.mmau_pro.diversity_probe import make_cli
from benchmarking.mmsu.loader import DEFAULT_DATA_ROOT, SUBSET_FILES, load_mmsu_mcq

main = make_cli(loader_fn=load_mmsu_mcq, subset_choices=SUBSET_FILES,
                default_data_root=DEFAULT_DATA_ROOT, default_subset="full",
                subset_help="MMSU subset: full = MMSU-meta.json (5,000 single-audio MCQ)")

if __name__ == "__main__":
    main()
