"""MMAR EPF grid runner — the mmau_pro diversity probe bound to the MMAR loader.

    python -m benchmarking.mmar.diversity_probe \
        --endpoints http://localhost:8100/v1,http://localhost:8101/v1 --model-name qwen-omni \
        --prompts 4,5,7,9 --signals mean_logprob,entropy --budgets 1,8,16,32 \
        --select all --limit 1100 --max-inflight 64 \
        --jsonl benchmarking/mmar/results/run01_epf_grid/mmar_run01.jsonl \
        --csv   benchmarking/mmar/results/run01_epf_grid/mmar_run01.csv \
        --log   benchmarking/mmar/results/run01_epf_grid/mmar_run01.log
"""

from benchmarking.mmar.loader import DEFAULT_DATA_ROOT, SUBSET_FILES, load_mmar_mcq
from benchmarking.mmau_pro.diversity_probe import make_cli

main = make_cli(loader_fn=load_mmar_mcq, subset_choices=SUBSET_FILES,
                default_data_root=DEFAULT_DATA_ROOT, default_subset="full",
                subset_help="MMAR subset: full = MMAR-meta.json (1,000 single-audio MCQ)")

if __name__ == "__main__":
    main()
