"""MMAU A/B causality check — mmau_pro's ab_causality bound to the MMAU loader.

    python -m benchmarking.mmau.ab_causality \
        --endpoint http://localhost:8100/v1 --model-name qwen-omni --limit 15
"""

from benchmarking.mmau.loader import DEFAULT_DATA_ROOT, load_mmau_mcq
from benchmarking.mmau_pro.ab_causality import make_cli

main = make_cli(loader_fn=load_mmau_mcq, default_data_root=DEFAULT_DATA_ROOT,
                subset="full")

if __name__ == "__main__":
    main()
