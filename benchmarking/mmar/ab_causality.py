"""MMAR A/B causality check — mmau_pro's ab_causality bound to the MMAR loader.

    python -m benchmarking.mmar.ab_causality \
        --endpoint http://localhost:8100/v1 --model-name qwen-omni --limit 15
"""

from benchmarking.mmar.loader import DEFAULT_DATA_ROOT, load_mmar_mcq
from benchmarking.mmau_pro.ab_causality import make_cli

main = make_cli(loader_fn=load_mmar_mcq, default_data_root=DEFAULT_DATA_ROOT,
                subset="full")

if __name__ == "__main__":
    main()
