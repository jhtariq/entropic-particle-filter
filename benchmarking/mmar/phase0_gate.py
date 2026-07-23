"""MMAR phase-0 gates — mmau_pro's phase0_gate bound to the MMAR loader.

    python -m benchmarking.mmar.phase0_gate \
        --endpoint http://localhost:8100/v1 --model-name qwen-omni
"""

from benchmarking.mmar.loader import DEFAULT_DATA_ROOT, load_mmar_mcq
from benchmarking.mmau_pro.phase0_gate import make_cli

main = make_cli(loader_fn=load_mmar_mcq, default_data_root=DEFAULT_DATA_ROOT,
                gate_subset="full")

if __name__ == "__main__":
    main()
