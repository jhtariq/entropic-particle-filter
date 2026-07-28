# Policy: Mellow v0 (167M params: HTSAT -> SmolLM2-135M). Not vLLM-supported;
# served by the bespoke OpenAI-compatible shim benchmarking/mmau_pro/serve_mellow.py.
# Expect degraded P4/'Answer: <letter>' format-following (tiny ALM) — disclosed
# in results.md; scoring falls back to choice-text matching.
MODEL_ID="soham97/mellow"
MODEL_REV="83672db0dae28764e283210d5bb732621e903d8a"
SERVED_NAME="mellow"
SERVE_KIND=mellow
MELLOW_CODE_DIR="/home/exx/inference-time-scaling/mellow_code"
MELLOW_VARIANT="v0"
SUBSET=full
GPUS="0"
