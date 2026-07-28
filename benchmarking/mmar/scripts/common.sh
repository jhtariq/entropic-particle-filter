# Shared constants for the MMAR run scripts. Sourced by serve/watchdog/run_grid;
# each of those sets REPO_ROOT first, then sources this, then the run config.
EPF_PY=/home/exx/miniconda3/envs/epf/bin/python
EPF_VLLM=/home/exx/miniconda3/envs/epf/bin/vllm
HF_HOME_DIR=/media/exx/68031955-bdaa-4e71-9687-916b9876dfc6/hf_cache
DATA_ROOT=/home/exx/inference-time-scaling/mmar
MEDIA_ROOT=/home/exx/inference-time-scaling   # --allowed-local-media-path: common parent of ALL audio roots
ENDPOINTS="http://localhost:8100/v1,http://localhost:8101/v1"
