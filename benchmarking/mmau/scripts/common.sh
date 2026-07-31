# Shared constants for the MMAU (original, test-mini) run scripts. Sourced by
# serve/watchdog/run_grid; each of those sets REPO_ROOT first, then sources this,
# then the run config.
EPF_PY=/home/exx/miniconda3/envs/epf/bin/python
EPF_VLLM=/home/exx/miniconda3/envs/epf/bin/vllm
HF_HOME_DIR=/media/exx/68031955-bdaa-4e71-9687-916b9876dfc6/hf_cache
DATA_ROOT=/home/exx/inference-time-scaling/data/mmau
MEDIA_ROOT=/home/exx/inference-time-scaling   # --allowed-local-media-path: common parent of ALL audio roots
# Single-GPU session (2026-07-31): GPU 1 / :8101 belongs to the live Kimi MMAR job.
# If GPU 1 frees up, override GPUS="0 1" + both endpoints in the config and resume.
GPUS="0"
ENDPOINTS="http://localhost:8100/v1"
