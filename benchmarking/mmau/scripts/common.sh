# Shared constants for the MMAU (original, test-mini) run scripts. Sourced by
# serve/watchdog/run_grid; each of those sets REPO_ROOT first, then sources this,
# then the run config.
EPF_PY=/home/exx/miniconda3/envs/epf/bin/python
EPF_VLLM=/home/exx/miniconda3/envs/epf/bin/vllm
HF_HOME_DIR=/media/exx/68031955-bdaa-4e71-9687-916b9876dfc6/hf_cache
DATA_ROOT=/home/exx/inference-time-scaling/data/mmau
MEDIA_ROOT=/home/exx/inference-time-scaling   # --allowed-local-media-path: common parent of ALL audio roots
# Single-GPU session default (2026-07-31): GPU 1 / :8101 was the live Kimi MMAR job.
# Both are env-overridable so that when GPU 1 frees up the grid can be resumed across
# both GPUs without editing this file, e.g.:
#   GPUS="0 1" ENDPOINTS="http://localhost:8100/v1,http://localhost:8101/v1" bash run_grid.sh ...
GPUS="${GPUS:-0}"
ENDPOINTS="${ENDPOINTS:-http://localhost:8100/v1}"
