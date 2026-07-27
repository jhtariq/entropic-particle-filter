# Shared constants for the MMSU run scripts. Sourced by serve/watchdog/run_grid;
# each of those sets SCRIPT_DIR + REPO_ROOT first, then sources this, then the run config.
# Every value is env-overridable; a gitignored scripts/local.sh (plain VAR=value lines)
# is sourced FIRST and wins — set your machine's paths there, never edit this file.
[ -f "$SCRIPT_DIR/local.sh" ] && source "$SCRIPT_DIR/local.sh"
: "${EPF_PY:=/home/exx/miniconda3/envs/epf/bin/python}"
: "${EPF_VLLM:=/home/exx/miniconda3/envs/epf/bin/vllm}"
: "${HF_HOME_DIR:=/media/exx/68031955-bdaa-4e71-9687-916b9876dfc6/hf_cache}"
: "${DATA_ROOT:=/home/exx/inference-time-scaling/data/mmsu}"   # MMSU-meta.json + audio/ live here
: "${MEDIA_ROOT:=/home/exx/inference-time-scaling}"  # --allowed-local-media-path / shim media root: must be a parent of DATA_ROOT
: "${ENDPOINTS:=http://localhost:8100/v1,http://localhost:8101/v1}"
