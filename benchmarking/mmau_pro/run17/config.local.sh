# Local overrides for THIS box (gitignored — the collaborator box uses the defaults).
EPF_DATA_ROOT=/home/exx/inference-time-scaling
HF_HOME=/home/exx/.cache/huggingface
MELLOW_CODE_DIR=/home/exx/inference-time-scaling/mellow_code
# NOTE: don't set BUDGETS here — plain assignments in this file override even
# explicit env vars (it is sourced first). config.sh's default drives the local
# b{1,8,16} phase; the collaborator package flips that default to the extension.
