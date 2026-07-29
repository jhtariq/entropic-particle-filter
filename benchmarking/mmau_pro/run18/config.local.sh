# Local overrides for THIS box (gitignored — the collaborator box uses the defaults).
EPF_DATA_ROOT=/home/exx/inference-time-scaling
# Phi-4-MM already lives in the DEFAULT HF cache on this box — do NOT point HF_HOME
# at the big volume or fetch/serve re-downloads the 12 GB checkpoint.
HF_HOME=/home/exx/.cache/huggingface
# NOTE: don't set BUDGETS here — plain assignments in this file override even
# explicit env vars (it is sourced first). config.sh's default is the full ladder;
# pass BUDGETS="1 8 16" on the command line for the local seed phase.
