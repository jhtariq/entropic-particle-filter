#!/bin/bash
#SBATCH --account=bcey-delta-gpu
#SBATCH --partition=gpuA40x4-preempt,gpuA100x4-preempt
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200g
#SBATCH --time=08:00:00
#SBATCH --job-name=mmar_sc_pre
#SBATCH --output=/u/awaheed/slurm_logs/mmar_sc-%j.out
#SBATCH --requeue
#SBATCH --open-mode=append
#
# Comma partition list: Slurm starts us on whichever of H200 (141 GB) / A40 (48 GB)
# frees first, and mmar_sc/lib.sh kv_plan() re-derives the serving knobs from the card
# it actually lands on. NOT `set -e` — one dead cell must not kill the whole job, and
# every cell is resumable, so a re-submit is the entire recovery procedure.
set -uo pipefail

REPO=/work/hdd/bcey/awaheed/its-for-audio-reasoning/entropic-particle-filter

module load cuda-compat/13.0 2>/dev/null   # cu130 wheels vs Delta's 570.x/CUDA-12.8 driver

# node-local NVMe for every cache, so nothing touches the Lustre quotas
export TMPDIR="/tmp/$USER/${SLURM_JOB_ID:-0}"
mkdir -p "$TMPDIR"
export PIP_CACHE_DIR="$TMPDIR/pip" HF_HUB_OFFLINE=1

echo "=== $(date) job=${SLURM_JOB_ID:-none} node=$(hostname) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo

cd "$REPO" && exec bash mmar_sc/run_all.sh
