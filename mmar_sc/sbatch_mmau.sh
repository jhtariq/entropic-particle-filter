#!/bin/bash
#SBATCH --account=bcey-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200g
#SBATCH --time=08:00:00
#SBATCH --job-name=mmau_sc
#SBATCH --output=/u/awaheed/slurm_logs/mmau_sc-%j.out
#SBATCH --requeue
#SBATCH --open-mode=append
#
# 8 h on 4x A40, NON-preempt (gpuA40x4 only — no -preempt partition, so once this
# starts nothing can evict it). Runs the 12-cell MMAU test-mini sweep, then HOLDS the
# allocation for follow-up work rather than returning it to a 300-deep queue.
set -uo pipefail

REPO=/work/hdd/bcey/awaheed/its-for-audio-reasoning/entropic-particle-filter
module load cuda-compat/13.0 2>/dev/null   # cu130 wheels vs Delta's CUDA-12.8 driver

export TMPDIR="/tmp/$USER/${SLURM_JOB_ID:-0}"
mkdir -p "$TMPDIR"
export PIP_CACHE_DIR="$TMPDIR/pip" HF_HUB_OFFLINE=1

echo "=== $(date) job=${SLURM_JOB_ID:-none} node=$(hostname) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo

cd "$REPO" && BENCH=mmau exec bash mmar_sc/run_bench.sh
