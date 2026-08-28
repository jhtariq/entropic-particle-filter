#!/bin/bash
#SBATCH --account=bcey-delta-gpu
#SBATCH --partition=gpuA40x4,gpuA100x4
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=100g
#SBATCH --time=06:00:00
#SBATCH --job-name=d1k_sc_7b
#SBATCH --output=/u/awaheed/slurm_logs/d1k_sc-%x-%j.out
#SBATCH --requeue
#SBATCH --open-mode=append
set -uo pipefail
module load cuda-compat/13.0 2>/dev/null || true
export TMPDIR="/tmp/$USER/${SLURM_JOB_ID:-0}"; mkdir -p "$TMPDIR"
export HF_HUB_OFFLINE=1
export BENCH=mmau_pro SUBSET=d1k
export OUT_ROOT=/work/hdd/bcey/awaheed/its-for-audio-reasoning/mmau_pro_sc_out_7b
export MMAU_PRO_MODELS="qwen_omni_7b"
cd /work/hdd/bcey/awaheed/its-for-audio-reasoning/entropic-particle-filter
exec bash mmar_sc/run_bench.sh
