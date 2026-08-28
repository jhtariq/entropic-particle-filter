#!/bin/bash
#SBATCH --account=bcey-delta-gpu
#SBATCH --partition=gpuA40x4,gpuA100x4
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=100g
#SBATCH --time=06:00:00
#SBATCH --job-name=mmau_sc_7b
#SBATCH --output=/u/awaheed/slurm_logs/mmau_sc-%x-%j.out
#SBATCH --requeue
#SBATCH --open-mode=append
# Omni-7B SC + greedy on MMAU test-mini (the cells missing from the Aug-3 MMAU SC run).
# Own OUT_ROOT so the 3B and 7B jobs never contend for the single-writer lock.
set -uo pipefail
module load cuda-compat/13.0 2>/dev/null || true
export TMPDIR="/tmp/$USER/${SLURM_JOB_ID:-0}"; mkdir -p "$TMPDIR"
export HF_HUB_OFFLINE=1
export BENCH=mmau SUBSET=full
export OUT_ROOT=/u/awaheed/mmau_sc_out_7b
export MMAU_MODELS="qwen_omni_7b"
export BASE_PORT=8120
export HOLD_AFTER_RUN=0     # do NOT idle to the wall after the cells finish
cd /work/hdd/bcey/awaheed/its-for-audio-reasoning/entropic-particle-filter
exec bash mmar_sc/run_bench.sh
