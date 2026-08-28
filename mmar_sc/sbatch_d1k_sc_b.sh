#!/bin/bash
#SBATCH --account=bcey-delta-gpu
#SBATCH --partition=gpuH200x8,gpuA40x4,gpuA100x4
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200g
#SBATCH --time=10:00:00
#SBATCH --job-name=d1k_sc_b
#SBATCH --output=/u/awaheed/slurm_logs/d1k_sc-%x-%j.out
#SBATCH --requeue
#SBATCH --open-mode=append
# D1K SC+greedy for: qwen2_audio phi4mm. Arms: greedy_nocot greedy_p4 sc_nocot sc_p4
# (SC n=128 t=0.8; ladder subsampled offline at R=200). Resumable cells.
set -uo pipefail
module load cuda-compat/13.0 2>/dev/null || true
export TMPDIR="/tmp/$USER/${SLURM_JOB_ID:-0}"; mkdir -p "$TMPDIR"
export HF_HUB_OFFLINE=1
export BENCH=mmau_pro SUBSET=d1k
export OUT_ROOT=/work/hdd/bcey/awaheed/its-for-audio-reasoning/mmau_pro_sc_out_b
export MMAU_PRO_MODELS="qwen2_audio phi4mm"
cd /work/hdd/bcey/awaheed/its-for-audio-reasoning/entropic-particle-filter
exec bash mmar_sc/run_bench.sh
