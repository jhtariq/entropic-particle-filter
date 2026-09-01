#!/bin/bash
#SBATCH --account=bcey-delta-gpu
#SBATCH --partition=gpuA40x4,gpuA100x4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --output=/u/awaheed/slurm_logs/epfdd-%x-%j.out
# Probe-EPF budget curve (env: MODEL=phi|kimi|3b|7b|q2a, BENCH=mmar|mmau_pro_d1k,
# optional FORCE_PM / KIMI_APC / APC_TOL — see run_newmodel_curve.sh).
# Time limit is set at submission (--time): most models 12h, kimi 36h.
set -uo pipefail
D=/work/hdd/bcey/awaheed/its-for-audio-reasoning/epf_deepdive
: "${MODEL:?phi|kimi|3b|7b|q2a}"
PORT=$((8300 + SLURM_JOB_ID % 700))  # job-unique: avoids SO_REUSEPORT collisions
exec bash "$D/run_newmodel_curve.sh" "$MODEL" "$PORT"
