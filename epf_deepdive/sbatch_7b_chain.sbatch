#!/bin/bash
#SBATCH --account=bcey-delta-gpu
#SBATCH --partition=gpuA40x4,gpuA100x4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --output=/u/awaheed/slurm_logs/epfdd-%x-%j.out
# Resume/complete the 7B MMAR chain (greedy arms auto-skip; probe + SC arms resume).
bash /work/hdd/bcey/awaheed/its-for-audio-reasoning/epf_deepdive/run_7b_experiments.sh
