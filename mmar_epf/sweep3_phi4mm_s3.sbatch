#!/bin/bash
#SBATCH --account=bcey-delta-gpu
#SBATCH --partition=gpuA40x4,gpuA100x4
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=100g
#SBATCH --time=02:30:00
#SBATCH --job-name=mmar_epf3_phi4mm_s3of8
#SBATCH --output=/u/awaheed/slurm_logs/mmar_epf3-%x-%j.out
#SBATCH --requeue
#SBATCH --open-mode=append
# FULL-TRACE EPF rerun (pruned lineages included), full MMAR 1000 items.
set -uo pipefail
export BASE_PORT=8412
export OUT_ROOT_OVERRIDE=/u/awaheed/mmar_epf3_out
export SUBSET=full SELECT=all LIMIT=1000
export SIGNALS=mean_logprob,entropy
export BUDGETS=1,2,4,8,16,32,64,128
export PROMPTS=4 SEED=1234
export STORE_TEXT=1 STORE_STEPS=1 SAVE_TRACES=1
export MAX_INFLIGHT=128
bash /work/hdd/bcey/awaheed/its-for-audio-reasoning/entropic-particle-filter/mmar_epf/run_node.sh phi4mm 3 8 2
