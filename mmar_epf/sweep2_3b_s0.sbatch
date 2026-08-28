#!/bin/bash
#SBATCH --account=bcey-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200g
#SBATCH --time=03:30:00
#SBATCH --job-name=mmar_epf2_3b_s0
#SBATCH --output=/u/awaheed/slurm_logs/mmar_epf2-%x-%j.out
#SBATCH --requeue
#SBATCH --open-mode=append
# Real-signal EPF sweep, full MMAR (1000 items): budgets 1,2,4,8,16,32,64,128,
# signals mean_logprob+entropy, P4, per-step trajectories stored. Resumable —
# a requeued/rerun job skips rows already in this shard's JSONL.
set -uo pipefail
export OUT_ROOT_OVERRIDE=/u/awaheed/mmar_epf2_out
export SUBSET=full SELECT=all LIMIT=1000
export SIGNALS=mean_logprob,entropy
export BUDGETS=1,2,4,8,16,32,64,128
export PROMPTS=4 SEED=1234
export STORE_TEXT=1 STORE_STEPS=1
bash /work/hdd/bcey/awaheed/its-for-audio-reasoning/entropic-particle-filter/mmar_epf/run_node.sh 3b 0 2 4
