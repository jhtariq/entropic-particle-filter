#!/bin/bash
#SBATCH --account=bcey-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200g
#SBATCH --time=24:00:00
#SBATCH --job-name=epf_hold2
#SBATCH --output=/u/awaheed/slurm_logs/epf_hold-%j.out
#SBATCH --requeue
#SBATCH --open-mode=append
#
# 24 h on 4x A40, NON-preempt. Reserves the GPUs and holds them so work can be pushed in
# with `srun --jobid=<ID> --overlap <cmd>`. Release by deleting the HOLD file or scancel.
set -uo pipefail
module load cuda-compat/13.0 2>/dev/null || true
HOLDFILE=/u/awaheed/epf_hold_${SLURM_JOB_ID}.HOLD
: > "$HOLDFILE"
echo "=== $(date) job=$SLURM_JOB_ID node=$(hostname) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
cat <<EOM

================ HOLDING ALLOCATION (24 h) ================
  use:      srun --jobid=$SLURM_JOB_ID --overlap <command>
  release:  rm $HOLDFILE      (or scancel $SLURM_JOB_ID)
===========================================================
EOM
while [ -e "$HOLDFILE" ]; do sleep 30; done
echo "HOLD released at $(date) — exiting"
