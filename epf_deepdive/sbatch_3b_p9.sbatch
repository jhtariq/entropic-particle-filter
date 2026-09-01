#!/bin/bash
#SBATCH --account=bcey-delta-gpu
#SBATCH --partition=gpuA40x4,gpuA100x4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --output=/u/awaheed/slurm_logs/epfdd-%x-%j.out
# Serve 3B, then resume/complete the 3B P9 chain (probe P9, greedy_p9, sc_p9).
set -uo pipefail
D=/work/hdd/bcey/awaheed/its-for-audio-reasoning/epf_deepdive
export SERVE_TAG="3bp9_${SLURM_JOB_ID}"
bash "$D/serve_model.sh" 3b 8500
t=0; while [ $t -lt 1800 ]; do
  curl -sf http://localhost:8500/v1/models >/dev/null 2>&1 && break; sleep 10; t=$((t+10)); done
[ $t -lt 1800 ] || { echo "FATAL server not healthy"; exit 1; }
echo "server healthy after ${t}s"
bash "$D/run_3b_p9.sh"
rc=$?
kill "$(cat "$D/serve_${SERVE_TAG}.pid")" 2>/dev/null
exit $rc
