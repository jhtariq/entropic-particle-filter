#!/bin/bash
#SBATCH --account=bcey-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200g
#SBATCH --time=24:00:00
#SBATCH --job-name=kimi_mmau
#SBATCH --output=/u/awaheed/slurm_logs/kimi_mmau-%j.out
#SBATCH --requeue
#SBATCH --open-mode=append
#
# The Kimi-Audio arm of the MMAU test-mini SC sweep — deferred when the other two models
# ran. MMAU_MODELS is overridden so qwen2-audio/phi4mm (already complete) are not re-served.
#
# Kimi needs three things that config.sh already encodes, all learned the hard way:
#   * run19 serve wrapper + jinja chat template (vLLM's stock kimi path can't do this)
#   * --stop "[EOS]"  — the model emits the literal text "[EOS]"; generation_config.json
#     has no eos_token_id, so without this it loops "A[EOS][EOS]AAAA..." until truncation
#   * MAX_INFLIGHT=1  — with >=2 audio items per engine step the output degenerates to
#     "AAAA..." (non-deterministic even at temp 0). Cost: slow. Benefit: correct.
set -uo pipefail
module load cuda-compat/13.0 2>/dev/null   # cu130 wheels vs Delta's CUDA-12.8 driver
export TMPDIR="/tmp/$USER/${SLURM_JOB_ID:-0}"; mkdir -p "$TMPDIR"
export PIP_CACHE_DIR="$TMPDIR/pip" HF_HUB_OFFLINE=1
echo "=== $(date) job=${SLURM_JOB_ID:-none} node=$(hostname) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
cd /work/hdd/bcey/awaheed/its-for-audio-reasoning/entropic-particle-filter
BENCH=mmau MMAU_MODELS=kimi_audio exec bash mmar_sc/run_bench.sh
