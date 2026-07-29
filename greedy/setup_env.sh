#!/bin/bash
# Create the ONE conda env that serves and drives all six models (vLLM for the
# Qwen/Phi/Kimi family, the FastAPI shim for Mellow) and runs the greedy driver.
set -euo pipefail
GREEDY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$GREEDY_DIR/config.sh"
cd "$REPO_ROOT"

command -v conda > /dev/null 2>&1 || {
  echo "FATAL: conda not found. Install Miniconda first:" >&2
  echo "  https://docs.conda.io/en/latest/miniconda.html" >&2
  exit 1
}

if [ ! -x "$EPF_PY" ]; then
  echo "creating conda env '$EPF_ENV_NAME' (python 3.11)"
  conda create -n "$EPF_ENV_NAME" python=3.11 -y
fi
"$EPF_PY" -m pip install -r requirements-epf.txt
"$EPF_PY" -m pip install -e ".[dev,benchmark]"
# server-side audio decode deps (PyAV is vLLM's fallback decoder — without it: "Invalid audio file")
# + hf_transfer for fast downloads of the 44 GiB data.zip and the checkpoints
"$EPF_PY" -m pip install librosa soundfile av resampy hf_transfer
# Mellow shim extras (run17): torchcodec needs a system ffmpeg (checked below)
"$EPF_PY" -m pip install torchlibrosa importlib_resources torchcodec

# system ffmpeg 4-7 for torchcodec (Mellow audio decode) — check-and-instruct, never auto-sudo
if command -v ffmpeg > /dev/null 2>&1; then
  FFV="$(ffmpeg -version 2>/dev/null | head -1 | sed 's/^ffmpeg version \([0-9]*\).*/\1/')"
  case "$FFV" in
    4|5|6|7) echo "ffmpeg OK (major version $FFV)" ;;
    *) echo "FATAL: ffmpeg major version '$FFV' — torchcodec needs 4-7. Install e.g.:" >&2
       echo "  sudo apt install ffmpeg" >&2; exit 1 ;;
  esac
else
  echo "FATAL: ffmpeg not found (the Mellow shim's torchcodec needs it). Install e.g.:" >&2
  echo "  sudo apt install ffmpeg" >&2
  exit 1
fi

# disk headroom: models ~85 GB (HF_HOME) + datasets ~115 GB peak (EPF_DATA_ROOT)
mkdir -p "$EPF_DATA_ROOT" "$HF_HOME"
FREE_GB="$(df -BG --output=avail "$EPF_DATA_ROOT" | tail -1 | tr -dc '0-9')"
if [ "${FREE_GB:-0}" -lt 250 ]; then
  echo "WARNING: only ${FREE_GB} GB free under $EPF_DATA_ROOT — ~250 GB needed for models+data." >&2
  echo "  Point EPF_DATA_ROOT (and HF_HOME) at a bigger volume via greedy/config.local.sh." >&2
fi

"$EPF_PY" - <<'PYEOF'
import torch, vllm
print(f"ENV OK: vllm {vllm.__version__} | torch {torch.__version__} | CUDA {torch.version.cuda} "
      f"| GPUs visible: {torch.cuda.device_count()}")
from vllm.tokenizers.kimi_audio import KimiAudioTokenizer  # noqa: F401 — kimi arch support probe
print("ENV OK: vLLM registers the Kimi-Audio stack (tokenizer/processor/model)")
import fastapi, torchaudio, torchcodec, torchlibrosa, uvicorn  # noqa: F401 — mellow shim deps
print("ENV OK: Mellow shim deps (fastapi/torchaudio/torchcodec/torchlibrosa/uvicorn)")
PYEOF

cat <<'NOTE'
--------------------------------------------------------------------------------
requirements-epf.txt pins CUDA-13.0 builds of torch 2.11.0 / vLLM 0.22.1 (the
reference Blackwell box). If the pip install failed on torch/vllm, install wheels
matching YOUR CUDA stack first, then re-run this script — the remaining pins apply.
vLLM must register MoonshotKimiaForCausalLM and Phi4MMForCausalLM (0.22.1 does).
--------------------------------------------------------------------------------
NOTE
