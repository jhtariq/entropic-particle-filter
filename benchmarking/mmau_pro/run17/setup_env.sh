#!/bin/bash
# Create the ONE conda env that serves (via the shim) and drives Mellow.
set -euo pipefail
RUN17_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUN17_DIR/config.sh"
cd "$REPO_ROOT"

if [ ! -x "$EPF_PY" ]; then
  echo "creating conda env '$EPF_ENV_NAME' (python 3.11)"
  conda create -n "$EPF_ENV_NAME" python=3.11 -y
fi
"$EPF_PY" -m pip install -r requirements-epf.txt
"$EPF_PY" -m pip install -e ".[dev,benchmark]"
# audio decode + fast downloads (as in run16)
"$EPF_PY" -m pip install librosa soundfile av resampy hf_transfer
# shim/Mellow extras: torchlibrosa (HTSAT features), importlib_resources (mellow code),
# torchcodec (torchaudio>=2.9 delegates .load to it; needs system ffmpeg 4-7)
"$EPF_PY" -m pip install torchlibrosa importlib_resources torchcodec

"$EPF_PY" - <<'PYEOF'
import fastapi, torch, torchaudio, torchcodec, torchlibrosa, uvicorn  # noqa: F401
print(f"ENV OK: torch {torch.__version__} | torchaudio {torchaudio.__version__} "
      f"| CUDA {torch.version.cuda} | GPUs visible: {torch.cuda.device_count()}")
PYEOF

cat <<'NOTE'
--------------------------------------------------------------------------------
requirements-epf.txt pins CUDA-13.0 builds of torch 2.11.0 (+ vLLM, unused here —
the shim serves Mellow with plain transformers). If the pip install failed on
torch, install wheels matching YOUR CUDA stack first, then re-run this script.
torchcodec needs the ffmpeg shared libraries (`ffmpeg -version` -> 4..7); if it
cannot be installed, the shim falls back to soundfile/librosa decoding, but the
reference-parity script (validate_shim.py) needs torchaudio.load to work.
Mellow is 167M params — any single >=8 GB GPU serves it comfortably.
--------------------------------------------------------------------------------
NOTE
