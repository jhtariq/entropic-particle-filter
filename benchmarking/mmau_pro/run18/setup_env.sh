#!/bin/bash
# Create the ONE conda env that serves and drives Phi-4-multimodal-instruct.
set -euo pipefail
RUN18_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUN18_DIR/config.sh"
cd "$REPO_ROOT"

if [ ! -x "$EPF_PY" ]; then
  echo "creating conda env '$EPF_ENV_NAME' (python 3.11)"
  conda create -n "$EPF_ENV_NAME" python=3.11 -y
fi
"$EPF_PY" -m pip install -r requirements-epf.txt
"$EPF_PY" -m pip install -e ".[dev,benchmark]"
# server-side audio decode deps (PyAV is vLLM's fallback decoder — without it: "Invalid audio file")
# + hf_transfer for fast downloads of the 44 GiB data.zip and the checkpoint
"$EPF_PY" -m pip install librosa soundfile av resampy hf_transfer
# SERVE_MODE=merged needs peft for the offline speech-lora merge (harmless otherwise)
[ "$SERVE_MODE" = "merged" ] && "$EPF_PY" -m pip install peft

"$EPF_PY" - <<'PYEOF'
import torch, vllm
print(f"ENV OK: vllm {vllm.__version__} | torch {torch.__version__} | CUDA {torch.version.cuda} "
      f"| GPUs visible: {torch.cuda.device_count()}")
PYEOF

cat <<'NOTE'
--------------------------------------------------------------------------------
requirements-epf.txt pins CUDA-13.0 builds of torch 2.11.0 / vLLM 0.22.1 (the
reference Blackwell box). If the pip install failed on torch/vllm, install wheels
matching YOUR CUDA stack first, then re-run this script — the remaining pins apply.
vLLM must register Phi4MMForCausalLM (0.22.1 does); the speech-lora adapter is
attached at serve time via --lora-modules (see config.sh SERVE_MODE).
--------------------------------------------------------------------------------
NOTE
