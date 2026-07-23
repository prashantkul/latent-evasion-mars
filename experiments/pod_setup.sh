#!/usr/bin/env bash
# One-shot setup for the Qwen3.5-27B AgentHarm run on a RunPod A100 pod.
# Uses the image's CUDA torch; installs the rest to match the DGX venv versions.
# Downloads the model. Does NOT need the OpenAI key (that's for the run step).
set -euo pipefail

cd /workspace 2>/dev/null || cd "$HOME"
echo "[setup] workdir: $(pwd)"

# repo (public)
if [ ! -d latent-evasion-mars ]; then
  git clone https://github.com/prashantkul/latent-evasion-mars.git
fi
cd latent-evasion-mars

echo "[setup] torch already in image:"; python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

echo "[setup] installing deps (matching DGX versions, keeping image torch) ..."
# numpy/scikit-learn left to pip's resolver: the DGX pins (numpy 2.5.1) need py3.12,
# but the CUDA-12.4 pod image is py3.11 -> use py3.11-compatible builds.
pip install -q \
  "transformers==5.14.1" \
  "inspect_ai==0.3.249" \
  "inspect_evals[agentharm]==0.15.0" \
  "numpy<2.5" \
  "scikit-learn" \
  "openai==2.46.0" \
  "huggingface_hub==1.16.1" \
  "accelerate==1.14.0"

# The CUDA-12.4 pod image ships torch 2.4.1, too old for transformers 5.14.1 (needs DTensor,
# torch >=2.5). Driver 555 caps us at CUDA 12.5, so upgrade the whole torch stack to a cu124
# build >=2.6 (torch+torchvision+torchaudio must match, else torchvision::nms / libtorchaudio
# undefined-symbol errors when transformers loads the multimodal Qwen3.5).
echo "[setup] upgrading torch stack to 2.6.0+cu124 (image torch too old for transformers 5.14.1) ..."
pip install -q torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; from torch.distributed.tensor import DTensor; print('torch', torch.__version__, 'DTensor OK')"

echo "[setup] fast-path kernels for Qwen3.5 linear attention (best-effort) ..."
pip install -q flash-linear-attention causal-conv1d || echo "[setup] fla/causal-conv1d install failed -> torch fallback (ok)"

echo "[setup] downloading Qwen3.5-27B (~55GB) ..."
hf download Qwen/Qwen3.5-27B

echo "[setup] DONE"
