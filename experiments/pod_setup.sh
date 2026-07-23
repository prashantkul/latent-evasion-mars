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
pip install -q \
  "transformers==5.14.1" \
  "inspect_ai==0.3.249" \
  "inspect_evals[agentharm]==0.15.0" \
  "numpy==2.5.1" \
  "scikit-learn==1.9.0" \
  "openai==2.46.0" \
  "huggingface_hub==1.16.1" \
  "accelerate==1.14.0"

echo "[setup] fast-path kernels for Qwen3.5 linear attention (best-effort) ..."
pip install -q flash-linear-attention causal-conv1d || echo "[setup] fla/causal-conv1d install failed -> torch fallback (ok)"

echo "[setup] downloading Qwen3.5-27B (~55GB) ..."
hf download Qwen/Qwen3.5-27B

echo "[setup] DONE"
