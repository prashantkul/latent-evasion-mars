#!/usr/bin/env bash
# Setup for a CUDA-13 pod (driver >=580, image runpod/pytorch:*-cu1300-torch291-*).
# torch 2.9.1+cu130 already present & new enough for transformers -> NO torch dance.
# Two lanes: system python (hf extraction + Inspect eval client) + venv for vLLM.
set -uo pipefail

cd /workspace
[ -d latent-evasion-mars ] || git clone https://github.com/prashantkul/latent-evasion-mars.git
cd latent-evasion-mars

echo "[hf] torch:"; python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
echo "[hf] installing eval + extraction deps (system python, PEP668 override; keep image torch) ..."
pip install --break-system-packages -q "transformers==5.14.1" "inspect_ai==0.3.249" \
  "inspect_evals[agentharm]==0.15.0" "openai==2.46.0" "huggingface_hub==1.16.1" \
  "accelerate==1.14.0" "scikit-learn" "numpy<2.6"
python -c "from torch.distributed.tensor import DTensor; import transformers, inspect_ai; print('[hf ok]', transformers.__version__, inspect_ai.__version__)"

# venv MUST live on local container disk (/opt), not /workspace (network MFS).
# Importing torch/triton/vllm from MFS stalls and looks hung.
echo "[vllm] local-disk venv at /opt/vllm-env (do NOT put site-packages on /workspace) ..."
export PIP_CACHE_DIR=/workspace/pip-cache
mkdir -p "$PIP_CACHE_DIR"
python -m venv /opt/vllm-env
/opt/vllm-env/bin/pip install -q --upgrade pip
/opt/vllm-env/bin/pip install -q vllm
/opt/vllm-env/bin/python -c "import torch,vllm; print('[vllm]', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)"

echo "[model] downloading Qwen3.5-27B (~55GB) onto volume HF cache ..."
export HF_HOME=/workspace/.cache/huggingface
hf download Qwen/Qwen3.5-27B

echo "SETUP_DONE"
