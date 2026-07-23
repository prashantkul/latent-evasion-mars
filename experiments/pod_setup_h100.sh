#!/usr/bin/env bash
# Setup for the CUDA-12.8 pod (driver 570, torch 2.8+cu128 already in image).
# Two lanes:
#   - system python: transformers/inspect (probe extraction via hf + Inspect eval CLIENT)
#   - /workspace/vllm-env: vLLM (serves Qwen3.5/Gemma-4 with native tool parsers)
# No torch dance needed (image torch 2.8 is new enough for transformers 5.14.1).
set -uo pipefail

cd /workspace
[ -d latent-evasion-mars ] || git clone https://github.com/prashantkul/latent-evasion-mars.git
cd latent-evasion-mars

echo "[hf] torch present:"; python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
echo "[hf] installing eval + extraction deps (keep image torch) ..."
pip install -q "transformers==5.14.1" "inspect_ai==0.3.249" "inspect_evals[agentharm]==0.15.0" \
  "openai==2.46.0" "huggingface_hub==1.16.1" "accelerate==1.14.0" "scikit-learn" "numpy<2.6"
python -c "from torch.distributed.tensor import DTensor; import transformers; print('[hf] ok', transformers.__version__)"

echo "[vllm] creating isolated venv + installing vllm (own torch) ..."
python -m venv /workspace/vllm-env
/workspace/vllm-env/bin/pip install -q --upgrade pip
/workspace/vllm-env/bin/pip install -q vllm
/workspace/vllm-env/bin/python - <<'PY'
import vllm
print("[vllm] version", vllm.__version__)
try:
    from vllm.tool_parsers import ToolParserManager as M
    ks = sorted(M.tool_parsers.keys())
except Exception:
    try:
        from vllm.entrypoints.openai.tool_parsers import ToolParserManager as M
        ks = sorted(M.tool_parsers.keys())
    except Exception as e:
        ks = ["<introspect failed: %s>" % e]
print("[vllm] parsers w/ qwen/gemma:", [k for k in ks if "qwen" in str(k) or "gemma" in str(k)])
PY

echo "[model] downloading Qwen3.5-27B (~55GB) ..."
hf download Qwen/Qwen3.5-27B

echo "SETUP_DONE"
