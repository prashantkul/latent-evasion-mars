#!/usr/bin/env bash
# Serve Qwen3.5-27B via vLLM. Run on the pod:
#   bash /workspace/serve_qwen_vllm.sh
# or in tmux: tmux new -d -s vllm 'bash /workspace/serve_qwen_vllm.sh'
#
# venv lives on LOCAL disk (/opt/vllm-env). /workspace is MFS and stalls imports.
# HF model cache stays on the volume so weights survive pod resets.
export PATH=/opt/vllm-env/bin:$PATH
export HF_HOME=/workspace/.cache/huggingface
export PYTHONUNBUFFERED=1
cd /opt
# Qwen3.5-27B is a hybrid Mamba/attention model: max_num_seqs must be <= available
# Mamba cache blocks (~345) or CUDA-graph capture fails. 256 is ample for our eval.
exec /opt/vllm-env/bin/vllm serve Qwen/Qwen3.5-27B \
  --served-model-name qwen-3.5-27b \
  --tool-call-parser qwen3_xml \
  --enable-auto-tool-choice \
  --max-model-len 16384 \
  --max-num-seqs 256 \
  --host 0.0.0.0 --port 8000 > /opt/vllm_serve.log 2>&1
