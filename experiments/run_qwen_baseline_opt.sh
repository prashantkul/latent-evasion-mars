#!/usr/bin/env bash
# Run the Qwen3.5-27B AgentHarm baseline via Inspect -> local vLLM (openai-api).
# Everything on LOCAL disk (/opt). LIMIT env: 2 for a mini, unset/0 for the full run.
cd /opt/latent-evasion-mars
set -a; . ./.env 2>/dev/null; set +a
export VLLM_BASE_URL=http://localhost:8000/v1 VLLM_API_KEY=EMPTY
export HF_HOME=/opt/.cache/huggingface
export PYTHONUNBUFFERED=1
exec python experiments/qwen35_baseline_vllm.py
