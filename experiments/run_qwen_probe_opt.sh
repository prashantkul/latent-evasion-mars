#!/usr/bin/env bash
# Agentic probe extraction + dissociation for Qwen3.5-27B (Inspect hf provider, hidden_states).
# Reads behavioral outcomes from the vLLM AgentHarm run. Runs on local /opt; model cache on MFS.
cd /opt/latent-evasion-mars
set -a; . ./.env 2>/dev/null; set +a
export HF_HOME=/workspace/.cache/huggingface           # 52GB model cache lives here
export GRADED_LOG_DIR=/opt/latent-evasion-mars/logs_qwen35v_harmful
export OUT=/opt/qwen35_dissociation.json
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec python experiments/qwen35_probe_dissociation.py
