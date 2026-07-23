#!/usr/bin/env bash
# Robust launcher for the Qwen3.5-27B AgentHarm baseline on the DGX.
# Usage: run_qwen_baseline.sh [LIMIT]   (LIMIT=2 for a mini; 0/absent for full)
# Sources the OpenAI judge key from .env; keeps thinking off (baseline default).
cd "$HOME/latent-evasion" || exit 1
set -a; . ./.env 2>/dev/null; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LIMIT="${1:-0}"
exec .venv/bin/python qwen35_baseline.py
