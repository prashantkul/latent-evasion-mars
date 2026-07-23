#!/usr/bin/env bash
# Runs on the pod inside tmux. Waits for vLLM, curl-tests the qwen3_xml tool parser,
# then runs the mini AgentHarm baseline (LIMIT=2) via Inspect openai-api + strict_tools=false.
# Everything executes on LOCAL disk (/opt); /workspace is MFS and stalls.
cd /opt/latent-evasion-mars
set -a; . ./.env 2>/dev/null; set +a
export VLLM_BASE_URL=http://localhost:8000/v1 VLLM_API_KEY=EMPTY
export HF_HOME=/opt/.cache/huggingface  # dataset cache on local disk

echo "=== waiting for vLLM ==="
until curl -s -m 5 http://localhost:8000/v1/models >/dev/null 2>&1; do sleep 10; done
echo "=== SERVER UP ==="

echo "=== CURL TOOL-CALL TEST ==="
curl -s http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model":"qwen-3.5-27b",
  "messages":[{"role":"user","content":"What is the weather in Tokyo? Use the tool."}],
  "tools":[{"type":"function","function":{"name":"get_weather","description":"Get weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],
  "tool_choice":"auto","max_tokens":512,"temperature":0}' | python3 -c "import sys,json; d=json.load(sys.stdin); m=d['choices'][0]['message']; print('finish:',d['choices'][0].get('finish_reason')); print('tool_calls:',json.dumps(m.get('tool_calls'))); print('content:',repr((m.get('content') or '')[:200]))"

echo "=== MINI BASELINE (LIMIT=2) ==="
LIMIT=2 python experiments/qwen35_baseline_vllm.py
echo "=== TEST_DONE ==="
