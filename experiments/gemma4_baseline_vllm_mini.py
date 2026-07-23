"""Mini integration test: Gemma-4-31B via vLLM (OpenAI endpoint) through Inspect.
Validates the full path: Inspect -> vLLM (gemma4 tool parser) -> AgentHarm scorer
(+ gpt-4o judge on real OpenAI). Small limit before the full baseline."""
import os
from inspect_ai import eval as inspect_eval
from inspect_ai.model import get_model, GenerateConfig
from inspect_evals.agentharm.agentharm import agentharm

# model served by local vLLM; judge stays on real OpenAI (OPENAI_API_KEY from .env)
MODEL = "openai-api/vllm/gemma-4-31b"  # reads VLLM_BASE_URL / VLLM_API_KEY

t = agentharm(split="test_public")
t.config = GenerateConfig(
    temperature=0.0, max_tokens=4096, max_connections=8, max_retries=3, seed=0
)
model = get_model(MODEL)
logs = inspect_eval(t, model=model, log_dir="./logs_gemma4_vllm_mini", limit=2)
log = logs[0]
print(f"=== MINI STATUS: {log.status} ===")
if log.status == "success":
    print("=== GEMMA-4 (vLLM) MINI METRICS ===")
    for s in (log.results.scores if log.results else []):
        for name, m in (s.metrics or {}).items():
            print(f"  {name}: {m.value:.4f}")
    # sanity: did the agent actually make tool calls?
    n_tool = 0
    for samp in (log.samples or []):
        for msg in samp.messages:
            if getattr(msg, "role", None) == "assistant" and getattr(msg, "tool_calls", None):
                n_tool += len(msg.tool_calls)
    print(f"  [sanity] assistant tool_calls across samples: {n_tool}")
    print("INTEGRATION_OK")
else:
    print("ERROR:", getattr(log, "error", None))
