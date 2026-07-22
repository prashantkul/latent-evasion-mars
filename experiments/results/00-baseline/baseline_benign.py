import os
from inspect_ai import eval as inspect_eval
from inspect_ai.model import get_model, GenerateConfig
from inspect_evals.agentharm.agentharm import agentharm_benign

t = agentharm_benign(split="test_public")
t.config = GenerateConfig(max_tokens=4096, max_connections=2, max_retries=3, seed=0)
model = get_model("hf/unsloth/Llama-3.2-3B-Instruct", device="cuda:0")
logs = inspect_eval(t, model=model, log_dir="./logs_baseline_benign")
log = logs[0]
print("=== BENIGN BASELINE METRICS (official AgentHarm scorer) ===")
for s in (log.results.scores if log.results else []):
    for name, m in (s.metrics or {}).items():
        print(f"  {name}: {m.value:.4f}")
