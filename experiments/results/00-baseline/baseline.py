import os
from inspect_ai import eval as inspect_eval
from inspect_ai.model import get_model, GenerateConfig
from inspect_evals.agentharm.agentharm import agentharm

t = agentharm(split="test_public")                 # official task, default agent + combined_scorer
# official config hardcodes temperature=0.0 which transformers 5.x rejects -> override (greedy)
t.config = GenerateConfig(max_tokens=4096, max_connections=4, max_retries=3, seed=0)
model = get_model("hf/unsloth/Llama-3.2-3B-Instruct", device="cuda:0")
logs = inspect_eval(t, model=model, log_dir="./logs_baseline")
log = logs[0]
print("=== BASELINE METRICS (official AgentHarm scorer) ===")
for s in (log.results.scores if log.results else []):
    for name, m in (s.metrics or {}).items():
        print(f"  {name}: {m.value:.4f}")
