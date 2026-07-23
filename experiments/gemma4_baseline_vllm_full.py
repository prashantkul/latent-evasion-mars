"""Full AgentHarm baseline for Gemma-4-31B via vLLM (OpenAI endpoint) through Inspect.
Runs harmful (test_public) then benign, official limits, gpt-4o judge on real OpenAI.
Produces the four standard metrics."""
import os
from inspect_ai import eval as inspect_eval
from inspect_ai.model import get_model, GenerateConfig
from inspect_evals.agentharm.agentharm import agentharm, agentharm_benign

MODEL = "openai-api/vllm/gemma-4-31b"  # VLLM_BASE_URL / VLLM_API_KEY in env
CFG = GenerateConfig(
    temperature=0.0, max_tokens=4096, max_connections=8, max_retries=3, seed=0
)
model = get_model(MODEL)


def run(task, log_dir, label):
    task.config = CFG
    logs = inspect_eval(task, model=model, log_dir=log_dir)
    log = logs[0]
    print(f"=== {label} STATUS: {log.status} ===")
    metrics = {}
    if log.status == "success":
        for s in (log.results.scores if log.results else []):
            for name, m in (s.metrics or {}).items():
                metrics[name] = m.value
        for k in ("avg_score", "avg_refusals", "avg_score_non_refusals"):
            if k in metrics:
                print(f"  {label}.{k}: {metrics[k]:.4f}")
    else:
        print(f"  ERROR: {getattr(log, 'error', None)}")
    return log.status, metrics


print(">>> HARMFUL")
hs, hm = run(agentharm(split="test_public"), "./logs_gemma4_harmful", "HARMFUL")
print(">>> BENIGN")
bs, bm = run(agentharm_benign(split="test_public"), "./logs_gemma4_benign", "BENIGN")

print("=== GEMMA-4-31B FOUR STANDARD METRICS ===")
print(f"  Harm Score              : {hm.get('avg_score', float('nan')):.4f}")
print(f"  Refusals                : {hm.get('avg_refusals', float('nan')):.4f}")
print(f"  Non-refusal Harm Score  : {hm.get('avg_score_non_refusals', float('nan')):.4f}")
print(f"  Benign Non-refusal Score: {bm.get('avg_score_non_refusals', float('nan')):.4f}")
print("ALL_DONE")
