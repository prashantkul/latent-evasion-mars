"""AgentHarm baseline for Qwen3.5-27B via Inspect's hf provider (single clean path).

Qwen tool-calling is Inspect-native (input formatting + output parsing), so no vLLM and no
custom parser. Runs harmful (test_public) then benign, official limits, gpt-4o judge on real
OpenAI. Produces the four standard metrics. Greedy decoding (transformers 5.x rejects
temperature=0.0, so we omit it in the config override).
"""
import os
from inspect_ai import eval as inspect_eval
from inspect_ai.model import get_model, GenerateConfig
from inspect_evals.agentharm.agentharm import agentharm, agentharm_benign

MID = os.environ.get("QWEN35_27B_MODEL", "Qwen/Qwen3.5-27B")
# Qwen3.5 is a reasoning model; disable thinking for a clean baseline vs the non-reasoning 3B
# (override with ENABLE_THINKING=1). transformers 5.x rejects temperature=0.0 -> omit (greedy).
ENABLE_THINKING = os.environ.get("ENABLE_THINKING", "0") == "1"
CFG = GenerateConfig(max_tokens=4096, max_connections=4, max_retries=3, seed=0)
model = get_model(f"hf/{MID}", device="cuda:0", enable_thinking=ENABLE_THINKING)

import sys
LIMIT = int(os.environ.get("LIMIT", "0")) or None  # LIMIT=2 for a mini validation


def run(task, log_dir, label):
    task.config = CFG
    logs = inspect_eval(task, model=model, log_dir=log_dir, limit=LIMIT)
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
        # sanity: did the agent make tool calls?
        n_tool = sum(
            len(msg.tool_calls)
            for samp in (log.samples or [])
            for msg in samp.messages
            if getattr(msg, "role", None) == "assistant" and getattr(msg, "tool_calls", None)
        )
        print(f"  [sanity] assistant tool_calls: {n_tool}")
    else:
        print(f"  ERROR: {getattr(log, 'error', None)}")
    return log.status, metrics


print(">>> HARMFUL")
hs, hm = run(agentharm(split="test_public"), "./logs_qwen35_harmful", "HARMFUL")
print(">>> BENIGN")
bs, bm = run(agentharm_benign(split="test_public"), "./logs_qwen35_benign", "BENIGN")

print("=== QWEN3.5-27B FOUR STANDARD METRICS ===")
print(f"  Harm Score              : {hm.get('avg_score', float('nan')):.4f}")
print(f"  Refusals                : {hm.get('avg_refusals', float('nan')):.4f}")
print(f"  Non-refusal Harm Score  : {hm.get('avg_score_non_refusals', float('nan')):.4f}")
print(f"  Benign Non-refusal Score: {bm.get('avg_score_non_refusals', float('nan')):.4f}")
print("ALL_DONE")
