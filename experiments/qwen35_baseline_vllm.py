"""AgentHarm baseline for Qwen3.5-27B via vLLM (OpenAI endpoint) through Inspect.

vLLM serves Qwen3.5 with --tool-call-parser qwen3_xml (its native <function=...> format),
which Inspect's hf provider could NOT parse. strict_tools=False stops Inspect stamping
strict:true on the tool schemas (which fights vLLM guided decoding). Judge stays on real OpenAI.

LIMIT=2 for a mini validation; unset/0 for the full run. Reports the four standard metrics
and a tool-call sanity count (should be >0 and NOT 'unknown' this time).
"""
import os
from inspect_ai import eval as inspect_eval
from inspect_ai.model import get_model, GenerateConfig
from inspect_evals.agentharm.agentharm import agentharm, agentharm_benign

MODEL = "openai-api/vllm/qwen-3.5-27b"      # reads VLLM_BASE_URL / VLLM_API_KEY
LIMIT = int(os.environ.get("LIMIT", "0")) or None
# Qwen3.5 is a reasoning model; disable thinking for a clean (and much faster) baseline vs the
# non-reasoning 3B. vLLM honors chat_template_kwargs.enable_thinking via extra_body.
CFG = GenerateConfig(temperature=0.0, max_tokens=4096, max_connections=8, max_retries=3, seed=0,
                     extra_body={"chat_template_kwargs": {"enable_thinking": False}})
model = get_model(MODEL, strict_tools=False)


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
        # sanity: real tool calls (not parse-errored 'unknown')
        n_tool = n_unknown = 0
        for samp in (log.samples or []):
            for msg in samp.messages:
                for tc in (getattr(msg, "tool_calls", None) or []):
                    n_tool += 1
                    if tc.function in ("unknown", None) or getattr(tc, "parse_error", None):
                        n_unknown += 1
        print(f"  [sanity] tool_calls: {n_tool}  (unknown/parse_error: {n_unknown})")
    else:
        print(f"  ERROR: {getattr(log, 'error', None)}")
    return log.status, metrics


print(">>> HARMFUL")
hs, hm = run(agentharm(split="test_public"), "./logs_qwen35v_harmful", "HARMFUL")
print(">>> BENIGN")
bs, bm = run(agentharm_benign(split="test_public"), "./logs_qwen35v_benign", "BENIGN")

print("=== QWEN3.5-27B (vLLM) FOUR STANDARD METRICS ===")
print(f"  Harm Score              : {hm.get('avg_score', float('nan')):.4f}")
print(f"  Refusals                : {hm.get('avg_refusals', float('nan')):.4f}")
print(f"  Non-refusal Harm Score  : {hm.get('avg_score_non_refusals', float('nan')):.4f}")
print(f"  Benign Non-refusal Score: {bm.get('avg_score_non_refusals', float('nan')):.4f}")
print("ALL_DONE")
