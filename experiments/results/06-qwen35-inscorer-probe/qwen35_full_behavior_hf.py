"""FULL official AgentHarm behavioural eval for Qwen3.5-27B through Inspect's **hf** provider.

This is the run the parser fix makes possible: previously Inspect's hf handler turned every Qwen3.5
tool call into function="unknown", so the behavioural baseline had to go through vLLM. With
qwen35_inspect_patch installed the hf provider drives the agent loop in-process.

Same official task + limits as the vLLM baseline (agentharm / agentharm_benign, test_public,
max_tokens=4096, official message limit) so the four standard metrics are comparable.

NOTE: no hidden_states here — storing per-token hidden states across 65 layers for 4096-token
generations would blow GPU memory. The probe read is a separate pass (qwen35_full_probe_pass.py).

  source /opt/latent-evasion-mars/.env      # OPENAI_API_KEY for the judge
  export HF_HOME=/workspace/.cache/huggingface PYTHONPATH=/opt/latent-evasion-mars
  python3 experiments/qwen35_full_behavior_hf.py --out /opt/qwen35_full_behavior_hf.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen35_inspect_patch  # noqa: E402

from inspect_ai import eval as inspect_eval  # noqa: E402
from inspect_ai.model import GenerateConfig, get_model  # noqa: E402
from inspect_evals.agentharm.agentharm import agentharm, agentharm_benign  # noqa: E402


def run(task, model, cfg, log_dir, label, limit):
    task.config = cfg
    log = inspect_eval(task, model=model, log_dir=log_dir, limit=limit)[0]
    print(f"=== {label} STATUS: {log.status} ===", flush=True)
    metrics = {}
    n_tool = n_unknown = 0
    if log.status == "success":
        for s in (log.results.scores if log.results else []):
            for name, m in (s.metrics or {}).items():
                metrics[name] = m.value
        for k in ("avg_score", "avg_refusals", "avg_score_non_refusals"):
            if k in metrics:
                print(f"  {label}.{k}: {metrics[k]:.4f}", flush=True)
        for samp in (log.samples or []):
            for msg in samp.messages:
                for tc in (getattr(msg, "tool_calls", None) or []):
                    n_tool += 1
                    if tc.function in ("unknown", None) or getattr(tc, "parse_error", None):
                        n_unknown += 1
        print(f"  [parser sanity] tool_calls {n_tool}  unknown/parse_error {n_unknown}", flush=True)
    else:
        print(f"  ERROR: {getattr(log, 'error', None)}", flush=True)
    return log.status, metrics, n_tool, n_unknown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("QWEN35_27B_MODEL", "Qwen/Qwen3.5-27B"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-connections", type=int, default=4)
    ap.add_argument("--out", default="/opt/qwen35_full_behavior_hf.json")
    args = ap.parse_args()
    limit = args.limit or None

    qwen35_inspect_patch.install()
    print("installed Qwen3.5 tool-call parser patch", flush=True)

    # no hidden_states: this pass generates long agentic rollouts
    model = get_model(f"hf/{args.model}", device=args.device, enable_thinking=False)
    cfg = GenerateConfig(max_tokens=4096, max_connections=args.max_connections)

    print(">>> HARMFUL", flush=True)
    hs, hm, ht, hu = run(agentharm(split="test_public"), model, cfg,
                         "./logs_qwen35hf_harmful", "HARMFUL", limit)
    print(">>> BENIGN", flush=True)
    bs, bm, bt, bu = run(agentharm_benign(split="test_public"), model, cfg,
                         "./logs_qwen35hf_benign", "BENIGN", limit)

    print("\n=== QWEN3.5-27B (hf provider, patched) FOUR STANDARD METRICS ===")
    print(f"  Harm Score              : {hm.get('avg_score', float('nan')):.4f}")
    print(f"  Refusals                : {hm.get('avg_refusals', float('nan')):.4f}")
    print(f"  Non-refusal Harm Score  : {hm.get('avg_score_non_refusals', float('nan')):.4f}")
    print(f"  Benign Non-refusal Score: {bm.get('avg_score_non_refusals', float('nan')):.4f}")
    print(f"  parser: harmful {ht} calls / {hu} unknown | benign {bt} calls / {bu} unknown")

    json.dump({"model": args.model, "harmful_status": hs, "benign_status": bs,
               "harmful_metrics": hm, "benign_metrics": bm,
               "tool_calls": {"harmful": ht, "harmful_unknown": hu,
                              "benign": bt, "benign_unknown": bu}},
              open(args.out, "w"), indent=2)
    print(f"saved {args.out}\nALL_DONE")


if __name__ == "__main__":
    main()
