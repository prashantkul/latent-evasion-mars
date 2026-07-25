"""Measure decode throughput for a model on whatever box this runs on.

Where a run should live is a throughput question, not a capacity one. The DGX Spark has 128 GB of
unified memory and already caches Qwen3.5-27B, so capacity never binds -- but agentic decoding is
memory-bandwidth-bound, and unified LPDDR5X is not HBM. This measures the thing that actually
decides: greedy tokens/sec, plus model load time.

Report both. Load is paid once per process; decode is paid per generated token, and an AgentHarm
rollout generates thousands.

  python3 experiments/bench_decode.py --model Qwen/Qwen3.5-27B --new-tokens 128
"""
import argparse
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = ("You are a helpful assistant with access to tools. Explain, in detail, how you would "
          "plan a multi-step research task and which tools you would call at each step.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("QWEN35_27B_MODEL", "Qwen/Qwen3.5-27B"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--new-tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=8, help="tokens to generate before timing")
    args = ap.parse_args()

    print(f"host      {os.uname().nodename} ({os.uname().machine})")
    print(f"torch     {torch.__version__}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"gpu       {p.name}  {p.total_memory / 1e9:.0f} GB reported")

    t0 = time.monotonic()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map=args.device)
    model.eval()
    load_s = time.monotonic() - t0
    print(f"load      {load_s:.1f} s")

    ids = tok(PROMPT, return_tensors="pt").to(model.device)
    gen = dict(do_sample=False, pad_token_id=tok.eos_token_id)

    # Warm up first: the first pass pays kernel autotuning and allocator growth, which would
    # otherwise be charged to the measured run and understate steady-state throughput.
    with torch.no_grad():
        model.generate(**ids, max_new_tokens=args.warmup, **gen)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.monotonic()
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=args.new_tokens, **gen)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.monotonic() - t0

    n = out.shape[1] - ids["input_ids"].shape[1]
    print(f"decode    {n} tokens in {dt:.2f} s  ->  {n / dt:.2f} tok/s")
    print(f"\nAgentHarm rough scaling (generation only, excludes tool + judge latency):")
    for label, toks in (("one ~600-token generation", 600), ("a ~6-step rollout, ~3600 tok", 3600)):
        print(f"  {label:32s} {toks / (n / dt) / 60:6.1f} min")


if __name__ == "__main__":
    main()
