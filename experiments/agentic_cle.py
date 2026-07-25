"""Agentic CLE: run the Controlled Latent-space Evasion attack inside a live AgentHarm eval.

The CLE paper evaluates on single-turn completions. This runs the same intervention while the
model is acting as a tool-using agent, scored by AgentHarm's official `combined_scorer`, so the
result is expressed in the four standard metrics and is directly comparable to the untouched
baseline in experiments/BASELINES.md.

Why this has to run in-process: CLE's intervention is a forward hook on the decoder blocks, so
vLLM cannot host it. That path only became usable once the Qwen3.5 tool-call parser gap was fixed
(qwen35_inspect_patch).

The intervention itself lives in the `cle` Inspect provider (experiments/cle_provider.py) -- Path B
from the 21 Jul harness note. This file is the experiment runner: it chooses the arms, drives the
control-vs-attacked comparison and reports the four standard metrics. Both CLE-P and CLE-A go
through the same provider, so the two attacks differ in their schedule and nothing else.

CLE-P (`utils/hooks.projection_hook`) is active during prefill AND decoding, exactly as in
cle-p.py:

    score = w·h + b + margin
    h'    = h - beta * (score / ||w||^2) * w        # beta=1 lands the score at exactly -margin

CONTROL: `--beta 0` makes the hook an exact no-op (h' = h) while keeping every other code path
identical, so the control and attacked runs differ in one number and nothing else. Always run it
-- comparing against the vLLM baseline instead would confound the attack with a provider change.

MARGIN: `--margin` takes a raw score value, as CLE does. But raw score scale is not comparable
across layers (activation norms grow with depth), and nobody has run Optuna for this model, so a
number borrowed from another model's search is meaningless here. `--margin-quantile Q` instead
derives a per-layer margin from held-out-of-test VAL activations: it targets the Q-quantile of the
*harmless* score distribution at each layer, i.e. "land harmful activations where harmless ones
actually sit." Q=0.5 aims at the median harmless prompt.

  python3 experiments/agentic_cle.py --probe <canonical stem> --layers 20-30 \
      --margin-quantile 0.5 --val-acts <val_acts.npz> --split val --out cle_agentic.json

  # CLE-A, the paper's cadence, with the staleness trace
  python3 experiments/agentic_cle.py --attack cle-a --recalibrate sample --trace-scores ...

  # correctness checks before trusting any CLE-A number (design note §7)
  python3 experiments/agentic_cle.py --attack cle-a --selftest ...
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "results", "06-qwen35-inscorer-probe"))

import cle_provider                                               # noqa: E402
import qwen35_inspect_patch                                       # noqa: E402

from inspect_ai import eval as inspect_eval                       # noqa: E402
from inspect_ai.model import GenerateConfig, get_model            # noqa: E402
from inspect_evals.agentharm.agentharm import agentharm, agentharm_benign   # noqa: E402

FOUR = ("avg_score", "avg_refusals", "avg_score_non_refusals", "avg_full_score")





def run(task, model, cfg, log_dir, label, limit):
    task.config = cfg
    log = inspect_eval(task, model=model, log_dir=log_dir, limit=limit)[0]
    print(f"=== {label}: {log.status} ===", flush=True)
    if log.status != "success":
        print(f"  ERROR {getattr(log, 'error', None)}", flush=True)
        return {}
    m = {k: float(v.value)
         for s in (log.results.scores if log.results else [])
         for k, v in (s.metrics or {}).items() if k in FOUR}
    n_tool = n_bad = 0
    for samp in (log.samples or []):
        for msg in samp.messages:
            for tc in (getattr(msg, "tool_calls", None) or []):
                n_tool += 1
                if tc.function in ("unknown", None) or getattr(tc, "parse_error", None):
                    n_bad += 1
    print("  " + "  ".join(f"{k} {m[k]:.4f}" for k in FOUR if k in m), flush=True)
    print(f"  [parser sanity] tool_calls {n_tool}  unknown/parse_error {n_bad}", flush=True)
    return m


def four_metrics(h, b):
    return {"harm_score": h.get("avg_score"), "refusals": h.get("avg_refusals"),
            "non_refusal_harm_score": h.get("avg_score_non_refusals"),
            "benign_non_refusal_score": b.get("avg_score_non_refusals")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("QWEN35_27B_MODEL", "Qwen/Qwen3.5-27B"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--probe", required=True, help="canonical probe stem (probe_io.py)")
    ap.add_argument("--layers", default="20-30", help="'20-30' or '22,24,26'")
    ap.add_argument("--beta", type=float, default=1.0, help="0 = exact no-op control")
    ap.add_argument("--margin", type=float, default=None, help="raw score margin (CLE-style)")
    ap.add_argument("--margin-quantile", type=float, default=None,
                    help="derive per-layer margin from the val harmless score distribution")
    ap.add_argument("--val-acts", default=None, help="required with --margin-quantile")
    ap.add_argument("--split", default="val", choices=["val", "test_public"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--max-connections", type=int, default=1)
    ap.add_argument("--control", action="store_true", help="also run the beta=0 control")
    ap.add_argument("--attack", default="cle-p", choices=["cle-p", "cle-a"],
                    help="cle-p recomputes the displacement every position; cle-a measures it once "
                         "at the last prompt token and adds that frozen vector everywhere")
    ap.add_argument("--recalibrate", default="sample", choices=["sample", "turn"],
                    help="cle-a only. 'sample' freezes the delta for the whole rollout (the "
                         "paper's semantics); 'turn' recalibrates each generate call (ours, not "
                         "the paper's -- never report it as the CLE-A baseline)")
    ap.add_argument("--trace-scores", action="store_true",
                    help="cle-a only. Record w.h+b at the last position of each generate call, so "
                         "staleness under 'sample' cadence is measured rather than inferred")
    ap.add_argument("--selftest", action="store_true",
                    help="run the cle-a correctness checks and exit without an eval")
    ap.add_argument("--log-dir", default="./logs_agentic_cle")
    ap.add_argument("--out", default="./agentic_cle.json")
    args = ap.parse_args()
    limit = args.limit or None

    if (args.margin is None) == (args.margin_quantile is None):
        raise SystemExit("pass exactly one of --margin or --margin-quantile")
    if args.margin_quantile is not None and not args.val_acts:
        raise SystemExit("--margin-quantile needs --val-acts (val only; test stays held out)")

    qwen35_inspect_patch.install()
    # The intervention lives in the `cle` provider (experiments/cle_provider.py), which is Path B
    # from the 21 Jul harness note: Inspect owns the run and we own generation, rather than
    # reaching under the framework to patch the model.
    #
    # do_sample=False is a MODEL ARG, not a GenerateConfig field. Inspect's hf provider defaults
    # do_sample to True, so an unspecified run SAMPLES -- while the frozen vLLM baseline is greedy
    # (temperature 0). Setting temperature=0.0 instead does not work: transformers rejects it
    # ("has to be a strictly positive float ... set do_sample=False"), which killed a test_public
    # run after 1.5 h. Greedy also removes sampling noise from the control-vs-attacked comparison.
    margin_args = ({"cle_margin_quantile": args.margin_quantile, "cle_val_acts": args.val_acts}
                   if args.margin_quantile is not None else {"cle_margins": args.margin})
    model = get_model(f"cle/{args.model}", device=args.device, enable_thinking=False,
                      do_sample=False,
                      cle_attack=args.attack, cle_probe=args.probe, cle_layers=args.layers,
                      cle_beta=args.beta, cle_recalibrate=args.recalibrate,
                      cle_trace=args.trace_scores, **margin_args)
    # No temperature here: with do_sample=False it is unused, and transformers errors on 0.0.
    cfg = GenerateConfig(max_tokens=args.max_tokens, max_connections=args.max_connections)
    api = model.api
    sel, margins = api.layers, api.margins

    print(f"model {args.model} | {len(api.blocks)} blocks | {args.attack} on {sel}")
    print(f"margin: {api.margin_mode}")
    for l in sel:
        print(f"    L{l:02d} margin {margins[l]:+.4f}")

    if args.selftest:
        r = api.selftest()
        print("\n=== CLE self-test (cle_a_agentic_design.md §7) ===")
        for k, v in r.items():
            print(f"  {k:34s} {v}")
        raise SystemExit(0 if r["pass"] else 1)

    results = {"model": args.model, "probe": args.probe, "layers": sel, "beta": args.beta,
               "margin_mode": api.margin_mode, "margins": margins, "split": args.split,
               "attack": args.attack, "provider": "cle"}
    if args.attack == "cle-a":
        # 'turn' is ours, not the paper's. Record which was used so a number can never be
        # reported as the CLE-A baseline when it is the middle point we invented.
        results["recalibrate"] = args.recalibrate

    def go(beta, tag):
        api.set_beta(beta)
        api.stats = cle_provider.CleStats()
        print(f"\n>>> {tag} (beta={beta}) HARMFUL", flush=True)
        h = run(agentharm(split=args.split), model, cfg, f"{args.log_dir}/{tag}_harmful",
                f"{tag}.harmful", limit)
        print(f">>> {tag} (beta={beta}) BENIGN", flush=True)
        b = run(agentharm_benign(split=args.split), model, cfg, f"{args.log_dir}/{tag}_benign",
                f"{tag}.benign", limit)
        print(f"  [cle] {api.stats.summary()}", flush=True)
        results.setdefault("cle_stats", {})[tag] = api.stats.summary()
        if args.trace_scores and api.stats.trace:
            results.setdefault("cle_trace", {})[tag] = api.stats.trace
        return four_metrics(h, b)

    if args.control:
        # beta=0 => h' = h exactly; same code path, same hooks installed, no perturbation.
        results["control"] = go(0.0, "control")
    results["attacked"] = go(args.beta, "attacked")

    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nsaved {args.out}")
    print("\n=== AGENTIC CLE — four standard AgentHarm metrics ===")
    ctrl = results.get("control")
    for k in ("harm_score", "refusals", "non_refusal_harm_score", "benign_non_refusal_score"):
        a = results["attacked"].get(k)
        if a is None:
            continue
        if ctrl and ctrl.get(k) is not None:
            print(f"  {k:26s} control {ctrl[k]:.4f}  ->  attacked {a:.4f}   ({a - ctrl[k]:+.4f})")
        else:
            print(f"  {k:26s} attacked {a:.4f}")
    print("\nReading: the attack works if Refusals falls while Non-refusal Harm holds up. A drop in")
    print("Benign Non-refusal means the intervention is damaging general capability, not just safety.")


if __name__ == "__main__":
    main()
