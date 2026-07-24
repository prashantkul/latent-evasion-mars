"""Agentic CLE: run the Controlled Latent-space Evasion attack inside a live AgentHarm eval.

The CLE paper evaluates on single-turn completions. This runs the same intervention while the
model is acting as a tool-using agent, scored by AgentHarm's official `combined_scorer`, so the
result is expressed in the four standard metrics and is directly comparable to the untouched
baseline in experiments/BASELINES.md.

Why this has to go through Inspect's **hf** provider: CLE's intervention is a forward hook on the
decoder blocks, so the model must run in-process. vLLM cannot host it. That path only became
usable once the Qwen3.5 tool-call parser gap was fixed (qwen35_inspect_patch).

The intervention is CLE-P (`utils/hooks.projection_hook`), active during prefill AND decoding,
exactly as in cle-p.py:

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
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "results", "06-qwen35-inscorer-probe"))

import qwen35_inspect_patch                                       # noqa: E402
from probe_adapter import to_probe_dict                           # noqa: E402
from probe_io import load_probe                                   # noqa: E402
from utils.hooks import projection_hook, remove_hooks             # noqa: E402

from inspect_ai import eval as inspect_eval                       # noqa: E402
from inspect_ai.model import GenerateConfig, get_model            # noqa: E402
from inspect_evals.agentharm.agentharm import agentharm, agentharm_benign   # noqa: E402

FOUR = ("avg_score", "avg_refusals", "avg_score_non_refusals", "avg_full_score")


def parse_layers(spec, n_layers):
    """'20-30' -> [20..30]; '22,24' -> [22, 24]. Matches utils/args.parse_layers_arg."""
    sel = (list(range(int(spec.split("-")[0]), int(spec.split("-")[1]) + 1))
           if "-" in spec else [int(x) for x in spec.split(",")])
    bad = [l for l in sel if not 0 <= l < n_layers]
    if bad:
        raise SystemExit(f"layers out of range for a {n_layers}-layer model: {bad}")
    return sel


def geometric_margins(val_acts, probe_stem, layers, q):
    """Per-layer margin that lands a pushed activation at the q-quantile of the HARMLESS scores.

    With beta=1 the post-intervention score is exactly -margin, so margin = -quantile(harmless).
    Derived from val activations only -- the test split stays held out.
    """
    w, b, idx, _ = load_probe(probe_stem)
    z = np.load(val_acts, allow_pickle=True)
    X, y = z["X"], z["y"].astype(int)
    pos = {int(l): n for n, l in enumerate(idx)}
    out = {}
    for l in layers:
        s_harmless = X[y == 0, l, :] @ w[pos[l]] + b[pos[l]]
        out[l] = float(-np.quantile(s_harmless, q))
    return out


def hf_layers(model):
    """The decoder blocks of the model Inspect is actually driving."""
    hf = getattr(model.api, "model", None)
    if hf is None:
        raise SystemExit("could not reach the underlying HF model — is this the hf provider?")
    for path in (("model", "layers"), ("transformer", "h"), ("layers",)):
        obj = hf
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise SystemExit(f"could not locate decoder layers on {type(hf).__name__}")


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
    ap.add_argument("--log-dir", default="./logs_agentic_cle")
    ap.add_argument("--out", default="./agentic_cle.json")
    args = ap.parse_args()
    limit = args.limit or None

    if (args.margin is None) == (args.margin_quantile is None):
        raise SystemExit("pass exactly one of --margin or --margin-quantile")
    if args.margin_quantile is not None and not args.val_acts:
        raise SystemExit("--margin-quantile needs --val-acts (val only; test stays held out)")

    qwen35_inspect_patch.install()
    model = get_model(f"hf/{args.model}", device=args.device, enable_thinking=False)
    # Pin decoding explicitly to match the frozen vLLM run (temperature 0.0, seed 0). Leaving
    # these unset lets the model's own generation_config supply defaults -- Qwen ships sampling
    # defaults -- so an unpinned run may silently sample while the baseline it is read against
    # did not. Control and attacked share whatever is set here, so this does not affect an
    # already-collected comparison; it removes a confound from the next one.
    cfg = GenerateConfig(max_tokens=args.max_tokens, max_connections=args.max_connections,
                         temperature=0.0, seed=0)
    layers_mod = hf_layers(model)
    sel = parse_layers(args.layers, len(layers_mod))

    probes = to_probe_dict(args.probe, sel, device=args.device)
    if args.margin_quantile is not None:
        margins = geometric_margins(args.val_acts, args.probe, sel, args.margin_quantile)
        how = f"quantile {args.margin_quantile} of val harmless scores (per layer)"
    else:
        margins = {l: args.margin for l in sel}
        how = f"fixed {args.margin}"
    print(f"model {args.model} | {len(layers_mod)} blocks | intervening on {sel}")
    print(f"margin: {how}")
    for l in sel:
        print(f"    L{l:02d} margin {margins[l]:+.4f}")

    results = {"model": args.model, "probe": args.probe, "layers": sel, "beta": args.beta,
               "margin_mode": how, "margins": margins, "split": args.split,
               "intervention": "CLE-P projection hook (prefill + decoding)"}

    def go(beta, tag):
        handles = [layers_mod[l].register_forward_hook(
            projection_hook(probes[l]["w"], probes[l]["b"], beta, margins[l])) for l in sel]
        try:
            print(f"\n>>> {tag} (beta={beta}) HARMFUL", flush=True)
            h = run(agentharm(split=args.split), model, cfg, f"{args.log_dir}/{tag}_harmful",
                    f"{tag}.harmful", limit)
            print(f">>> {tag} (beta={beta}) BENIGN", flush=True)
            b = run(agentharm_benign(split=args.split), model, cfg, f"{args.log_dir}/{tag}_benign",
                    f"{tag}.benign", limit)
        finally:
            remove_hooks(handles)
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
