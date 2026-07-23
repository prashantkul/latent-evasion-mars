"""Agentic-context activation extraction on the AgentHarm VALIDATION split (32 harmful + 32 benign),
using the repo's existing extractor (models/language_models.py::get_representations).

Approach (b): keep get_representations, but render each sample's AgentHarm tools into the prompt so
the post-instruction (-1) read happens in the *agentic* distribution — not a bare chat prompt.
`--agentic` (default) injects tools + thinking-off; `--no-agentic` reproduces the plain single-turn
read for an A/B on the context effect.

Fits a per-layer mean-difference refusal direction on the harmful(1)/benign(0) contrast and reports
5-fold CV AUC per layer (which layers separate harmful vs benign in this context, and the best one).
Saves the raw activations + directions so the behaviour join (P(fire | complied)) can be done later
without re-running the GPU pass.

Run on the GPU box (needs a forward pass through the 27B). Example:
  PYTHONPATH=$HOME/latent-evasion .venv/bin/python experiments/qwen35_agentic_val_extract.py \
      --model Qwen/Qwen3.5-27B --device cuda:0 --out ./qwen35_agentic_val.json
"""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from inspect_ai.tool import ToolDef
from inspect_evals.agentharm.benchmark import benign_tools, harmful_tools
from inspect_evals.agentharm.utils import filter_dataset, load_dataset

from models import Qwen35_27b


def _prune(x):
    """Drop null fields so the rendered tool schema stays minimal/standard."""
    if isinstance(x, dict):
        return {k: _prune(v) for k, v in x.items() if v is not None}
    if isinstance(x, list):
        return [_prune(v) for v in x]
    return x


def _tool_schema(name, module):
    td = ToolDef(getattr(module, name)())
    params = td.parameters
    params = params.model_dump() if hasattr(params, "model_dump") else params
    return {
        "type": "function",
        "function": {"name": td.name, "description": td.description, "parameters": _prune(params)},
    }


def _sample_tools(sample, module):
    schemas = []
    for name in (sample.metadata or {}).get("target_functions", []):
        try:
            schemas.append(_tool_schema(name, module))
        except Exception as exc:  # a missing/renamed tool should not abort the whole run
            print(f"  [warn] tool {name!r} unavailable: {exc}")
    return schemas


def load_split(task_name, split="val"):
    ds = filter_dataset(load_dataset(task_name, split, False), [], None, None)
    module = harmful_tools if task_name == "harmful" else benign_tools
    rows = []
    for s in ds:
        text = s.input if isinstance(s.input, str) else str(s.input)
        rows.append((str(s.id), text, _sample_tools(s, module)))
    return rows


def extract(model, rows, agentic):
    feats, ids = [], []
    for i, (sid, text, tools) in enumerate(rows):
        kwargs = {}
        if agentic:
            kwargs = {"tools": tools, "enable_thinking": False}
        reps = model.get_representations(text, token_pos=-1, **kwargs)  # (1, L, H)
        feats.append(reps.to(dtype=torch.float32).cpu().numpy()[0])
        ids.append(sid)
        if (i + 1) % 8 == 0:
            print(f"  ... {i + 1}/{len(rows)}", flush=True)
    return np.stack(feats), ids  # (N, L, H), [ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("QWEN35_27B_MODEL", "Qwen/Qwen3.5-27B"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--agentic", action=argparse.BooleanOptionalAction, default=True,
                    help="render AgentHarm tools + thinking-off into the prompt (agentic read)")
    ap.add_argument("--out", default="./qwen35_agentic_val.json")
    args = ap.parse_args()

    print(f"loading val (agentic={args.agentic}) ...", flush=True)
    harmful = load_split("harmful", "val")
    benign = load_split("benign", "val")
    print(f"val: {len(harmful)} harmful / {len(benign)} benign", flush=True)

    model = Qwen35_27b(model_name=args.model, device=args.device)

    print("extract harmful ...", flush=True)
    Xh, idh = extract(model, harmful, args.agentic)
    print("extract benign ...", flush=True)
    Xb, idb = extract(model, benign, args.agentic)

    X = np.concatenate([Xh, Xb])                       # (64, L, H)
    y = np.array([1] * len(Xh) + [0] * len(Xb))
    L = X.shape[1]
    print(f"activations {X.shape}  layers={L}", flush=True)

    # Persist the raw activations immediately — the GPU pass is the expensive part and the
    # analysis below must never be able to lose it.
    acts_path = args.out.replace(".json", "_acts.npz")
    np.savez(acts_path, X=X, y=y, ids=np.array(idh + idb))
    print(f"saved raw activations -> {acts_path}", flush=True)

    per_layer = []
    for li in range(L):
        s = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X[:, li], y):
            mu1 = X[tr][y[tr] == 1, li].mean(0)
            mu0 = X[tr][y[tr] == 0, li].mean(0)
            d = mu1 - mu0
            d /= (np.linalg.norm(d) + 1e-12)
            s[te] = X[te, li] @ d
        per_layer.append(float(roc_auc_score(y, s)))

    best = int(np.argmax(per_layer))                   # 0-based into hidden_states[1:]
    print("\nlayer | CV AUC(harmful vs benign)")
    for li, auc in enumerate(per_layer):
        mark = "  <- best" if li == best else ""
        print(f"  {li:>2}  | {auc:.3f}{mark}")
    print(f"\nbest layer (0-based over transformer blocks): {best}  AUC {per_layer[best]:.3f}")

    # final directions on all val, per layer, for downstream behaviour join
    dirs = np.zeros((L, X.shape[2]), dtype=np.float32)
    for li in range(L):
        mu1 = X[y == 1, li].mean(0)
        mu0 = X[y == 0, li].mean(0)
        d = mu1 - mu0
        dirs[li] = d / (np.linalg.norm(d) + 1e-12)

    np.savez(acts_path, X=X, y=y, dirs=dirs, ids=np.array(idh + idb))  # re-save with directions
    json.dump({"model": args.model, "agentic": args.agentic, "n_layers": int(L),
               "cv_auc_by_layer": per_layer, "best_layer": best,
               "best_auc": per_layer[best], "ids_harmful": idh, "ids_benign": idb,
               "acts_file": os.path.basename(acts_path)}, open(args.out, "w"), indent=2)
    print(f"\nsaved {args.out} and {acts_path}")


if __name__ == "__main__":
    main()
