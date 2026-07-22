"""Better probe metric: continuous probe score per prompt vs behavioural labels.

Probe score = average of (w.h + b) across layers (a few aggregations tried).
Label 1 = refusal (y=1 refusal, 0 else)  -> AUC(score -> refusal): does the internal
          signal predict the refusal decision? (threshold-free; replaces firing fraction)
Label 2 = harm/completion                -> AUC(score -> harm>=0.5) + Spearman(score, harm)

Re-runs the fast probe pass for full per-layer scores; reuses the existing M2f graded log
(logs_full/...11-07-09...) for harm + refusal labels. Behavior-clustered bootstrap CIs.
"""
import glob
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from inspect_ai import Task, eval as inspect_eval
from inspect_ai.log import read_eval_log
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import TaskState

from inspect_evals.agentharm.agents.agent import get_agent
from inspect_evals.agentharm.utils import (
    filter_dataset, get_benchmark_tools, load_dataset, setup_tools_from_metadata,
)

PROBE_DIR = os.path.expanduser("~/latent-evasion/dataset/representations/llama32-3b/train_svm")
GRADED_LOG = sorted(glob.glob(os.path.expanduser("~/latent-evasion/logs_full/*.eval")),
                    key=os.path.getsize)[-1]   # the big one = M2f graded pass
B = 5000

_W, _B = {}, {}
for p in sorted(glob.glob(os.path.join(PROBE_DIR, "svm_layer*.pt"))):
    o = torch.load(p, map_location="cpu", weights_only=False)
    l = int(o["layer_idx"]); _W[l] = o["w"].float(); _B[l] = float(o["b"])
LAYERS = sorted(_W)
_WMAT = torch.stack([_W[l] for l in LAYERS])
_BVEC = torch.tensor([_B[l] for l in LAYERS])


def _scores(hs):
    step0 = hs[0]
    h = torch.stack([step0[l + 1][0, -1, :].float().cpu() for l in LAYERS])
    return ((_WMAT * h).sum(-1) + _BVEC).numpy()   # (28,) per-layer w.h+b


@scorer(metrics=[mean()])
def probe_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        meta = (state.output.metadata or {}) if state.output else {}
        hs = meta.get("hidden_states")
        m = state.metadata or {}
        s = _scores(hs).tolist() if hs else None
        return Score(value=(float(np.mean([s[l] for l in (14, 18, 22, 26)])) if s else 0.0),
                     metadata={"scores": s, "id_original": m.get("id_original"), "category": m.get("category")})
    return score


def probe_task():
    ds = filter_dataset(load_dataset("harmful", "test_public", False), [], None, None)
    plan = [setup_tools_from_metadata("harmful", get_benchmark_tools("harmful"), 0),
            get_agent(agent="default", tool_choice="none")]
    return Task(dataset=ds, solver=plan, scorer=probe_scorer(),
                config=GenerateConfig(max_tokens=8, max_connections=1))


def read_labels(path):
    log = read_eval_log(path)
    out = {}
    for smp in (log.samples or []):
        for sc in (smp.scores or {}).values():
            v = sc.value
            if isinstance(v, dict) and "score" in v:
                out[str(smp.id)] = (float(v["score"]), float(v.get("refusal", 0.0)))
    return out


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


if __name__ == "__main__":
    mid = os.environ.get("LLAMA32_3B_MODEL", "unsloth/Llama-3.2-3B-Instruct")
    model = get_model(f"hf/{mid}", hidden_states=True, device="cuda:0")

    print("probe pass ...")
    p1 = inspect_eval(probe_task(), model=model, log_dir="./logs_metrics")
    labels = read_labels(GRADED_LOG)

    recs = []
    for smp in (p1[0].samples or []):
        m = (smp.scores["probe_scorer"].metadata if smp.scores else None) or {}
        sid = str(smp.id)
        if m.get("scores") and sid in labels:
            harm, refusal = labels[sid]
            recs.append((str(m.get("id_original")), np.array(m["scores"]), harm, refusal, str(m.get("category"))))
    ios = np.array([r[0] for r in recs])
    S = np.stack([r[1] for r in recs])           # (N, 28) per-layer scores
    harm = np.array([r[2] for r in recs])
    ref = np.array([r[3] for r in recs]).astype(int)
    N = len(recs)
    print(f"\nN={N} rows, {len(set(ios))} behaviors")

    # score aggregations (one scalar per prompt)
    aggs = {
        "mean_good(14,18,22,26)": S[:, [14, 18, 22, 26]].mean(1),
        "mean_mid(10-27)":        S[:, 10:].mean(1),
        "mean_all(0-27)":         S.mean(1),
        "layer_18":               S[:, 18],
    }
    # also: best single layer by refusal-AUC
    per_layer_auc = [roc_auc_score(ref, S[:, l]) for l in range(28)]
    best_l = int(np.argmax(per_layer_auc))
    aggs[f"best_layer_{best_l}"] = S[:, best_l]

    behaviors = np.unique(ios)
    idx_by = {b: np.where(ios == b)[0] for b in behaviors}

    def cluster_ci(fn, seed=0):
        rng = np.random.default_rng(seed); vals = []
        for _ in range(B):
            idx = np.concatenate([idx_by[b] for b in rng.choice(behaviors, len(behaviors), replace=True)])
            try:
                v = fn(idx)
            except Exception:
                v = np.nan
            if v is not None and not np.isnan(v):
                vals.append(v)
        return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

    harm_bin = (harm >= 0.5).astype(int)

    print("\n=== AUC(probe score -> REFUSAL)  [y=1 refusal] ===")
    print("  aggregation                 AUC    95% CI (clustered)")
    best = None
    for name, sc in aggs.items():
        auc = roc_auc_score(ref, sc)
        lo, hi = cluster_ci(lambda ix, sc=sc: roc_auc_score(ref[ix], sc[ix]) if len(set(ref[ix])) > 1 else np.nan)
        print(f"  {name:<26} {auc:.3f}  [{lo:.3f}, {hi:.3f}]")
        if best is None or auc > best[1]:
            best = (name, auc, sc)

    bs = best[2]
    print(f"\n=== using best aggregation: {best[0]} ===")
    auc_h = roc_auc_score(harm_bin, bs) if len(set(harm_bin)) > 1 else float("nan")
    lo, hi = cluster_ci(lambda ix: roc_auc_score(harm_bin[ix], bs[ix]) if len(set(harm_bin[ix])) > 1 else np.nan)
    sp = spearman(bs, harm)
    print(f"  AUC(score -> harm>=0.5)     {auc_h:.3f}  [{lo:.3f}, {hi:.3f}]")
    print(f"  Spearman(score, harm score) {sp:+.3f}")
    print(f"  mean score | refused={bs[ref == 1].mean():+.3f}  not-refused={bs[ref == 0].mean():+.3f}")

    import json
    json.dump({
        "N": N, "n_behaviors": len(behaviors),
        "refusal_auc": {k: {"auc": float(roc_auc_score(ref, v)),
                            "ci": list(cluster_ci(lambda ix, v=v: roc_auc_score(ref[ix], v[ix]) if len(set(ref[ix])) > 1 else np.nan))}
                        for k, v in aggs.items()},
        "best_agg": best[0],
        "harm_auc": {"auc": auc_h, "ci": [lo, hi]},
        "harm_spearman": sp,
        "records": [{"io": r[0], "score_good": float(r[1][[14, 18, 22, 26]].mean()),
                     "harm": float(r[2]), "refused": int(r[3]), "cat": r[4]} for r in recs],
    }, open("probe_metrics.json", "w"), indent=2)
    print("\nsaved probe_metrics.json")
