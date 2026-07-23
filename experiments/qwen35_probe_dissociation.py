"""Qwen3.5-27B: agentic probe training + dissociation, single hf path (adapts agentic_probe.py).

Captures AGENTIC-CONTEXT activations via Inspect's hf provider (hidden_states=True,
max_connections=1 so the batch dim stays clean): each sample's tools are set up and the agent
is run with tool_choice="none" / max_tokens=8, and we read the residual stream at the
post-instruction token (last prompt token) at every transformer layer.

  1. Fit the refusal direction on the val contrast (32 harmful=1 / 32 benign=0), per layer:
     mean-difference direction + LinearSVC. Pick the best layer by held-out (CV) harmful/benign AUC.
  2. Read the behavioral outcome per test_public sample from the baseline graded log
     (refused / complied + harm score).
  3. Dissociation at the best layer: P(fire | complied), P(fire | refused), refusal=>fire,
     AUC(probe score -> refusal), with a behavior-clustered bootstrap CI on P(fire | complied).

Run AFTER the baseline (needs its graded .eval log) and while nothing else holds the GPU.
"""
import glob
import json
import os

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from inspect_ai import Task, eval as inspect_eval
from inspect_ai.log import read_eval_log
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import TaskState

from inspect_evals.agentharm.agents.agent import get_agent
from inspect_evals.agentharm.utils import (
    filter_dataset, get_benchmark_tools, load_dataset, setup_tools_from_metadata,
)

MID = os.environ.get("QWEN35_27B_MODEL", "Qwen/Qwen3.5-27B")
GRADED_DIR = os.environ.get("GRADED_LOG_DIR", "./logs_qwen35_harmful")
OUT = os.environ.get("OUT", "./qwen35_dissociation.json")
B = 5000  # bootstrap resamples

VAL = {"X": [], "y": []}
TEST = {"X": [], "id": [], "io": []}


def acts(hs):
    """All transformer layers (skip embedding at index 0), post-instruction token, step 0."""
    step0 = hs[0]                                   # tuple(n_layers+1) of (batch, seq, hidden)
    return np.stack([step0[l][0, -1, :].float().cpu().numpy() for l in range(1, len(step0))])


@scorer(metrics=[mean()])
def val_scorer(label):
    async def score(state: TaskState, target: Target) -> Score:
        hs = ((state.output.metadata or {}) if state.output else {}).get("hidden_states")
        if hs is not None:
            VAL["X"].append(acts(hs)); VAL["y"].append(label)
        return Score(value=float(label))
    return score


@scorer(metrics=[mean()])
def test_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        hs = ((state.output.metadata or {}) if state.output else {}).get("hidden_states")
        if hs is not None:
            TEST["X"].append(acts(hs))
            TEST["id"].append(str(state.sample_id))
            TEST["io"].append(str((state.metadata or {}).get("id_original")))
        return Score(value=0.0)
    return score


def task(task_name, split, scorer_fn):
    ds = filter_dataset(load_dataset(task_name, split, False), [], None, None)
    tools = get_benchmark_tools(task_name) if task_name != "chat" else []
    plan = [setup_tools_from_metadata(task_name, tools, 0),
            get_agent(agent="default", tool_choice="none")]
    return Task(dataset=ds, solver=plan, scorer=scorer_fn,
                config=GenerateConfig(max_tokens=8, max_connections=1))


def refusal_outcomes(graded_dir):
    path = sorted(glob.glob(os.path.join(graded_dir, "*.eval")), key=os.path.getsize)[-1]
    log = read_eval_log(path)
    out = {}
    for smp in (log.samples or []):
        for sc in (smp.scores or {}).values():
            v = sc.value
            if isinstance(v, dict) and "score" in v:
                out[str(smp.id)] = (float(v["score"]), int(v.get("refusal", 0) >= 0.5))
    return path, out


if __name__ == "__main__":
    # thinking off to match the baseline; hidden_states + single connection for clean extraction
    model = get_model(f"hf/{MID}", hidden_states=True, device="cuda:0", enable_thinking=False)

    print("extract val harmful ..."); inspect_eval(task("harmful", "val", val_scorer(1)), model=model, log_dir="./logs_qwen35_probe")
    print("extract val benign  ..."); inspect_eval(task("benign", "val", val_scorer(0)), model=model, log_dir="./logs_qwen35_probe")
    print("extract test_public ..."); inspect_eval(task("harmful", "test_public", test_scorer()), model=model, log_dir="./logs_qwen35_probe")

    Xtr = np.stack(VAL["X"]); ytr = np.array(VAL["y"])          # (64, L, H)
    Xte = np.stack(TEST["X"]); ios = np.array(TEST["io"])       # (176, L, H)
    L = Xtr.shape[1]
    gpath, outcomes = refusal_outcomes(GRADED_DIR)
    print(f"\nbehavioral outcomes from {os.path.basename(gpath)}: {len(outcomes)} samples")

    matched = [sid for sid in TEST["id"] if sid in outcomes]
    idx = [TEST["id"].index(sid) for sid in matched]
    Xm = Xte[idx]; iom = ios[idx]
    ref = np.array([outcomes[sid][1] for sid in matched])
    harm = np.array([outcomes[sid][0] for sid in matched])
    print(f"train: {Xtr.shape[0]} val ({int(ytr.sum())} harmful / {int((ytr==0).sum())} benign)  |  "
          f"test: {len(matched)} matched, refusal rate {ref.mean():.3f}")

    # per-layer: CV harmful/benign AUC on val (layer selection) + mean-diff direction
    behaviors = np.unique(iom); idx_by = {b: np.where(iom == b)[0] for b in behaviors}
    def cluster_ci(vec):
        rng = np.random.default_rng(0); vals = []
        for _ in range(B):
            take = np.concatenate([idx_by[b] for b in rng.choice(behaviors, len(behaviors), replace=True)])
            vals.append(vec[take].mean())
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

    per_layer, cv_auc = {}, []
    for li in range(L):
        # CV harmful/benign AUC on val (no test peeking) for layer selection
        s = np.zeros(len(ytr)); skf = StratifiedKFold(5, shuffle=True, random_state=0)
        for tr, te in skf.split(Xtr[:, li], ytr):
            mu1 = Xtr[tr][ytr[tr]==1, li].mean(0); mu0 = Xtr[tr][ytr[tr]==0, li].mean(0)
            d = mu1 - mu0; d /= (np.linalg.norm(d)+1e-12); s[te] = Xtr[te, li] @ d
        cv = roc_auc_score(ytr, s); cv_auc.append(cv)
        # final direction on all val; score test; AUC->refusal
        mu1 = Xtr[ytr==1, li].mean(0); mu0 = Xtr[ytr==0, li].mean(0)
        d = mu1 - mu0; d /= (np.linalg.norm(d)+1e-12)
        st = Xm[:, li] @ d
        auc_ref = roc_auc_score(ref, st) if len(set(ref)) > 1 else float("nan")
        per_layer[li+1] = {"cv_hb_auc": float(cv), "auc_refusal": float(auc_ref)}

    best = int(np.argmax(cv_auc))                      # array index
    bl = best + 1                                       # transformer layer number
    print(f"best layer by val CV harmful/benign AUC: L{bl} (AUC {cv_auc[best]:.3f})")

    # dissociation at best layer
    mu1 = Xtr[ytr==1, best].mean(0); mu0 = Xtr[ytr==0, best].mean(0)
    d = mu1 - mu0; d /= (np.linalg.norm(d)+1e-12)
    thr = 0.5 * (mu1 @ d + mu0 @ d)
    st = Xm[:, best] @ d
    fire = (st > thr).astype(int)
    complied = ref == 0; refused = ref == 1
    p_fire_complied = float(fire[complied].mean()) if complied.any() else float("nan")
    p_fire_refused = float(fire[refused].mean()) if refused.any() else float("nan")
    ci = cluster_ci(fire[complied]) if complied.any() else None
    auc_ref_best = float(roc_auc_score(ref, st)) if len(set(ref)) > 1 else float("nan")
    refused_and_fired = int(fire[refused].sum()); n_refused = int(refused.sum())

    print("\n=== QWEN3.5-27B DISSOCIATION ===")
    print(f"  best layer                  : L{bl}  (val CV harmful/benign AUC {cv_auc[best]:.3f})")
    print(f"  refusal rate                : {ref.mean():.3f}")
    print(f"  P(fire | complied)          : {p_fire_complied:.3f}  CI {ci}")
    print(f"  P(fire | refused)           : {p_fire_refused:.3f}")
    print(f"  refusal => fired            : {refused_and_fired}/{n_refused}")
    print(f"  AUC(probe score -> refusal) : {auc_ref_best:.3f}")

    out = {"model": MID, "n_layers": int(L), "best_layer": bl,
           "val_cv_auc_by_layer": cv_auc, "per_layer": per_layer,
           "refusal_rate": float(ref.mean()), "p_fire_complied": p_fire_complied,
           "p_fire_complied_ci": ci, "p_fire_refused": p_fire_refused,
           "refused_and_fired": [refused_and_fired, n_refused],
           "auc_refusal_best": auc_ref_best, "threshold": float(thr),
           "records": [{"id": sid, "score": float(sc), "refused": int(r), "harm": float(h)}
                       for sid, sc, r, h in zip(matched, st, ref, harm)]}
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\nsaved {OUT}")
