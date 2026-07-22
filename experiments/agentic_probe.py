"""Train an agentic-native probe on the AgentHarm VALIDATION split (32 harmful + 32 benign = 64),
evaluate on the held-out test_public set (176 rows, disjoint behaviors), at 4 layers.

Probe learns harmful(1) vs harmless(0) from val activations (last prompt token, agentic context).
Evaluated by AUC(probe score -> refusal) on test, with behavior-clustered CIs. Both LinearSVC and
the mean-difference (single-direction) probe are reported.
"""
import glob
import os

import numpy as np
import torch
from sklearn.svm import LinearSVC
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

LAYERS = [14, 18, 22, 26]
GRADED_LOG = sorted(glob.glob(os.path.expanduser("~/latent-evasion/logs_full/*.eval")),
                    key=os.path.getsize)[-1]
B = 5000

VAL = {"X": [], "y": []}
TEST = {"X": [], "id": [], "io": []}


def act4(hs):
    step0 = hs[0]
    return np.stack([step0[l + 1][0, -1, :].float().cpu().numpy() for l in LAYERS])  # (4, 3072)


@scorer(metrics=[mean()])
def val_scorer(label):
    async def score(state: TaskState, target: Target) -> Score:
        hs = ((state.output.metadata or {}) if state.output else {}).get("hidden_states")
        if hs is not None:
            VAL["X"].append(act4(hs)); VAL["y"].append(label)
        return Score(value=float(label))
    return score


@scorer(metrics=[mean()])
def test_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        hs = ((state.output.metadata or {}) if state.output else {}).get("hidden_states")
        if hs is not None:
            TEST["X"].append(act4(hs))
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


def refusal_labels(path):
    log = read_eval_log(path)
    out = {}
    for smp in (log.samples or []):
        for sc in (smp.scores or {}).values():
            v = sc.value
            if isinstance(v, dict) and "score" in v:
                out[str(smp.id)] = (float(v["score"]), int(v.get("refusal", 0) >= 0.5))
    return out


if __name__ == "__main__":
    mid = os.environ.get("LLAMA32_3B_MODEL", "unsloth/Llama-3.2-3B-Instruct")
    model = get_model(f"hf/{mid}", hidden_states=True, device="cuda:0")

    print("extract val harmful ...");  inspect_eval(task("harmful", "val", val_scorer(1)), model=model, log_dir="./logs_valprobe")
    print("extract val benign ...");   inspect_eval(task("benign", "val", val_scorer(0)), model=model, log_dir="./logs_valprobe")
    print("extract test_public ...");  inspect_eval(task("harmful", "test_public", test_scorer()), model=model, log_dir="./logs_valprobe")

    Xtr = np.stack(VAL["X"]); ytr = np.array(VAL["y"])            # (64, 4, 3072)
    Xte = np.stack(TEST["X"]); ios = np.array(TEST["io"])         # (176, 4, 3072)
    lab = refusal_labels(GRADED_LOG)
    harm = np.array([lab[i][0] for i in TEST["id"]])
    ref = np.array([lab[i][1] for i in TEST["id"]])
    print(f"\ntrain: {Xtr.shape[0]} val ({int(ytr.sum())} harmful / {int((ytr==0).sum())} benign)  |  "
          f"test: {Xte.shape[0]} rows, refusal rate {ref.mean():.2f}")

    behaviors = np.unique(ios); idx_by = {b: np.where(ios == b)[0] for b in behaviors}

    def cluster_ci(scores, y):
        rng = np.random.default_rng(0); vals = []
        for _ in range(B):
            idx = np.concatenate([idx_by[b] for b in rng.choice(behaviors, len(behaviors), replace=True)])
            if len(set(y[idx])) > 1:
                vals.append(roc_auc_score(y[idx], scores[idx]))
        return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

    print("\nlayer | SVM AUC(->refusal)        | SD AUC(->refusal) | SVM AUC(->harm>=.5)")
    out = {"layers": LAYERS, "n_train": int(Xtr.shape[0]), "n_test": int(Xte.shape[0]),
           "refusal_rate": float(ref.mean()), "per_layer": {}, "records": {}}
    harm_bin = (harm >= 0.5).astype(int)
    for i, L in enumerate(LAYERS):
        clf = LinearSVC(C=0.1, dual="auto", max_iter=500000).fit(Xtr[:, i], ytr)
        s_svm = clf.decision_function(Xte[:, i])
        auc_svm = roc_auc_score(ref, s_svm); lo, hi = cluster_ci(s_svm, ref)
        mu1 = Xtr[:, i][ytr == 1].mean(0); mu0 = Xtr[:, i][ytr == 0].mean(0)
        sd = mu1 - mu0; sd = sd / (np.linalg.norm(sd) + 1e-12)
        s_sd = Xte[:, i] @ sd
        auc_sd = roc_auc_score(ref, s_sd)
        auc_h = roc_auc_score(harm_bin, s_svm) if len(set(harm_bin)) > 1 else float("nan")
        print(f"  {L:>2}  | {auc_svm:.3f} [{lo:.3f}, {hi:.3f}] |      {auc_sd:.3f}      |    {auc_h:.3f}")
        out["per_layer"][L] = {"svm_auc_refusal": auc_svm, "svm_ci": [lo, hi],
                               "sd_auc_refusal": auc_sd, "svm_auc_harm": auc_h}
        out["records"][L] = [{"score": float(s), "refused": int(r), "harm": float(h)}
                             for s, r, h in zip(s_svm, ref, harm)]

    import json
    json.dump(out, open("agentic_probe.json", "w"), indent=2)
    print("\nsaved agentic_probe.json")
