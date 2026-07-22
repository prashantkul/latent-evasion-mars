"""All-layer version: agentic-native probe (mean-difference, trained on 64 val samples) at EVERY
layer 0..27, evaluated on 176 test_public rows. Per-layer AUC(->refusal) with behavior-clustered CI,
plus per-layer test scores and the mean-across-layers score, saved for ROC plotting."""
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

NL = 28
GRADED_LOG = sorted(glob.glob(os.path.expanduser("~/latent-evasion/logs_full/*.eval")),
                    key=os.path.getsize)[-1]
PROBE_DIR = os.path.expanduser("~/latent-evasion/dataset/representations/llama32-3b/train_svm")
B = 5000

# Arditi probes (all 28 layers) for a comparison line
AW, AB = [], []
for l in range(NL):
    o = torch.load(os.path.join(PROBE_DIR, f"svm_layer{l:02d}.pt"), map_location="cpu", weights_only=False)
    AW.append(o["w"].float().numpy()); AB.append(float(o["b"]))
AW = np.stack(AW)   # (28, 3072)
AB = np.array(AB)

VAL = {"X": [], "y": []}
TEST = {"X": [], "id": [], "io": []}


def actL(hs):
    step0 = hs[0]
    return np.stack([step0[l + 1][0, -1, :].float().cpu().numpy() for l in range(NL)])  # (28, 3072)


@scorer(metrics=[mean()])
def val_scorer(label):
    async def score(state: TaskState, target: Target) -> Score:
        hs = ((state.output.metadata or {}) if state.output else {}).get("hidden_states")
        if hs is not None:
            VAL["X"].append(actL(hs)); VAL["y"].append(label)
        return Score(value=float(label))
    return score


@scorer(metrics=[mean()])
def test_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        hs = ((state.output.metadata or {}) if state.output else {}).get("hidden_states")
        if hs is not None:
            TEST["X"].append(actL(hs))
            TEST["id"].append(str(state.sample_id))
            TEST["io"].append(str((state.metadata or {}).get("id_original")))
        return Score(value=0.0)
    return score


def task(task_name, split, scorer_fn):
    ds = filter_dataset(load_dataset(task_name, split, False), [], None, None)
    tools = get_benchmark_tools(task_name) if task_name != "chat" else []
    plan = [setup_tools_from_metadata(task_name, tools, 0), get_agent(agent="default", tool_choice="none")]
    return Task(dataset=ds, solver=plan, scorer=scorer_fn, config=GenerateConfig(max_tokens=8, max_connections=1))


def refusal_labels(path):
    log = read_eval_log(path); out = {}
    for smp in (log.samples or []):
        for sc in (smp.scores or {}).values():
            if isinstance(sc.value, dict) and "score" in sc.value:
                out[str(smp.id)] = int(sc.value.get("refusal", 0) >= 0.5)
    return out


if __name__ == "__main__":
    mid = os.environ.get("LLAMA32_3B_MODEL", "unsloth/Llama-3.2-3B-Instruct")
    model = get_model(f"hf/{mid}", hidden_states=True, device="cuda:0")
    print("extract val harmful ..."); inspect_eval(task("harmful", "val", val_scorer(1)), model=model, log_dir="./logs_alllayers")
    print("extract val benign ...");  inspect_eval(task("benign", "val", val_scorer(0)), model=model, log_dir="./logs_alllayers")
    print("extract test ...");        inspect_eval(task("harmful", "test_public", test_scorer()), model=model, log_dir="./logs_alllayers")

    Xtr = np.stack(VAL["X"]); ytr = np.array(VAL["y"])       # (64, 28, 3072)
    Xte = np.stack(TEST["X"])                                # (176, 28, 3072)
    lab = refusal_labels(GRADED_LOG)
    ref = np.array([lab[i] for i in TEST["id"]])
    ios = np.array(TEST["io"])
    behaviors = np.unique(ios); idx_by = {b: np.where(ios == b)[0] for b in behaviors}
    print(f"train {Xtr.shape[0]} val, test {Xte.shape[0]} rows, refusal {ref.mean():.2f}")

    def ci(scores):
        rng = np.random.default_rng(0); v = []
        for _ in range(B):
            idx = np.concatenate([idx_by[b] for b in rng.choice(behaviors, len(behaviors), replace=True)])
            if len(set(ref[idx])) > 1:
                v.append(roc_auc_score(ref[idx], scores[idx]))
        return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

    # agentic-native mean-difference probe per layer -> test scores
    md_scores = np.zeros((Xte.shape[0], NL))
    per_layer = {}
    for l in range(NL):
        sd = Xtr[ytr == 1, l].mean(0) - Xtr[ytr == 0, l].mean(0)
        sd = sd / (np.linalg.norm(sd) + 1e-12)
        s = Xte[:, l] @ sd
        md_scores[:, l] = s
        auc = roc_auc_score(ref, s); lo, hi = ci(s)
        a_auc = roc_auc_score(ref, Xte[:, l] @ AW[l] + AB[l])
        per_layer[l] = {"md_auc": auc, "md_ci": [lo, hi], "arditi_auc": a_auc}
        print(f"  layer {l:02d}: md AUC {auc:.3f} [{lo:.3f},{hi:.3f}]  | arditi {a_auc:.3f}")

    mean_score = md_scores.mean(1)
    mlo, mhi = ci(mean_score)
    print(f"  MEAN over layers: AUC {roc_auc_score(ref, mean_score):.3f} [{mlo:.3f},{mhi:.3f}]")

    import json
    json.dump({
        "n_train": int(Xtr.shape[0]), "n_test": int(Xte.shape[0]), "refusal_rate": float(ref.mean()),
        "per_layer": per_layer,
        "mean_auc": float(roc_auc_score(ref, mean_score)), "mean_ci": [mlo, mhi],
        "refused": ref.astype(int).tolist(),
        "md_scores": md_scores.tolist(),        # (176, 28) per-layer agentic-native probe scores
        "mean_score": mean_score.tolist(),      # (176,) mean across layers
    }, open("probe_all_layers.json", "w"), indent=2)
    print("saved probe_all_layers.json")
