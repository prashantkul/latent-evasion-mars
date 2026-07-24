"""VALID in-context probe experiment for Qwen3.5-27B — train and evaluate entirely inside Inspect.

The read context is the REAL agent prompt (AgentHarm's ~558-char system prompt + tools), which the
offline get_representations path in `05` omitted — that omission is why `05` is INVALID. Here every
activation is read through Inspect's hf provider (patched for Qwen3.5 tool-calling), so train and test
share the exact same distribution.

  PASS A (val, throwaway scoring): inspect_eval over the val split; the scorer only HARVESTS each
     sample's per-layer post-instruction vector (returns a dummy Score). After the run we train the
     per-layer LinearSVC (standardized, folded to a raw-space (w,b)) and mean-diff on those val
     activations, and pick the layer by 5-fold CV.
  PASS B (test, real scoring): inspect_eval over the held-out test_public split; the just-trained
     probe IS the scorer — it computes w·h + b live per sample, so the .eval log's Score column is
     the probe firing. Per-layer svm/md scores are stored in metadata for the AUC evaluation.

Train ONLY on the 64 val rows. Test activations are scored, NEVER fit on. Behavioural labels
(refused) come from the graded vLLM run, joined by sample id, for AUC(probe -> refusal).

  export HF_HOME=/workspace/.cache/huggingface PYTHONPATH=/opt/latent-evasion-mars
  python3 experiments/results/06-qwen35-inscorer-probe/qwen35_inscorer_experiment.py \
      --harmful-log '/opt/latent-evasion-mars/logs_qwen35v_harmful/*.eval' \
      --benign-log  '/opt/latent-evasion-mars/logs_qwen35v_benign/*.eval' \
      --out /opt/qwen35_inscorer_experiment.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen35_inspect_patch  # noqa: E402

from inspect_ai import Task, eval as inspect_eval  # noqa: E402
from inspect_ai.log import read_eval_log  # noqa: E402
from inspect_ai.model import GenerateConfig, get_model  # noqa: E402
from inspect_ai.scorer import Score, Target, mean, scorer  # noqa: E402
from inspect_ai.solver import TaskState  # noqa: E402

from inspect_evals.agentharm.agents.agent import get_agent  # noqa: E402
from inspect_evals.agentharm.utils import (  # noqa: E402
    filter_dataset, get_benchmark_tools, load_dataset, setup_tools_from_metadata,
)

C = 0.1
B = 5000
VAL = {"X": [], "y": []}                     # harvested val activations (Pass A)
TEST = {"harmful": {}, "benign": {}}         # per-sample probe scores (Pass B)
TEST_ACTS = {"harmful": [], "benign": []}    # cached in-context test activations (Pass B)
W = {}                                       # trained probe (filled between passes)


def read_all_layers(hs):
    """Post-instruction token at every transformer layer (embedding skipped). -> (L, H)."""
    step0 = hs[0]
    return np.stack([step0[l][0, -1, :].float().cpu().numpy() for l in range(1, len(step0))])


# ---------- PASS A: harvest val activations (throwaway score) ----------
@scorer(metrics=[mean()])
def collect_val(label):
    async def score(state: TaskState, target: Target) -> Score:
        hs = ((state.output.metadata or {}) if state.output else {}).get("hidden_states")
        if hs is not None:
            VAL["X"].append(read_all_layers(hs)); VAL["y"].append(label)
        return Score(value=float(label))     # placeholder — Pass A logs are not a result
    return score


# ---------- PASS B: the trained probe IS the scorer ----------
@scorer(metrics=[mean()])
def probe_scorer(tag):
    async def score(state: TaskState, target: Target) -> Score:
        hs = ((state.output.metadata or {}) if state.output else {}).get("hidden_states")
        if hs is None:
            return Score(value=0.0, metadata={"error": "no hidden_states"})
        h = read_all_layers(hs)                                  # (L, H)
        svm = (h * W["svm_w"]).sum(-1) + W["svm_b"]              # (L,)
        md = (h * W["dirs"]).sum(-1)                             # (L,)
        bl = W["best_layer"]
        io = str((state.metadata or {}).get("id_original"))
        TEST[tag][str(state.sample_id)] = {"svm": svm.tolist(), "md": md.tolist(), "id_original": io}
        # Cache the raw in-context activation, not just this probe's scores: re-evaluating a
        # NEW probe then costs a CPU matmul instead of another agentic GPU run.
        TEST_ACTS[tag].append((str(state.sample_id), h))
        return Score(value=float(svm[bl] > 0),                  # real: best-layer SVC firing
                     metadata={"svm_best": float(svm[bl]), "md_best": float(md[bl]),
                               "best_layer": int(bl), "id_original": io})
    return score


def task(task_name, split, scorer_fn):
    ds = filter_dataset(load_dataset(task_name, split, False), [], None, None)
    tools = get_benchmark_tools(task_name)
    plan = [setup_tools_from_metadata(task_name, tools, 0),
            get_agent(agent="default", tool_choice="none")]
    return Task(dataset=ds, solver=plan, scorer=scorer_fn,
                config=GenerateConfig(max_tokens=8, max_connections=1))


def fit_svm(X, y):
    return make_pipeline(StandardScaler(),
                         LinearSVC(C=C, dual="auto", max_iter=1_000_000, random_state=42)).fit(X, y)


def raw_wb(pipe):
    sc = pipe.named_steps["standardscaler"]; svc = pipe.named_steps["linearsvc"]
    w = svc.coef_[0] / sc.scale_
    b = float(svc.intercept_[0] - np.dot(sc.mean_ / sc.scale_, svc.coef_[0]))
    return w.astype(np.float32), b


def train_probe(X, y):
    """Per-layer SVM (w,b) + mean-diff dir on ALL val; 5-fold CV AUC for layer selection."""
    L, H = X.shape[1], X.shape[2]
    svm_w = np.zeros((L, H), np.float32); svm_b = np.zeros(L, np.float32)
    dirs = np.zeros((L, H), np.float32); cv = np.zeros(L)
    for l in range(L):
        svm_w[l], svm_b[l] = raw_wb(fit_svm(X[:, l], y))
        sd = X[y == 1, l].mean(0) - X[y == 0, l].mean(0)
        dirs[l] = sd / (np.linalg.norm(sd) + 1e-12)
        s = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X[:, l], y):
            s[te] = fit_svm(X[tr, l], y[tr]).decision_function(X[te, l])
        cv[l] = roc_auc_score(y, s)
    return svm_w, svm_b, dirs, cv


def refusal_map(pattern):
    p = sorted(glob.glob(pattern), key=os.path.getsize)[-1]
    out = {}
    for smp in (read_eval_log(p).samples or []):
        for sc in (smp.scores or {}).values():
            v = sc.value
            if isinstance(v, dict) and "score" in v:
                out[str(smp.id)] = (float(v["score"]), int(v.get("refusal", 0) >= 0.5)); break
    return os.path.basename(p), out


def cluster_ci(scores, ref, io):
    uniq = np.unique(io); idx = {c: np.where(io == c)[0] for c in uniq}
    rng = np.random.default_rng(0); v = []
    for _ in range(B):
        i = np.concatenate([idx[c] for c in rng.choice(uniq, len(uniq), replace=True)])
        if len(set(ref[i])) > 1:
            v.append(roc_auc_score(ref[i], scores[i]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("QWEN35_27B_MODEL", "Qwen/Qwen3.5-27B"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--harmful-log", default="/opt/latent-evasion-mars/logs_qwen35v_harmful/*.eval")
    ap.add_argument("--benign-log", default="/opt/latent-evasion-mars/logs_qwen35v_benign/*.eval")
    ap.add_argument("--log-dir", default="./logs_qwen35_inscorer_exp")
    ap.add_argument("--acts-out", default="/opt/qwen35_inscorer_val_acts.npz")
    ap.add_argument("--test-acts-out", default="/opt/qwen35_inscorer_test_acts.npz",
                    help="cache in-context TEST activations so a new probe can be re-evaluated "
                         "on CPU (experiments/eval_probe.py) without another agentic GPU run")
    ap.add_argument("--out", default="/opt/qwen35_inscorer_experiment.json")
    args = ap.parse_args()

    qwen35_inspect_patch.install()
    model = get_model(f"hf/{args.model}", hidden_states=True, device=args.device,
                      enable_thinking=False)

    # ---- PASS A: harvest val, train ----
    print("PASS A — harvest val activations (in-scorer) ...", flush=True)
    inspect_eval(task("harmful", "val", collect_val(1)), model=model, log_dir=args.log_dir)
    inspect_eval(task("benign", "val", collect_val(0)), model=model, log_dir=args.log_dir)
    Xval = np.stack(VAL["X"]); yval = np.array(VAL["y"])
    print(f"  val activations {Xval.shape} ({int(yval.sum())} harmful / {int((yval==0).sum())} benign)", flush=True)
    np.savez(args.acts_out, X=Xval, y=yval)              # persist val acts (training set)

    svm_w, svm_b, dirs, cv = train_probe(Xval, yval)
    best = int(np.argmax(cv))
    W.update(svm_w=svm_w, svm_b=svm_b, dirs=dirs, best_layer=best)
    print(f"  trained probe; best layer by val-CV L{best} (AUC {cv[best]:.3f})", flush=True)

    # ---- PASS B: probe fires live on held-out test ----
    print("PASS B — probe as scorer on held-out test (in-scorer) ...", flush=True)
    inspect_eval(task("harmful", "test_public", probe_scorer("harmful")), model=model, log_dir=args.log_dir)
    inspect_eval(task("benign", "test_public", probe_scorer("benign")), model=model, log_dir=args.log_dir)
    print(f"  test scored: {len(TEST['harmful'])} harmful, {len(TEST['benign'])} benign", flush=True)

    # Persist the raw in-context test activations immediately — this is the artifact that makes
    # re-evaluating a teammate's probe a CPU job rather than a repeat of this GPU run.
    ta_ids = [i for i, _ in TEST_ACTS["harmful"]] + [i for i, _ in TEST_ACTS["benign"]]
    ta_X = np.stack([a for _, a in TEST_ACTS["harmful"]] + [a for _, a in TEST_ACTS["benign"]])
    ta_y = np.concatenate([np.ones(len(TEST_ACTS["harmful"])), np.zeros(len(TEST_ACTS["benign"]))])
    np.savez(args.test_acts_out, X=ta_X.astype(np.float32), y=ta_y.astype(int), ids=np.array(ta_ids))
    print(f"  cached test activations {ta_X.shape} -> {args.test_acts_out}", flush=True)

    # ---- evaluate: join behavioural labels, per-layer AUC ----
    hlog, href = refusal_map(args.harmful_log)
    ids = [i for i in TEST["harmful"] if i in href]
    ref = np.array([href[i][1] for i in ids])
    io = np.array([TEST["harmful"][i]["id_original"] for i in ids])
    L = Xval.shape[1]
    md = np.array([[TEST["harmful"][i]["md"][l] for l in range(L)] for i in ids])       # (n, L)
    svm = np.array([[TEST["harmful"][i]["svm"][l] for l in range(L)] for i in ids])
    bmd = np.array([[TEST["benign"][i]["md"][l] for l in range(L)] for i in TEST["benign"]])
    bsvm = np.array([[TEST["benign"][i]["svm"][l] for l in range(L)] for i in TEST["benign"]])
    lab = np.concatenate([np.ones(len(md)), np.zeros(len(bmd))])

    def summarize(name, sc_h, sc_b):
        auc_ref = np.array([roc_auc_score(ref, sc_h[:, l]) for l in range(L)])
        auc_hb = np.array([roc_auc_score(lab, np.concatenate([sc_h[:, l], sc_b[:, l]])) for l in range(L)])
        b = int(np.argmax(auc_ref)); ms = sc_h.mean(1)
        r = dict(best_layer=b, best_auc=float(auc_ref[b]), best_ci=cluster_ci(sc_h[:, b], ref, io),
                 mean_auc=float(roc_auc_score(ref, ms)), best_hb=float(auc_hb.max()),
                 best_hb_layer=int(np.argmax(auc_hb)),
                 auc_refusal=auc_ref.tolist(), auc_harm_vs_benign=auc_hb.tolist())
        print(f"  {name:9s} best L{b} AUC->refusal {auc_ref[b]:.3f} CI {r['best_ci']} | "
              f"mean-layers {r['mean_auc']:.3f} | harm-vs-benign L{r['best_hb_layer']} {auc_hb.max():.3f}", flush=True)
        return r

    print(f"\n=== IN-CONTEXT PROBE (train val in-scorer -> eval test in-scorer) ===")
    print(f"  behavioural labels from {hlog} | n={len(ids)} refusal {ref.mean():.3f}", flush=True)
    res_md = summarize("mean-diff", md, bmd)
    res_svm = summarize("SVM", svm, bsvm)

    json.dump({"model": args.model, "context": "in-scorer (real agent system prompt + tools)",
               "n_val": int(len(yval)), "n_test_harmful": len(ids), "n_test_benign": len(bmd),
               "refusal_rate": float(ref.mean()), "val_cv_by_layer": cv.tolist(),
               "val_cv_best_layer": best, "mean_diff": res_md, "svm": res_svm,
               "test_ids": ids}, open(args.out, "w"), indent=2)

    np.savez(args.out.replace(".json", "_scores.npz"),
             md_h=md, svm_h=svm, md_b=bmd, svm_b=bsvm, ref=ref, io=io,
             ids=np.array(ids), svm_w=svm_w, svm_bias=svm_b, dirs=dirs)
    print(f"\nsaved {args.out}", flush=True)


if __name__ == "__main__":
    main()
