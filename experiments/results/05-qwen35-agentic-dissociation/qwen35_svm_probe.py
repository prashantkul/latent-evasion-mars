"""Qwen3.5-27B: train BOTH a per-layer LinearSVC and a mean-difference probe on the val agentic
activations, evaluate on held-out test_public. Parallels experiments/agentic_probe.py (which reports
SVM and SD columns) but on the get_representations path (Qwen-compatible).

Saves: the raw test activations (so future probe variants need no re-extraction), per-layer test
scores for both probes (harmful + benign), the trained SVM weights/biases and mean-diff directions,
and per-layer AUC(->refusal)/AUC(harmful-vs-benign) with best-layer and mean-over-layers for each.

Run on the GPU box. Ex:
  export HF_HOME=/workspace/.cache/huggingface PYTHONPATH=/opt/latent-evasion-mars
  python3 experiments/qwen35_svm_probe.py --val-acts /opt/qwen35_agentic_val_acts.npz \
      --harmful-log 'logs_qwen35v_harmful/*.eval' --benign-log 'logs_qwen35v_benign/*.eval' \
      --out /opt/qwen35_svm_probe.json --save-test-acts /opt/qwen35_test_acts.npz
"""
import argparse
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
from qwen35_agentic_val_extract import extract, load_split  # noqa: E402
from qwen35_agentic_dissociation import refusal_outcomes, id_original_map  # noqa: E402

from models import Qwen35_27b  # noqa: E402

C = 0.1


def fit_svm(X, y):
    # Standardize first: Qwen's late-layer residual stream has massive-magnitude outlier dims
    # that make an unscaled LinearSVC numerically unstable (overflow in decision_function).
    return make_pipeline(StandardScaler(),
                         LinearSVC(C=C, dual="auto", max_iter=1_000_000, random_state=42)).fit(X, y)


def raw_wb(pipe):
    """Fold StandardScaler into the linear head -> a single raw-space (w, b) so the probe is one
    direction: decision(X) = X @ w + b, identical to pipe.decision_function(X)."""
    sc = pipe.named_steps["standardscaler"]; svc = pipe.named_steps["linearsvc"]
    w = svc.coef_[0] / sc.scale_
    b = float(svc.intercept_[0] - np.dot(sc.mean_ / sc.scale_, svc.coef_[0]))
    return w.astype(np.float32), b


def val_train(val_acts):
    """Per-layer SVM (w, b) + mean-diff direction on ALL val, plus 5-fold CV AUC for layer choice."""
    z = np.load(val_acts, allow_pickle=True)
    X, y = z["X"], z["y"]                                    # (64, L, H), (64,)
    L, H = X.shape[1], X.shape[2]
    svm_w = np.zeros((L, H), np.float32); svm_b = np.zeros(L, np.float32)
    dirs = np.zeros((L, H), np.float32)
    cv_svm = np.zeros(L); cv_md = np.zeros(L)
    for l in range(L):
        clf = fit_svm(X[:, l], y)
        svm_w[l], svm_b[l] = raw_wb(clf)
        sd = X[y == 1, l].mean(0) - X[y == 0, l].mean(0)
        dirs[l] = sd / (np.linalg.norm(sd) + 1e-12)
        s_svm = np.zeros(len(y)); s_md = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X[:, l], y):
            c = fit_svm(X[tr, l], y[tr]); s_svm[te] = c.decision_function(X[te, l])
            m = X[tr][y[tr] == 1, l].mean(0) - X[tr][y[tr] == 0, l].mean(0)
            m /= (np.linalg.norm(m) + 1e-12); s_md[te] = X[te, l] @ m
        cv_svm[l] = roc_auc_score(y, s_svm); cv_md[l] = roc_auc_score(y, s_md)
    return dict(svm_w=svm_w, svm_b=svm_b, dirs=dirs, cv_svm=cv_svm, cv_md=cv_md, L=L)


def test_acts(model, agentic, task_name, log_pattern):
    rows = load_split(task_name, "test_public")
    X, ids = extract(model, rows, agentic)                  # (N, L, H)
    logname, outc = refusal_outcomes(log_pattern)
    iomap = id_original_map(task_name)
    keep = [i for i, sid in enumerate(ids) if sid in outc]
    X = X[keep]; ids = [ids[i] for i in keep]
    ref = np.array([outc[sid][1] for sid in ids])
    io = np.array([iomap.get(sid, sid) for sid in ids])
    print(f"{task_name}: {len(ids)} rows, refusal {ref.mean():.3f}", flush=True)
    return X, ref, io, ids


def layer_scores(X, P):
    """(N, L) SVM decision + mean-diff projection per layer."""
    L = X.shape[1]
    svm = np.stack([X[:, l] @ P["svm_w"][l] + P["svm_b"][l] for l in range(L)], 1)
    md = np.stack([X[:, l] @ P["dirs"][l] for l in range(L)], 1)
    return svm, md


def summarize(name, sc_h, ref, sc_b):
    L = sc_h.shape[1]
    lab = np.concatenate([np.ones(len(sc_h)), np.zeros(len(sc_b))])
    auc_ref = np.array([roc_auc_score(ref, sc_h[:, l]) for l in range(L)])
    auc_hb = np.array([roc_auc_score(lab, np.concatenate([sc_h[:, l], sc_b[:, l]])) for l in range(L)])
    best = int(np.argmax(auc_ref))
    mean_score = sc_h.mean(1); mean_auc = float(roc_auc_score(ref, mean_score))
    print(f"  {name:8s} best L{best} AUC->refusal {auc_ref[best]:.3f} | mean-over-layers {mean_auc:.3f} "
          f"| best harm-vs-benign L{int(np.argmax(auc_hb))} {auc_hb.max():.3f}", flush=True)
    return dict(auc_refusal=auc_ref.tolist(), auc_harm_vs_benign=auc_hb.tolist(),
                best_layer=best, best_auc=float(auc_ref[best]), mean_auc=mean_auc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("QWEN35_27B_MODEL", "Qwen/Qwen3.5-27B"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--agentic", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--val-acts", default="/opt/qwen35_agentic_val_acts.npz")
    ap.add_argument("--harmful-log", default="logs_qwen35v_harmful/*.eval")
    ap.add_argument("--benign-log", default="logs_qwen35v_benign/*.eval")
    ap.add_argument("--out", default="/opt/qwen35_svm_probe.json")
    ap.add_argument("--save-test-acts", default="/opt/qwen35_test_acts.npz")
    args = ap.parse_args()

    P = val_train(args.val_acts)
    print(f"trained per-layer SVM + mean-diff on val ({P['L']} layers); "
          f"val-CV best: SVM L{int(np.argmax(P['cv_svm']))} {P['cv_svm'].max():.3f} | "
          f"MD L{int(np.argmax(P['cv_md']))} {P['cv_md'].max():.3f}", flush=True)

    model = Qwen35_27b(model_name=args.model, device=args.device)
    print("extract test harmful ...", flush=True)
    Xh, ref, io, idh = test_acts(model, args.agentic, "harmful", args.harmful_log)
    print("extract test benign ...", flush=True)
    Xb, _rb, _iob, idb = test_acts(model, args.agentic, "benign", args.benign_log)

    # raw test activations first (expensive; reuse for any future probe)
    np.savez(args.save_test_acts, h_acts=Xh, b_acts=Xb, ref=ref, io=io,
             h_ids=np.array(idh), b_ids=np.array(idb))
    print(f"saved raw test activations -> {args.save_test_acts}", flush=True)

    svm_h, md_h = layer_scores(Xh, P)
    svm_b, md_b = layer_scores(Xb, P)
    np.savez(args.out.replace(".json", "_scores.npz"),
             svm_h=svm_h, md_h=md_h, svm_b=svm_b, md_b=md_b, ref=ref, io=io,
             svm_w=P["svm_w"], svm_b_bias=P["svm_b"], dirs=P["dirs"])

    print("\n=== QWEN3.5-27B PROBE COMPARISON (test) ===")
    res_svm = summarize("SVM", svm_h, ref, svm_b)
    res_md = summarize("mean-diff", md_h, ref, md_b)

    json.dump({"model": args.model, "agentic": args.agentic, "n_layers": int(P["L"]), "C": C,
               "refusal_rate": float(ref.mean()),
               "val_cv_svm": P["cv_svm"].tolist(), "val_cv_md": P["cv_md"].tolist(),
               "svm": res_svm, "mean_diff": res_md,
               "refused": ref.astype(int).tolist()}, open(args.out, "w"), indent=2)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
