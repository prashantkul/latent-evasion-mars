"""Per-layer test scores for the Qwen3.5-27B agentic probe (for the ROC-fan / best-layer / mean plot).

Mirrors experiments/probe_all_layers.py: per-layer unit-normalised mean-difference direction fit on
val, projected onto the test_public rows at EVERY layer. Saves per-layer AUC(->refusal) with
behaviour-clustered CI, the (n_test, n_layers) score matrix, and the mean-across-layers score.

Reuses get_representations via the val-extract helpers; directions come from the val activations
(recomputed here from X_val/y_val so it doesn't depend on the saved dirs). Run on the GPU box. Ex:
  export HF_HOME=/workspace/.cache/huggingface PYTHONPATH=/opt/latent-evasion-mars
  python3 experiments/qwen35_perlayer_test.py --val-acts /opt/qwen35_agentic_val_acts.npz \
      --harmful-log 'logs_qwen35v_harmful/*.eval' --benign-log 'logs_qwen35v_benign/*.eval' \
      --out /opt/qwen35_perlayer_test.json
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen35_agentic_val_extract import extract, load_split  # noqa: E402
from qwen35_agentic_dissociation import refusal_outcomes, id_original_map  # noqa: E402

from models import Qwen35_27b  # noqa: E402

B = 5000


def val_directions(val_acts):
    z = np.load(val_acts, allow_pickle=True)
    X, y = z["X"], z["y"]                       # (64, L, H), (64,)
    L = X.shape[1]
    dirs = np.zeros((L, X.shape[2]), dtype=np.float32)
    for l in range(L):
        sd = X[y == 1, l].mean(0) - X[y == 0, l].mean(0)
        dirs[l] = sd / (np.linalg.norm(sd) + 1e-12)
    return dirs


def cluster_auc_ci(scores, ref, clusters):
    uniq = np.unique(clusters); idx_by = {c: np.where(clusters == c)[0] for c in uniq}
    rng = np.random.default_rng(0); v = []
    for _ in range(B):
        idx = np.concatenate([idx_by[c] for c in rng.choice(uniq, len(uniq), replace=True)])
        if len(set(ref[idx])) > 1:
            v.append(roc_auc_score(ref[idx], scores[idx]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def score_test(model, agentic, dirs, task_name, log_pattern):
    rows = load_split(task_name, "test_public")
    X, ids = extract(model, rows, agentic)                 # (N, L, H)
    logname, outc = refusal_outcomes(log_pattern)
    iomap = id_original_map(task_name)
    keep = [i for i, sid in enumerate(ids) if sid in outc]
    X = X[keep]; ids = [ids[i] for i in keep]
    L = X.shape[1]
    md = np.stack([X[:, l] @ dirs[l] for l in range(L)], axis=1)   # (N, L)
    ref = np.array([outc[sid][1] for sid in ids])
    io = np.array([iomap.get(sid, sid) for sid in ids])
    print(f"{task_name}: {len(ids)} rows, refusal {ref.mean():.3f}", flush=True)
    return {"md": md, "ref": ref, "io": io, "ids": ids}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("QWEN35_27B_MODEL", "Qwen/Qwen3.5-27B"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--agentic", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--val-acts", default="/opt/qwen35_agentic_val_acts.npz")
    ap.add_argument("--harmful-log", default="logs_qwen35v_harmful/*.eval")
    ap.add_argument("--benign-log", default="logs_qwen35v_benign/*.eval")
    ap.add_argument("--out", default="/opt/qwen35_perlayer_test.json")
    args = ap.parse_args()

    dirs = val_directions(args.val_acts)
    L = dirs.shape[0]
    print(f"val directions {dirs.shape}", flush=True)

    model = Qwen35_27b(model_name=args.model, device=args.device)
    print("extract test harmful ...", flush=True)
    H = score_test(model, args.agentic, dirs, "harmful", args.harmful_log)
    print("extract test benign ...", flush=True)
    Bn = score_test(model, args.agentic, dirs, "benign", args.benign_log)

    md = H["md"]; ref = H["ref"]; io = H["io"]
    # save the score matrices immediately
    np.savez(args.out.replace(".json", "_scores.npz"),
             h_md=md, h_ref=ref, h_io=io, h_ids=np.array(H["ids"]),
             b_md=Bn["md"], b_io=Bn["io"], b_ids=np.array(Bn["ids"]), dirs=dirs)

    # per-layer AUC(->refusal) on harmful + harmful-vs-benign
    per_layer = {}
    lab_hb = np.concatenate([np.ones(md.shape[0]), np.zeros(Bn["md"].shape[0])])
    for l in range(L):
        s = md[:, l]
        auc_ref = roc_auc_score(ref, s) if len(set(ref)) > 1 else float("nan")
        ci = cluster_auc_ci(s, ref, io) if len(set(ref)) > 1 else [float("nan")] * 2
        sco_hb = np.concatenate([md[:, l], Bn["md"][:, l]])
        auc_hb = roc_auc_score(lab_hb, sco_hb)
        per_layer[l] = {"auc_refusal": float(auc_ref), "auc_refusal_ci": ci, "auc_harm_vs_benign": float(auc_hb)}

    aucs = np.array([per_layer[l]["auc_refusal"] for l in range(L)])
    best = int(np.nanargmax(aucs))
    mean_score = md.mean(1)                                     # raw mean across layers (matches 03)
    mean_auc = float(roc_auc_score(ref, mean_score))
    mean_ci = cluster_auc_ci(mean_score, ref, io)

    print("\n=== QWEN3.5-27B PER-LAYER (test) ===")
    print(f"  best layer (AUC->refusal): L{best}  AUC {aucs[best]:.3f}  CI {per_layer[best]['auc_refusal_ci']}")
    print(f"  mean over layers:          AUC {mean_auc:.3f}  CI {mean_ci}")
    print(f"  (val-CV best was L54; this best is by test AUC->refusal)")

    json.dump({"model": args.model, "agentic": args.agentic, "n_layers": int(L),
               "n_test_harmful": int(md.shape[0]), "refusal_rate": float(ref.mean()),
               "best_layer": best, "best_auc": float(aucs[best]),
               "mean_auc": mean_auc, "mean_ci": mean_ci,
               "per_layer": per_layer,
               "refused": ref.astype(int).tolist(),
               "md_scores": md.tolist(),
               "mean_score": mean_score.tolist()}, open(args.out, "w"), indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
