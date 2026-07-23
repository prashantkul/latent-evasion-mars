"""Qwen3.5-27B agentic dissociation, built on the repo's get_representations extractor.

Direction is the val harmful/benign mean-difference (reuses the saved val activations if present,
else extracts them). Evaluation extracts test_public harmful + benign in agentic context and joins
the behavioural outcome (refused / complied) from the vLLM graded logs:

  harmful test -> P(fire | complied)  [the dissociation vs the 3B's 0.81], P(fire | refused),
                  refusal=>fire, AUC(score -> refusal)
  benign  test -> P(fire | complied) == the probe's false-positive rate (benign refusal is ~0)
  monitor      -> AUC(score -> harmful class) over harmful+benign test, recall/FPR at val threshold

Reuses helpers from qwen35_agentic_val_extract (load_split, extract, tool schemas) so extraction
logic is not duplicated. Run on the GPU box after the baseline graded logs exist. Example:
  export HF_HOME=/workspace/.cache/huggingface PYTHONPATH=/opt/latent-evasion-mars
  python3 experiments/qwen35_agentic_dissociation.py \
      --val-acts /opt/qwen35_agentic_val_acts.npz \
      --harmful-log 'logs_qwen35v_harmful/*.eval' --benign-log 'logs_qwen35v_benign/*.eval' \
      --out /opt/qwen35_agentic_dissociation.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from inspect_ai.log import read_eval_log
from inspect_evals.agentharm.utils import filter_dataset, load_dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen35_agentic_val_extract import extract, load_split  # noqa: E402

from models import Qwen35_27b  # noqa: E402

B = 5000


def refusal_outcomes(pattern):
    path = sorted(glob.glob(pattern), key=os.path.getsize)[-1]
    log = read_eval_log(path)
    out = {}
    for smp in (log.samples or []):
        for sc in (smp.scores or {}).values():
            v = sc.value
            if isinstance(v, dict) and "score" in v:
                out[str(smp.id)] = (float(v["score"]), int(v.get("refusal", 0) >= 0.5))
                break
    return os.path.basename(path), out


def id_original_map(task_name):
    ds = filter_dataset(load_dataset(task_name, "test_public", False), [], None, None)
    return {str(s.id): str((s.metadata or {}).get("id_original")) for s in ds}


def cluster_ci(vec, clusters):
    """Behaviour-clustered bootstrap CI on vec.mean(), resampling whole behaviours."""
    uniq = np.unique(clusters)
    idx_by = {c: np.where(clusters == c)[0] for c in uniq}
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(B):
        take = np.concatenate([idx_by[c] for c in rng.choice(uniq, len(uniq), replace=True)])
        vals.append(vec[take].mean())
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def train_direction(model, agentic, val_acts):
    if val_acts and os.path.exists(val_acts):
        print(f"reuse val activations {val_acts}", flush=True)
        z = np.load(val_acts, allow_pickle=True)
        X, y = z["X"], z["y"]
    else:
        print("extract val harmful/benign ...", flush=True)
        Xh, _ = extract(model, load_split("harmful", "val"), agentic)
        Xb, _ = extract(model, load_split("benign", "val"), agentic)
        X = np.concatenate([Xh, Xb]); y = np.array([1] * len(Xh) + [0] * len(Xb))
    L = X.shape[1]
    cv = []
    for li in range(L):
        s = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X[:, li], y):
            mu1 = X[tr][y[tr] == 1, li].mean(0); mu0 = X[tr][y[tr] == 0, li].mean(0)
            d = mu1 - mu0; d /= (np.linalg.norm(d) + 1e-12)
            s[te] = X[te, li] @ d
        cv.append(float(roc_auc_score(y, s)))
    best = int(np.argmax(cv))
    mu1 = X[y == 1, best].mean(0); mu0 = X[y == 0, best].mean(0)
    d = mu1 - mu0; d /= (np.linalg.norm(d) + 1e-12)
    thr = float(0.5 * (mu1 @ d + mu0 @ d))
    print(f"best layer {best} val-CV AUC {cv[best]:.3f}  thr {thr:.3f}", flush=True)
    return best, d, thr, cv


def score_split(model, task_name, agentic, best, d, log_pattern):
    rows = load_split(task_name, "test_public")
    X, ids = extract(model, rows, agentic)
    logname, outc = refusal_outcomes(log_pattern)
    iomap = id_original_map(task_name)
    keep = [i for i, sid in enumerate(ids) if sid in outc]
    ids = [ids[i] for i in keep]; X = X[keep]
    scores = X[:, best] @ d
    ref = np.array([outc[sid][1] for sid in ids])
    harm = np.array([outc[sid][0] for sid in ids])
    io = np.array([iomap.get(sid, sid) for sid in ids])
    print(f"{task_name}: log {logname}  matched {len(ids)}  refusal {ref.mean():.3f}", flush=True)
    return {"ids": ids, "scores": scores, "ref": ref, "harm": harm, "io": io}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("QWEN35_27B_MODEL", "Qwen/Qwen3.5-27B"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--agentic", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--val-acts", default="/opt/qwen35_agentic_val_acts.npz")
    ap.add_argument("--harmful-log", default="logs_qwen35v_harmful/*.eval")
    ap.add_argument("--benign-log", default="logs_qwen35v_benign/*.eval")
    ap.add_argument("--out", default="/opt/qwen35_agentic_dissociation.json")
    args = ap.parse_args()

    model = Qwen35_27b(model_name=args.model, device=args.device)
    best, d, thr, cv = train_direction(model, args.agentic, args.val_acts)

    print("extract test_public harmful ...", flush=True)
    H = score_split(model, "harmful", args.agentic, best, d, args.harmful_log)
    print("extract test_public benign ...", flush=True)
    Bn = score_split(model, "benign", args.agentic, best, d, args.benign_log)

    # persist raw test scores before metrics so nothing is lost
    np.savez(args.out.replace(".json", "_scores.npz"),
             h_scores=H["scores"], h_ref=H["ref"], h_harm=H["harm"], h_io=H["io"], h_ids=np.array(H["ids"]),
             b_scores=Bn["scores"], b_ref=Bn["ref"], b_io=Bn["io"], b_ids=np.array(Bn["ids"]),
             direction=d, best_layer=best, threshold=thr)

    # --- harmful dissociation ---
    hfire = (H["scores"] > thr).astype(int)
    comp = H["ref"] == 0; refd = H["ref"] == 1
    p_fire_complied = float(hfire[comp].mean()) if comp.any() else float("nan")
    ci = cluster_ci(hfire[comp], H["io"][comp]) if comp.any() else None
    p_fire_refused = float(hfire[refd].mean()) if refd.any() else float("nan")
    auc_ref = float(roc_auc_score(H["ref"], H["scores"])) if len(set(H["ref"])) > 1 else float("nan")

    # --- benign false-positive rate (benign refusal ~0 => all complied) ---
    bfire = (Bn["scores"] > thr).astype(int)
    bcomp = Bn["ref"] == 0
    fpr = float(bfire[bcomp].mean()) if bcomp.any() else float("nan")
    fpr_ci = cluster_ci(bfire[bcomp], Bn["io"][bcomp]) if bcomp.any() else None

    # --- monitor: harmful(1) vs benign(0) separation on test ---
    mon_scores = np.concatenate([H["scores"], Bn["scores"]])
    mon_label = np.concatenate([np.ones(len(H["scores"])), np.zeros(len(Bn["scores"]))])
    auc_hb = float(roc_auc_score(mon_label, mon_scores))
    recall = float(hfire.mean())  # fraction of harmful flagged at val threshold

    print("\n=== QWEN3.5-27B AGENTIC DISSOCIATION ===")
    print(f"  best layer               L{best}  (val-CV AUC {cv[best]:.3f})")
    print(f"  harmful refusal rate     {H['ref'].mean():.3f}")
    print(f"  P(fire | complied) harm  {p_fire_complied:.3f}  CI {ci}   [vs 3B 0.81]")
    print(f"  P(fire | refused)  harm  {p_fire_refused:.3f}   ({int(hfire[refd].sum())}/{int(refd.sum())})")
    print(f"  AUC(score -> refusal)    {auc_ref:.3f}")
    print(f"  benign FPR P(fire)       {fpr:.3f}  CI {fpr_ci}")
    print(f"  monitor AUC(harm vs ben) {auc_hb:.3f}")
    print(f"  harmful flagged @thr     {recall:.3f}")

    json.dump({"model": args.model, "agentic": args.agentic, "best_layer": best,
               "val_cv_auc": cv, "threshold": thr,
               "harmful": {"n": len(H["ids"]), "refusal_rate": float(H["ref"].mean()),
                           "p_fire_complied": p_fire_complied, "p_fire_complied_ci": ci,
                           "p_fire_refused": p_fire_refused, "auc_refusal": auc_ref,
                           "flagged_at_thr": recall},
               "benign": {"n": len(Bn["ids"]), "refusal_rate": float(Bn["ref"].mean()),
                          "fpr": fpr, "fpr_ci": fpr_ci},
               "monitor_auc_harm_vs_benign": auc_hb}, open(args.out, "w"), indent=2)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
