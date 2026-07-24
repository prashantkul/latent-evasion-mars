"""Which layer-ensemble should the Inspect scorer emit?

A probe artifact is 64 independent per-layer (w, b) pairs, so "the probe fires" needs a
combining rule. The pinned metric already uses one -- the unweighted mean of raw scores --
but the Pass B scorer still writes best-layer into the .eval Score column. Before making the
scorer agree, check that the mean is the right rule rather than merely the first one.

The concern with a raw mean is scale: activation norms grow with depth, so a handful of late
high-magnitude layers could dominate the average and the "ensemble" would be a single layer
wearing a disguise. Rules compared here:

  raw mean       mean_l (w_l.h + b_l)          -- current pinned metric
  z-scored mean  standardize each layer on VAL first, then average
  median         robust to a few extreme layers
  vote fraction  mean_l 1[score_l > 0]         -- scale-free by construction
  best layer     the selection-dependent baseline we are trying to leave behind

All statistics for calibration come from VAL only. Test is scored, never fit on.

  uv run --with numpy --with scikit-learn python experiments/ensemble_rules.py
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_io import load_probe, score as probe_score

HERE = os.path.dirname(os.path.abspath(__file__))
R06 = os.path.join(HERE, "results", "06-qwen35-inscorer-probe")
RB = os.path.join(R06, "rebaseline_2026-07-24")


def confusion(y, pred):
    TP = int(((pred == 1) & (y == 1)).sum()); FN = int(((pred == 0) & (y == 1)).sum())
    FP = int(((pred == 1) & (y == 0)).sum()); TN = int(((pred == 0) & (y == 0)).sum())
    return (TP + TN) / len(y), TP / (TP + FN or 1), FP / (TN + FP or 1)


def rules(s_val, s_test, y_val):
    """Each rule -> (val score fn applied to test, threshold derived from val)."""
    mu, sd = s_val.mean(0), s_val.std(0) + 1e-9

    def thresh_at(v):
        """Midpoint between the val class means -- the same convention the probes use."""
        return 0.5 * (v[y_val == 1].mean() + v[y_val == 0].mean())

    out = {}
    out["raw mean"] = (s_test.mean(1), 0.0)                     # canonical boundary, no fitting
    zv, zt = (s_val - mu) / sd, (s_test - mu) / sd
    out["z-scored mean"] = (zt.mean(1), thresh_at(zv.mean(1)))
    out["median"] = (np.median(s_test, 1), 0.0)
    out["vote fraction"] = ((s_test > 0).mean(1), thresh_at((s_val > 0).mean(1)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default=os.path.join(R06, "probe_canonical", "qwen35_svm"))
    ap.add_argument("--val-acts", default=os.path.join(RB, "val_acts.npz"))
    ap.add_argument("--test-acts", default=os.path.join(RB, "test_acts.npz"))
    ap.add_argument("--scores", default=os.path.join(RB, "scores_and_probe.npz"),
                    help="supplies the behavioural refusal labels for the second objective")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    w, b, _layers, meta = load_probe(args.probe)
    v = np.load(args.val_acts, allow_pickle=True)
    t = np.load(args.test_acts, allow_pickle=True)
    s_val, y_val = probe_score(v["X"], w, b), v["y"].astype(int)
    s_test, y_test = probe_score(t["X"], w, b), t["y"].astype(int)
    print(f"probe {os.path.basename(args.probe)}  val {s_val.shape}  test {s_test.shape}")

    # --- is the raw mean actually an ensemble, or one loud layer? ---
    scale = s_test.std(0)
    order = np.argsort(scale)[::-1]
    share = scale**2 / (scale**2).sum()
    print(f"\nper-layer score scale (test std): min {scale.min():.2f}  median {np.median(scale):.2f}  "
          f"max {scale.max():.2f}  ({scale.max()/scale.min():.0f}x spread)")
    print(f"  loudest layers {order[:5].tolist()} carry {share[order[:5]].sum():.1%} of total variance")
    m = s_test.mean(1)
    corr = np.array([np.corrcoef(m, s_test[:, l])[0, 1] for l in range(s_test.shape[1])])
    print(f"  raw mean correlates most with L{int(np.argmax(corr))} (r={corr.max():.3f}); "
          f"median r over layers {np.median(corr):.3f}")

    # --- the rules, on held-out test ---
    ref = np.load(args.scores, allow_pickle=True)["ref"].astype(int) if args.scores else None
    nh = int(y_test.sum())
    print(f"\n{'rule':16s} {'AUC h-v-b':>10s} {'acc':>7s} {'recall':>7s} {'FPR':>7s}   {'AUC->refusal':>12s}")
    res = {}
    for name, (v_test, thr) in rules(s_val, s_test, y_val).items():
        auc = roc_auc_score(y_test, v_test)
        acc, rec, fpr = confusion(y_test, (v_test > thr).astype(int))
        ar = roc_auc_score(ref, v_test[:nh]) if ref is not None and 0 < ref.sum() < len(ref) else float("nan")
        res[name] = {"auc": auc, "acc": acc, "recall": rec, "fpr": fpr, "auc_refusal": ar,
                     "threshold": float(thr)}
        print(f"{name:16s} {auc:10.4f} {acc:7.3f} {rec:7.3f} {fpr:7.3f}   {ar:12.4f}")

    # best single layer, chosen on val -- the incumbent scorer
    bl = int(meta.get("val_cv_best_layer", 0))
    auc = roc_auc_score(y_test, s_test[:, bl])
    acc, rec, fpr = confusion(y_test, (s_test[:, bl] > 0).astype(int))
    ar = roc_auc_score(ref, s_test[:nh, bl]) if ref is not None else float("nan")
    res[f"best layer L{bl}"] = {"auc": auc, "acc": acc, "recall": rec, "fpr": fpr, "auc_refusal": ar}
    print(f"{'best layer L'+str(bl):16s} {auc:10.4f} {acc:7.3f} {rec:7.3f} {fpr:7.3f}   {ar:12.4f}"
          "   <- what the scorer emits today")

    # how much does the ensemble depend on WHICH layers? drop the loudest few.
    print("\nrobustness — raw-mean AUC after dropping the loudest layers:")
    for k in (0, 1, 3, 5, 10):
        keep = np.setdiff1d(np.arange(s_test.shape[1]), order[:k])
        print(f"  drop {k:2d} loudest -> {roc_auc_score(y_test, s_test[:, keep].mean(1)):.4f}")

    # --- is any gap between rules real, or one lucky sample? ---
    # Paired cluster bootstrap over behaviour ids (id_original), matching the CI convention in
    # qwen35_inscorer_experiment.cluster_ci: samples sharing a behaviour are not independent.
    io = np.load(args.scores, allow_pickle=True)["io"] if args.scores else None
    if io is not None and ref is not None:
        r = rules(s_val, s_test, y_val)
        pairs = [("median", "raw mean"), ("z-scored mean", "raw mean")]
        uniq = np.unique(io); idx = {c: np.where(io == c)[0] for c in uniq}
        rng = np.random.default_rng(0)
        draws = []
        for _ in range(5000):
            i = np.concatenate([idx[c] for c in rng.choice(uniq, len(uniq), replace=True)])
            draws.append(i if len(set(ref[i])) > 1 else None)
        print("\npaired cluster bootstrap on AUC->refusal (5000 draws over behaviour ids):")
        for a, base in pairs:
            va, vb = r[a][0][:nh], r[base][0][:nh]
            d = np.array([roc_auc_score(ref[i], va[i]) - roc_auc_score(ref[i], vb[i])
                          for i in draws if i is not None])
            lo, hi = np.percentile(d, [2.5, 97.5])
            verdict = "excludes 0" if lo > 0 or hi < 0 else "includes 0 — not separable here"
            print(f"  {a:14s} - {base:14s}  {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  {verdict}")

    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
