"""Acceptance test for the cached in-context test activations.

The cache is only useful if scoring it with the committed canonical probe reproduces the
frozen baseline. This checks exactly that, so a corrupted or mis-ordered cache cannot
silently become the new reference.

    uv run python experiments/verify_test_acts_cache.py --acts <qwen35_inscorer_test_acts.npz>
"""
import argparse
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_io import load_probe, score as probe_score

HERE = os.path.dirname(os.path.abspath(__file__))
R06 = os.path.join(HERE, "results", "06-qwen35-inscorer-probe")

# Frozen in BASELINES.md, produced from the committed per-layer scores npz.
EXPECTED = {
    "qwen35_single_direction": {"auc_mean_layers": 0.9304, "auc_best": 0.9428, "best_layer": 24},
    "qwen35_svm": {"auc_mean_layers": 0.9561, "auc_best": 0.9479, "best_layer": 28},
}
TOL = 0.02          # generation is greedy but not bit-identical across runs/stacks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--tol", type=float, default=TOL)
    args = ap.parse_args()

    z = np.load(args.acts, allow_pickle=True)
    X, y = z["X"], z["y"].astype(int)
    print(f"cache {args.acts}")
    print(f"  X {X.shape}  y {int(y.sum())} harmful / {int((y == 0).sum())} benign")
    if X.ndim != 3:
        raise SystemExit(f"FAIL: expected (N, L, H), got {X.shape}")
    if X.shape[0] != 352:
        print(f"  [warn] expected 352 test_public rows, got {X.shape[0]}")
    if not np.isfinite(X).all():
        raise SystemExit("FAIL: non-finite values in cached activations")

    ok = True
    for name, exp in EXPECTED.items():
        w, b, _, _ = load_probe(os.path.join(R06, "probe_canonical", name))
        s = probe_score(X, w, b)
        auc_layer = np.array([roc_auc_score(y, s[:, l]) for l in range(s.shape[1])])
        got_mean = float(roc_auc_score(y, s.mean(1)))
        got_best, best_l = float(auc_layer.max()), int(auc_layer.argmax())
        d_mean, d_best = abs(got_mean - exp["auc_mean_layers"]), abs(got_best - exp["auc_best"])
        good = d_mean <= args.tol and d_best <= args.tol
        ok &= good
        print(f"  {name:26s} {'PASS' if good else 'FAIL'}  "
              f"mean-layers {got_mean:.4f} (exp {exp['auc_mean_layers']:.4f}, d {d_mean:.4f}) | "
              f"best L{best_l} {got_best:.4f} (exp L{exp['best_layer']} {exp['auc_best']:.4f}, d {d_best:.4f})")

    print("\nCACHE VERIFIED — eval_probe.py --acts is now the re-evaluation path." if ok else
          "\nCACHE MISMATCH — do not adopt as the baseline input until explained.")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
