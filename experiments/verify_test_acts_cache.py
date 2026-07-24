"""Sanity-check a cached in-context activation set, and report drift vs the frozen baseline.

Two separable questions, deliberately not conflated:

1. **Is the cache structurally sound?** Right shape, right N, finite, labels present. A failure
   here is a real defect and exits non-zero.
2. **How far does it sit from the previous baseline?** Scored with the *previously committed*
   probe. This is a cross-run number -- a probe fit on one run's val activations, applied to
   another run's test activations -- so a gap measures stack/environment drift, not an error.
   Reported, never gating.

Re-baselining is the expected workflow: a fresh two-pass run produces a probe AND activations
from one environment, so the self-consistent pair is what teammates should be compared against.
Canonicalize the new probe with `probe_io.py --scores ... --val-acts ...`, then use
`eval_probe.py` to produce the new frozen numbers. This script exists to make the size of that
shift visible rather than silent.

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
BASELINE = {
    "qwen35_single_direction": {"auc_mean_layers": 0.9304, "auc_best": 0.9428, "best_layer": 24},
    "qwen35_svm": {"auc_mean_layers": 0.9561, "auc_best": 0.9479, "best_layer": 28},
}
NOTABLE = 0.02          # drift beyond this is worth explaining before re-baselining


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--probe-dir", default=os.path.join(R06, "probe_canonical"),
                    help="probes to score with (default: the previously committed ones)")
    ap.add_argument("--expect-n", type=int, default=352)
    args = ap.parse_args()

    z = np.load(args.acts, allow_pickle=True)
    X, y = z["X"], z["y"].astype(int)
    print(f"cache {args.acts}")
    print(f"  X {X.shape}  y {int(y.sum())} harmful / {int((y == 0).sum())} benign")

    # ---- 1. structural integrity (gating) ----
    problems = []
    if X.ndim != 3:
        problems.append(f"expected (N, L, H), got {X.shape}")
    elif X.shape[0] != len(y):
        problems.append(f"X has {X.shape[0]} rows but y has {len(y)}")
    elif X.shape[0] != args.expect_n:
        problems.append(f"expected {args.expect_n} rows, got {X.shape[0]}")
    if X.size and not np.isfinite(X).all():
        problems.append("non-finite values in activations")
    if len(set(y.tolist())) < 2:
        problems.append("only one class present in y")
    if problems:
        for p in problems:
            print(f"  STRUCTURAL FAILURE: {p}")
        raise SystemExit(1)
    print("  structure OK — shape, labels and finiteness all sound")

    # ---- 2. drift vs the previous baseline (informational) ----
    print(f"\ndrift vs frozen BASELINES.md, scored with probes in {os.path.relpath(args.probe_dir, HERE)}")
    print("  (cross-run: previous run's probe on this run's activations — a gap is stack drift, not an error)")
    worth_a_look = False
    for name, exp in BASELINE.items():
        w, b, _, _ = load_probe(os.path.join(args.probe_dir, name))
        s = probe_score(X, w, b)
        auc_layer = np.array([roc_auc_score(y, s[:, l]) for l in range(s.shape[1])])
        got_mean, got_best, best_l = float(roc_auc_score(y, s.mean(1))), float(auc_layer.max()), int(auc_layer.argmax())
        d = got_mean - exp["auc_mean_layers"]
        worth_a_look |= abs(d) > NOTABLE
        print(f"  {name:26s} mean-layers {got_mean:.4f} (was {exp['auc_mean_layers']:.4f}, {d:+.4f})"
              f" | best L{best_l} {got_best:.4f} (was L{exp['best_layer']} {exp['auc_best']:.4f})")

    print("\nCache is usable." + (
        f"\nDrift exceeds {NOTABLE} on at least one probe — worth understanding before re-baselining "
        "(transformers version and generation nondeterminism are the usual causes)."
        if worth_a_look else
        f"\nDrift within {NOTABLE} — this run reproduces the previous baseline."))
    print("\nTo re-baseline on this run (recommended — probe and activations from one environment):")
    print("  uv run python experiments/probe_io.py --scores <run>_scores.npz --val-acts <run>_val_acts.npz \\")
    print("      --out-dir <dest> --note 're-baselined <date>'")
    print("  uv run python experiments/eval_probe.py --probe <dest>/qwen35_svm --acts <this file>")


if __name__ == "__main__":
    main()
