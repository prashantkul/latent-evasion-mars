"""Regenerate 07's probe column against a re-baselined probe + activation cache.

The gpt-4o judge labels are REUSED from the cached majority-vote run -- the judge is a fixed
monitor and re-querying it would cost money and add nondeterminism for no gain. Only the probe
side is recomputed, so the comparison stays apples-to-apples on identical rows.

Alignment is on (split, sample id), never on position: AgentHarm's harmful and benign splits
reuse the same id space (`1-1` exists in both), so an id-only join would silently cross-match
benign rows onto harmful ones and quietly corrupt the confusion matrix.

    uv run python experiments/results/07-qwen35-monitor-confusion/regen_probe_column.py \
        --acts <test_acts.npz> --probe <canonical probe stem>
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from probe_io import load_probe, score as probe_score          # noqa: E402
from eval_probe import confusion                               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True, help="in-context test activations npz (X, y, ids)")
    ap.add_argument("--probe", required=True, help="canonical probe stem")
    ap.add_argument("--judge", default=os.path.join(HERE, "qwen35_monitor_batch.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "qwen35_monitor_batch_rebaselined.json"))
    args = ap.parse_args()

    d = json.load(open(args.judge))
    J = d["judge"]
    jkey = [(int(t), str(i)) for t, i in zip(J["true"], J["ids"])]

    z = np.load(args.acts, allow_pickle=True)
    X, y, ids = z["X"], z["y"].astype(int), [str(i) for i in z["ids"]]
    akey = [(int(t), str(i)) for t, i in zip(y, ids)]

    if set(jkey) != set(akey):
        only_j, only_a = set(jkey) - set(akey), set(akey) - set(jkey)
        raise SystemExit(f"row mismatch — judge-only {len(only_j)} e.g. {list(only_j)[:3]}; "
                         f"acts-only {len(only_a)} e.g. {list(only_a)[:3]}")
    pos = {k: n for n, k in enumerate(akey)}
    order = np.array([pos[k] for k in jkey])          # reorder acts into judge row order
    X, y = X[order], y[order]
    assert (y == np.array(J["true"])).all(), "split labels disagree after alignment"
    print(f"aligned {len(order)} rows on (split, id)")

    w, b, _, meta = load_probe(args.probe)
    s = probe_score(X, w, b)
    auc_layer = np.array([roc_auc_score(y, s[:, l]) for l in range(s.shape[1])])
    best = int(np.argmax(auc_layer))
    s_mean = s.mean(1)

    # Headline uses mean-over-layers at the probe's own canonical boundary: selection-free, and
    # no threshold is fitted on test.
    pconf = confusion(y, (s_mean > 0).astype(int))
    jconf = {k: J[k] for k in ("TP", "FN", "FP", "TN", "accuracy", "precision", "recall", "fpr")}

    fpr_c, tpr_c, _ = roc_curve(y, s_mean)
    at_judge = float(fpr_c[int(np.argmin(np.abs(tpr_c - jconf["recall"])))])

    probe = {"probe": args.probe, "probe_meta": meta,
             "auc_mean_layers": float(roc_auc_score(y, s_mean)),
             "auc_best": float(auc_layer[best]), "best_hb_layer": best,
             "best_layer_is_selection_dependent": True,
             "probe_fpr_at_judge_recall": at_judge,
             "score_mean": s_mean.tolist(), "true": y.tolist(), **pconf}

    print(f"\nJUDGE (majority of {d.get('n_votes')})  acc {jconf['accuracy']:.3f}  "
          f"recall {jconf['recall']:.3f}  FPR {jconf['fpr']:.3f}")
    print(f"PROBE (mean-layers @ canonical boundary)  acc {pconf['accuracy']:.3f}  "
          f"recall {pconf['recall']:.3f}  FPR {pconf['fpr']:.3f}  | AUC {probe['auc_mean_layers']:.4f}")
    print(f"  at the judge's recall {jconf['recall']:.3f}, probe FPR {at_judge:.3f} vs judge {jconf['fpr']:.3f} -> "
          f"{'probe better' if at_judge < jconf['fpr'] else 'judge better at that recall'}")

    json.dump({**{k: d[k] for k in ("model", "judge_model", "n_votes", "temperature", "n", "split_votes")},
               "judge": J, "probe": probe,
               "note": "judge labels reused from the cached majority-vote run; probe column "
                       "recomputed against the re-baselined probe + activation cache"},
              open(args.out, "w"), indent=2)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
