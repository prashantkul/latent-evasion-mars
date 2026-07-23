"""SVM vs mean-difference probe on Qwen3.5-27B (held-out test). Left: per-layer AUC(->refusal) for
both probes. Right: test ROC at each probe's best layer, plus its mean-over-layers ensemble.

Shows the small-N result: the mean-difference direction beats the LinearSVC on this transfer task
(refusal prediction), and its mean-over-layers ensemble is far more stable.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "qwen35_svm_probe.json")))
z = np.load(os.path.join(HERE, "qwen35_svm_probe_scores.npz"), allow_pickle=True)
ref = z["ref"].astype(int)
svm_h, md_h = z["svm_h"], z["md_h"]           # (176, 64) per-layer test scores
NL = svm_h.shape[1]

OK = dict(blue="#0072B2", orange="#E69F00", grey="#B0B0B0", black="#111111")
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130})
fig, ax = plt.subplots(1, 2, figsize=(14, 6.4))
fig.suptitle("Qwen3.5-27B · mean-difference vs LinearSVC probe  ·  64 val → 176 test rows",
             fontsize=13, fontweight="bold", x=0.02, ha="left")

# ---- A: per-layer AUC -> refusal ----
a = ax[0]
md_auc = np.array(d["mean_diff"]["auc_refusal"]); svm_auc = np.array(d["svm"]["auc_refusal"])
a.axhline(0.5, ls="--", color=OK["grey"], lw=1, label="chance (0.50)")
a.plot(range(NL), md_auc, color=OK["blue"], lw=2.2, label="mean-difference")
a.plot(range(NL), svm_auc, color=OK["orange"], lw=2.2, label="LinearSVC")
bm, bs = d["mean_diff"]["best_layer"], d["svm"]["best_layer"]
a.scatter([bm], [md_auc[bm]], color=OK["blue"], s=80, zorder=5, edgecolor="white")
a.scatter([bs], [svm_auc[bs]], color=OK["orange"], s=80, zorder=5, edgecolor="white")
a.set_xlabel("transformer layer"); a.set_ylabel("test AUC (probe score → refusal)")
a.set_xlim(-1, NL); a.set_ylim(0.45, 1.02)
a.set_title("Per-layer refusal AUC · mean-diff dominates", loc="left", fontweight="bold", fontsize=12)
a.legend(loc="lower right", framealpha=0.92)

# ---- B: ROC at best layer + mean-over-layers ----
b = ax[1]
b.plot([0, 1], [0, 1], ls="--", color=OK["grey"], lw=1, label="chance (0.50)")
for scores, col, name, bl in [(md_h, OK["blue"], "mean-diff", bm), (svm_h, OK["orange"], "LinearSVC", bs)]:
    fpr, tpr, _ = roc_curve(ref, scores[:, bl])
    b.plot(fpr, tpr, color=col, lw=2.8, label=f"{name} · best L{bl}  (AUC {roc_auc_score(ref, scores[:, bl]):.3f})")
    ms = scores.mean(1)
    fpr, tpr, _ = roc_curve(ref, ms)
    b.plot(fpr, tpr, color=col, lw=1.8, ls=":", label=f"{name} · mean-layers  (AUC {roc_auc_score(ref, ms):.3f})")
b.set_xlabel("false positive rate"); b.set_ylabel("true positive rate")
b.set_xlim(-0.02, 1.02); b.set_ylim(-0.02, 1.02)
b.set_title("Test ROC · best layer (solid) & mean-over-layers (dotted)", loc="left", fontweight="bold", fontsize=12)
b.legend(loc="lower right", framealpha=0.92, fontsize=9.5)

fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(HERE, "probe_svm_vs_md.png")
fig.savefig(out, bbox_inches="tight")
print("saved", out, f"| MD best {md_auc[bm]:.3f} mean {roc_auc_score(ref, md_h.mean(1)):.3f} | "
      f"SVM best {svm_auc[bs]:.3f} mean {roc_auc_score(ref, svm_h.mean(1)):.3f}")
