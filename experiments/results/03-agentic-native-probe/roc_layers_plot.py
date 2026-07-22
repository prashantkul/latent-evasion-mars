"""ROC curves: all 28 individual layers (faint), the best layer (red), and the mean-across-layers
(black). Agentic-native mean-difference probe (trained on 64 val samples), tested on 176 test rows."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "probe_all_layers.json")))
ref = np.array(d["refused"])
md = np.array(d["md_scores"])            # (176, 28)
mean_score = np.array(d["mean_score"])   # (176,)
NL = md.shape[1]
md_auc = np.array([d["per_layer"][str(l)]["md_auc"] for l in range(NL)])
best_l = int(np.argmax(md_auc))

OK = dict(grey="#B0B0B0", red="#D7263D", black="#111111")
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130})
fig, ax = plt.subplots(figsize=(7.6, 7))

ax.plot([0, 1], [0, 1], ls="--", color="#999999", lw=1, label="chance (0.50)")

# all individual layers — faint grey background
for l in range(NL):
    if l == best_l:
        continue
    fpr, tpr, _ = roc_curve(ref, md[:, l])
    ax.plot(fpr, tpr, color=OK["grey"], lw=0.8, alpha=0.45,
            label="individual layers" if l == 0 else None)

# mean over layers — black
fpr, tpr, _ = roc_curve(ref, mean_score)
ax.plot(fpr, tpr, color=OK["black"], lw=2.6,
        label=f"mean over layers  (AUC {roc_auc_score(ref, mean_score):.2f})")

# best layer — red
fpr, tpr, _ = roc_curve(ref, md[:, best_l])
ax.plot(fpr, tpr, color=OK["red"], lw=2.8,
        label=f"best layer {best_l}  (AUC {md_auc[best_l]:.2f})")

ax.set_xlabel("false positive rate")
ax.set_ylabel("true positive rate")
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
ax.set_title(f"ROC — probe score → refusal, per layer\n"
             f"agentic-native mean-diff probe · {d['n_train']} val → {d['n_test']} test rows",
             loc="left", fontweight="bold", fontsize=12)
ax.legend(loc="lower right", framealpha=0.92)

fig.tight_layout()
out = os.path.join(HERE, "probe_roc_layers.png")
fig.savefig(out, bbox_inches="tight")
print("saved", out, "| best layer", best_l, f"AUC {md_auc[best_l]:.3f}")
