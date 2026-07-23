"""ROC — probe score -> refusal, per layer. All 64 layers (faint), best layer (red), and the
mean-across-layers score (black). Qwen3.5-27B agentic mean-difference probe, val-trained (64),
tested on 176 test_public harmful rows. Matches the 03-agentic-native-probe figure.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(HERE, "qwen35_perlayer_test_scores.npz"), allow_pickle=True)
md = z["h_md"]                       # (176, 64) per-layer probe scores on test harmful
ref = z["h_ref"].astype(int)         # (176,) refused
NL = md.shape[1]
aucs = np.array([roc_auc_score(ref, md[:, l]) for l in range(NL)])
best = int(np.argmax(aucs))
mean_score = md.mean(1)
mean_auc = roc_auc_score(ref, mean_score)

OK = dict(grey="#B0B0B0", red="#D7263D", black="#111111")
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130})
fig, ax = plt.subplots(figsize=(7.6, 7))

ax.plot([0, 1], [0, 1], ls="--", color="#999999", lw=1, label="chance (0.50)")

for l in range(NL):
    if l == best:
        continue
    fpr, tpr, _ = roc_curve(ref, md[:, l])
    ax.plot(fpr, tpr, color=OK["grey"], lw=0.8, alpha=0.40,
            label="individual layers" if l == 0 else None)

fpr, tpr, _ = roc_curve(ref, mean_score)
ax.plot(fpr, tpr, color=OK["black"], lw=2.6, label=f"mean over layers  (AUC {mean_auc:.3f})")

fpr, tpr, _ = roc_curve(ref, md[:, best])
ax.plot(fpr, tpr, color=OK["red"], lw=2.8, label=f"best layer {best}  (AUC {aucs[best]:.3f})")

ax.set_xlabel("false positive rate")
ax.set_ylabel("true positive rate")
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
ax.set_title("ROC — probe score → refusal, per layer\n"
             f"Qwen3.5-27B agentic mean-diff probe · 64 val → {md.shape[0]} test rows",
             loc="left", fontweight="bold", fontsize=12)
ax.legend(loc="lower right", framealpha=0.92)

fig.tight_layout()
out = os.path.join(HERE, "probe_roc_layers.png")
fig.savefig(out, bbox_inches="tight")
print("saved", out, f"| best L{best} AUC {aucs[best]:.3f} | mean AUC {mean_auc:.3f}")
