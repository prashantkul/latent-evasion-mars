"""Per-layer ROC curves + the mean-across-layers ROC, and per-layer AUC(->refusal) with CI band.
Agentic-native mean-difference probe (trained on 64 val samples), tested on 176 test_public rows."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import roc_curve, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "probe_all_layers.json")))
ref = np.array(d["refused"])
md = np.array(d["md_scores"])          # (176, 28)
mean_score = np.array(d["mean_score"])  # (176,)
NL = md.shape[1]
layers = list(range(NL))
md_auc = np.array([d["per_layer"][str(l)]["md_auc"] for l in layers])
md_lo = np.array([d["per_layer"][str(l)]["md_ci"][0] for l in layers])
md_hi = np.array([d["per_layer"][str(l)]["md_ci"][1] for l in layers])
ard = np.array([d["per_layer"][str(l)]["arditi_auc"] for l in layers])

OK = dict(blue="#0072B2", orange="#E69F00", grey="#666666", verm="#D55E00", green="#009E73")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130})
fig, ax = plt.subplots(1, 2, figsize=(14, 5.8))
fig.suptitle(f"Agentic-native probe (mean-diff, trained on {d['n_train']} val samples) — "
             f"per-layer AUC & ROC on {d['n_test']} test rows", fontsize=13, fontweight="bold")

# --- A: per-layer AUC with CI band ---
a = ax[0]
a.fill_between(layers, md_lo, md_hi, color=OK["blue"], alpha=0.18, label="95% CI (clustered)")
a.plot(layers, md_auc, color=OK["blue"], lw=2.2, marker="o", ms=3, label="agentic-native (mean-diff)")
a.plot(layers, ard, color=OK["orange"], lw=1.6, ls="--", label="Arditi probe (transferred)")
a.axhline(d["mean_auc"], color=OK["grey"], lw=1.4, ls=":", label=f"mean over layers ({d['mean_auc']:.2f})")
a.axhline(0.5, color=OK["grey"], lw=0.8, alpha=0.6)
a.set_xlabel("layer"); a.set_ylabel("AUC (probe score → refusal)")
a.set_ylim(0.45, 1.0); a.set_title("Per-layer AUC(→refusal) with CI", loc="left", fontweight="bold")
a.legend(loc="lower right", framealpha=0.9, fontsize=8.5)

# --- B: ROC curves, individual layers + mean ---
b = ax[1]
b.plot([0, 1], [0, 1], ls="--", color=OK["grey"], lw=1)
colors = cm.viridis(np.linspace(0, 1, NL))
for l in layers:
    fpr, tpr, _ = roc_curve(ref, md[:, l])
    b.plot(fpr, tpr, color=colors[l], lw=0.8, alpha=0.55)
fpr, tpr, _ = roc_curve(ref, mean_score)
b.plot(fpr, tpr, color="black", lw=2.6, label=f"mean over layers  (AUC {roc_auc_score(ref, mean_score):.2f})")
b.set_xlabel("false positive rate"); b.set_ylabel("true positive rate")
b.set_title("ROC — individual layers (viridis) + mean (black)", loc="left", fontweight="bold")
b.legend(loc="lower right", framealpha=0.9)
sm = cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, NL - 1))
cb = fig.colorbar(sm, ax=b, fraction=0.046, pad=0.04); cb.set_label("layer")

fig.tight_layout(rect=[0, 0, 1, 0.95])
out = os.path.join(HERE, "probe_roc_layers.png")
fig.savefig(out, bbox_inches="tight")
print("saved", out)
print(f"best layer: {int(np.argmax(md_auc))} (AUC {md_auc.max():.3f})  | mean-over-layers AUC {d['mean_auc']:.3f}")
