"""ROC curves: probe score (predictor) vs judge-derived labels (refusal, harm). Okabe-Ito."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "probe_metrics.json")))
R = d["records"]
score = np.array([r["score_good"] for r in R])   # mean probe score over good layers
harm = np.array([r["harm"] for r in R])
ref = np.array([r["refused"] for r in R]).astype(int)
harm_bin = (harm >= 0.5).astype(int)

OK = dict(blue="#0072B2", orange="#E69F00", grey="#666666", verm="#D55E00")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130})

fig, ax = plt.subplots(1, 2, figsize=(13, 5.6))
fig.suptitle(f"Probe score as a predictor of judge labels  (Llama-3.2-3B · AgentHarm · "
             f"N={d['N']} rows, {d['n_behaviors']} behaviors)", fontsize=13, fontweight="bold")

# --- A: ROC curves ---
a = ax[0]
a.plot([0, 1], [0, 1], ls="--", color=OK["grey"], lw=1, label="chance (0.50)")
for y, lab, col in [(ref, "refusal", OK["blue"]), (harm_bin, "harm ≥ 0.5", OK["orange"])]:
    if len(set(y)) > 1:
        fpr, tpr, _ = roc_curve(y, score)
        auc = roc_auc_score(y, score)
        a.plot(fpr, tpr, color=col, lw=2.2, label=f"{lab}  (AUC {auc:.2f})")
a.set_xlabel("false positive rate"); a.set_ylabel("true positive rate")
a.set_title("ROC — probe score vs judge label", loc="left", fontweight="bold")
a.legend(loc="lower right", framealpha=0.9); a.set_xlim(-0.02, 1.02); a.set_ylim(-0.02, 1.02)

# --- B: score distributions by refusal ---
b = ax[1]
bins = np.linspace(min(score), max(score), 22)
b.hist(score[ref == 0], bins=bins, color=OK["verm"], alpha=0.65, label="did NOT refuse")
b.hist(score[ref == 1], bins=bins, color=OK["blue"], alpha=0.65, label="refused")
b.axvline(0, ls="--", color=OK["grey"], lw=1)
b.set_xlabel("mean probe score over good layers (w·h + b)")
b.set_ylabel("count")
b.set_title("Score distribution by refusal", loc="left", fontweight="bold")
b.legend(framealpha=0.9)

fig.tight_layout(rect=[0, 0, 1, 0.95])
out = os.path.join(HERE, "probe_roc.png")
fig.savefig(out, bbox_inches="tight")
print("saved", out)
print(f"AUC refusal={roc_auc_score(ref, score):.3f}  AUC harm>=0.5={roc_auc_score(harm_bin, score):.3f}")
