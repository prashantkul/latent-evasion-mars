"""Confusion matrix plot for the judge-as-classifier (harmful/harmless) on the val set."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "judge_confusion.json")))
M = np.array([[d["TP"], d["FN"]], [d["FP"], d["TN"]]])   # rows true harmful/harmless, cols pred harmful/harmless

fig, ax = plt.subplots(figsize=(6.2, 5.6))
ax.imshow(M, cmap="Blues", vmin=0, vmax=M.max() * 1.15)
for (i, j), v in np.ndenumerate(M):
    ax.text(j, i, str(v), ha="center", va="center", fontsize=34,
            color="white" if v > M.max() * 0.6 else "#333333", fontweight="bold")
ax.set_xticks([0, 1]); ax.set_xticklabels(["pred: harmful", "pred: harmless"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["true: harmful", "true: harmless"])
ax.set_title(f"Judge as harmful/harmless classifier — AgentHarm val (N={d['n']})\n"
             f"acc {d['accuracy']:.2f} · precision {d['precision']:.2f} · recall {d['recall']:.2f} · "
             f"FPR {d['fpr']:.2f}", fontsize=11, fontweight="bold", loc="left")
ax.add_patch(Rectangle((-0.5, -0.5), 1, 1, fill=False, edgecolor="#009E73", lw=2.5))   # TP
ax.add_patch(Rectangle((-0.5, 0.5), 1, 1, fill=False, edgecolor="#D55E00", lw=2.5))    # FP
ax.grid(False)
fig.tight_layout()
out = os.path.join(HERE, "judge_confusion.png")
fig.savefig(out, bbox_inches="tight")
print("saved", out)

# which benign were flagged harmful (false positives)?
true = np.array(d["true"]); pred = np.array(d["pred"]); cat = d["cat"]
fp = [(i, cat[i]) for i in range(len(true)) if true[i] == 0 and pred[i] == 1]
print("false positives (benign flagged harmful):", fp)
