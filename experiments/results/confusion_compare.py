"""Side-by-side confusion matrices: LLM judge vs. linear probe, harmful/harmless on AgentHarm val."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
J = json.load(open(os.path.join(HERE, "judge_confusion.json")))
P = json.load(open(os.path.join(HERE, "probe_confusion.json")))


def panel(ax, d, title):
    M = np.array([[d["TP"], d["FN"]], [d["FP"], d["TN"]]])
    ax.imshow(M, cmap="Blues", vmin=0, vmax=M.max() * 1.15)
    for (i, j), v in np.ndenumerate(M):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=30,
                color="white" if v > M.max() * 0.6 else "#333333", fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred: harmful", "pred: harmless"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["true: harmful", "true: harmless"])
    ax.add_patch(Rectangle((-0.5, -0.5), 1, 1, fill=False, edgecolor="#009E73", lw=2.4))
    if d["FP"] > 0:
        ax.add_patch(Rectangle((-0.5, 0.5), 1, 1, fill=False, edgecolor="#D55E00", lw=2.4))
    ax.set_title(title + f"\nacc {d['accuracy']:.2f} · prec {d['precision']:.2f} · "
                 f"rec {d['recall']:.2f} · FPR {d['fpr']:.2f}", fontsize=11, fontweight="bold", loc="left")
    ax.grid(False)


fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.6))
fig.suptitle(f"Harmful/harmless monitor — LLM judge vs. linear probe  (AgentHarm val, N={J['n']})",
             fontsize=13, fontweight="bold")
panel(ax[0], J, "LLM judge  J(response, prompt)")
panel(ax[1], P, f"Probe  (Arditi, layer {P['layer']}, AUC {P.get('auc', float('nan')):.2f})")
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = os.path.join(HERE, "confusion_compare.png")
fig.savefig(out, bbox_inches="tight")
print("saved", out)
print(f"judge: acc {J['accuracy']:.3f} rec {J['recall']:.3f} FPR {J['fpr']:.3f}  |  "
      f"probe: acc {P['accuracy']:.3f} rec {P['recall']:.3f} FPR {P['fpr']:.3f} AUC {P.get('auc'):.3f}")
