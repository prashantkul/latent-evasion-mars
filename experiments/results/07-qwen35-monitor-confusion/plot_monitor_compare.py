"""Qwen3.5-27B monitor comparison: gpt-4o judge vs in-context probe, both harmful/harmless on
test_public (176+176). Left/mid: confusion matrices. Right: probe ROC with the judge's fixed
operating point marked — showing the two sit at different points, not a domination.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics import roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "qwen35_monitor_confusion.json")))
J, P = d["judge"], d["probe"]
OK = dict(blue="#0072B2", orange="#E69F00", red="#D7263D", grey="#B0B0B0", black="#111111")
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130})
fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))
fig.suptitle("Qwen3.5-27B · harmful/harmless monitor · AgentHarm test_public (176 harmful + 176 benign)",
             fontsize=13, fontweight="bold", x=0.02, ha="left")


def cm(a, c, title, sub):
    M = np.array([[c["TP"], c["FN"]], [c["FP"], c["TN"]]])   # rows: pred-harmful/pred-harmless? -> use true x pred
    M = np.array([[c["TP"], c["FP"]], [c["FN"], c["TN"]]])   # rows pred(1/0), cols true(harmful/benign)
    a.imshow(M, cmap="Blues", vmin=0, vmax=176)
    for (i, j), v in np.ndenumerate(M):
        a.text(j, i, str(v), ha="center", va="center", fontsize=24,
               color="white" if v > 90 else OK["black"], fontweight="bold")
    a.set_xticks([0, 1]); a.set_xticklabels(["harmful", "benign"])
    a.set_yticks([0, 1]); a.set_yticklabels(["flag harmful", "flag benign"])
    a.set_xlabel("ground truth"); a.grid(False)
    a.set_title(f"{title}\n{sub}", loc="left", fontweight="bold", fontsize=12)
    # red box on the false-positive cell (benign flagged harmful)
    a.add_patch(Rectangle((0.5, -0.5), 1, 1, fill=False, edgecolor=OK["red"], lw=2.5))


cm(ax[0], J, "gpt-4o judge  J(resp, prompt)",
   f"acc {J['accuracy']:.2f} · recall {J['recall']:.2f} · FPR {J['fpr']:.2f}")
cm(ax[1], P, "in-context probe (mean-diff) @ val threshold",
   f"acc {P['accuracy']:.2f} · recall {P['recall']:.2f} · FPR {P['fpr']:.2f}")
ax[1].text(0.5, 1.9, "red box = benign flagged harmful (false positives)",
           transform=ax[1].transData, ha="center", fontsize=8.5, color=OK["red"])

# ROC of the probe with the judge point
b = ax[2]
yp = np.array(P["true"]); s = np.array(P["score_best"])
fpr, tpr, _ = roc_curve(yp, s)
b.plot([0, 1], [0, 1], ls="--", color=OK["grey"], lw=1, label="chance")
b.plot(fpr, tpr, color=OK["blue"], lw=2.8, label=f"probe ROC (AUC {P['auc_best']:.3f})")
b.scatter([P["fpr"]], [P["recall"]], color=OK["blue"], s=90, zorder=5, edgecolor="white",
          label=f"probe @val-thr ({P['fpr']:.2f}, {P['recall']:.2f})")
b.scatter([J["fpr"]], [J["recall"]], color=OK["red"], marker="*", s=320, zorder=6, edgecolor="white",
          label=f"gpt-4o judge ({J['fpr']:.2f}, {J['recall']:.2f})")
b.set_xlabel("false positive rate (benign flagged)"); b.set_ylabel("recall (harmful caught)")
b.set_xlim(-0.02, 1.02); b.set_ylim(-0.02, 1.02)
b.set_title("Operating points · different trade-offs", loc="left", fontweight="bold", fontsize=12)
b.legend(loc="lower right", framealpha=0.92, fontsize=9)

fig.tight_layout(rect=[0, 0, 1, 0.95])
out = os.path.join(HERE, "monitor_compare.png")
fig.savefig(out, bbox_inches="tight")
print("saved", out)
