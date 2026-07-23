"""Qwen3.5-27B agentic probe-monitor figure (tracked metrics: per-layer AUC + test ROC).

Left  — where the refusal direction lives: val-CV AUC(harmful vs benign) across all 64 layers,
        best layer highlighted, ≥0.99 plateau shaded.
Right — how it monitors on the held-out test set (176 harmful + 176 benign): ROC of probe score
        -> refusal, and ROC of harmful vs benign, with the val-threshold operating point marked.

Matches the 03-agentic-native-probe house style. No firing-fraction metric (retired).
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "qwen35_agentic_dissociation.json")))
z = np.load(os.path.join(HERE, "qwen35_agentic_dissociation_scores.npz"), allow_pickle=True)

cv = np.array(d["val_cv_auc"])                 # (64,) val-CV harmful/benign AUC per layer
best = int(d["best_layer"]); thr = float(d["threshold"])
hs, hr = z["h_scores"], z["h_ref"]             # test harmful (176): probe score, refused
bs = z["b_scores"]                             # test benign  (176): probe score

OK = dict(blue="#0072B2", orange="#E69F00", green="#009E73", verm="#D55E00",
          red="#D7263D", grey="#B0B0B0", black="#111111")
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130})
fig, ax = plt.subplots(1, 2, figsize=(14, 6.4))
fig.suptitle("Qwen3.5-27B · agentic refusal probe (mean-difference)  ·  64 val → 352 test rows",
             fontsize=13, fontweight="bold", x=0.02, ha="left")

# ---- A: per-layer AUC sweep (val) ----
a = ax[0]
plateau = np.where(cv >= 0.99)[0]
if len(plateau):
    a.axvspan(plateau.min(), plateau.max(), color=OK["green"], alpha=0.08,
              label=f"≥0.99 plateau (L{plateau.min()}–{plateau.max()})")
a.axhline(0.5, ls="--", color=OK["grey"], lw=1, label="chance (0.50)")
a.plot(np.arange(len(cv)), cv, color=OK["blue"], lw=2.2)
a.scatter([best], [cv[best]], color=OK["red"], s=90, zorder=5,
          label=f"best layer {best}  (AUC {cv[best]:.3f})")
a.set_xlabel("transformer layer"); a.set_ylabel("val-CV AUC (harmful vs benign)")
a.set_xlim(-1, len(cv)); a.set_ylim(0.45, 1.02)
a.set_title("Where the direction lives · per-layer separation", loc="left", fontweight="bold", fontsize=12)
a.legend(loc="lower right", framealpha=0.92)

# ---- B: test ROC (best layer) ----
b = ax[1]
b.plot([0, 1], [0, 1], ls="--", color=OK["grey"], lw=1, label="chance (0.50)")

fpr_r, tpr_r, _ = roc_curve(hr, hs)
auc_r = roc_auc_score(hr, hs)
b.plot(fpr_r, tpr_r, color=OK["red"], lw=2.8, label=f"probe → refusal  (AUC {auc_r:.3f})")

lab = np.concatenate([np.ones(len(hs)), np.zeros(len(bs))])
sco = np.concatenate([hs, bs])
fpr_h, tpr_h, _ = roc_curve(lab, sco)
auc_h = roc_auc_score(lab, sco)
b.plot(fpr_h, tpr_h, color=OK["blue"], lw=2.8, label=f"harmful vs benign  (AUC {auc_h:.3f})")

# val-threshold operating point on the harmful-vs-benign curve
op_tpr = float((hs > thr).mean()); op_fpr = float((bs > thr).mean())
b.scatter([op_fpr], [op_tpr], color=OK["black"], s=80, zorder=6,
          label=f"val threshold  (recall {op_tpr:.2f} · FPR {op_fpr:.2f})")

b.set_xlabel("false positive rate"); b.set_ylabel("true positive rate")
b.set_xlim(-0.02, 1.02); b.set_ylim(-0.02, 1.02)
b.set_title("How it monitors · held-out test ROC", loc="left", fontweight="bold", fontsize=12)
b.legend(loc="lower right", framealpha=0.92)

fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(HERE, "probe_monitor.png")
fig.savefig(out, bbox_inches="tight")
print("saved", out, f"| best L{best} val-AUC {cv[best]:.3f} | test AUC→refusal {auc_r:.3f} | harm-vs-benign {auc_h:.3f}")
