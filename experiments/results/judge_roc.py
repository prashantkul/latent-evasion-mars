"""ROC: the judge's continuous harm score as predictor vs the behavioural label (refusal).
No probe. Positive class = complied (did NOT refuse) — a consistency check on the judge's
two signals. Data = the M2f full run (results_full.json)."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "results_full.json")))
R = d["records"]
harm = np.array([r["harm"] for r in R])              # judge harm score (0..1)
refused = np.array([r["refused"] for r in R]).astype(int)
complied = 1 - refused                               # positive class: did NOT refuse
n = len(R)

OK = dict(blue="#0072B2", orange="#E69F00", grey="#666666", verm="#D55E00")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130})

fig, ax = plt.subplots(1, 2, figsize=(13, 5.6))
fig.suptitle(f"Judge harm score vs behavioural label  (Llama-3.2-3B · AgentHarm · N={n} rows)",
             fontsize=13, fontweight="bold")

# --- A: ROC ---
a = ax[0]
a.plot([0, 1], [0, 1], ls="--", color=OK["grey"], lw=1, label="chance (0.50)")
fpr, tpr, _ = roc_curve(complied, harm)
auc = roc_auc_score(complied, harm)
a.plot(fpr, tpr, color=OK["blue"], lw=2.4, label=f"harm → complied  (AUC {auc:.2f})")
a.set_xlabel("false positive rate  (refused, but high harm)")
a.set_ylabel("true positive rate  (complied, high harm)")
a.set_title("ROC — judge harm score predicts compliance", loc="left", fontweight="bold")
a.legend(loc="lower right", framealpha=0.9); a.set_xlim(-0.02, 1.02); a.set_ylim(-0.02, 1.02)

# --- B: harm-score distribution by refusal ---
b = ax[1]
bins = np.linspace(0, 1, 22)
b.hist(harm[complied == 1], bins=bins, color=OK["verm"], alpha=0.65, label="complied (not refused)")
b.hist(harm[refused == 1], bins=bins, color=OK["blue"], alpha=0.65, label="refused")
b.set_xlabel("judge harm score")
b.set_ylabel("count")
b.set_title("Harm score by refusal label", loc="left", fontweight="bold")
b.legend(framealpha=0.9)

fig.tight_layout(rect=[0, 0, 1, 0.95])
out = os.path.join(HERE, "judge_roc.png")
fig.savefig(out, bbox_inches="tight")
print("saved", out)
print(f"AUC(harm -> complied) = {auc:.3f}")
print(f"harm mean | refused={harm[refused==1].mean():.3f}  complied={harm[complied==1].mean():.3f}")
print(f"refused-but-harm>=0.5: {int(((refused==1)&(harm>=0.5)).sum())} of {int(refused.sum())} refusals")
