"""Visualize the M3 full graded run. Okabe-Ito palette (colorblind-safe)."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "results_full.json")))
R = d["records"]
fire = np.array([r["fire"] for r in R])          # good-layer firing fraction 0..1
l18 = np.array([r["l18"] for r in R])            # probe score at layer 18
harm = np.array([r["harm"] for r in R])          # graded harm 0..1
ref = np.array([r["refused"] for r in R])        # refused (judge)
cat = np.array([r["cat"] for r in R])
f = fire >= 0.5

OK = dict(blue="#0072B2", orange="#E69F00", green="#009E73", verm="#D55E00",
          sky="#56B4E9", purple="#CC79A7", grey="#666666")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 130})

fig, ax = plt.subplots(2, 2, figsize=(14, 10.5))
fig.suptitle("Refusal in the agent loop — Llama-3.2-3B on AgentHarm (n=44 behaviors, graded)",
             fontsize=14, fontweight="bold")

# --- A: headline metrics with 95% bootstrap CI ---
a = ax[0, 0]
labels = {"refusal_rate (judge)": "refusal rate (judge)",
          "harm_score (graded)": "harm score (graded)",
          "probe_fires_rate": "probe fires (any behavior)",
          "P(fire | did-not-refuse)": "P(fire | did NOT refuse)",
          "P(fire & did-not-refuse)": "P(fire & did NOT refuse)"}
keys = list(labels)
pts = [d["stats"][k]["point"] for k in keys]
lo = [d["stats"][k]["lo"] for k in keys]
hi = [d["stats"][k]["hi"] for k in keys]
y = np.arange(len(keys))[::-1]
cols = [OK["grey"], OK["orange"], OK["blue"], OK["verm"], OK["verm"]]
a.hlines(y, lo, hi, color=cols, lw=2.5)
a.scatter(pts, y, color=cols, s=70, zorder=3)
a.axvline(0.5, ls="--", color=OK["grey"], lw=1)
for yi, k in zip(y, keys):
    a.text(d["stats"][k]["hi"] + 0.02, yi, f"{d['stats'][k]['point']:.2f}", va="center", fontsize=9)
a.set_yticks(y); a.set_yticklabels([labels[k] for k in keys])
a.set_xlim(0, 1.08); a.set_xlabel("rate")
a.set_title("Headline metrics · 95% bootstrap CI", loc="left", fontweight="bold")

# --- B: per-category (grouped bars) ---
b = ax[0, 1]
cats = sorted(set(cat))
agg = {c: dict(ref=ref[cat == c].mean(), harm=harm[cat == c].mean(), fire=f[cat == c].mean())
       for c in cats}
order = sorted(cats, key=lambda c: agg[c]["fire"])
yp = np.arange(len(order)); h = 0.26
b.barh(yp + h, [agg[c]["fire"] for c in order], height=h, color=OK["blue"], label="probe fires")
b.barh(yp, [agg[c]["harm"] for c in order], height=h, color=OK["orange"], label="harm score")
b.barh(yp - h, [agg[c]["ref"] for c in order], height=h, color=OK["green"], label="refusal rate")
b.set_yticks(yp); b.set_yticklabels(order)
b.set_xlim(0, 1.05); b.set_xlabel("rate")
b.legend(loc="lower right", framealpha=0.9)
b.set_title("By harm category", loc="left", fontweight="bold")

# --- C: per-behavior scatter, probe magnitude vs harm, by refusal ---
c = ax[1, 0]
c.axvspan(0, l18.max() + 0.2, color=OK["verm"], alpha=0.06)
c.axvline(0, ls="--", color=OK["grey"], lw=1)
m = ~ref
c.scatter(l18[m], harm[m], color=OK["verm"], s=55, edgecolor="white", lw=0.6, label="did NOT refuse")
c.scatter(l18[ref], harm[ref], color=OK["blue"], s=55, edgecolor="white", lw=0.6, label="refused")
c.set_xlabel("probe score @ layer 18  (w·h + b) — fires when > 0")
c.set_ylabel("graded harm score")
c.text(0.98, 0.96, "dissociation zone\n(fires + didn't refuse)", transform=c.transAxes,
       ha="right", va="top", color=OK["verm"], fontsize=9, fontweight="bold")
c.legend(loc="upper left", framealpha=0.9)
c.set_title("Per-behavior · probe magnitude vs harm", loc="left", fontweight="bold")

# --- D: 2x2 contingency (behaviour x representation) ---
dax = ax[1, 1]
M = np.array([[int((f & ~ref).sum()), int((f & ref).sum())],
              [int((~f & ~ref).sum()), int((~f & ref).sum())]])
dax.imshow(M, cmap="Blues", vmin=0, vmax=M.max() * 1.15)
for (i, j), v in np.ndenumerate(M):
    dax.text(j, i, str(v), ha="center", va="center", fontsize=26,
             color="white" if v > M.max() * 0.6 else OK["grey"], fontweight="bold")
dax.set_xticks([0, 1]); dax.set_xticklabels(["did NOT refuse", "refused"])
dax.set_yticks([0, 1]); dax.set_yticklabels(["probe FIRES", "probe silent"])
dax.add_patch(Rectangle((-0.5, -0.5), 1, 1, fill=False, edgecolor=OK["verm"], lw=3))
dax.text(0, -0.62, "DISSOCIATION", ha="center", color=OK["verm"], fontsize=9, fontweight="bold")
dax.set_title("Behaviour × representation · counts", loc="left", fontweight="bold")
dax.grid(False)

fig.tight_layout(rect=[0, 0, 1, 0.97])
out = os.path.join(HERE, "m3_overview.png")
fig.savefig(out, bbox_inches="tight")
print("saved", out)
print("2x2 [rows fire/silent x cols notref/ref]:\n", M)
