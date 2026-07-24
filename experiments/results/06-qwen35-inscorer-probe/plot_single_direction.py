"""What the single-direction (mean-difference) probe actually is, drawn from real activations.

Uses the HELD-OUT test cache (352 in-context activations), not the val split the probe was fit
on — an in-sample picture would show separation the probe didn't earn.

  uv run --with matplotlib --with numpy --with scikit-learn python \
      experiments/results/06-qwen35-inscorer-probe/plot_single_direction.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from probe_io import load_probe, score as probe_score          # noqa: E402

LAYER = 24                     # mean-difference best harm-vs-benign layer
OK = dict(blue="#0072B2", orange="#E69F00", red="#D7263D", grey="#B0B0B0",
          black="#111111", green="#009E73", purple="#CC79A7")
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130})

acts = os.path.join(HERE, "rebaseline_2026-07-24", "test_acts.npz")
z = np.load(acts, allow_pickle=True)
X, y = z["X"], z["y"].astype(int)
w_sd, b_sd, _, _ = load_probe(os.path.join(HERE, "probe_canonical", "qwen35_single_direction"))
w_svm, b_svm, _, _ = load_probe(os.path.join(HERE, "probe_canonical", "qwen35_svm"))

H = X[:, LAYER, :].astype(np.float64)          # (N, 5120) activations at one layer
d = w_sd[LAYER].astype(np.float64)             # the unit direction
b = float(b_sd[LAYER])
proj = H @ d                                   # signed position along the direction
boundary = -b                                  # score = proj + b > 0  <=>  proj > -b

# Second axis: the widest direction ORTHOGONAL to d, so the plane is a faithful slice
# (nothing of the probe's axis leaks into the vertical spread).
R = H - np.outer(proj, d)
R -= R.mean(0)
_, _, Vt = np.linalg.svd(R, full_matrices=False)
perp = R @ Vt[0]

fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.0))
fig.suptitle("The single-direction probe — one vector, from the difference of two class means"
             f"   ·   Qwen3.5-27B, layer {LAYER}, held-out AgentHarm test_public",
             fontsize=12.5, fontweight="bold", x=0.012, ha="left")

# ---------------- Panel A: the geometry ----------------
a = ax[0]
a.scatter(proj[y == 0], perp[y == 0], s=26, c=OK["blue"], alpha=.62, label="harmless (benign)",
          edgecolor="none")
a.scatter(proj[y == 1], perp[y == 1], s=26, c=OK["orange"], alpha=.62, label="harmful", edgecolor="none")
mu1, mu0 = proj[y == 1].mean(), proj[y == 0].mean()
p1, p0 = perp[y == 1].mean(), perp[y == 0].mean()
a.annotate("", xy=(mu1, p1), xytext=(mu0, p0),
           arrowprops=dict(arrowstyle="-|>", lw=2.6, color=OK["black"], shrinkA=0, shrinkB=0))
a.scatter([mu0, mu1], [p0, p1], s=190, marker="X", c=[OK["blue"], OK["orange"]],
          edgecolor=OK["black"], linewidth=1.6, zorder=6)
a.text((mu0 + mu1) / 2, (p0 + p1) / 2 + 3.4, "w = μ$_{harmful}$ − μ$_{harmless}$",
       ha="center", fontsize=10.5, fontweight="bold")
a.axvline(boundary, color=OK["red"], lw=2.2, ls="--")
a.annotate("boundary: midpoint\nbetween the centroids", xy=(boundary, perp.min() * .78),
           xytext=(boundary - (proj.max() - proj.min()) * .30, perp.min() * .60),
           color=OK["red"], fontsize=9, ha="center",
           arrowprops=dict(arrowstyle="->", color=OK["red"], lw=1.2))
a.set_xlabel("projection onto w  (signed distance along the direction)")
a.set_ylabel("widest orthogonal direction")
a.set_title("A · Two clouds, one line between their centres", loc="left", fontweight="bold", fontsize=11.5)
a.legend(loc="lower right", framealpha=.92, fontsize=9)

# ---------------- Panel B: collapse to the 1-D score ----------------
c = ax[1]
bins = np.linspace(proj.min(), proj.max(), 34)
c.hist(proj[y == 0], bins=bins, color=OK["blue"], alpha=.72, label="harmless")
c.hist(proj[y == 1], bins=bins, color=OK["orange"], alpha=.72, label="harmful")
c.axvline(boundary, color=OK["red"], lw=2.2, ls="--")
pred = (proj > boundary).astype(int)
acc = (pred == y).mean()
rec = pred[y == 1].mean()
fpr = pred[y == 0].mean()
c.set_title("B · That is the whole classifier", loc="left", fontweight="bold", fontsize=11.5)
c.set_xlabel("score = x·w + b   (fires when > 0)")
c.set_ylabel("prompts")
c.text(.03, .97, f"AUC {roc_auc_score(y, proj):.3f}\nacc {acc:.3f}   recall {rec:.3f}   FPR {fpr:.3f}",
       transform=c.transAxes, va="top", fontsize=9.5,
       bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=OK["grey"]))
c.legend(loc="upper right", framealpha=.92, fontsize=9)

# ---------------- Panel C: it is not the same vector the SVM finds ----------------
e = ax[2]
L = X.shape[1]
cos = np.array([float(w_sd[l] @ w_svm[l] / (np.linalg.norm(w_sd[l]) * np.linalg.norm(w_svm[l]) + 1e-12))
                for l in range(L)])
s_sd, s_svm = probe_score(X, w_sd, b_sd), probe_score(X, w_svm, b_svm)
auc_sd = np.array([roc_auc_score(y, s_sd[:, l]) for l in range(L)])
auc_svm = np.array([roc_auc_score(y, s_svm[:, l]) for l in range(L)])
e.plot(auc_sd, color=OK["blue"], lw=2.4, label="single-direction AUC")
e.plot(auc_svm, color=OK["green"], lw=2.4, label="LinearSVC AUC")
e.plot(cos, color=OK["purple"], lw=1.9, ls=":", label="cos(w$_{sd}$, w$_{svm}$)")
e.axhline(.5, color=OK["grey"], lw=1, ls="--")
e.axvline(LAYER, color=OK["red"], lw=1.4, alpha=.55)
e.text(LAYER + 1, .06, f"layer {LAYER}", color=OK["red"], fontsize=9)
e.set_xlabel("layer")
e.set_ylabel("AUC  /  cosine similarity")
e.set_ylim(-0.05, 1.05)
e.set_title("C · Two different vectors, both work", loc="left", fontweight="bold", fontsize=11.5)
e.legend(loc="lower right", framealpha=.92, fontsize=9)

fig.tight_layout(rect=[0, 0, 1, 0.93])
out = os.path.join(HERE, "single_direction_probe.png")
fig.savefig(out, bbox_inches="tight")
print("saved", out)
print(f"  layer {LAYER}: AUC {roc_auc_score(y, proj):.4f}  acc {acc:.3f}  recall {rec:.3f}  FPR {fpr:.3f}")
print(f"  cos(w_sd, w_svm) at L{LAYER}: {cos[LAYER]:.3f}   |  mean over layers {cos.mean():.3f}")
