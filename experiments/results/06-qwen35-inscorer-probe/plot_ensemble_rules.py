"""Is the mean-over-layers score a real ensemble, and is it the right combining rule?

Both panels come from the held-out test activations scored by the frozen probe, so this is the
decision we actually made, not an illustration of it.

  uv run --with matplotlib --with numpy --with scikit-learn python \
      experiments/results/06-qwen35-inscorer-probe/plot_ensemble_rules.py
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
from probe_io import load_probe, score as probe_score

OK = dict(blue="#0072B2", orange="#E69F00", red="#D7263D", grey="#B0B0B0",
          black="#111111", green="#009E73", purple="#CC79A7")
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130})

RB = os.path.join(HERE, "rebaseline_2026-07-24")
w, b, _layers, meta = load_probe(os.path.join(HERE, "probe_canonical", "qwen35_svm"))
t = np.load(os.path.join(RB, "test_acts.npz"), allow_pickle=True)
v = np.load(os.path.join(RB, "val_acts.npz"), allow_pickle=True)
s = probe_score(t["X"], w, b); y = t["y"].astype(int)
sv, yv = probe_score(v["X"], w, b), v["y"].astype(int)
ref = np.load(os.path.join(RB, "scores_and_probe.npz"), allow_pickle=True)["ref"].astype(int)
nh, L = int(y.sum()), s.shape[1]
bl = int(meta["val_cv_best_layer"])

fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.1))
fig.suptitle("Combining 64 per-layer probes: the mean is a real ensemble, and it beats picking one"
             "   ·   Qwen3.5-27B, held-out test n=352",
             fontsize=12.5, fontweight="bold", x=0.012, ha="left")

# ---------------- Panel A: is the mean dominated by a few loud layers? ----------------
a = ax[0]
scale = s.std(0)
corr = np.array([np.corrcoef(s.mean(1), s[:, l])[0, 1] for l in range(L)])
a.bar(np.arange(L), scale, color=OK["grey"], width=.85, label="per-layer score scale (std)")
a.set_xlabel("layer"); a.set_ylabel("score std on test", color=OK["black"])
a.set_ylim(0, scale.max() * 1.62)                     # headroom so annotations clear the data
a2 = a.twinx(); a2.spines["top"].set_visible(False)
a2.plot(corr, color=OK["blue"], lw=2.4, label="corr(layer, mean-over-layers)")
a2.set_ylabel("correlation with the ensemble score", color=OK["blue"])
a2.set_ylim(0, 1.42); a2.set_yticks([0, .25, .5, .75, 1.0])
a2.tick_params(axis="y", colors=OK["blue"])
a2.axhline(corr.max(), color=OK["blue"], lw=1, ls="--", alpha=.55)
a2.text(L - 1, corr.max() + .04, f"max r={corr.max():.3f} at L{int(np.argmax(corr))} — "
        "the mean is not a stand-in for any one layer", ha="right", fontsize=9, color=OK["blue"])
a.text(.98, .55, f"per-layer scale spans only {scale.max()/scale.min():.1f}× min-to-max,\n"
       "so no layer shouts over the others", transform=a.transAxes, ha="right", va="top",
       fontsize=9, color="#555")
a.set_title("A · No single layer stands in for the mean", loc="left", fontweight="bold", fontsize=11.5)
h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
a.legend(h1 + h2, l1 + l2, loc="upper left", framealpha=.93, fontsize=9)

# ---------------- Panel B: the rules, on both objectives ----------------
mu, sd = sv.mean(0), sv.std(0) + 1e-9
rules = {
    "best layer\n(val-CV pick)": s[:, bl],
    "vote fraction": (s > 0).mean(1),
    "median": np.median(s, 1),
    "raw mean": s.mean(1),
    "z-scored mean": ((s - mu) / sd).mean(1),
}
names = list(rules)
hb = [roc_auc_score(y, val) for val in rules.values()]
rf = [roc_auc_score(ref, val[:nh]) for val in rules.values()]
yy = np.arange(len(names))
b_ = ax[1]
X0 = .58
# Lollipops rather than bars: the panel stays mostly white, so the annotations that carry the
# argument are readable instead of sitting on top of ink.
for i, (u, r) in enumerate(zip(hb, rf)):
    b_.plot([X0, u], [i + .17] * 2, color=OK["blue"], lw=1.1, alpha=.35, zorder=1)
    b_.plot([X0, r], [i - .17] * 2, color=OK["purple"], lw=1.1, alpha=.35, zorder=1)
    b_.text(u + .008, i + .17, f"{u:.4f}", va="center", fontsize=9, color=OK["blue"])
    b_.text(r + .008, i - .17, f"{r:.4f}", va="center", fontsize=9, color=OK["purple"])
b_.scatter(hb, yy + .17, s=95, color=OK["blue"], zorder=3, label="harmful vs benign")
b_.scatter(rf, yy - .17, s=95, color=OK["purple"], zorder=3, label="→ model's own refusal")
b_.set_yticks(yy); b_.set_yticklabels(names)
b_.set_xlim(X0, 1.03); b_.set_ylim(-.75, len(names) - .25)
b_.set_xlabel("held-out test AUC")
b_.axvline(max(hb), color=OK["blue"], lw=1, ls=":", alpha=.6)
b_.get_yticklabels()[names.index("raw mean")].set_fontweight("bold")
i_mean, i_med = names.index("raw mean"), names.index("median")
b_.annotate("pinned rule", xy=(hb[i_mean] - .002, i_mean + .17), xytext=(.80, i_mean + .62),
            fontsize=9.5, color=OK["green"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=OK["green"], lw=1.4))
b_.annotate("median leads here: +0.055,\nCI [+0.002, +0.118] — open", xy=(rf[i_med], i_med - .17),
            xytext=(.615, i_med - .95), fontsize=9, color=OK["purple"],
            arrowprops=dict(arrowstyle="->", color=OK["purple"], lw=1.2))
b_.set_title("B · Every ensemble beats the selected layer", loc="left", fontweight="bold", fontsize=11.5)
b_.legend(loc="lower right", framealpha=.95, fontsize=9)

fig.tight_layout(rect=[0, 0, 1, 0.93])
out = os.path.join(HERE, "ensemble_rules.png")
fig.savefig(out, bbox_inches="tight")
print("saved", out)
print(f"  scale spread {scale.max()/scale.min():.2f}x | max corr with mean r={corr.max():.3f} (L{int(np.argmax(corr))})")
order = np.argsort(scale)[::-1]
for k in (0, 5, 10):
    keep = np.setdiff1d(np.arange(L), order[:k])
    print(f"  drop {k:2d} loudest -> AUC {roc_auc_score(y, s[:, keep].mean(1)):.4f}")
for n, u, r in zip(names, hb, rf):
    print(f"  {n.replace(chr(10), ' '):22s} h-v-b {u:.4f}   ->refusal {r:.4f}")
