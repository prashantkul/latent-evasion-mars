"""Why val-CV layer selection is a lottery at N=64, and what to do instead.

Everything comes from the committed run summary (val CV per layer) and its held-out test AUCs,
so the picture is the actual selection problem we faced, not an illustration of one.

  uv run --with matplotlib --with numpy python \
      experiments/results/06-qwen35-inscorer-probe/plot_layer_selection.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OK = dict(blue="#0072B2", orange="#E69F00", red="#D7263D", grey="#B0B0B0",
          black="#111111", green="#009E73", purple="#CC79A7")
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130})

d = json.load(open(os.path.join(HERE, "rebaseline_2026-07-24", "inscorer_experiment.json")))
cv = np.array(d["val_cv_by_layer"])                       # what selection can see (val, n=64)
hb = np.array(d["svm"]["auc_harm_vs_benign"])             # held-out test, harm vs benign
ref = np.array(d["svm"]["auc_refusal"])                   # held-out test, -> model's own refusal
tied = np.where(cv >= 0.9999)[0]                          # layers indistinguishable on val
pick = int(cv.argmax())                                   # the layer val-CV actually chose
MEAN_HB, MEAN_REF = 0.9561, 0.7708                        # selection-free mean-over-layers

fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.1))
fig.suptitle("Layer selection at N=64 is a lottery — 38 of 64 layers tie at val CV AUC 1.000"
             "   ·   Qwen3.5-27B, LinearSVC",
             fontsize=12.5, fontweight="bold", x=0.012, ha="left")

# ---------------- Panel A: what selection sees vs what is true ----------------
a = ax[0]
for l in tied:
    a.axvspan(l - .5, l + .5, color=OK["orange"], alpha=.13, lw=0)
a.plot(cv, color=OK["orange"], lw=2.4, label="val CV AUC  (what selection sees)")
a.plot(hb, color=OK["blue"], lw=2.4, label="held-out test AUC  (the truth)")
a.axhline(MEAN_HB, color=OK["green"], lw=2.0, ls="-.",
          label=f"mean-over-layers, test  ({MEAN_HB:.3f})")
a.scatter([pick], [cv[pick]], s=110, color=OK["red"], zorder=6, edgecolor="white")
a.scatter([pick], [hb[pick]], s=110, color=OK["red"], zorder=6, edgecolor="white")
a.annotate(f"val-CV picks L{pick}\ntest {hb[pick]:.3f}", xy=(pick, hb[pick]),
           xytext=(pick + 6, hb[pick] - .22), color=OK["red"], fontsize=9.5,
           arrowprops=dict(arrowstyle="->", color=OK["red"], lw=1.2))
a.set_xlabel("layer")
a.set_ylabel("AUC (harmful vs benign)")
a.set_ylim(.45, 1.03)
a.set_title("A · A flat ceiling on val hides real variation on test", loc="left",
            fontweight="bold", fontsize=11.5)
a.legend(loc="lower right", framealpha=.93, fontsize=9)

# ---------------- Panel B: the spread you are gambling over ----------------
b = ax[1]
rng = np.random.default_rng(0)
jit = rng.uniform(-.09, .09, len(tied))
b.scatter(hb[tied], np.full(len(tied), 1) + jit, s=52, color=OK["blue"], alpha=.75,
          edgecolor="none", label="harmful vs benign")
b.scatter(ref[tied], np.zeros(len(tied)) + jit, s=52, color=OK["purple"], alpha=.75,
          edgecolor="none", label="→ model's own refusal")
for yv, arr, col in ((1, hb, OK["blue"]), (0, ref, OK["purple"])):
    lo, hi = arr[tied].min(), arr[tied].max()
    b.plot([lo, hi], [yv - .22, yv - .22], color=col, lw=2.2)
    b.text((lo + hi) / 2, yv - .34, f"spread {hi - lo:.3f}", ha="center", color=col, fontsize=9.5)
    b.scatter([arr[pick]], [yv + jit[list(tied).index(pick)]], s=150, marker="X",
              color=OK["red"], zorder=7, edgecolor="white", linewidth=1.2)
b.axvline(MEAN_HB, color=OK["blue"], lw=1.6, ls="-.", alpha=.8)
b.axvline(MEAN_REF, color=OK["purple"], lw=1.6, ls="-.", alpha=.8)
b.text(MEAN_HB, 1.42, " mean-over-\n layers", color=OK["blue"], fontsize=8.5, ha="center")
b.text(MEAN_REF, 1.42, " mean-over-\n layers", color=OK["purple"], fontsize=8.5, ha="center")
b.set_yticks([0, 1])
b.set_yticklabels(["→ refusal", "harm vs benign"])
b.set_xlabel("held-out test AUC of the 38 layers that tie on val")
b.set_ylim(-.55, 1.72)
b.set_title("B · Red X = the layer val-CV picked", loc="left", fontweight="bold", fontsize=11.5)
b.legend(loc="lower left", framealpha=.93, fontsize=9)

fig.tight_layout(rect=[0, 0, 1, 0.93])
out = os.path.join(HERE, "layer_selection_lottery.png")
fig.savefig(out, bbox_inches="tight")
print("saved", out)
print(f"  {len(tied)}/64 layers tie at val CV 1.000; val-CV picked L{pick}")
print(f"  among tied — harm-vs-benign test {hb[tied].min():.3f}..{hb[tied].max():.3f} "
      f"(spread {hb[tied].max()-hb[tied].min():.3f}); L{pick} lands at {hb[pick]:.3f}")
print(f"  among tied — ->refusal test {ref[tied].min():.3f}..{ref[tied].max():.3f} "
      f"(spread {ref[tied].max()-ref[tied].min():.3f}); L{pick} lands at {ref[pick]:.3f}")
print(f"  mean-over-layers beats every tied layer on harm-vs-benign: "
      f"{MEAN_HB:.4f} vs best tied {hb[tied].max():.4f}")
