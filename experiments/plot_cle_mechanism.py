"""How CLE-P and CLE-A actually move an activation — drawn from first principles.

Panel A  the one idea both attacks share: a probe is a ruler, and steering slides a point along it.
Panel B  CLE-P recomputes the push at every token; CLE-A measures it once and reuses it.
Panel C  the consequence on a real sequence: CLE-P lands every token exactly on target, CLE-A
         lands the token it was measured on and drifts elsewhere.

Panels A and B are schematic (a 2-D stand-in for a 5120-D space). Panel C uses the real probe and
the real cached activations, so the drift is measured rather than drawn.

  uv run --with matplotlib --with numpy python experiments/plot_cle_mechanism.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from probe_io import load_probe

R06 = os.path.join(HERE, "results", "06-qwen35-inscorer-probe")
RB = os.path.join(R06, "rebaseline_2026-07-24")
R08 = os.path.join(HERE, "results", "08-agentic-cle-pilot")
RED, BLUE, GREEN, GREY = "#D7263D", "#0072B2", "#009E73", "#9099A2"
np.seterr(all="ignore")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130})

fig = plt.figure(figsize=(16.5, 5.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.15, 1.25], wspace=.28)
fig.suptitle("What steering actually does to one number", fontsize=13.5, fontweight="bold",
             x=0.012, ha="left")

# ---------------------------------------------------------------- A: the ruler
a = fig.add_subplot(gs[0])
a.set_title("A · the probe is a ruler", loc="left", fontweight="bold", fontsize=11.5)
a.axhline(0, color="#333", lw=1.6)
a.axvline(0, color="#111", lw=2.2)
a.text(.12, 1.16, "boundary\nscore = 0", ha="left", fontsize=8.5, color="#111")
a.axvspan(0, 3.2, color=RED, alpha=.07, lw=0)
a.axvspan(-3.2, 0, color=BLUE, alpha=.07, lw=0)
a.text(1.6, -1.55, "probe says\nHARMFUL", ha="center", color=RED, fontsize=9, fontweight="bold")
a.text(-1.6, -1.55, "probe says\nHARMLESS", ha="center", color=BLUE, fontsize=9, fontweight="bold")
a.scatter([2.1], [0], s=150, color=RED, zorder=5, edgecolor="white", linewidth=1.4)
a.text(2.1, .34, "a harmful\nrequest", ha="center", color=RED, fontsize=8.5)
a.scatter([-1.3], [0], s=150, color=BLUE, zorder=5, edgecolor="white", linewidth=1.4)
a.text(-1.3, .34, "a harmless\nrequest", ha="center", color=BLUE, fontsize=8.5)
a.annotate("", xy=(-.9, -.55), xytext=(2.1, -.55),
           arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.6, mutation_scale=18))
a.text(.6, -.92, "steering slides it\nalong the ruler", ha="center", color=GREEN, fontsize=9,
       fontweight="bold")
a.scatter([-.9], [0], s=150, facecolor="none", edgecolor=RED, linewidth=1.8, zorder=5)
a.set_xlim(-3.2, 3.2); a.set_ylim(-2.1, 2.1); a.axis("off")
a.text(0, 1.88, "one direction in the model's internal state", ha="center", fontsize=8.5,
       color="#555", style="italic")

# ---------------------------------------------------------------- B: measure once vs every time
b = fig.add_subplot(gs[1])
b.set_title("B · when is the push measured?", loc="left", fontweight="bold", fontsize=11.5)
toks = ["prompt", "tok 1", "tok 2", "tok 3", "tok 4"]
xs = np.arange(len(toks))
need = np.array([2.1, 1.4, 2.6, 0.9, 1.9])          # how far each token actually needs to move
b.plot(xs, need, "o-", color=GREY, lw=1.6, ms=7, label="distance each token needs to move")
b.plot(xs, need, "o", color=RED, ms=7, zorder=4)
for x, n in zip(xs, need):
    b.annotate("", xy=(x, 0), xytext=(x, n),
               arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.2, mutation_scale=13))
b.axhline(need[0], color="#B36B00", lw=2.0, ls="--")
b.text(len(toks) - .05, need[0] + .12, "CLE-A: one push, measured on the prompt,\nreused for every token",
       ha="right", fontsize=8.8, color="#B36B00", fontweight="bold")
for x, n in zip(xs, need):
    if abs(n - need[0]) > .05:
        b.annotate("", xy=(x + .12, n), xytext=(x + .12, need[0]),
                   arrowprops=dict(arrowstyle="-", color="#B36B00", lw=1.4, ls=":"))
b.axhline(0, color="#111", lw=2.0)
b.text(-.42, .13, "target", fontsize=8.5, color="#111")
b.text(.5, .045, "CLE-P: each green arrow is recomputed per token,\nso every one lands on target",
       transform=b.transAxes, ha="center", fontsize=8.8, color=GREEN, fontweight="bold")
b.set_xticks(xs); b.set_xticklabels(toks, fontsize=9)
b.set_ylabel("how far from target")
b.set_ylim(-.95, 3.15); b.set_xlim(-.5, len(toks) - .3)
b.legend(loc="upper left", fontsize=8.5, framealpha=.95)

# ---------------------------------------------------------------- C: measured drift, real data
c = fig.add_subplot(gs[2])
c.set_title("C · the cost of reusing one push (real activations)", loc="left", fontweight="bold",
            fontsize=11.5)
w_all, b_all, _idx, _m = load_probe(os.path.join(R06, "probe_canonical", "qwen35_svm"))
t = np.load(os.path.join(RB, "test_acts.npz"), allow_pickle=True)
v = np.load(os.path.join(RB, "val_acts.npz"), allow_pickle=True)
L = 25
w, bb = w_all[L].astype(np.float64), float(b_all[L])
X, y = t["X"][:, L, :].astype(np.float64), t["y"].astype(int)
sv = v["X"][:, L, :].astype(np.float64) @ w + bb
margin = float(-np.quantile(sv[v["y"].astype(int) == 0], 0.5))
pre = X @ w + bb
harm = pre[y == 1]

# CLE-P: every point lands at -margin. CLE-A: the push sized for a REFERENCE point is applied to
# all of them, so each lands at its own pre-score minus that one fixed shift.
ref = np.median(harm)                                # the point the delta was measured on
shift = ref + margin                                 # the fixed displacement, in score units
post_p = np.full_like(harm, -margin)
post_a = harm - shift

c.axvline(-margin, color=GREEN, lw=2.2)

c.axvline(0, color="#111", lw=1.6, ls="--")
bins = np.linspace(min(post_a.min(), -margin) - .4, max(post_a.max(), harm.max()) + .4, 44)
c.hist(harm, bins=bins, color=RED, alpha=.32, label="before steering")
c.hist(post_a, bins=bins, color="#B36B00", alpha=.62, label="after CLE-A (one reused push)")
c.plot([-margin, -margin], [0, 1], lw=0)
c.axvspan(-margin - .012, -margin + .012, color=GREEN, alpha=1.0,
          label="after CLE-P (all land here)")
c.text(0, c.get_ylim()[1] * .99, " boundary ", fontsize=8.5, color="#111", ha="left",
       va="top", bbox=dict(boxstyle="round,pad=.2", fc="white", ec="none", alpha=.85))
c.annotate("target", xy=(-margin, c.get_ylim()[1] * .38), xytext=(-3.4, c.get_ylim()[1] * .46),
           color=GREEN, fontsize=9, fontweight="bold",
           arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5))
frac = float((post_a > 0).mean())
c.text(.98, .55, f"CLE-A leaves {frac:.0%} of harmful prompts\nstill scoring harmful;\nCLE-P leaves 0%",
       transform=c.transAxes, ha="right", fontsize=9, color="#333",
       bbox=dict(boxstyle="round,pad=.4", facecolor="white", edgecolor="#ccc"))
c.set_xlabel(f"probe score at layer {L}  (harmful prompts, n={len(harm)})")
c.set_ylabel("count")
c.legend(loc="upper left", fontsize=8.3, framealpha=.95)

fig.subplots_adjust(top=.86, bottom=.12, left=.04, right=.985)
out = os.path.join(R08, "cle_mechanism.png")
fig.savefig(out, bbox_inches="tight")
print("saved", out)
print(f"  layer {L}: target score {-margin:.3f}; fixed push sized on the median harmful point")
print(f"  CLE-P: every harmful prompt lands at {-margin:.3f} (spread 0)")
print(f"  CLE-A: lands {post_a.mean():.3f} +- {post_a.std():.3f}; {frac:.1%} still score > 0")
