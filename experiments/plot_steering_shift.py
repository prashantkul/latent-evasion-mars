"""What CLE-P steering does to the probe score, as a layers x samples heat map.

Computed analytically from the cached in-context activations -- no GPU. At beta=1 the projection
hook lands the score at exactly -margin, so the post-intervention score is known in closed form:

    pre  = w.h + b
    post = pre                      outside the intervened layer band
         = -margin                  inside it   (beta=1; verified numerically to 2.66e-15)

Colour is the probe score itself on a diverging scale: RED = positive = the probe firing =
harmful side of the boundary; BLUE = negative = harmless side. So the intervention is visible as
a band that turns red rows blue.

NOTE ON THE AXES. The activation cache holds ONE vector per layer, read at the post-instruction
token, so the second axis here is samples, not tokens. A layers x TOKENS view needs a fresh
extraction that keeps every position -- the same trajectory cache the adaptive-probe direction
needs. It is not derivable from this cache.

  uv run --with matplotlib --with numpy python experiments/plot_steering_shift.py
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from probe_io import load_probe, score as probe_score

R06 = os.path.join(HERE, "results", "06-qwen35-inscorer-probe")
RB = os.path.join(R06, "rebaseline_2026-07-24")
R08 = os.path.join(HERE, "results", "08-agentic-cle-pilot")   # these plots are about the CLE
                                                                 # intervention, not the 06 probe fit
plt.rcParams.update({"font.size": 10, "figure.dpi": 130})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default=os.path.join(R06, "probe_canonical", "qwen35_svm"))
    ap.add_argument("--acts", default=os.path.join(RB, "test_acts.npz"))
    ap.add_argument("--val-acts", default=os.path.join(RB, "val_acts.npz"))
    ap.add_argument("--layers", default="20-30")
    ap.add_argument("--quantile", type=float, default=0.5)
    ap.add_argument("--out", default=os.path.join(R08, "steering_shift.png"))
    args = ap.parse_args()

    lo, hi = (int(x) for x in args.layers.split("-"))
    band = np.arange(lo, hi + 1)

    w, b, _layers, _meta = load_probe(args.probe)
    t = np.load(args.acts, allow_pickle=True)
    v = np.load(args.val_acts, allow_pickle=True)
    pre = probe_score(t["X"], w, b)                       # (N, L)
    y = t["y"].astype(int)

    # Margins exactly as the pilot derived them: the q-quantile of the VAL harmless scores.
    sv = probe_score(v["X"], w, b)
    yv = v["y"].astype(int)
    margin = -np.quantile(sv[yv == 0], args.quantile, axis=0)          # (L,)

    post = pre.copy()
    post[:, band] = -margin[band]                          # beta=1 lands exactly here

    # Order samples by pre-score within each class so structure is visible rather than shuffled.
    hidx = np.argsort(pre[y == 1].mean(1))
    bidx = np.argsort(pre[y == 0].mean(1))
    order = np.concatenate([np.where(y == 1)[0][hidx], np.where(y == 0)[0][bidx]])
    split = int((y == 1).sum())
    P, Q = pre[order].T, post[order].T                     # (L, N)
    lim = float(np.abs(pre).max())

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.3),
                           gridspec_kw={"width_ratios": [1, 1, 1]})
    fig.suptitle(f"CLE-P steering, layers {lo}-{hi}, beta=1, margin = median val-harmless score"
                 f"   ·   Qwen3.5-27B, held-out test n={len(y)}",
                 fontsize=12, fontweight="bold", x=0.012, ha="left")

    for a, M, title, cmap, vmax in (
            (ax[0], P, "A · before — probe score", "RdBu_r", lim),
            (ax[1], Q, "B · after — score inside the band is pinned", "RdBu_r", lim),
            (ax[2], Q - P, "C · change (after − before)", "PuOr_r", float(np.abs(Q - P).max()))):
        im = a.imshow(M, aspect="auto", origin="lower", cmap=cmap, vmin=-vmax, vmax=vmax,
                      interpolation="nearest")
        a.axvline(split - .5, color="k", lw=1.4)
        for edge in (lo - .5, hi + .5):
            a.axhline(edge, color="#111", lw=1.1, ls="--")
        a.set_xlabel("samples  (left: harmful  |  right: benign)")
        a.set_ylabel("layer" if a is ax[0] else "")
        a.set_title(title, loc="left", fontweight="bold", fontsize=10.5)
        fig.colorbar(im, ax=a, fraction=.046, pad=.02)
    ax[0].text(split * .5, 67, "harmful", ha="center", color="#B2182B", fontweight="bold")
    ax[0].text(split + (len(y) - split) * .5, 67, "benign", ha="center", color="#2166AC",
               fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(args.out, bbox_inches="tight")
    print("saved", args.out)

    # The numbers behind the picture -- benign is moved too, which is the capability risk.
    d = np.abs(post - pre)[:, band].mean(1)
    fired_before = (pre[:, band] > 0).mean(1)
    fired_after = (post[:, band] > 0).mean(1)
    print(f"\nmargin over band: {margin[band].min():.2f} .. {margin[band].max():.2f} "
          f"(landing score {-margin[band].max():.2f} .. {-margin[band].min():.2f})")
    for name, m in (("harmful", y == 1), ("benign ", y == 0)):
        print(f"  {name}  mean |score shift| in band {d[m].mean():6.2f}   "
              f"fraction of band-layers firing {fired_before[m].mean():.3f} -> {fired_after[m].mean():.3f}")
    print("\nBenign activations are pushed as hard as harmful ones: the hook is unconditional,")
    print("it cannot know the class. That is the mechanism behind any benign capability loss.")


if __name__ == "__main__":
    main()
