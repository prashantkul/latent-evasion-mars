"""Where CLE-P moves activations, in 2D. Red = harmful, blue = benign.

Exact, no GPU: at beta=1 the projection hook's output is closed-form,

    h' = h - ((w.h + b + margin) / ||w||^2) * w

so the post-intervention activation can be computed from the cached pre-intervention one.

Two frames, because they answer different questions:

  A  PCA(2) fit on the PRE activations, with POST projected into that SAME basis. This is the
     conventional view -- natural structure of the data. Its axes are chosen by variance and need
     not align with w, so the displacement can look oblique even though it is purely along w.

  B  Probe-aligned: x = signed distance along w/||w||, y = the leading direction of the residual
     after removing w. Steering is then exactly horizontal by construction, the decision boundary
     is a vertical line at 0, and the landing coordinate is a second vertical line. Nothing about
     the geometry is hidden by a basis choice.

  uv run --with matplotlib --with numpy --with scikit-learn python experiments/plot_steering_pca.py
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# numpy 2.2 on macOS/Accelerate raises divide-by-zero / overflow / invalid FP flags from BLAS
# matmul on these arrays even though every result is finite and correct: checked against einsum
# (max|diff| 4.9e-15) and against a pure-Python dot loop (8.9e-16). Suppressed narrowly, with the
# landing assert below as the real numerical guard -- it would catch anything genuinely wrong.
np.seterr(divide="ignore", over="ignore", invalid="ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from probe_io import load_probe

R06 = os.path.join(HERE, "results", "06-qwen35-inscorer-probe")
RB = os.path.join(R06, "rebaseline_2026-07-24")
R08 = os.path.join(HERE, "results", "08-agentic-cle-pilot")   # these plots are about the CLE
                                                                 # intervention, not the 06 probe fit
RED, BLUE = "#D7263D", "#0072B2"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default=os.path.join(R06, "probe_canonical", "qwen35_svm"))
    ap.add_argument("--acts", default=os.path.join(RB, "test_acts.npz"))
    ap.add_argument("--val-acts", default=os.path.join(RB, "val_acts.npz"))
    ap.add_argument("--layer", type=int, default=25, help="a layer inside the intervened band")
    ap.add_argument("--quantile", type=float, default=0.5)
    ap.add_argument("--out", default=os.path.join(R08, "steering_pca.png"))
    args = ap.parse_args()
    L = args.layer

    w_all, b_all, _lay, _meta = load_probe(args.probe)
    w, b = w_all[L].astype(np.float64), float(b_all[L])
    t = np.load(args.acts, allow_pickle=True)
    v = np.load(args.val_acts, allow_pickle=True)
    X, y = t["X"][:, L, :].astype(np.float64), t["y"].astype(int)

    # Margin exactly as the pilot derives it: q-quantile of the VAL harmless score distribution.
    sv = v["X"][:, L, :].astype(np.float64) @ w + b
    margin = float(-np.quantile(sv[v["y"].astype(int) == 0], args.quantile))

    pre = X @ w + b
    Xp = X - ((pre + margin) / (w @ w))[:, None] * w[None, :]        # beta = 1
    post = Xp @ w + b
    assert np.allclose(post, -margin, atol=1e-6), "beta=1 must land every point at -margin"

    fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.8))
    fig.suptitle(f"CLE-P moves activations along one direction — layer {L}, beta=1, "
                 f"margin {margin:.2f}   ·   Qwen3.5-27B, held-out test n={len(y)}",
                 fontsize=12, fontweight="bold", x=0.012, ha="left")

    # ---- A: conventional PCA, shared basis ----
    p = PCA(2).fit(X)
    A, B = p.transform(X), p.transform(Xp)
    a = ax[0]
    for m, c, lab in ((y == 1, RED, "harmful"), (y == 0, BLUE, "benign")):
        a.scatter(*A[m].T, s=16, c=c, alpha=.55, edgecolor="none", label=f"{lab} — before")
        a.scatter(*B[m].T, s=16, facecolor="none", edgecolor=c, linewidth=.6, alpha=.55,
                  label=f"{lab} — after")
    step = max(1, len(y) // 60)
    for i in range(0, len(y), step):
        a.annotate("", xy=B[i], xytext=A[i],
                   arrowprops=dict(arrowstyle="->", lw=.5, alpha=.4,
                                   color=RED if y[i] == 1 else BLUE))
    a.set_xlabel(f"PC1 ({p.explained_variance_ratio_[0]:.1%} var)")
    a.set_ylabel(f"PC2 ({p.explained_variance_ratio_[1]:.1%} var)")
    a.set_title("A · PCA(2) — filled = before, hollow = after", loc="left", fontweight="bold",
                fontsize=10.5)
    a.legend(fontsize=8, framealpha=.93, loc="upper right", ncol=2)
    a.text(.02, .03, f"|cos(PC1, w)| = {abs(p.components_[0] @ (w / np.linalg.norm(w))):.3f}"
           "\nthe steering direction is nearly ORTHOGONAL to PC1,\n"
           "so a PCA view barely shows the movement",
           transform=a.transAxes, fontsize=8.5, va="bottom", color="#444")

    # ---- B: probe-aligned frame ----
    u = w / np.linalg.norm(w)
    resid = X - np.outer(X @ u, u)                     # strip the probe direction
    v2 = PCA(1).fit(resid).components_[0]
    bx = ax[1]
    xa, ya = X @ u, resid @ v2
    xb = Xp @ u                                        # y is unchanged: motion is along u only
    for m, c, lab in ((y == 1, RED, "harmful"), (y == 0, BLUE, "benign")):
        bx.scatter(xa[m], ya[m], s=16, c=c, alpha=.55, edgecolor="none", label=f"{lab} — before")
        bx.scatter(xb[m], ya[m], s=16, facecolor="none", edgecolor=c, linewidth=.6, alpha=.55,
                   label=f"{lab} — after")
    for i in range(0, len(y), step):
        bx.annotate("", xy=(xb[i], ya[i]), xytext=(xa[i], ya[i]),
                    arrowprops=dict(arrowstyle="->", lw=.5, alpha=.4,
                                    color=RED if y[i] == 1 else BLUE))
    bnd = -b / np.linalg.norm(w)                       # w.h + b = 0
    land = (-margin - b) / np.linalg.norm(w)           # w.h + b = -margin
    bx.axvline(bnd, color="#111", lw=1.4, ls="--")
    bx.axvline(land, color="#009E73", lw=1.8)
    y0, y1 = bx.get_ylim(); bx.set_ylim(y0, y1 + (y1 - y0) * .18)
    bx.text(bnd, y1 + (y1 - y0) * .03, " decision boundary", fontsize=8.5, va="bottom")
    bx.text(land, y0, " landing point (−margin)", fontsize=8.5, va="bottom", color="#009E73")
    bx.set_xlabel("distance along the probe direction  w/||w||")
    bx.set_ylabel("leading residual direction (w removed)")
    bx.set_title("B · probe-aligned — every point moves horizontally, onto one line", loc="left",
                 fontweight="bold", fontsize=10.5)
    bx.legend(fontsize=8, framealpha=.93, loc="lower right", ncol=2)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(args.out, bbox_inches="tight")
    print("saved", args.out)

    d = np.abs(xb - xa)
    print(f"\nlayer {L}: boundary at {bnd:.2f}, landing at {land:.2f} (both in ||w||-normalised units)")
    for name, m in (("harmful", y == 1), ("benign ", y == 0)):
        print(f"  {name}  moved {d[m].mean():6.3f} +- {d[m].std():.3f}   "
              f"fired before {(pre[m] > 0).mean():.3f} -> after {(post[m] > 0).mean():.3f}")
    print(f"  PCA captures {p.explained_variance_ratio_.sum():.1%} of variance; "
          f"|cos(PC1, w)| = {abs(p.components_[0] @ u):.3f}")


if __name__ == "__main__":
    main()
