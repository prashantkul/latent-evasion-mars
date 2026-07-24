"""Layer-specific SVM score per token: tokens on y, layers on x, harmful beside harmless.

Reads the npz from experiments/extract_token_scores.py. Colour is tanh(score) on a diverging
red/blue scale -- red = positive = the probe firing = harmful side; blue = harmless side. tanh
squashes the tail so one extreme token cannot wash out the rest of the grid, at the cost of
compressing differences far from the boundary.

Pass two npz files (--pre --post) to get a third column showing what steering changed.

  uv run --with matplotlib --with numpy python experiments/plot_token_layer_scores.py \
      --pre tok_beta0.npz --post tok_beta1.npz --out token_layer_scores.png
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R08 = os.path.join(HERE, "results", "08-agentic-cle-pilot")
plt.rcParams.update({"font.size": 9, "figure.dpi": 130})


def tidy(t):
    """Readable token labels: byte-BPE markers out, whitespace made visible."""
    t = str(t).replace("Ġ", " ").replace("Ċ", "\\n").replace("Ġ", " ").replace("Ċ", "\\n")
    return t if len(t) <= 18 else t[:17] + "…"


def panel(ax, S, toks, title, layers, cmap="RdBu_r", vmax=1.0, tail=None):
    if tail and S.shape[1] > tail:                 # last N tokens: the tail is where the agent acts
        S, toks = S[:, -tail:], toks[-tail:]
    im = ax.imshow(np.tanh(S).T, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax,
                   interpolation="nearest")
    ax.set_xlabel("Layer Index")
    ax.set_title(title, fontsize=11)
    step = max(1, len(layers) // 32)
    ax.set_xticks(range(0, S.shape[0], step))
    ax.set_xticklabels([str(l) for l in range(0, S.shape[0], step)], fontsize=6.5)
    ax.set_yticks(range(len(toks)))
    ax.set_yticklabels([tidy(t) for t in toks], fontsize=6.5)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True, help="npz from extract_token_scores.py (beta=0)")
    ap.add_argument("--post", default=None, help="npz at beta=1, to show the change")
    ap.add_argument("--tail", type=int, default=40, help="show only the last N tokens (0 = all)")
    ap.add_argument("--out", default=os.path.join(R08, "token_layer_scores.png"))
    args = ap.parse_args()

    pre = np.load(args.pre, allow_pickle=True)
    post = np.load(args.post, allow_pickle=True) if args.post else None
    ncol = 2 if post is None else 4

    fig, ax = plt.subplots(1, ncol, figsize=(5.2 * ncol, 8.4))
    fig.suptitle("Layer Specific SVM Score" + ("" if post is None else "  —  before and after CLE-P"),
                 fontsize=13)
    cols = [("harmful", pre, "Harmful"), ("harmless", pre, "Harmless")]
    if post is not None:
        cols = [("harmful", pre, "Harmful — before"), ("harmful", post, "Harmful — after"),
                ("harmless", pre, "Harmless — before"), ("harmless", post, "Harmless — after")]
    for a, (tag, src, title) in zip(ax, cols):
        im = panel(a, src[f"{tag}_scores"], src[f"{tag}_tokens"], title, src["layers"],
                   tail=args.tail or None)
        a.set_ylabel("Tokens" if a is ax[0] else "")
        cb = fig.colorbar(im, ax=a, fraction=.046, pad=.02)
        cb.set_label("SVM Score (tanh)", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, bbox_inches="tight")
    print("saved", args.out)
    for tag, src, title in cols:
        S = src[f"{tag}_scores"]
        print(f"  {title:22s} grid {S.shape} (layers x tokens)  "
              f"fires at {(S > 0).mean():.1%} of (layer, token) cells")


if __name__ == "__main__":
    main()
