# 04 — Monitor confusion matrices: judge vs probe (harmful/harmless)

Both as harmful/harmless classifiers on the AgentHarm **val** set (32 harmful + 32 benign).

- `judge_confusion.{json,png}` (`judge_conf_plot.py`) — gpt-4o `J(response, prompt) → 1/0`.
  Recall 1.00, FPR 0.06, acc 0.97. Only misses: 2 benign twins (Sexual, Hate) over-flagged.
- `probe_confusion.json` — Arditi probe (layer 11, threshold 0). Recall 0.59, FPR 0.00, precision 1.00,
  **AUC 0.92** — strong ranking, but the default threshold is miscalibrated for the agentic distribution.
- `confusion_compare.py` → `confusion_compare.png` — the two side by side.

**Result:** opposite error profiles. Judge = near-perfect but a paid LLM call/prompt; probe = free and
discriminative (AUC 0.92) but needs threshold recalibration to be a usable binary monitor.
