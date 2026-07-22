# 02 — Score-as-predictor ROC/AUC (M5)

Replaces the ad-hoc "firing fraction" with continuous **score → label** AUC (threshold-free).

- `probe_metrics.json` + `roc_plot.py` → `probe_roc.png` — the probe score (mean over good layers)
  as a predictor of **refusal** (AUC 0.89) and of **harm≥0.5** (AUC 0.27, anti-predicts).
- `judge_roc.py` → `judge_roc.png` — the judge's harm score as a predictor of **compliance**
  (AUC 0.89); consistency check between the judge's harm and refusal signals.
  (reads `../01-dissociation/results_full.json`.)

**Result:** the probe predicts refusal about as well as the judge's harm score predicts compliance.
