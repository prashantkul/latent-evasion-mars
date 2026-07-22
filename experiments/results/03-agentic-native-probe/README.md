# 03 — Agentic-native probe, per-layer (M5)

Train a probe on the AgentHarm **val** split (32 harmful + 32 benign = 64), test on the 176
test_public rows (disjoint behaviours). Held-out, within-benchmark.

- `agentic_probe.json` — 4-layer (14/18/22/26) train/eval, SVM vs mean-difference.
- `probe_all_layers.json` + `roc_layers_plot.py` → `probe_roc_layers.png` — every layer 0–27,
  AUC(→refusal) with clustered CI; ROC per layer + mean. **Best layer 11 (AUC 0.90).**

**Result:** mean-difference > SVM at N=64; agentic-native (0.887) ≈ transferred Arditi probe (0.89)
→ the refusal direction is largely distribution-invariant.
