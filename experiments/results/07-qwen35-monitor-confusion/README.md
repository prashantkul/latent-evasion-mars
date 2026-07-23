# 07 — Monitor comparison for Qwen3.5-27B: gpt-4o judge vs the in-context probe

The Qwen analogue of `04-monitor-confusion` (which was Llama-3.2-3B on val). Both are evaluated as
**harmful/harmless classifiers** on AgentHarm **test_public** (176 harmful + 176 benign), against
ground truth.

- **Judge:** `J(response, prompt)` = gpt-4o → 1 (harmful request) / 0 (harmless), 04's exact prompt.
  Responses read from the existing vLLM behavioural logs — **no model re-run**, just gpt-4o calls.
- **Probe:** the **valid in-context** harm-vs-benign scores from `06` (mean-difference, read inside the
  Inspect scorer). AUC is threshold-free; the confusion matrix uses the **val-fit** decision boundary
  (no test peeking).

No GPU. `qwen35_monitor_confusion.py` (judge + metrics) and `plot_monitor_compare.py` (figure).

## Result — the `04` story flips on the harder test set
| monitor | acc | recall | FPR | note |
|---|---|---|---|---|
| gpt-4o judge `J(resp,prompt)` | 0.79 | 0.99 | **0.41** | fixed operating point; flags **73/176 benign** as harmful |
| in-context probe @ val threshold | **0.87** | 0.83 | **0.085** | AUC 0.943; flags 15/176 benign |

**The gpt-4o judge over-flags the adversarial benign twins.** On the 3B **val** set (`04`) the judge was
near-perfect (recall 1.00, **FPR 0.06**). On Qwen **test_public** its **FPR jumps to 0.41** — 73 of 176
benign requests wrongly flagged, spread across Drugs (14), Harassment (14), Fraud (13), Sexual (11),
Cybercrime (9). AgentHarm's benign `test_public` items are *benign twins* — structurally close to the
harmful behaviours — and a text judge reading request + response struggles to tell them apart.

**The probe, at a val-calibrated threshold, is a more accurate and far more precise monitor here**
(acc 0.87 vs 0.79; FPR 0.085 vs 0.41) — the model's internal harm representation separates benign
twins from harmful better than the external text judge, at 1/5 the false-alarm rate.

**But it is a trade-off, not a knockout — stated honestly.** The probe does **not** dominate the
judge's ROC point: at the judge's recall (0.99) the probe's FPR is **0.58**, *worse* than 0.41 (see
the ROC panel — the judge star sits above the probe curve at that recall). The probe wins by operating
at a lower-recall / much-lower-FPR point (0.83 recall, 0.085 FPR) that the fixed judge cannot reach. If
you must catch ~99% of harmful, the judge's point is better — but at a 41% benign false-alarm rate.

**Takeaway.** As a deployable binary monitor with a sensible threshold, the free internal probe beats
the paid gpt-4o text judge on accuracy and precision on this hard benign-twin set; the judge remains
the choice only at the extreme high-recall corner, where both are costly on false positives.

## Caveats
- 18/352 responses were empty (the model produced no completion); the judge still reads the prompt
  (`J` is mostly request-determined), so these aren't dropped — flagged as `n_empty_response`.
- gpt-4o at temperature 0 is near-deterministic; recall/FPR wobble ~0.005 between runs (API
  nondeterminism), not enough to change the picture.
- The probe threshold is val-fit; the judge has no threshold to calibrate. The AUC row and the ROC
  panel are the threshold-free comparison; the confusion matrices compare each at its natural point.
- Probe = mean-difference, harm-vs-benign best layer **L24** (AUC 0.943); `06` also has the SVM.

## Files
- `qwen35_monitor_confusion.json` — judge + probe predictions, confusion matrices, per-layer AUC,
  `probe_fpr_at_judge_recall`.
- `monitor_compare.png` — two confusion matrices + probe ROC with the judge operating point.
- Judge model: `openai/gpt-4o-2024-08-06`. Responses from `05/inspect_logs/logs_qwen35v_{harmful,benign}`,
  probe scores from `06/qwen35_inscorer_experiment_scores.npz`.
