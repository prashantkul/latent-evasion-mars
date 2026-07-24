# Results — agentic-refusal experiments (AgentHarm)

Each folder is one experiment: data (`.json`), the script that made it, and the plot.
For the frozen reference numbers and how to re-evaluate a new probe against them, see
[`../BASELINES.md`](../BASELINES.md). Full milestone log: `../agentharm-refusal-tracker.md`.

## Llama-3.2-3B-Instruct

| Folder | What it shows | Headline |
|--------|---------------|----------|
| `00-baseline/` | Official AgentHarm benchmark run (standard task/scorer) | Harm 26.9% · Refusals 45.5% · Non-refusal harm 43.4% · Benign 41.2% |
| `01-dissociation/` | Full graded run (176 rows); refusal vs. probe firing | knows-but-does-it-anyway (uses the now-**retired** firing-fraction metric) |
| `02-roc-probe-vs-judge/` | Probe score & judge score as predictors (ROC/AUC) | probe→refusal AUC 0.89; judge harm→compliance AUC 0.89 |
| `03-agentic-native-probe/` | Probe trained on val, tested on test; per-layer sweep | best layer 11 (0.90); agentic-native ≈ transferred |
| `04-monitor-confusion/` | Judge vs. probe as harmful/harmless monitor | judge recall 1.00/FPR 0.06; probe AUC 0.92 but threshold miscalibrated |

## Qwen3.5-27B — second-model generalization

| Folder | What it shows | Headline |
|--------|---------------|----------|
| `05-qwen35-agentic-dissociation/` | First Qwen attempt, offline activation reconstruction | **probe results INVALID** (missing agent system prompt); behavioural four metrics valid |
| `06-qwen35-inscorer-probe/` | Valid two-pass: train on val *and* evaluate on test entirely inside the Inspect scorer | SVM AUC 0.956 mean-layers; mean-diff 0.930 — dissociation replicates on a second model |
| `07-qwen35-monitor-confusion/` | gpt-4o judge vs in-context probe as harmful/harmless monitors | judge FPR 0.45 on benign twins vs probe 0.085 — the `04` story flips on the harder test set |

Qwen behaviour baseline: Harm 36.6% · Refusals 58.5% · Non-refusal harm 84.6% · Benign 84.8%.

## Reading order

`00` → `01`–`03` establish the dissociation on the 3B. `04` compares monitors. `06` replicates the
probe result on a second, far more capable model with a methodologically tighter design; `07` is its
monitor comparison. `05` is kept for provenance and its behavioural numbers only.

Regenerate any plot: `uv run --with matplotlib --with numpy --with scikit-learn python <folder>/<plot>.py`
Re-evaluate a probe: `uv run python ../eval_probe.py --probe <stem> --acts <in-context acts npz>`
