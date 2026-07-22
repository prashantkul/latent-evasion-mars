# Results — agentic-refusal experiments (Llama-3.2-3B on AgentHarm)

Each folder is one experiment: data (`.json`), the script that made it, and the plot.
See `../agentharm-refusal-tracker.md` for the full milestone log.

| Folder | What it shows | Headline |
|--------|---------------|----------|
| `00-baseline/` | Official AgentHarm benchmark run (standard task/scorer) | the model's by-the-book harm/refusal numbers |
| `01-dissociation/` | Full graded run (176 rows); refusal vs. probe firing | P(fire\|did-not-refuse)=0.81 — knows but does it anyway |
| `02-roc-probe-vs-judge/` | Probe score & judge score as predictors (ROC/AUC) | probe→refusal AUC 0.89; judge harm→compliance AUC 0.89 |
| `03-agentic-native-probe/` | Probe trained on val, tested on test; per-layer sweep | best layer 11 (0.90); agentic-native ≈ transferred |
| `04-monitor-confusion/` | Judge vs. probe as harmful/harmless monitor (confusion matrices) | judge recall 1.00/FPR 0.06; probe AUC 0.92 but threshold miscalibrated |

Regenerate any plot: `uv run --with matplotlib --with numpy --with scikit-learn python <folder>/<plot>.py`
