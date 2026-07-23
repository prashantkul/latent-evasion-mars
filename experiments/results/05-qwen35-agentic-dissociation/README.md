# 05 — Qwen3.5-27B agentic dissociation (second model)

Does the behaviour/representation dissociation replicate on a more capable agent? Same design as the
Llama-3.2-3B work (`01-dissociation`), on **Qwen3.5-27B** (bf16, thinking off), extracted with the
repo's `models/language_models.py::get_representations` primitive — extended so the post-instruction
token is read **in agentic context** (AgentHarm tool schemas rendered into the prompt), not bare chat.

## Pipeline
- `experiments/qwen35_agentic_val_extract.py` — extract val (32 harmful + 32 benign) agentic activations,
  fit per-layer mean-difference direction, pick best layer by 5-fold CV AUC. `--agentic/--no-agentic`.
- `experiments/qwen35_agentic_dissociation.py` — reuse the val direction, extract test_public
  harmful + benign (176 each) in agentic context, join behaviour from the vLLM graded logs.

## Files
- `qwen35_agentic_val.json` / `_acts.npz` — val CV AUC by layer; raw activations `(64, 64, 5120)`, y, per-layer dirs, ids.
- `qwen35_agentic_dissociation.json` / `_scores.npz` — dissociation metrics; per-row scores/ref/harm/id_original for both classes.
- `*.log` — run logs. `inspect_logs/logs_qwen35v_{harmful,benign}/` — the vLLM AgentHarm graded `.eval` logs (behaviour source).

## Result

We report in the two tracked blocks: the four standard AgentHarm behavioural metrics, and the
probe-monitor metrics (AUC of probe score → refusal, and probe-as-classifier recall/FPR). We no
longer report the firing-fraction `P(fire | complied)`.

### Behavioural — four standard AgentHarm metrics (agentic, direct request)
| Model | Attack | Harm Score | Refusals | Non-refusal Harm | Benign Non-refusal |
|---|---|---|---|---|---|
| Qwen3.5-27B | none (direct) | 36.6% | 58.5% | 84.6% | 84.8% |
| Llama-3.2-3B | none (direct) | 26.9% | 45.5% | 43.4% | 41.2% |

From the official combined_scorer (`avg_score`, `avg_refusals`, `avg_score_non_refusals`; benign col
from the benign run). Qwen3.5 is a genuinely capable agent (Benign Non-refusal 84.8% → no capability
ceiling), refuses more (58.5%), and executes when it complies (Non-refusal Harm 84.6%).

### Probe-monitor — representation (best layer L54, val-CV harmful/benign AUC 0.999)
| metric | Qwen3.5-27B | Llama-3.2-3B |
|---|---|---|
| AUC(probe score → refusal) | 0.959 | ~0.89 |
| AUC(harmful vs benign) | 0.902 | ~0.92 |
| probe-as-classifier @val-thr | recall 0.784 · FPR 0.080 · precision 0.908 | recall 0.59 · FPR 0.06 (thr 0) |

**Reading.** The probe is an even stronger monitor on the capable model than on the 3B: it ranks the
behavioural refusal decision at AUC 0.959 (vs ~0.89) and separates harmful from benign at AUC 0.902.
At a val-calibrated threshold it flags 78.4% of harmful behaviours at an 8% benign false-positive
rate. Per-layer separation emerges ~L24–33 and plateaus ≥0.99 across L45–63. (Threshold-comparison to
the 3B is loose — the 3B row used the untuned Arditi threshold 0 — so the AUC row is the clean
apples-to-apples comparison.)

### Per-layer test ROC (best layer & mean)
Re-scored every layer on the test set (`qwen35_perlayer_test.py` → `_perlayer_test_scores.npz`,
`h_md` (176×64) probe scores → refusal):
- **Best layer by test AUC→refusal: L54, AUC 0.959** (CI [0.923, 0.985]) — the *same* layer val-CV
  picked, so the choice is robust across selection criteria. The late layers L54–63 all sit at
  0.954–0.959 (a plateau, not a sharp peak).
- **Mean over layers: AUC 0.955** (CI [0.915, 0.984]) — averaging all 64 layers is within noise of
  the best single layer, so layer choice barely matters for this monitor.
- Aside: harmful-vs-benign *content* separation peaks earlier (L36, AUC 0.922) than the refusal
  *decision* (L54) — the decision is read best in late layers.

## Figures
- `probe_monitor.png` (`plot_monitor.py`): left — per-layer val-CV AUC(harmful vs benign), best L54,
  ≥0.99 plateau L45–63; right — held-out test ROC for probe→refusal (0.959) and harmful-vs-benign
  (0.902), val-threshold operating point (recall 0.78 · FPR 0.08).
- `probe_roc_layers.png` (`roc_layers_plot.py`): ROC fan — all 64 layers (faint), best layer 54 (red,
  0.959), mean over layers (black, 0.955). Matches the 03 figure.

## Reproduce (GPU box)
```
export HF_HOME=/workspace/.cache/huggingface PYTHONPATH=/opt/latent-evasion-mars
python3 experiments/qwen35_agentic_val_extract.py --out qwen35_agentic_val.json
python3 experiments/qwen35_agentic_dissociation.py --val-acts qwen35_agentic_val_acts.npz \
    --harmful-log 'logs_qwen35v_harmful/*.eval' --benign-log 'logs_qwen35v_benign/*.eval' \
    --out qwen35_agentic_dissociation.json
```
