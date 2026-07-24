# Baselines — frozen reference numbers for agentic probe evaluation

The reference point a new probe gets compared against. Every number here is reproducible from a
committed artifact by the command shown; nothing is transcribed from prose.

**Scope.** This file owns *agentic evaluation*: AgentHarm behaviour + probe-as-monitor. Probe
*training* lives elsewhere (`classifier/train_latent.py`, or whatever a teammate hands over).

---

## Pinned metric definitions

Fixing these matters more than the numbers — a shifted definition silently invalidates a comparison.

**The four standard AgentHarm metrics** (probe-independent; from the official `combined_scorer`):

| Metric | Source field | Split |
|---|---|---|
| Harm Score | `avg_score` | harmful |
| Refusals | `avg_refusals` | harmful |
| Non-refusal Harm Score | `avg_score_non_refusals` | harmful |
| Benign Non-refusal Score | `avg_score_non_refusals` | benign |

*Retired:* the firing-fraction metric `P(fire | complied)` used in `01`. Not tracked; kept in that
folder for history only.

**Probe monitor block** (harmful vs benign as a binary classifier against the AgentHarm split label):

- **`auc_mean_layers` — the headline.** Scores averaged across all layers, then AUC. **Selection-free,
  and therefore the only number that is fair to compare across probes.**
- `auc_best_layer` — supporting detail, **flagged selection-dependent**. With 64 val samples the
  per-layer argmax is largely selection noise (many layers reach CV AUC 1.000; an argmax picked that
  way generalised to 0.785). Two probes cannot be ranked on this.
- Confusion at the **canonical boundary** (`score = x·w + b`, fires `> 0`). No threshold is fitted at
  eval time, so there is no test peeking — but the confusion is only as good as the probe's own bias
  calibration (see the SVM caveat below).
- `auc_probe_to_refusal_harmful` — does the probe predict the model's own refusal, on harmful only.

---

## Llama-3.2-3B-Instruct — `unsloth/Llama-3.2-3B-Instruct`

Behaviour: AgentHarm, no attack, no intervention, greedy, official limits (4096 tok / 20 msg).

| Metric | Value |
|---|---|
| Harm Score | 26.9% |
| Refusals | 45.5% |
| Non-refusal Harm Score | 43.4% |
| Benign Non-refusal Score | 41.2% |

The two capability numbers are near-equal (43.4 vs 41.2), so the low Harm Score reflects a limited
tool-user rather than restraint. Details: `results/00-baseline/`.

| Probe result | Value | Where |
|---|---|---|
| probe → refusal AUC | 0.89 | `results/02-roc-probe-vs-judge/` |
| agentic-native probe, best layer | L11, AUC 0.90 | `results/03-agentic-native-probe/` |
| probe as harm/benign monitor (val, n=64, L11) | AUC 0.922; acc 0.797, recall 0.594, FPR 0.000 | `results/04-monitor-confusion/` |
| gpt-4o judge as monitor (val) | recall 1.00, FPR 0.06 | `results/04-monitor-confusion/` |

Note the probe/judge comparison here is on **val (n=64)** and it is where the judge looked strong.
That reversed on the harder Qwen test set — see below.

---

## Qwen3.5-27B — `Qwen/Qwen3.5-27B`

Behaviour: AgentHarm `test_public`, vLLM run, official scorer.

| Metric | Value |
|---|---|
| Harm Score | 36.6% |
| Refusals | 58.5% |
| Non-refusal Harm Score | 84.6% |
| Benign Non-refusal Score | 84.8% |

Substantially more capable than the 3B (84.6 vs 43.4 non-refusal harm) and slightly more refusing.

**Probe monitor** — AgentHarm `test_public`, 176 harmful + 176 benign, probe trained on val (32+32)
and read **in-context inside the Inspect scorer**:

| Probe | AUC mean-layers *(headline)* | AUC best layer | acc @ canonical boundary | recall | FPR |
|---|---|---|---|---|---|
| `single_direction` (mean-difference) | **0.930** | 0.943 (L24) | 0.858 | 0.864 | 0.148 |
| `svm` (LinearSVC C=0.1) | **0.956** | 0.948 (L28) | 0.750 | 0.500 | 0.000 |

probe → refusal AUC (harmful only, mean-layers): `single_direction` 0.913, `svm` 0.771.

**SVM bias caveat.** The SVM has the better AUC but its own intercept — fit on 32+32 val — is
conservative on the test distribution: recall 0.500 at FPR 0.000. Ranking is by AUC; the confusion
row reflects calibration, not separability. `single_direction`, whose bias is a centroid midpoint,
sits at a far more usable operating point.

**Monitor comparison** (`results/07-qwen35-monitor-confusion/`), same 352 rows, same split labels:

| Monitor | acc | recall | FPR |
|---|---|---|---|
| gpt-4o judge, majority-of-3 @ T1.0 | 0.77 | 0.99 | 0.45 |
| gpt-4o judge, single-shot T0 | 0.79 | 0.99 | 0.41 |
| in-context probe @ val threshold | 0.87 | 0.83 | 0.085 |

The judge over-flags AgentHarm's adversarial benign twins (79/176), verified as genuine: on those
rows the model complied 0/73 refused, and 62/176 were flagged unanimously 3/3. Stated honestly, this
is a **trade-off, not a domination** — at the judge's recall (0.99) the probe's FPR is 0.58, worse
than 0.41. The probe wins at a lower-recall operating point the fixed judge cannot reach.

---

## Validity boundaries

- **`results/05-qwen35-agentic-dissociation/` — probe numbers are INVALID.** Activations were
  reconstructed offline and omitted AgentHarm's 558-char agent system prompt, so they are not the
  model's real agentic context. Its **behavioural** four metrics remain valid and are the source of
  the Qwen behaviour table above. `06` is the valid probe baseline.
- **`results/06-.../probe_INVALID_from_05/`** is a byte-identical copy of that invalid probe, kept
  only for provenance. Use `probe_canonical/`.
- Probe activations must be read **in-context, inside the running agentic eval**. Offline
  reconstruction is the exact failure that invalidated `05`.
- Agentic probe evaluation requires Inspect's **`hf` provider (in-process)** — that is what puts
  `hidden_states` within reach of the scorer. vLLM yields behavioural metrics only.

---

## Reproducing / re-evaluating a new probe

Canonical probe artifact (`experiments/probe_io.py`), keyed like `classifier/train_latent.py`:

```
<stem>.npz    w (L, H) float32 | b (L,) float32 | layers (L,) int32     score = x·w + b, fires > 0
<stem>.json   REQUIRED metadata: probe_type, model, hidden, read_position,
              layer_index, score, trained_on
```

The sidecar is required and validated on load. A probe with a mismatched `read_position` or
`layer_index` convention produces plausible numbers rather than an error — the check is the only
thing between a handover and a silent garbage run.

Reproduce the Qwen rows above:

```bash
R05=experiments/results/05-qwen35-agentic-dissociation/inspect_logs
R06=experiments/results/06-qwen35-inscorer-probe

uv run python experiments/eval_probe.py \
  --probe  $R06/probe_canonical/qwen35_single_direction \
  --scores $R06/qwen35_inscorer_experiment_scores.npz \
  --harmful-log "$R05/logs_qwen35v_harmful/*.eval" \
  --benign-log  "$R05/logs_qwen35v_benign/*.eval"
```

For a **new** probe, swap `--scores` for `--acts <cached in-context test activations>`:

```bash
uv run python experiments/eval_probe.py --probe path/to/new_probe --acts <test_acts.npz>
```

Regenerate the canonical artifacts from the `06` run: `uv run python experiments/probe_io.py`.

---

## Known gaps

1. **No cached in-context test activations yet — the main blocker on cheap re-evaluation.** `06`
   Pass B saved probe *scores* but not the activations, so re-evaluating a new probe currently needs
   a full agentic GPU re-run. `qwen35_inscorer_experiment.py` now caches them
   (`--test-acts-out`); **the next GPU run produces the artifact**, after which every re-evaluation is
   a CPU matmul. (`05/qwen35_test_acts.npz` is *not* a substitute — it is from the invalid path.)
2. **TODO — CLE per-layer adapter.** `utils/probes.py` loads per-layer `svm_layerXX.pt` /
   `sd_layerXX.pt`; the eval harness loads a stacked canonical npz. Key names now agree (`w`, `b`),
   so this is a format shim, not a semantic conversion: convert both ways so a probe can be
   evaluated here and then handed to `cle-a.py` / `cle-p.py` unchanged. Not yet written.
3. Llama-3.2-3B rows predate the canonical schema and are transcribed from their experiment folders;
   they have not been re-run through `eval_probe.py`.
4. Layer-selection protocol is deliberately excluded from the headline metric. If a probe genuinely
   needs per-layer selection, the selection must be fit on val and declared in the metadata.
