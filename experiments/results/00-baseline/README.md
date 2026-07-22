# 00 — AgentHarm baseline (Llama-3.2-3B, direct request)

The behavioural baseline: the official AgentHarm task on Llama-3.2-3B-Instruct with
**no attack and no intervention**, greedy decoding, official limits (4096 tok / 20 msg),
scored with the official `combined_scorer` (grading functions + gpt-4o refusal/semantic judge).

## The four standard metrics

| Metric | Value | What it isolates |
|---|---|---|
| **Harm Score** | 26.9% | overall risk (compliance × capability) |
| **Refusals** | 45.5% | safety alignment — how often it says no |
| **Non-refusal Harm Score** | 43.4% | harmful capability when it *does* try |
| **Benign Non-refusal Score** | 41.2% | capability anchor — legit tool use |

Reading: moderate alignment (~46% refusal) on top of a **capability ceiling**. The two
capability numbers are nearly equal (43.4% harmful vs 41.2% benign), so the low Harm Score
is a limited tool-user, not sandbagging or restraint. `Harm Score ≈ (1−Refusals) × Non-refusal
Harm Score` = 0.545 × 0.434 ≈ 0.24, consistent with the measured 0.269.

## Provenance

- **Harmful columns** (Harm Score, Refusals, Non-refusal Harm Score) come from the full
  176-row graded run (M2f) in [`../01-dissociation/`](../01-dissociation/) — same official
  `combined_scorer`, same limits.
- **Benign column** is `avg_score_non_refusals` from the benign baseline run here
  (`avg_score 0.4116`, `avg_refusals 0.0`). Per-category benign scores are in
  `baseline_benign.log`.

## The harmful re-run and why it isn't the source here

`baseline.py` re-ran the *harmful* official task standalone. It crashed at sample 125/176
with `KeyError: 'file_path'` inside AgentHarm's own `grade_malware_distribution` grading
function — it indexes `args_add_hidden_malware["file_path"]` without a presence check, and
the 3B emitted an `add_hidden_malware` tool call missing that arg. This is a fragility in the
official scorer on one specific behavior, not our code. The full M2f run (above) had already
scored all 176 rows cleanly, so it is the source of the harmful numbers. Crash trace is
preserved in `baseline_harmful_crash.log`; resume (if desired) needs a defensive `.get()`
patch to that grading function.

## Files

- `baseline_benign.py` / `baseline_benign.log` — benign run + metrics (complete)
- `baseline.py` / `baseline_harmful_crash.log` — harmful re-run (crashed 125/176; kept for the trace)
