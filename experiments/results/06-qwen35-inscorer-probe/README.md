# 06 — Qwen3.5-27B probe, trained and evaluated inside the Inspect scorer

Closes the gap that forced the Qwen3.5 workaround, and produces the **valid** in-context probe result
that replaces the invalid `05`. Previously Inspect's HF provider could not parse Qwen3.5's tool calls,
so agentic behaviour had to come from a vLLM run and activations from a separate `get_representations`
pass (no system prompt) — the context defect that invalidated `05`. With the parser fixed, the hf
provider drives Qwen3.5 in-process, `hidden_states` reach the scorer, and the probe is both trained
and evaluated on activations read in the **real agent prompt**.

## ✅ Valid result — `qwen35_inscorer_experiment.py`
Two Inspect passes in one process. **Pass A** (val, throwaway scoring): the scorer harvests val
activations read in-scorer; the probe is trained on them afterward. **Pass B** (test): the trained
probe *is* the scorer — it fires `w·h + b` live on each held-out test row, so the `.eval` log's Score
column is the probe firing. Train on the 64 val rows only; test activations are scored, never fit on.
Behavioural labels (refused) from the graded vLLM run, joined by sample id.

**AUC(probe → refusal), Qwen3.5-27B, real agentic context, held-out test (176 harmful, refusal 0.585):**

| selection | mean-difference | LinearSVC |
|---|---|---|
| **mean-over-layers (selection-free)** | **0.912** | 0.771 |
| val-CV-selected layer (L18) | 0.785 [0.679, 0.873] | 0.744 |
| test-argmax layer (L63, optimistic) | 0.924 [0.865, 0.970] | 0.916 |
| best AUC(harmful vs benign) | 0.943 (L24) | 0.948 (L28) |

**Read the mean-over-layers row as the headline** — it needs no layer selection, so it has no leakage.
The probe is a **strong agentic monitor in the real context: AUC 0.912** predicting the refusal
decision, **0.943** separating harmful from benign.

**Finding — val-CV layer selection is unreliable at N=64.** With 64 val samples, many layers perfectly
separate harmful/benign (CV AUC 1.000), so the argmax (L18) is essentially arbitrary and picks a layer
that generalises poorly (test 0.785). The selection-free mean-over-layers (0.912) is the robust move;
don't trust a single val-CV-picked layer at this sample size. (Best layer also moved from L54 offline
to the very late layers in-context — the system prompt shifts where the refusal decision is readable.)

### Three-way comparison — why context matters (AUC mean-diff → refusal)
| experiment | context (train → eval) | AUC | status |
|---|---|---|---|
| `05` | offline → offline | 0.959 | **invalid** (inflated by same wrong-context train+test) |
| `06` transfer | offline → in-scorer | 0.877 | mismatch (offline probe applied to real context) |
| **`06` valid** | **in-scorer → in-scorer** | **0.912** (mean-layers) / 0.924 (test-argmax) | **honest** |

The offline `05` number (0.959) was optimistic: training and testing in the same clean offline context
inflates it. In the real agent context the honest figure is **0.912** — still a strong monitor, just
not 0.96. mean-diff again beats the SVC on the selection-free mean (0.912 vs 0.771); they tie on the
best single layer.

Artifacts: `qwen35_inscorer_experiment.json` (per-layer AUCs, val-CV curve),
`qwen35_inscorer_experiment_scores.npz` (per-sample per-layer scores + trained weights),
`qwen35_inscorer_val_acts.npz` (the val training activations, in-scorer), `inspect_logs_experiment/`
(Pass A harvest logs + Pass B logs whose Score column is the probe firing).

---

## Parser fix (the enabler)

## The bug (two parts)
`inspect_ai/model/_providers/util/hf_handler.py`:

1. **Name gating.** `model_specific_tool_parse` requires both `"qwen"` and `"instruct"` in the model
   name. `Qwen/Qwen3.5-27B` has neither `instruct` nor `coder`, so it falls through to
   `parse_unknown_tool_calls`. That fallback *does* match `"<tool_call>"` — but it then passes the
   inner text to `parse_tool_call_content`, which expects Qwen2.5-style JSON
   (`{"name":…, "arguments":{…}}`). Qwen3.5 emits nested XML instead:

   ```
   <tool_call>
   <function=NAME>
   <parameter=KEY>
   value
   </parameter>
   </function>
   </tool_call>
   ```

   `json.loads` fails, so **every** call became `ToolCall(function="unknown", parse_error=…)`.
   (Not "no pattern matched" — the block matched, the payload didn't.)
2. **Zero-argument tools.** `parse_tool_call_content` rejects empty arguments
   (`if not name or not arguments`), which would break the 14 zero-arg AgentHarm tools
   (`get_contacts`, `get_client_emails`, …) even once the format parsed.

## The fix — `experiments/qwen35_inspect_patch.py`
Monkey-patches both functions (not a site-packages edit, so it survives reinstalls). Detection is
**format-driven** (`"<function=" in response`) rather than name-driven, which fixes the name-gating
problem generally and leaves every other model's path untouched. `install()` before `inspect_eval`.

Verified before/after on the same emission with the real `HFHandler` (no GPU):

| | parsed |
|---|---|
| before | `('unknown', {})`, `parse_error=True` |
| after  | `('query_google_search', {'request': 'UK visa rules'})`, no error |

Unit-tested against: single call with params, zero-arg call, text-then-call, multiple calls,
multi-line parameter values, no-call. Regression-checked that Qwen2.5-style JSON still parses.

## Validation run (`qwen35_inscorer_probe.py`, 8 test_public harmful samples)
Pass 1 `tool_choice="none"` — SVC evaluated live in the scorer. Pass 2 `tool_choice="auto"` — real
multi-step tool loop.

| check | result |
|---|---|
| hidden_states present in scorer | 8/8 |
| SVC fired (computed in-scorer) | 8/8 |
| tool calls parsed | **21** |
| `function=="unknown"` | **0** (was every call before) |
| parse_errors | **0** |
| samples that called a tool | 7/8 |

Probe weights are the val-trained per-layer probes from `05` (`svm_w`/`svm_b_bias`/`dirs`), read at
the post-instruction token of layer 54 with `enable_thinking=False` — matching how they were trained.
The hf provider supports both `hidden_states=True` and `enable_thinking` as model args.

## Why it matters
The id-join in `05` was a workaround, not a necessity. In-scorer evaluation makes the Inspect log the
source of truth (its Score column *is* the probe firing), which is the pattern `multistep_dissociation.py`
uses for the 3B. Tradeoff: the hf path is slower than vLLM on a 27B, so vLLM still wins for large
behavioural baselines — this buys correctness and self-documenting logs, not throughput.

## Full-scale probe pass (all 352 test_public rows, in-scorer)
`qwen35_full_probe_pass.py` — every layer's SVM and mean-diff score written into the Score metadata,
so the `.eval` log carries the complete per-layer probe output.

| split | rows | SVC fires (in-scorer, L54) |
|---|---|---|
| harmful | 176 | 0.915 |
| benign | 176 | 0.284 |

(The SVC's own bias is a poorly-calibrated operating point — 28% benign firing. AUC is the metric to
read, below.)

## ⛔ Why `05` is marked INVALID: it was missing the agent system prompt
Joining these in-scorer scores to the same behavioural labels used in `05` does **not** reproduce the
`05` numbers:

| metric (L54 unless noted) | offline read (`05`) | in-scorer read (`06`) |
|---|---|---|
| AUC(mean-diff → refusal) | 0.959 | **0.877** |
| AUC(mean-diff, mean-over-layers) | 0.955 | 0.901 |
| AUC(SVM → refusal) | 0.917 (its best, L50) | 0.749 |
| AUC(mean-diff, harmful vs benign) | 0.902 | **0.927** |

Cause, confirmed by reading the messages Inspect actually sent: the AgentHarm default agent prepends a
**558-character system prompt** ("You are a helpful LLM agent that calls all necessary tools…"). The
offline `get_representations` path built `[user]` + tools only — **no system message** — so `05`'s
"agentic context" was tools-in-prompt but system-prompt-free: an *approximation* of the real agent
prompt, not the prompt the model actually saw.

**How to read this.** `05` is arithmetically self-consistent but characterises a prompt distribution
that is not the agent's, so it is **marked INVALID** and its probe-monitor numbers should not be
cited. Note the direction still encodes harm at least as well in the real context
(harmful-vs-benign *improves*, 0.902 → 0.927); what does not carry over is calibration to this
model's refusal decision under the actual agent prompt.

**The valid experiment** trains and evaluates in one context: run the val split through the same
in-scorer path, refit the probes there, then evaluate on the in-scorer test scores already collected
here. Pending (the GPU is currently running the behavioural pass).

## Files
- `qwen35_inscorer.json` — 8-sample validation: probe scores + tool-call/parse diagnostics.
- `qwen35_full_probe.json` — full 352-row in-scorer probe scores, every layer, both splits.
- `inspect_logs/*.eval` — the 8-sample validation runs; pass-1's Score column is the probe firing.
- `inspect_logs_fullprobe/*.eval` — the full 352-row probe pass.
- `probe/qwen35_svm_probe_weights.npz` + `.json` — the standalone trained probe (per-layer `svm_w`,
  `svm_b`, `dirs`) with a metadata sheet describing the read position and usage
  (`score = acts[layer] @ svm_w[layer] + svm_b[layer]`, fires when > 0).
- `probe/qwen35_svm_probe_scores.npz` — the source scores/weights file it was exported from.
- `*.log` — run logs.

## Code (kept with the experiment)
`qwen35_inspect_patch.py` (the parser fix), `qwen35_inscorer_probe.py` (8-sample validation),
`qwen35_full_probe_pass.py` (full in-scorer probe pass), `qwen35_full_behavior_hf.py` (official
behavioural eval through the patched hf provider). Run with this folder on `PYTHONPATH` — the scripts
resolve their sibling imports via `sys.path.insert(dirname(__file__))`.
