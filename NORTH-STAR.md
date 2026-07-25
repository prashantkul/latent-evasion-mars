# North star

The one-page statement of what this project is. Read it before designing an experiment, and check
the drift log at the bottom before running one — every entry there is a real detour we took.

---

## What we are doing

Piras et al. 2026 (["Controlled latent evasion", arXiv:2605.21706](https://arxiv.org/pdf/2605.21706))
suppress LLM (Large Language Model) refusal by projecting activations past a linear probe's decision
boundary, with the per-layer margin set by BO (Bayesian Optimization) over ~500-700 trials per model.

**We extend that attack in two ways.**

1. **Adaptive probes.** The paper fits one static probe per layer on post-instruction-token
   activations only. We train and update probes on activations *throughout generation*, so the
   decision boundary tracks how harmful/harmless representations drift as tokens are produced —
   not only at the prompt boundary.

2. **BO-free geometric margins.** Instead of searching per-layer margins, we derive them from the
   harmful/harmless activation distributions directly, under the explicit constraint that a steered
   harmful point is not pushed further into the compliant region than harmless prompts typically
   sit. Implemented as `--margin-quantile` in `experiments/agentic_cle.py`: the per-layer margin is
   the q-quantile of the harmless score distribution, so `q=0.5` targets the harmless median.

**Stretch goal.** Evaluate in agentic (tool-use, multi-step) settings, which the original paper does
not cover.

**Side goal.** Better understanding of the refusal mechanism itself — in particular, relating
refusal directions to the loss optimized by prompt-level jailbreak attacks.

**Also planned.** The same attack delivered by two other routes: low-rank fine-tuning (to "unlearn"
refusal) and input-prompt modification.

## This is an attack-side project

The contributions above are **extensions of an attack**. Monitor quality, dissociation, and
probe-vs-judge comparisons are *testbed and instrumentation*, not the thesis. They matter because
they tell us the probe is a sound object to attack, and they gave us the in-context reading
machinery. They are not what the work is judged on.

## Our lane: agentic evaluation

This repo's workstream is **agentic evals**. That scope is narrow on purpose.

| ours | teammates' |
|---|---|
| CLE inside a live multi-step tool-use loop (AgentHarm) | probe design and training |
| the eval harness, frozen baselines, pinned metric definitions | chat-mode / single-turn evaluation of individual models |
| in-scorer activation reads in the real agent prompt | ASR + MMLU against the paper's BO-tuned numbers |
| token-level behaviour across a generated trajectory | per-model BO searches |

**We consume probes; we do not design them.** A teammate's probe arrives as a canonical artifact
(`experiments/probe_io.py`: stacked `w`/`b` plus a required metadata sidecar) and our job is to
re-evaluate it against a fixed reference. See known gap #1 in `experiments/BASELINES.md` — the
per-layer `svm_layerXX.pt` ↔ stacked-npz shim is still unwritten, and it is the handover surface.

**Consequence for planning.** Steps (1) and (2) of the plan below are largely *not our work*. They
are dependencies. Our agentic numbers cannot be interpreted until the chat-mode ground truth
exists, and the useful move when we are blocked on that is to say so — not to run the chat-mode
experiment ourselves.

**Where our lane is load-bearing rather than a stretch goal.** The case for adaptive probes rests
on representations drifting across generated tokens. That drift is largest in long multi-step
trajectories, which only exist in our setting. So the measurement that decides whether contribution
(2) is worth building is an *agentic* measurement, and it is ours to make. See Q2.

## Plan, in order

1. **Geometric margin rule on the paper's own frozen post-instruction probes.** Cheapest change,
   isolates its effect against published ground truth.
2. **Adaptive / online probes** trained on generated-token activations — *if* (1) is competitive.
3. **Port to AgentHarm** for agentic experiments.

**Next concrete step (as briefed):** implement the geometric margin rule on 1-2 models the paper
already covers — `llama2-7b`, `mistral-7b-rr`, both present in `utils/models_utils.py` — and compare
**ASR (Attack Success Rate)** and coherence (**MMLU**) against the paper's BO-tuned CLE numbers.
`optuna_search.py` is the BO baseline and already wires the HarmBench-Mistral judge and ASR.

## How we know it is working

| | target |
|---|---|
| **Geometric margin** | ASR within ~5-10 points of BO-tuned CLE, at **zero** per-model search cost |
| **Coherence** | MMLU / ARC / TruthfulQA no worse than the paper reports |
| **Adaptive probes** | better-sustained compliance confidence across generated tokens than the static probe, especially where static already drifts (long generations, agentic loops) |

## Stop / reconsider conditions

- The geometric rule causes large coherence collapse, or does not steer effectively. That would
  suggest BO was fitting real per-model / per-layer structure our heuristic misses.
  **Only concludable against a ground-truth BO baseline on a paper-covered model** — see drift D2.
- Adaptive probes add engineering complexity without beating CLE's existing result that a *single
  fixed post-instruction perturbation sustains evasion through generation*. If there is no gap left
  to close in non-agentic settings, do not build them there.

## Open questions we want challenged

**Q1 — Is latent steering a surrogate for prompt-level jailbreaks?** Latent steering pushes an
activation across a probe boundary; GCG optimizes input tokens to minimize cross-entropy on a fixed
affirmative target. Both aim at the same compliant region. Is raising a probe's compliance
confidence essentially a surrogate for lowering the loss on a fixed target sentence, and is the
equivalence real enough to formalize?

*Cheap decisive test:* `cos(w_probe, grad_h CE(affirmative target))` at the same layer and read
position. Alignment means the surrogate relationship is first-order and formalizable; no alignment
means any agreement between the two attacks is behavioural coincidence. One backward pass per
prompt, no training, no BO.

**Q2 — Is there a real gap to close?** CLE already shows a single fixed post-instruction
perturbation sustains evasion through generation with no reprojection. Does the distribution shift
across layers and generated tokens actually matter enough to improve the characterization of the
refusal subspace?

*What we know:* our own mechanism figure measured 4% of harmful prompts still reading harmful under
one reused CLE-A push, versus 0% under CLE-P, on real held-out activations. Real residual, but thin,
and measured on probe score rather than behaviour.

*Unverified assumption:* the case for adaptive probes is strongest where a fixed perturbation has
most time to go stale — long generations and agentic loops. If that holds, agentic is not a stretch
goal for contribution (2) but the setting where it has any headroom at all. **This is currently
assumed, not measured.** The cheap test is whether probe scores actually drift across generated
tokens in an agentic trajectory (`experiments/extract_token_scores.py`,
`experiments/plot_token_layer_scores.py`). No drift means contribution (2) has nothing to track.

---

## Drift log

Real detours. Each one looked locally reasonable.

**D1 — Framing the project as monitor defence.** Recurring. The dissociation and probe-vs-judge
results are strong, and it is easy to slide into "can a latent monitor be evaded" as the thesis.
It is not. This is an attack-side extension. Monitor work is instrumentation.

**D2 — Reading an agentic null as a verdict on the margin rule.** We ran agentic CLE on
Qwen3.5-27B while the chat-mode ground-truth comparison does not yet exist. A null there has four
candidate explanations at once — margin rule, agentic setting, layer band, and a model the paper
never covered — with no baseline to separate them. Step (1) is a **dependency owned by teammates**,
not work for us to pick up; the correct response to being blocked on it is to name the dependency.
What we *can* do in lane is build a comparator inside the agentic setting: run two margin rules
against each other on the same model and harness, which isolates the rule without leaving our scope.

**D3 — Metric-family confusion at the handover.** The contribution's success criterion is ASR +
MMLU against the paper's numbers, and that is measured in chat mode by teammates. AgentHarm's four
standard metrics are the *correct* output for our lane. The drift is not producing them — it is
presenting them as if they answered the ASR question.

**D4 — Undirected hyperparameter choices.** `--layers 20-30` was a default, not a derivation, and
propagated into four scripts before anyone measured that band 53-63 is where the refusal decision
is read (→refusal AUC 0.891 vs 0.771, zero layer overlap).

**D5 — Model choice drifting off the paper's set.** Qwen3.5-27B is our own baseline and has no
BO-tuned CLE numbers to compare against. Fine for our lane; it just means a cross-lane comparison
needs a model both sides have run.

**D6 — Proposing work outside our lane.** Recommending we run the chat-mode ASR/MMLU comparison on
`llama2-7b` ourselves. It is the right next experiment for the *project* and the wrong one for
*us*. When the blocking work belongs to someone else, flag the dependency and pick the highest-value
in-lane experiment instead.

### Check before running an experiment

1. **Is this in our lane — agentic?** If it is chat-mode, probe training, or a per-model BO search,
   it belongs to a teammate. Flag it, do not run it.
2. Which contribution does it inform — geometric margins, adaptive probes, or neither?
3. What is the comparator, and does it exist *for this model, in this setting*?
4. Does it produce AgentHarm's four standard metrics, and am I about to present them as if they
   answered an ASR question?
5. If it comes back null, how many explanations will there be?
