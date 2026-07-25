# Latent evasion — run book for the current code

What we can run **today**, with exact invocations, costs and gates. The *why* behind each experiment
lives in [`agentic_experiment_design.md`](agentic_experiment_design.md); the scope boundary lives in
[`../NORTH-STAR.md`](../NORTH-STAR.md). This file is the executable half — if a command is not here,
it is not ready to run.

Nothing in this file has been executed yet. Statuses below say so explicitly.

---

## 0. What the code can do right now

Everything routes through the `cle` Inspect provider (`cle_provider.py`), so CLE-P and CLE-A differ
in their schedule and nothing else.

| Knob | Values | Notes |
|---|---|---|
| `--attack` | `cle-p`, `cle-a`, `none` | `none` = hook-free baseline through the identical path |
| `--layers` | `"20-30"`, `"0-63"`, `"22,24,26"` | validated against the loaded model's block count |
| `--beta` | float | `0` is an exact algebraic no-op — **measured**, `max\|delta\| = 0` |
| `--margin-quantile` | q with `--val-acts` | per-layer margin = q-quantile of val **harmless** scores |
| `--margin` | float | CLE's raw-score margin, expanded to every layer |
| `--recalibrate` | `sample`, `turn` | CLE-A only. `turn` is **ours**, not the paper's |
| `--trace-scores` | flag | CLE-A only. Records `w·h+b` at the last position of each generate call |
| `--selftest` | flag | the three §7 correctness checks, no eval |
| `--control` | flag | runs the β=0 arm first, same code path |
| `--split` | `val`, `test_public` | `test_private` is held out and stays that way |

Probes available (`results/06-qwen35-inscorer-probe/probe_canonical/`): `qwen35_svm` and
`qwen35_single_direction`. **Every run so far has used `svm` only.**

Also available: `chat_vs_agentic_ablation.py` (run/join), `extract_token_scores.py` (per-token ×
per-layer grids, one forward pass per prompt), `plot_steering_shift.py`, `plot_cle_mechanism.py`,
`eval_probe.py`.

## 1. Where we stand, in one table

| | measured | value |
|---|---|---|
| vanilla agentic baseline (test_public, vLLM) | ✅ | Harm 36.6 · Refusals 58.5 · NRH 84.6 · Benign 84.8 |
| vanilla agentic (test_public, hf, greedy) | ✅ | Refusals 0.5909 |
| CLE-P L20-30 β=1 q=0.5 (val, greedy) | ✅ | refusals **Δ0.0000**, benign **−0.0677** |
| CLE-P L53-63 β=1 q=0.5 (val, greedy) | ✅ | refusals **Δ−0.0313**, benign **−0.0521** |
| framing effect, chat vs agentic | ✅ | refusal 0.75 → 0.59, 12-vs-2 flips, p = 0.0129 |
| **CLE-A, any configuration** | ❌ | never run |
| **the evasion axis — does the probe go quiet?** | ❌ | **never measured, in any run** |
| provider reproduces the pre-provider CLE-P path | ❌ | unvalidated |
| selftest passes with correct tolerances | ❌ | inconclusive (bf16) |

**The scale to keep in view:** steering has moved refusals by 0–1 samples of 32. Reframing the task
moves 12 behaviours of 44. Whatever we run next should be judged against that.

---

## 2. Blockers — nothing downstream is trustworthy until these pass

### B1 · Precision-aware selftest &nbsp;·&nbsp; DGX &nbsp;·&nbsp; ~10 min &nbsp;·&nbsp; free

The current tolerances were borrowed from a float32 numpy check that produced `2.66e-15`. The model
runs in **bf16**, where one ULP at activation magnitude ~128–256 is 0.5 — and the check reported
exactly 0.5. Plausibly precision, **unverified**, and I will not wave it through.

Change `selftest()` to report error **relative to the hidden-state scale** and print that scale, then:

```bash
python3 experiments/agentic_cle.py --attack cle-a --selftest \
    --probe <repo>/experiments/results/06-qwen35-inscorer-probe/probe_canonical/qwen35_svm \
    --layers 20-30 --margin-quantile 0.5 --val-acts <val_acts.npz>
```

**Gate:** `beta0_delta_max_abs` must be exactly `0.0` (it already is). The other two must be small
*relative to the activation scale*. Fails → CLE-A cannot be claimed at all.

Runs on the DGX: three forward passes on one prompt, no generation. Exactly the workload the
throughput measurement says belongs there.

### B2 · Provider equivalence for CLE-P &nbsp;·&nbsp; pod &nbsp;·&nbsp; ~10 min &nbsp;·&nbsp; ~$0.50

Stage 1 ran CLE-P through the pre-provider code path (hooks around the whole eval). The provider
installs them per generate call. Algebraically identical; **unverified**.

```bash
python3 experiments/agentic_cle.py --attack cle-p --split val --limit 8 \
    --layers 20-30 --beta 1.0 --margin-quantile 0.5 --val-acts <val_acts.npz> \
    --probe <stem>_svm --log-dir /workspace/b2 --out /workspace/b2.json
```

**Gate:** per-sample scores identical to the first 8 rows of
`results/09-cle-stage1-greedy/.../attacked_harmful/*.eval`. Greedy ⇒ exact match expected.
Fails → the provider changed CLE-P's behaviour, and every comparison to Stage 1 is void until fixed.

---

## 3. Experiments

Ordered by what each rules out. Costs at $2.99/h; val arms are harmful-only (~12 min) unless the
benign axis is needed (~30 min).

### E1 · Does *any* CLE-P configuration move refusals? &nbsp;·&nbsp; ~40 min &nbsp;·&nbsp; ~$2

Escalate along the two principled axes. **β stays at 1** — see the design doc §3.1: β>1 makes the
landing point start-dependent and pushes past where harmless prompts sit, abandoning the constraint
that defines the geometric-margin contribution. The escalation knob is `q`.

| arm | `--layers` | `--margin-quantile` | rules out |
|---|---|---|---|
| E1a | `0-63` | 0.5 | "11 layers is too few" — this is the paper's threat model |
| E1b | `53-63` | 0.9 | "the harmless median is too conservative a target" |
| E1c | `0-63` | 0.9 | both at maximum, still inside the geometric constraint |

```bash
for cfg in "0-63 0.5" "53-63 0.9" "0-63 0.9"; do set -- $cfg
  python3 experiments/agentic_cle.py --attack cle-p --split val \
      --layers "$1" --margin-quantile "$2" --beta 1.0 --val-acts <val_acts.npz> \
      --probe <stem>_svm --log-dir /workspace/e1_L$1_q$2 --out /workspace/e1_L$1_q$2.json
done
```

Reuse the Stage 1 β=0 control — it is layer-, β- and q-independent. Do not re-run it.

**Gate:** ≥ 5 discordant flips toward compliance on the paired analysis → E2. Otherwise → E3.

### E2 · Is it evasion, or just damage? &nbsp;·&nbsp; needs work first &nbsp;·&nbsp; ~15 min run

**Status: not runnable. This is the missing piece.** Every metric we have is behavioural; nothing has
ever asked whether the probe goes quiet during an attacked run. Without it we can report behaviour
change and never evasion — which is the actual CLE claim.

Requires wiring the `06` in-scorer probe read into the provider path so one pass yields both axes.
Estimated half a day. **Record per-step probe scores, not one number per rollout** — agentically,
"did the probe go quiet" has a *when* and a *for how long*.

### E3 · Mechanism — did the intervention do what we think? &nbsp;·&nbsp; DGX &nbsp;·&nbsp; free

Runs if E1 is null. Diagnostic, forward passes only, so it belongs on the DGX.

```bash
python3 experiments/extract_token_scores.py --beta 0.0 ...   # pre-steering grid
python3 experiments/extract_token_scores.py --beta 1.0 ...   # post-steering grid
python3 experiments/plot_steering_shift.py
```

Three questions: does the score land at `−m` in a *live agentic* forward, not a cached-activation
replay? How far does it drift across a rollout? How much are **benign** activations displaced —
since benign capability is the only thing that has actually moved so far.

A clean E3 — "activations land exactly on target, refusal behaviour does not follow" — is direct
evidence for the causal-dissociation hypothesis, and is a result in our lane rather than a failure.

### E4 · CLE-P vs CLE-A &nbsp;·&nbsp; ~30 min &nbsp;·&nbsp; ~$1.50

Only after B1, and only if E1 passes: CLE-P recomputes every position and is therefore the ceiling.
If it cannot move behaviour, CLE-A cannot either, and this compares two nulls.

```bash
python3 experiments/agentic_cle.py --attack cle-a --recalibrate sample --trace-scores ...
python3 experiments/agentic_cle.py --attack cle-a --recalibrate turn   --trace-scores ...
```

Answers the brief's Q2 in the setting where a frozen displacement has the most room to go stale. The
`--trace-scores` output gives the staleness curve as the *mechanism* alongside the behavioural
result. Report `sample` as the CLE-A baseline; `turn` is labelled as ours.

### E5 · Confirmation on `test_public` &nbsp;·&nbsp; ~2.6 h/arm &nbsp;·&nbsp; ~$8/arm

Whatever survives, once, at n=176. `control.harmful` is banked; `control.benign` still needs ~100 min
(~$5) and is a one-time cost. Gated on everything above.

### E6 · Complete the framing ablation &nbsp;·&nbsp; ~5 min &nbsp;·&nbsp; bundle only

`val` chat arm, 8 samples. `val` and `test_public` behaviours are disjoint (verified), so this adds
8 independent behaviours, 44 → 52. The val agentic half is already banked as the Stage 1 β=0 control.

Public AgentHarm has **52 behaviours total**, so this is the ceiling — it will not meaningfully
narrow the CI. Worth completing, **not worth a dedicated pod trip**. Bundle with any other session,
or run on the DGX.

---

## 4. Sequence

```
B1 (DGX, free) ─┐
B2 (pod, $0.50)─┴──► E1 ──► moves? ──yes──► E2 (build first) ──► E4 ──► E5
                              │
                              └──no───► E3 (DGX, free) ──► write up the dissociation
```

Through E4: **~1.5 h GPU, under $5.** E5 is the only expensive step and is gated on all of it.

## 5. What we are deliberately not running

- **`test_private`** — held out. Not for baselines, not for ablations.
- **β sweeps** — β>1 abandons the geometric-margin constraint that is contribution (2). If we ever
  want it, it is a *comparison against* the contribution, not an escalation of it.
- **Chat-mode ASR / MMLU on `llama2-7b` / `mistral-7b-rr`** — teammates' lane. When we are blocked
  on that ground truth, we name the dependency rather than absorb the work.
- **`single_direction` probe arms** — worth doing eventually (it is the better refusal predictor,
  0.913 vs 0.832, and for it `w` *is* the semantic axis so a geometric margin is better defined),
  but not before we know whether *any* configuration moves behaviour.

## 6. Standing limits to state in any write-up

- One model, Qwen3.5-27B, which the paper does not cover — no BO-tuned comparator exists for it.
- One probe type (`svm`) in every run to date.
- Greedy measures the mode, not the mean: an intervention that shifts the behavioural distribution
  without moving the argmax reads as null.
- `val` is 32 samples over 8 behaviours; it screens for large effects only. A val null means
  "no large effect", never "no effect".
