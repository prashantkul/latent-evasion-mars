# CLE-P agentic run plan (from 2026-07-25)

Where we go after `results/08-agentic-cle-pilot/`. Each stage says **why** it exists, because the
expensive mistake available here is spending a `test_public` arm (~4.3 h, ~$13) on a configuration
we had no reason to prefer — which is what happened with `--layers 20-30`.

**Standing position: we have no evidence yet that CLE-P steering works.** `08` established that the
harness runs end to end and that `--beta 0` is an exact no-op. Its deltas are not readable as an
effect: the refusal drop was 2 samples net of 6 flips (4 toward compliance, 2 away), and the run was
**sampling** (`do_sample` defaults True on Inspect's hf provider) while the frozen baseline is greedy.

---

## Stage 0 — bring the pod up

- Run `experiments/pod_bootstrap.sh`. **Why:** `/opt` is container disk and is wiped on every pod
  stop, taking the repo and pip installs with it; `/workspace` (incl. the ~52 GB HF cache) survives.
  The script restores one and never re-downloads the other.
- Pull the banked `control_harmful` off `/workspace` with `pull_pod_results.sh`. **Why:** it is the
  one condition the interrupted `test_public` run finished — refusals **0.5909**, within 0.6 pp of
  the frozen vLLM baseline's 58.5%. It is worth keeping on its own (it shows the hf provider
  reproduces baseline *refusal* behaviour, narrowing the validity boundary in `BASELINES.md`), and
  Stage 3 reuses it rather than re-running it.
- Confirm the frozen probe and val activation cache are present. **Why:** bootstrap fails loudly here
  rather than three hours into an eval — the failure mode this script was written for.

## Stage 1 — clean foundation + band comparison

`val`, greedy, β=1.0, ~90 min, ~$5.

- **One greedy control pair (harmful + benign).** **Why:** we do not have one. The `08` control was
  sampling, so control and attacked there differ by stochastic decoding *on top of* the intervention.
  Since β=0 is an exact no-op, the control is **layer-independent** — this one pair serves every
  subsequent val arm, so it is a fixed cost, not a per-arm cost.
- **L20-30, β=1, re-run greedy.** **Why:** makes the existing pilot number trustworthy, and keeps the
  band comparison apples-to-apples (same decoding, same control).
- **L53-63, β=1.** **Why:** this is the point of the stage. Measured on held-out test, band 20-30 has
  harm-vs-benign AUC 0.939 but **→refusal AUC 0.771**, while the best-11-layers-by-refusal are all
  late — **53-63, AUC 0.891** — with **zero overlap** between the two sets. `08` therefore intervened
  where *harm* is represented, not where the *refusal decision* is read, which is the thing we are
  trying to move. `--layers 20-30` was a default, not a derivation.

**Read this stage as triage, not evidence.** 32 harmful rows means 1 sample = 3.1 pp, so a 15 pp
refusal drop is 5 samples. Val can resolve a *large* effect and nothing subtler. A flat result means
"no large effect", not "no effect".

## Stage 2 — β sweep on whichever band moves

`val`, greedy, ~1 h, ~$3.

- **β ∈ {2, 4} on the Stage-1 winner.** **Why:** β is the step size of the projection
  (`h' = h − β·(score/‖w‖²)·w`, `utils/hooks.py:32`); β=1 lands the activation exactly on the target
  hyperplane. In the original CLE code β is an **Optuna-searched** hyperparameter
  (`optuna_search.py:88`) and nobody has run Optuna for Qwen3.5. We have only ever run β ∈ {0, 1}.
  Without this sweep, "β was too small" remains an unfalsified explanation for any null result.
- **If Stage 1 is flat everywhere, run the mechanism plots first** (`plot_steering_shift.py`,
  `plot_steering_pca.py`, `plot_cle_mechanism.py`). **Why:** before sweeping a knob, confirm the hook
  is actually moving activations where we think it is. Cheaper to check the mechanism than to buy
  more behavioural samples.

## Stage 3 — commit the winner to `test_public`

~4.3 h, ~$13.

- **`control.benign` + the winning attacked arm.** **Why:** `control.harmful` is already banked from
  the interrupted run and β=0 is layer-independent, so it is reusable as-is.
- **Why `test_public` at all:** `BASELINES.md` pins the frozen four metrics to `test_public` on vLLM.
  A `val` number differs from that table in *two* ways at once (split **and** provider), so a gap
  cannot be attributed to either. Only a `test_public` run can be placed beside the baseline.

---

## Cross-cutting gap — the evasion axis is missing

Every metric in Stages 1-3 is **behavioural**: does the model comply. CLE's actual claim is
**evasion** — the probe stops firing *while* the model still does the harmful thing.

- Nothing in `08` monitored the attacked run, so it says nothing about evasion.
- With 11 layers perturbed and a 64-layer monitor, the untouched layers would still fire — see the
  "a probe is 64 probes" lab note. The pinned combining rule is the mean of `score/‖coef_‖` across
  **all** layers.
- The in-scorer probe machinery from `results/06-qwen35-inscorer-probe/` is what reads both axes in
  one pass.

**Scope this before Stage 3**, so the expensive `test_public` arm returns the real result rather than
half of it.

## Known chores

- `run_agentic_cle_pilot.sh` hardcodes `--control`, so every arm re-runs it. A `CONTROL=` toggle
  saves ~33 min per arm after the first.
- Stale wreckage on the pod from the `temperature=0.0` crash: `cle_test_public_L0-63_beta1.0.json`
  and `cle_test_public_L20-30_beta1.0.json` written ~16:38-16:42 on 24 Jul have **all-None** metrics.
  Check the write timestamp before trusting any results json.

## Measured timings (from the `08` pilot log)

| condition | val (32+32) | test_public (176+176) |
|---|---|---|
| harmful | 12 min | ~57 min |
| benign | ~16-21 min | ~100 min |
| **per arm** | **~30 min** | **~2.6 h** |
