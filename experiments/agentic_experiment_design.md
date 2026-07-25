# Agentic CLE — experiment design

Supersedes `cle_p_agentic_plan.md`, which was a run schedule rather than a design: it listed arms
without stating what each would establish, what would count as an effect, or what a null would mean.
Stage 1 executed cleanly and still cannot answer its own question, which is the gap this closes.

Scope is `NORTH-STAR.md`'s lane: **agentic evaluation**. Chat-mode ASR/MMLU belongs to teammates.

---

## 1. What we know

**Established (val, 32 harmful + 32 benign, greedy, β=1, margin = harmless median):**

| | control (β=0) | CLE-P L20-30 | CLE-P L53-63 |
|---|---|---|---|
| Refusals | 0.6875 | 0.6875 | 0.6562 |
| Harm Score | 0.2244 | 0.2217 | 0.2244 |
| Non-refusal Harm | 0.5695 | 0.5838 | 0.5177 |
| Benign Non-refusal | 0.7418 | 0.6741 | 0.6897 |

Refusals move by 0 and 1 sample. Benign capability falls 5-7 points in both. **At this
configuration the intervention is pure cost.**

**Also established:** decoding must be greedy or the comparison is meaningless (a sampling control
differed from a greedy one by 10 points of non-refusal harm); β=0 is an exact algebraic no-op
(`max|delta| = 0`, measured).

**Not established, and currently assumed:**

- That the hook is displacing activations *as intended* in a live agentic run. The CLE-P selftest
  reported `score_landing_max_err = 0.023` and a CLE-A/CLE-P hidden-state gap of 0.5, against
  tolerances borrowed from a float32 numpy check. The model runs in **bf16**, where one ULP at
  activation magnitude ~128-256 is 0.5. Plausibly precision; **unverified**.
- That aggregate refusal equality means sample-level equality. 22/32 refusing in both arms could be
  a different 22.

## 2. The hypothesis, stated so it can fail

> **H1.** Projecting activations across the probe's decision boundary, at layers where the refusal
> decision is read, causes the agent to comply with harmful requests it would otherwise refuse.

H1 is what the CLE attack asserts, transposed to the agentic setting. Stage 1 is weak evidence
against it, confounded by low power and a single configuration.

The competing explanation is not "the attack is weak" but something the project has already
measured:

> **H0.** The probe direction encodes harmfulness but is not the causal lever for the refusal
> decision in an agent loop. Steering it changes the representation without changing the behaviour.

H0 is the causal form of our own dissociation result — representation and behaviour decoupled. If
H0 holds, that is a **finding, not a failed experiment**: it says a latent monitor reads something
the model's behaviour does not follow from, which bounds both the attack and the monitor.

**Distinguishing them requires measuring both axes.** Everything so far measures behaviour only.

| | probe still fires | probe silenced |
|---|---|---|
| **agent refuses** | intervention did nothing | **H0** — representation moved, behaviour did not |
| **agent complies** | behaviour moved for another reason | **H1** — the attack works |

Stage 1 cannot tell the top-left cell from the top-right, because no probe was read during the
attacked runs. **This is the single biggest gap in the design so far.**

## 3. Design decisions

### 3.1 β is the wrong escalation knob; the margin quantile is the right one

Post-intervention score is `(1−β)(w·h+b) − β·m`. At β=1 every point lands at exactly `−m`
regardless of origin — the margin is a destination, not a dose. At β=2 a point starting at `s`
lands at `−s − 2m`, which is **start-dependent**, so the property that makes the rule geometric is
gone. β>1 also pushes harmful points *further into harmless territory than harmless prompts sit*,
which is precisely the constraint contribution (2) is defined by.

**So we sweep `q`, not β.** `q=0.5` targets the harmless median; `q=0.9` targets deep harmless
territory while still landing where real harmless prompts actually are. That keeps every arm inside
the contribution's own constraint and directly tests whether that constraint is too conservative.

β stays at 1 except for the β=0 control.

### 3.2 CLE-P is the ceiling, so it is tested first

CLE-P recomputes the displacement at every position and every decode step; CLE-A applies one frozen
vector. CLE-P is therefore the stronger attack by construction. **If CLE-P cannot move behaviour,
CLE-A cannot either**, and a CLE-P-vs-CLE-A comparison would be comparing two nulls.

This orders the work: establish that *some* CLE-P configuration moves refusals before spending
anything on CLE-A. If none does, the CLE-A comparison is not the next experiment — the mechanism
question is.

### 3.3 Paired analysis, not rate differences

Greedy decoding is deterministic and both arms see the same 32 prompts, so control-vs-attacked is a
**paired** design. The informative quantity is the **discordant pairs** — prompts that refused in
one arm and complied in the other — not the difference of two rates.

Report `b` (refuse→comply) and `c` (comply→refuse) with McNemar's exact test. A net of 0 from
`b=0, c=0` means the intervention did nothing. A net of 0 from `b=4, c=4` means it did something
non-directional. These are completely different results and the rate difference hides both.

Costs nothing extra: the `.eval` logs already hold per-sample refusal labels.

### 3.4 Power, and what it licenses

At n=32 one sample is 3.1 pp. Under a paired test, detecting a genuine 6-flip effect (b=6, c=0) is
comfortable; a 3-flip effect is marginal. **Val screens for large effects only.** A val null
supports "no large effect," never "no effect" — and every val conclusion must be written that way.

n=176 (`test_public`) resolves ~3× finer and is where any headline claim has to land.

### 3.5 Screen on the harmful split alone

Refusals and Harm Score come from the harmful split; only Benign Non-refusal needs the benign run.
Harmful-only is ~12 min against ~30 for the pair. **Screening arms run harmful-only**; the benign
arm is added for configurations that survive screening, where the capability cost matters.

### 3.6 The control is layer-, β- and q-independent

β=0 is an exact no-op, so one control per (split, decoding) serves every arm. Already have it for
val greedy. Never re-run it.

## 4. Prerequisites before any new behavioural number

| P1 | Precision-aware selftest | rewrite tolerances as relative to hidden-state scale; report the scale. ~5 min GPU. Blocks all CLE-A claims. |
| P2 | Provider equivalence for CLE-P | the provider replaced the code path Stage 1 used. Re-run one Stage 1 arm at `--limit 8` and require identical per-sample scores. Greedy ⇒ exact match expected. ~8 min GPU. Blocks comparing new arms to Stage 1. |
| P3 | Flip analysis of Stage 1 | **no GPU.** Extract per-sample refusal labels from the pulled `.eval` logs and report b/c for both bands. Decides §3.3 for data already paid for. |

P3 runs now. P1 and P2 are the first GPU work of the next session.

## 5. The experiments

Each row states what it establishes and what a null means. Costs at $2.99/h.

### E1 — Does *any* CLE-P configuration move refusals? (val, harmful-only)

Escalate along the two principled axes, cheapest first. Stop at the first arm that moves refusals.

| arm | layers | q | why |
|---|---|---|---|
| E1a | 0-63 | 0.5 | the paper's threat model: perturb everything the monitor reads. 11 layers may simply be too few. |
| E1b | 53-63 | 0.9 | same band as Stage 1, pushed to deep harmless territory — tests whether the median target is too conservative |
| E1c | 0-63 | 0.9 | both axes at maximum, still inside the geometric constraint |

3 arms × ~12 min ≈ **36 min, ~$1.80**.

**Gate.** If any arm produces b ≥ 5 discordant flips toward compliance → H1 survives, proceed to E2.
If none does → H1 is in serious trouble at val resolution, and E3 becomes the priority instead of E2.

### E2 — Is it evasion, or just damage? (val, harmful-only, + probe read)

Only if E1 passes. Re-run the winning arm with the `06` in-scorer probe reading activations during
the attacked run, so both axes of the §2 table are populated in one pass. Requires wiring the
in-scorer probe into the CLE provider path — this is the one piece of engineering the design needs.

~12 min + ~half a day of work. **Without E2 we cannot claim evasion at all**, only behavioural change.

### E3 — Mechanism: did the intervention do what we think? (val, no eval)

Only if E1 fails. Diagnostic, not behavioural, and cheap.

- probe score at the post-instruction token, control vs attacked, per layer — does it land at `−m`
  in a *live agentic* forward, not just a cached-activation replay?
- how far the score drifts across a rollout's generate calls (`--trace-scores`)
- how much benign activations are displaced, since benign capability is what actually moved

Existing scripts cover most of this: `plot_steering_shift.py`, `plot_cle_mechanism.py`,
`extract_token_scores.py`. ~30 min GPU.

A clean E3 result — "activations land exactly on target, refusal behaviour does not follow" — is the
direct evidence for H0, and is a substantive result in our lane.

### E4 — CLE-P vs CLE-A (val, harmful-only)

Only if E1 passes; CLE-P is the ceiling (§3.2). Same probe, margin, layers, β — the two arms differ
in schedule and nothing else.

- `--attack cle-a --recalibrate sample` (the paper's semantics)
- `--recalibrate turn` as a labelled extra arm, ours not the paper's
- `--trace-scores` on both, giving the staleness curve as the mechanism

2 arms × ~12 min ≈ **24 min, ~$1.20**, plus P1 as a hard prerequisite.

Answers the brief's Q2 in the setting where a frozen displacement has the most room to go stale.

### E5 — Confirmation (test_public)

Whatever survives, run once at n=176 with both splits. `control.harmful` is already banked;
`control.benign` still needs ~100 min.

~2.6 h per attacked arm ≈ **$8/arm**, plus ~$5 once for the missing control half.

## 6. Sequence and total

```
P3 (free) ──► P1, P2 (~15 min GPU)
                 │
                 ▼
               E1  (36 min)
              /        \
        moves            does not
          │                 │
          ▼                 ▼
   E2 (evasion axis)     E3 (mechanism → H0)
          │
          ▼
   E4 (CLE-P vs CLE-A)
          │
          ▼
   E5 (test_public confirmation)
```

Through E4: **~1.5 h GPU, under $5.** E5 is the only expensive step and is gated on everything above.

## 7. What would make us stop

- **E1 null and E3 clean** (activations land on target, behaviour unmoved): H0. Write it up as a
  causal dissociation result. Do not spend on `test_public` for a null at val resolution without a
  power argument.
- **E1 passes but E2 shows the probe still fires**: behaviour moved without evasion. Interesting,
  but not the CLE claim — report it as such rather than as an attack success.
- **P2 fails**: the provider changed CLE-P's behaviour. Fix before anything else; every comparison
  to Stage 1 is void until it passes.

## 8. Known limits to state in any write-up

- One model (Qwen3.5-27B), which the paper does not cover, so no BO-tuned comparator exists.
- One probe type (`svm`). `single_direction` is the better refusal predictor (0.913 vs 0.832) and
  for it `w` *is* the semantic axis, so it is the more natural steering direction — untested here.
- Greedy measures the mode, not the mean. An intervention that shifts the behavioural distribution
  without moving the argmax reads as null.
- Val n=32 screens for large effects only.
