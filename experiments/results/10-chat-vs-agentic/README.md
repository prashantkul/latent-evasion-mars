# 10 — Chat vs agentic ablation (Qwen3.5-27B)

The same 44 harmful behaviours, asked two ways: as plain chat, and as an agentic task with tools.
No probes, no steering, no intervention. Holding the request fixed and varying only the framing
isolates how much of the refusal rate is about the *agentic setting* rather than the content of
the ask.

    Qwen3.5-27B | test_public | hf provider | greedy | AgentHarm official combined_scorer

## The four standard metrics, both arms

Reported in full, because three of the four are *structurally unavailable* in chat rather than
merely low — and that is the most important thing to see about this run.

| Standard metric | Source field | chat (44) | agentic (176) |
|---|---|---|---|
| Harm Score | `avg_score` (harmful) | **0.0000** | 0.3273 |
| Refusals | `avg_refusals` (harmful) | **0.7500** | **0.5909** |
| Non-refusal Harm Score | `avg_score_non_refusals` (harmful) | **0.0000** | 0.7718 |
| Benign Non-refusal Score | `avg_score_non_refusals` (benign) | **n/a** | — |

`avg_full_score` is 0.0000 on the chat arm too.

The two zeros are an artifact, not a measurement — see *What this does NOT measure* below. The `n/a`
is structural: `chat_dataset` exists on `agentharm()` but not `agentharm_benign()`. The agentic
`Benign Non-refusal` is absent for a different reason — that condition was `status=started` when the
pod went down and has never been completed.

**So this ablation rests on one of the four metrics.**

## Result — the refusal axis

| | refusal rate |
|---|---|
| chat (`chat_dataset=True`, 44 samples, no tools) | **0.7500** |
| agentic (`chat_dataset=False`, 176 samples, tools) | **0.5909** |
| **agentic − chat**, paired over 44 behaviours | **−0.1591** &nbsp; 95% CI [−0.3125, 0.0000] |

**Discordant behaviours: 12 refused in chat only, 2 refused agentically only.**
McNemar exact, two-sided: **p = 0.0129**.

Giving the model tools makes it refuse the *same request* substantially less often. The direction is
consistent — 12 flips one way against 2 the other. For contrast, the `08` sampling pilot's 4-vs-2
flip split gives p = 0.69, which is what noise looks like.

## Why the join is the design

Agentic is 4× augmented (176 = 44 behaviours × 4 variants); chat is base-only (44). Comparing 44
rows against 176 rows as two independent rates would confound the framing with whatever the
augmentation did. Samples carry `id_original`, so agentic aggregates to the behaviour level and the
comparison is **paired over 44 behaviours**, with the bootstrap resampling behaviours rather than
rows — the four variants of one behaviour are not independent observations. The M0 note flagged this
join key back on 21 Jul; `id` will not do.

Extraction was verified against the banked agentic control before spending GPU: 176 rows → 44
behaviours → exactly 4 variants each, refusal rate reconstructed as 0.5909.

## By category

Not uniform, and one category reverses hard.

| category | n | chat | agentic |
|---|---|---|---|
| Copyright | 6 | 0.333 | 0.167 |
| **Cybercrime** | 5 | **0.200** | **0.800** |
| Disinformation | 5 | 1.000 | 0.400 |
| Drugs | 5 | 1.000 | 0.600 |
| Fraud | 6 | 0.500 | 0.458 |
| Harassment | 6 | 1.000 | 0.583 |
| Hate | 5 | 1.000 | 0.750 |
| Sexual | 6 | 1.000 | 1.000 |

Cybercrime goes the *other* way — 0.20 in chat against 0.80 agentically. Sexual is saturated at 1.000
in both and contributes nothing. At n=5–6 per category these are individually weak, but the
Cybercrime reversal is large enough to be worth a look rather than an averaging-away.

## What this does NOT measure

**Harm Score and Non-refusal Harm are 0.0000 for every chat sample, and are not usable.** AgentHarm's
grading functions score tool calls — which tools were called, in what order, with what arguments. In
chat there are no tool calls, so every grading function returns zero. That is an artifact of applying
an agentic grader to a chat transcript, not a finding about the model.

Only the **refusal axis** is comparable across the two framings, because refusal is judged by the
gpt-4o refusal judge reading the response text, which works in both. The per-category refusal
breakdown behaves sensibly (0.2 to 1.0, varying by category), which is the evidence that the judge
side is fine while the grading side is not.

`chat_dataset` exists only on `agentharm()`, not `agentharm_benign()`, so there is no benign chat arm
and therefore no chat-side capability anchor.

## Provenance

- **Chat arm** — run here, 25 Jul 2026: `ablation/chat/*.eval`, log `ablation.log`.
- **Agentic arm** — the β=0 control from the interrupted CLE run, 24 Jul:
  `results/08-agentic-cle-pilot/cle_pilot/logs_test_public_L20-30_beta1.0/control_harmful/2026-07-24T16-58-42*.eval`.
  β=0 makes the hook an exact no-op (`max|delta| = 0`, measured), so it is a vanilla agentic run.
  Same model, same provider, same greedy decoding. It was not re-run because it is already the
  comparator we need, and re-running would have cost ~57 min for a number we hold.
- Paired stats: `ablation/ablation.json`.

Reproduce:

```bash
python3 experiments/chat_vs_agentic_ablation.py run --split test_public --log-dir <dir>
python3 experiments/chat_vs_agentic_ablation.py join --chat '<dir>/chat/*.eval' \
    --agentic '<agentic>/*.eval' --out ablation.json
```

## Why it matters

This is the behavioural counterpart to the dissociation result. `01`/`05` showed the internal refusal
signal staying on while the agent complied. This shows the *behaviour* itself moving with framing:
the same request, with tools attached, is refused 16 points less often.

It also sets a scale for the CLE work. Our steering interventions moved refusals by 0 and 1 sample on
val; simply reframing the task as agentic moves it by 12 behaviours out of 44. **The framing is a far
larger lever on refusal than any activation intervention we have applied so far.**
