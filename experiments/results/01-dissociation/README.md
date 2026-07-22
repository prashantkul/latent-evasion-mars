# 01 — Dissociation (full graded run, M2f)

**Question:** in the agent loop, does refusal behaviour track the internal refusal signal?

- `results_full.json` — per-row records (probe firing, L18 score, graded harm, refused) + bootstrap
  CIs, from all 176 test_public rows under AgentHarm's official generation limits (4096 tok / 20 msg),
  graded by the real combined_scorer (gpt-4o judge + grading functions).
- `plot_results.py` → `m3_overview.png` — 4-panel: headline metrics w/ CI, per-category bars,
  per-behaviour scatter, and the behaviour×representation 2×2.

**Result:** refusal rate 0.46, harm 0.27, probe fires 0.90; **P(fire | did-not-refuse) = 0.81
[0.70,0.92]** — the model represents harm as harmful yet complies. Refusal ⇒ firing (80/80).
