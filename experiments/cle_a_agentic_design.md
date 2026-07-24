# Adding CLE-A to the agentic harness — design note

`experiments/agentic_cle.py` runs **CLE-P only**. The paper evaluates both, so the agentic
baseline is partial until CLE-A is in. This note says what CLE-A actually does, why it does not
drop in as another flag, and what we have to decide before writing it.

Status: **not implemented**. Written 24 Jul 2026 while the CLE-P val pilot ran.

---

## 1. What CLE-A does, from the code

Two phases, both in `cle-a.py` / `utils/hooks.py`:

**Calibration.** Install `pipeline_delta_hook` on every selected layer, run *one* forward pass over
the prompt, and record

```python
delta[l] = h_mod[:, -1, :] - h[:, -1, :]        # displacement at the LAST prompt token only
```

The hook applies the full CLE-P projection while it measures, so `delta[l]` is exactly the CLE-P
displacement — captured at one position and then frozen.

**Generation.** Remove those hooks, install `add_hook(delta[l])`, which adds that fixed vector at
**every position and every decode step**.

So the two attacks share a displacement rule and differ in *where it is recomputed*:

| | CLE-P | CLE-A |
|---|---|---|
| displacement | recomputed per position, per step | computed once, then frozen |
| lands the score at | exactly `−margin` (β=1) | `−margin` at the calibration token only; elsewhere, wherever the fixed vector puts it |
| needs a pre-pass | no | **yes**, one forward per calibration |

Two details that are easy to miss and would silently change the method:

- **The deltas are a pipeline, not independent.** All calibration hooks are installed at once, so
  layer *l*'s delta is measured on a hidden state already modified by layers `< l`. Computing them
  one layer at a time gives different numbers. Hence `pipeline_delta_hook`.
- **The delta is per-prompt**, derived from that prompt's own last-token activation — not a global
  steering vector.

Note the calibration position is the last prompt token — the same post-instruction read position
our probe uses. That coherence is convenient and worth preserving.

---

## 2. Why it does not drop into the current harness

`agentic_cle.py` installs hooks once around the whole `inspect_eval` call, because CLE-P is
stateless — it needs nothing but `(w, b, β, margin)`.

CLE-A needs a calibration forward pass **per prompt, before that prompt's generation**, and Inspect
drives generation internally. There is no per-sample "before generate" seam at the hook level.

Worse in the agentic setting: an AgentHarm rollout is *many* generate calls, each with a longer
context than the last (tool results accumulate). "The prompt" is not a single thing.

---

## 3. The decision that actually matters: recalibration cadence

The paper is single-turn, so the question never arises there. In a rollout we must choose:

- **`sample`** — calibrate once on the initial prompt, freeze the delta for the entire rollout.
  Closest to the paper's semantics, and the sharpest contrast with CLE-P. As the context grows,
  the frozen delta becomes less and less matched to the activations it is being added to. If CLE-A
  underperforms CLE-P agentically, this is almost certainly why — and that is a *finding*, not a
  bug, since it is a property the paper's setting cannot exhibit.
- **`turn`** — recalibrate at each generate call. Tracks the growing context, but it is no longer
  the paper's CLE-A; it is a middle point between the two attacks that we would be inventing.

**Recommendation:** implement both behind `--recalibrate {sample,turn}`, default `sample`, and
report `sample` as the CLE-A baseline. `turn` is then a cheap extra arm that isolates exactly how
much the staleness costs. Do not silently pick `turn` because it scores better — label it as ours.

---

## 4. Integration options

**A. Solver-level.** Add an Inspect solver step before the agent that calibrates and installs
`add_hook`s per sample. Natural fit for `sample` cadence. Risk: we would tokenize the messages
ourselves to build the calibration input.

**B. Wrap the provider's generate. ← recommended.** Intercept the hf provider's generate, and on
each call: (1) if a delta is needed, run the calibration forward on the *exact* `input_ids` about
to be generated from, (2) install `add_hook`s, (3) generate, (4) remove. Cache by `sample_id` for
`sample` cadence; recompute every call for `turn`.

**Why B.** It calibrates on the exact token sequence the model is about to generate from, rather
than on a sequence we re-render. Re-rendering the agent's context and getting it subtly wrong is
precisely the failure that invalidated experiment `05` — the omitted 558-char agent system prompt.
We should not re-open that door for a second time in the same codebase.

**C. Global delta** (one vector from val, applied everywhere). Simplest, but it is a steering
vector, not CLE-A. Only worth it as a deliberate third arm, clearly named.

---

## 5. Control

`--beta 0` stays an exact no-op and needs no special handling: at β=0 the projection gives
`h_mod == h`, so `delta == 0` and `add_hook` adds a zero vector. The control still runs the
calibration pass, so it also controls for any effect of the extra forward pass. Keep it.

---

## 6. Gotchas to check before trusting a number

1. **Batch size.** `add_hook` broadcasts a 1-D delta over the batch, but a batched calibration
   returns one delta per row (2-D) and the rows must stay aligned. We run `max_connections=1`;
   assert batch==1 rather than assume it.
2. **Decode steps.** With a KV cache, `h` has `seq_len == 1` during decode. Adding the delta there
   is intended — verify it actually fires, since a silently-absent hook looks like "attack does
   not work".
3. **Layer ordering.** Install all calibration hooks before the pass, in layer order, so the
   pipeline property in §1 holds.
4. **Reaching the model.** `hf_layers()` in `agentic_cle.py` already walks to the decoder blocks;
   the calibration pass needs the same model object, under `torch.no_grad()`.
5. **Margins.** Identical to CLE-P — `geometric_margins()` from val, no change.

---

## 7. How we would know it is right

Before any behavioural claim:

- **β=0 ⇒ delta is exactly zero** at every selected layer (`max|delta| == 0`), and the four metrics
  match a no-hook run.
- **Single-position equivalence.** On one prompt, CLE-A and CLE-P must produce an *identical*
  hidden state at the calibration token, since the delta is defined as that displacement. Any gap
  means the calibration pass is not seeing what generation sees. This is the sharpest available
  test and it is cheap.
- **Score landing.** At the calibration token, post-hook `w·h + b == −margin` to float tolerance —
  the same check that gave 2.66e-15 for CLE-P.
- **Staleness is measurable.** Log `w·h + b` at the last position of each turn under `sample`
  cadence; it should drift away from `−margin` as the rollout grows. That number is the mechanism
  behind whatever the four metrics show, and it is what makes the result explanatory rather than
  just an outcome.

---

## 8. Effort

Roughly a day: the generate wrapper and delta cache are the real work; margins, control, runner and
reporting all come over unchanged from CLE-P. Sequencing — finish the CLE-P val pilot first (it
tells us the harness is sound), then CLE-A on val, then take whichever arms matter to `test_public`
together, so the expensive run happens once.
