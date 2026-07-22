# Experiments — refusal in the agentic setting (AgentHarm)

The experiment layer on top of `latent-evasion`: does the single-turn refusal probe still work when
Llama-3.2-3B runs as a tool-using agent? Uses AgentHarm (via `inspect_evals`) and the per-layer probes
trained by `../classifier/train_latent.py`.

## Contents
- `agentharm-refusal-tracker.md` — live experiment status, milestones (M0–M4), results.
- `dgx-setup.md` — the machine everything runs on (DGX Spark), env, keys, run recipe, gotchas.
- Scripts:
  - `m0a_hidden_states.py` — smoke test: Inspect `hf` provider returns per-layer hidden states.
  - `m0b_dataset.py` — smoke test: AgentHarm dataset loads; `chat_dataset` toggle.
  - `mini_run.py` — M2: chat vs agentic probe firing on matched behaviors.
  - `diagnose_separation.py` — M2b: harmful-vs-benign separation (AUC) vs the Arditi ceiling.
  - `check_logs.py` — read saved Inspect `.eval` logs at full precision.

## Dependency
These reuse `latent-evasion` (the probe artifacts in `dataset/representations/llama32-3b/train_svm/`
and the model wrappers) plus `inspect_ai` + `inspect_evals[agentharm]`. They are written to run on the
DGX where the repo lives at `~/latent-evasion`. See `dgx-setup.md` for the full recipe.

## Quick run (on the DGX)
```bash
cd ~/latent-evasion && source .venv/bin/activate
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export LLAMA32_3B_MODEL=unsloth/Llama-3.2-3B-Instruct
set -a; source ~/Documents/source-code/latent-adversarial-detection/.env; set +a   # judges
python experiments/diagnose_separation.py
```

## Headline result so far (M2b)
The single-turn refusal direction **transfers** to the agentic distribution — agentic AUC 0.97–0.99 at
layers 14–26 (≈ the Arditi training ceiling). The earlier "firing dropped 0.81→0.51" was a coarse
all-layers average; at good layers the direction cleanly separates harmful from benign. No
agentic-native probe needed on direction-failure grounds. Details in the tracker.
