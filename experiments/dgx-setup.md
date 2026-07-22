# DGX Setup — AgentHarm / refusal-probe experiments

Reference for the machine everything runs on. The CUDA/torch pipeline runs on the Nvidia DGX Spark,
**not** the Mac (the Mac env is MLX-only, no torch).

## Access
- `ssh nvidia-dgx` — alias in `~/.ssh/config` (Tailscale IP `100.114.188.13`, user `prashantkulkarni`,
  key `~/.ssh/id_prashant_orion-ai_io`).
- Hardware: DGX Spark, **GB10 Grace-Blackwell**, `aarch64`, 128 GB unified memory, single GPU (`cuda:0`).

## Layout (all under `~/latent-evasion`)
| What | Path |
|------|------|
| Repo (rsynced from Mac) | `~/latent-evasion/` |
| Virtualenv | `~/latent-evasion/.venv` |
| Trained probes (28 layers) | `~/latent-evasion/dataset/representations/llama32-3b/train_svm/` |
| Raw activations | `.../train_svm/HFx_train.pt`, `HLx_train.pt` |
| Experiment scripts | `~/latent-evasion/{m0a_hidden_states,m0b_dataset,mini_run,check_logs}.py` |
| Inspect eval logs | `~/latent-evasion/logs_mini/*.eval` |
| Model weights (cached, 6.1 GB) | `~/.cache/huggingface/hub/models--unsloth--Llama-3.2-3B-Instruct` |
| HF token file | `~/.cache/huggingface/token` |

## Environment
- Created with `uv venv --system-site-packages` so it reuses the **system torch** (currently
  `2.13.0+cu130`, aarch64 build) — do **not** reinstall the `torch==2.10` pinned in `requirements.txt`;
  the cu130 aarch64 wheel is special.
- Installed on top: `inspect_ai` (0.3.249), `inspect_evals[agentharm]`, `openai`, `transformers`
  (5.x), `scikit-learn`, `accelerate`, `sentencepiece`, `numpy`.

## Model
- `unsloth/Llama-3.2-3B-Instruct` — **ungated mirror**, weight-identical to `meta-llama/Llama-3.2-3B-Instruct`
  (the account `pskulkarni` is not on Meta's gated allow-list). Selected via env
  `LLAMA32_3B_MODEL`; the code default in `models/llama32.py` still points at the official repo.

## Keys
- `HF_TOKEN` — export from the token file: `export HF_TOKEN=$(cat ~/.cache/huggingface/token)`.
- `OPENAI_API_KEY` (gpt-4o refusal/semantic judges) — not in the shell env; sourced from a sibling
  project `.env`: `set -a; source ~/Documents/source-code/latent-adversarial-detection/.env; set +a`.
  ⚠️ **Rotate this key** — it was printed to a command log on 21 Jul 2026.

## Run recipe
```bash
ssh nvidia-dgx
cd ~/latent-evasion && source .venv/bin/activate
export PYTHONPATH=$HOME/latent-evasion            # so `models`/`utils` import (scripts under classifier/)
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export LLAMA32_3B_MODEL=unsloth/Llama-3.2-3B-Instruct
set -a; source ~/Documents/source-code/latent-adversarial-detection/.env; set +a   # judges

# (re)train single-turn probes — extraction + per-layer SVM/SD:
python classifier/train_latent.py --model_name llama32-3b --device cuda:0 --n_samples -1
# agentic mini-run — chat vs agentic probe firing:
python mini_run.py
# inspect a saved eval log:
inspect view --log-dir logs_mini
```

## Gotchas
- **PYTHONPATH:** `python classifier/train_latent.py` puts `classifier/` (not repo root) on the path →
  set `PYTHONPATH=$HOME/latent-evasion`, run from the repo root (relative `./dataset` paths).
- **transformers 5.x rejects `GenerateConfig(temperature=0.0)`** — omit for greedy (`do_sample=False`).
  The stock `agentharm` task hardcodes `temperature=0.0`, so it hits this too on this stack.
- **hf provider batches** concurrent samples → `metadata["hidden_states"]` has a batch dim; index the
  right row or set `max_connections=1` (byte-identical probe values across samples is the tell).
- Judges are constructed at **task-build time**, so `OPENAI_API_KEY` must be set even to *load* the
  agentharm task, not just to score.

## Sync
Push local edits from the Mac:
`rsync -az --exclude .git --exclude __pycache__ --exclude media latent-evasion/ nvidia-dgx:~/latent-evasion/`
The four experiment scripts currently live only on the DGX (and the Mac scratchpad); move them into the
repo tree if we want them versioned.
