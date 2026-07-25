# DGX Setup — AgentHarm / refusal-probe experiments

Reference for the machine. The CUDA/torch pipeline runs on the Nvidia DGX Spark, **not** the Mac
(the Mac env is MLX-only, no torch). For which runs belong here versus on a rented H100, read
**[Choosing the box](#choosing-the-box)** first — it is a throughput question, and the answer is
not "the free one".

## Access
- `ssh nvidia-dgx` — alias in `~/.ssh/config` (Tailscale IP `100.114.188.13`, user `prashantkulkarni`,
  key `~/.ssh/id_prashant_orion-ai_io`).
- Hardware: DGX Spark, **GB10 Grace-Blackwell**, `aarch64`, 128 GB unified memory, single GPU (`cuda:0`).
- Disk: 3.6 TB, ~2.6 TB free (as of 25 Jul 2026).

<a id="choosing-the-box"></a>
## Choosing the box — measured, 25 Jul 2026

**Capacity never binds here; bandwidth does.** 128 GB of unified memory holds anything we run, and
the HF cache already carries most of it. But GB10's memory is unified LPDDR5X, not HBM, and
autoregressive decoding is memory-bandwidth-bound. Measured with
`experiments/bench_decode.py` on **Qwen3.5-27B**, greedy, single stream:

| | DGX Spark (GB10) | RunPod H100 SXM |
|---|---|---|
| model load (52 GB) | 267 s | ~240 s |
| **decode** | **4.05 tok/s** | ~an order of magnitude faster |
| cost | owned | $2.99/h |

At 4 tok/s a ~600-token generation takes 2.5 min and a ~6-step rollout takes ~15 min of generation
alone. A `test_public` harmful arm (176 samples) runs in ~57 min on the H100; here it is roughly
**8–19 hours** depending on how many samples refuse immediately versus running full tool chains.
An hour becomes most of a day.

*Caveat on the ratio:* the DGX figure is measured, the H100 side is inferred from eval wall-clock
(which also contains tool and judge latency). A matched `bench_decode.py` run on the pod would
tighten it; not yet done.

**So route by token volume, not by who owns the hardware:**

| Run here (DGX) | Rent an H100 |
|---|---|
| correctness checks — `agentic_cle.py --selftest` is three forward passes on one prompt, no generation | any arm generating thousands of tokens over 44–176 samples (E1, E2, E4, E5) |
| mechanism diagnostics — activation reads, displacement, steering plots (E3) | `test_public` confirmation runs |
| anything on Llama-3.2-3B (already configured) | anything where wall-clock gates the next decision |
| all analysis — `eval_probe.py` on cached activations, flip analysis, ablation joins (CPU matmuls; the Mac can do these too) | |

The general rule: **forward passes are cheap here, generated tokens are not.**

## Model cache (`~/.cache/huggingface/hub`, 25 Jul 2026)

Far more than this doc used to claim — Qwen3.5-27B is already here, so an agentic run needs no
download:

| Model | Size |
|---|---|
| `meta-llama/Llama-3.1-70B-Instruct` | 132 G |
| `Qwen/Qwen2.5-Coder-32B-Instruct` | 62 G |
| `Qwen/Qwen2.5-32B-Instruct` | 62 G |
| `Qwen/Qwen3-Coder-30B-A3B-Instruct` | 57 G |
| **`Qwen/Qwen3.5-27B`** | **52 G** |
| `google/gemma-3-27b-it` | 52 G |
| `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | 45 G |
| `Qwen/Qwen2.5-72B-Instruct-AWQ` | 39 G |
| `unsloth/Llama-3.2-3B-Instruct` | 6.1 G |

## Known gaps on this box

- **`~/latent-evasion` is an rsync copy, not a git checkout** (no `.git`), so it cannot `git pull`
  and its scripts sit at the repo root rather than under `experiments/`. Every transfer is manual
  and versions drift silently. Converting it to a clone of `latent-evasion-mars` is ~10 minutes and
  would remove a whole class of "which version ran?" ambiguity.
- **The `OPENAI_API_KEY` below still needs rotating** — flagged 21 Jul, open since. It drives the
  AgentHarm judges on every box, not just this one.

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
