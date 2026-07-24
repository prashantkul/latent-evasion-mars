#!/usr/bin/env bash
# Agentic CLE pilot: run CLE-P inside a live AgentHarm eval and score it with the four standard
# metrics, so the result is comparable to the frozen baseline in BASELINES.md.
#
# Defaults to the CHEAP configuration on purpose -- val split, one layer band. A hook or scorer
# problem is much cheaper to find on 32+32 rows than on 176+176.
#
#   tmux new -s cle -d 'bash experiments/run_agentic_cle_pilot.sh 2>&1 | tee /workspace/cle_pilot.log'
#   tail -30 /workspace/cle_pilot.log
#
# Two configurations matter, and the pilot is not informative without both:
#   LAYERS=20-30  the attacker perturbs a band, the monitor reads all 64 -- untouched layers still fire
#   LAYERS=0-63   the attacker perturbs everything the monitor reads (the paper's threat model)
# Reporting only the first would measure the width of the band we happened to pick rather than the
# attack. See the "A probe is 64 probes" lab note.
set -euo pipefail

REPO=${REPO:-/workspace/latent-evasion-mars}
OUT=${OUT:-/workspace/cle_pilot}
SPLIT=${SPLIT:-val}
LAYERS=${LAYERS:-20-30}
BETA=${BETA:-1.0}
QUANTILE=${QUANTILE:-0.5}
VAL_ACTS=${VAL_ACTS:-/workspace/acts_cache/qwen35_inscorer_val_acts.npz}
PROBE=${PROBE:-$REPO/experiments/results/06-qwen35-inscorer-probe/probe_canonical/qwen35_svm}
TAG=${TAG:-${SPLIT}_L${LAYERS}_beta${BETA}}

mkdir -p "$OUT"
cd "$REPO"

# AgentHarm builds its gpt-4o judges at task-BUILD time, so the key must be set to LOAD the task.
set -a; source "$REPO/.env"; set +a
: "${OPENAI_API_KEY:?OPENAI_API_KEY not set — AgentHarm cannot even build its task}"
export QWEN35_27B_MODEL=${QWEN35_27B_MODEL:-Qwen/Qwen3.5-27B}
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
export PYTHONPATH="$REPO"

for f in "$VAL_ACTS" "${PROBE}.npz" "${PROBE}.json"; do
    [ -f "$f" ] || { echo "MISSING $f" >&2; exit 1; }
done

nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
echo "=== agentic CLE — split=$SPLIT layers=$LAYERS beta=$BETA margin=q$QUANTILE of val harmless ==="
echo "    margins come from VAL activations only; the test split stays held out"
echo "    --control runs beta=0 first: an exact no-op through the identical code path, so the"
echo "    comparison isolates the intervention rather than the hf-vs-vLLM provider change"

python3 experiments/agentic_cle.py \
    --model "$QWEN35_27B_MODEL" \
    --device cuda:0 \
    --probe "$PROBE" \
    --layers "$LAYERS" \
    --beta "$BETA" \
    --margin-quantile "$QUANTILE" \
    --val-acts "$VAL_ACTS" \
    --split "$SPLIT" \
    --control \
    --log-dir "$OUT/logs_$TAG" \
    --out "$OUT/cle_$TAG.json"

echo "=== done: $OUT/cle_$TAG.json ==="
