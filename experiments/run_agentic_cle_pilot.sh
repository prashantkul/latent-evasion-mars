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
# Env knobs: SPLIT LAYERS BETA QUANTILE CONTROL OUT TAG. Run the control once per split, then
# CONTROL=0 for every further arm on it -- see the CONTROL comment below.
#
# Two configurations matter, and the pilot is not informative without both:
#   LAYERS=20-30  the attacker perturbs a band, the monitor reads all 64 -- untouched layers still fire
#   LAYERS=0-63   the attacker perturbs everything the monitor reads (the paper's threat model)
# Reporting only the first would measure the width of the band we happened to pick rather than the
# attack. See the "A probe is 64 probes" lab note.
set -euo pipefail

REPO=${REPO:-/workspace/latent-evasion-mars}
# The pinned venv on the network volume (pod_bootstrap.sh). Falls back to
# system python so the script still runs on a box without it.
PY=${PY:-/workspace/.venv/bin/python}
[ -x "$PY" ] || PY=python3
OUT=${OUT:-/workspace/cle_pilot}
SPLIT=${SPLIT:-val}
LAYERS=${LAYERS:-20-30}
BETA=${BETA:-1.0}
QUANTILE=${QUANTILE:-0.5}
# CONTROL=0 skips the beta=0 arm. Safe to skip once a control exists for this SPLIT: beta=0 makes
# the hook an exact no-op (h' = h), so the control does not depend on LAYERS or BETA and one run of
# it serves every arm on that split. Skipping saves ~33 min on val, ~2.6 h on test_public per arm.
# Never skip it for a split that has no control yet -- the vLLM baseline is not a substitute
# (different provider, and on val a different split too).
CONTROL=${CONTROL:-1}
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

ctl=()
[ "$CONTROL" = "1" ] && ctl=(--control)

nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
echo "=== agentic CLE — split=$SPLIT layers=$LAYERS beta=$BETA margin=q$QUANTILE of val harmless ==="
echo "    margins come from VAL activations only; the test split stays held out"
if [ "$CONTROL" = "1" ]; then
    echo "    --control runs beta=0 first: an exact no-op through the identical code path, so the"
    echo "    control-vs-attacked comparison isolates the intervention. NOTE it is not a check"
    echo "    against BASELINES.md, which is test_public on vLLM -- two differences at once."
else
    echo "    CONTROL=0: attacked arm only. Compare against the beta=0 control already run on this"
    echo "    split -- beta=0 is layer- and beta-independent, so that control is the right one."
fi

"$PY" experiments/agentic_cle.py \
    --model "$QWEN35_27B_MODEL" \
    --device cuda:0 \
    --probe "$PROBE" \
    --layers "$LAYERS" \
    --beta "$BETA" \
    --margin-quantile "$QUANTILE" \
    --val-acts "$VAL_ACTS" \
    --split "$SPLIT" \
    ${ctl[@]+"${ctl[@]}"} \
    --log-dir "$OUT/logs_$TAG" \
    --out "$OUT/cle_$TAG.json"

echo "=== done: $OUT/cle_$TAG.json ==="
