#!/usr/bin/env bash
# Cache the in-context TEST activations for Qwen3.5-27B (RunPod H100).
#
# This is the artifact that turns every future probe re-evaluation from an agentic GPU run
# into a CPU matmul (experiments/eval_probe.py --acts). Re-runs the full valid two-pass
# experiment from `06`, which now also persists Pass B's raw activations for ALL layers
# (352 x 64 x 5120 float32 ~ 461 MB) -- all layers, because a new probe may live anywhere.
#
# Outputs go to /workspace (network volume) so they SURVIVE a pod stop. /opt is container
# disk and is wiped on restart -- that is how the previous run's artifacts were lost.
#
# Run under tmux -- SSH on this pod eats output from long-blocking commands, so everything
# goes to a file log and we poll it with short reads.
#
#   tmux new -s acts -d 'bash experiments/run_qwen35_test_acts_cache.sh 2>&1 | tee /workspace/test_acts.log'
#   tail -20 /workspace/test_acts.log
set -euo pipefail

REPO=${REPO:-/workspace/latent-evasion-mars}
OUT=${OUT:-/workspace/acts_cache}
mkdir -p "$OUT"
cd "$REPO"

# AgentHarm builds its gpt-4o judges at task-build time, so the key must be set merely to
# LOAD the task -- not just to score it.
set -a; source "$REPO/.env"; set +a
: "${OPENAI_API_KEY:?OPENAI_API_KEY not set — AgentHarm cannot even build its task without it}"
export QWEN35_27B_MODEL=${QWEN35_27B_MODEL:-Qwen/Qwen3.5-27B}
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}   # 52 GB cache lives here; do not re-download
export PYTHONPATH="$REPO"

# Graded vLLM logs supply the behavioural (refusal) labels for the join. They are committed
# to the repo under 05, so a `git pull` restores them even after a container-disk wipe.
L05="$REPO/experiments/results/05-qwen35-agentic-dissociation/inspect_logs"
H_LOG="$L05/logs_qwen35v_harmful/*.eval"
B_LOG="$L05/logs_qwen35v_benign/*.eval"
for pat in "$H_LOG" "$B_LOG"; do
    # shellcheck disable=SC2086
    if ! compgen -G $pat > /dev/null; then
        echo "MISSING graded log: $pat  (git pull in $REPO?)" >&2
        exit 1
    fi
done

# Reuse the frozen probe by default: the val split is a fixed 64 rows and the fit is
# deterministic, so Pass A would re-derive weights we already have bit for bit. This halves
# the agentic GPU time. Set PROBE_IN= (empty) to retrain from scratch instead.
PROBE_IN=${PROBE_IN-$REPO/experiments/results/06-qwen35-inscorer-probe/probe_canonical/qwen35}
if [ -n "$PROBE_IN" ]; then
    REUSE=(--probe-in "$PROBE_IN")
    echo "reusing probe: $PROBE_IN (Pass A skipped)"
else
    REUSE=()
    echo "retraining: Pass A will run"
fi

nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
echo "=== in-scorer experiment (Pass A: val/train, Pass B: test/eval + cache all layers) ==="

python3 experiments/results/06-qwen35-inscorer-probe/qwen35_inscorer_experiment.py \
    --model "$QWEN35_27B_MODEL" \
    --device cuda:0 \
    "${REUSE[@]}" \
    --harmful-log "$H_LOG" \
    --benign-log  "$B_LOG" \
    --log-dir     "$OUT/logs_qwen35_acts_cache" \
    --acts-out    "$OUT/qwen35_inscorer_val_acts.npz" \
    --test-acts-out "$OUT/qwen35_inscorer_test_acts.npz" \
    --out         "$OUT/qwen35_inscorer_experiment.json"

echo "=== done ==="
ls -la "$OUT"/qwen35_inscorer_*acts.npz
echo "Verify (on the Mac, after pulling): python experiments/verify_test_acts_cache.py --acts <path>"
