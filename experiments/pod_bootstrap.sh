#!/usr/bin/env bash
# One command to bring a restarted RunPod back to a state where the eval scripts run.
#
# Why this exists: /opt is CONTAINER disk and is wiped on every pod stop, taking the repo and any
# pip installs with it. /workspace is the network volume and survives, including the ~52 GB HF
# cache. So each restart needs the repo restored and the deps reinstalled -- but never the model
# re-downloaded, which is the expensive part.
#
#   printf 'bash /workspace/latent-evasion-mars/experiments/pod_bootstrap.sh\nexit\n' \
#     | ssh -tt <pod>@ssh.runpod.io -i ~/.ssh/id_ed25519
#
# Idempotent: safe to re-run on an already-healthy pod. Prints a readiness summary and exits
# non-zero if anything a run depends on is missing, so failures surface here rather than three
# hours into an eval.
set -uo pipefail

REPO=${REPO:-/workspace/latent-evasion-mars}
VENV=${VENV:-/workspace/.venv}
GIT_URL=${GIT_URL:-}
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
export PYTHONPATH="$REPO"
fail=0
note() { printf '  %-34s %s\n' "$1" "$2"; }

echo "=== pod bootstrap ==="
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || { echo "NO GPU"; fail=1; }

# --- repo on the network volume, not /opt ---
if [ -d "$REPO/.git" ]; then
    git -C "$REPO" pull --ff-only 2>&1 | tail -2
elif [ -n "$GIT_URL" ]; then
    git clone "$GIT_URL" "$REPO" 2>&1 | tail -2
else
    echo "  MISSING $REPO and no GIT_URL set — clone it or pass GIT_URL=..." >&2
    fail=1
fi
note "repo" "$REPO @ $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo MISSING)"

# --- deps into a venv ON THE NETWORK VOLUME, from a pinned file ---
# /workspace/.venv survives a pod restart, so this is a no-op on every run after the first. The
# pins live in experiments/pod-requirements.txt -- one declared environment rather than a package
# list maintained by hand in this script, which is how a pilot died on a missing `openai` after
# the 27B weights had loaded. --system-site-packages keeps the image's CUDA-matched torch visible;
# installing a PyPI torch over it risks a driver mismatch and a multi-GB download.
REQ="$REPO/experiments/pod-requirements.txt"
if [ ! -x "$VENV/bin/python" ]; then
    echo "  creating $VENV (first run on this volume)"
    uv venv --system-site-packages --python 3.12 "$VENV" 2>&1 | tail -2
fi
if [ -f "$REQ" ]; then
    UV_CACHE_DIR=/workspace/.uv-cache uv pip install --python "$VENV/bin/python" \
        -q -r "$REQ" 2>&1 | tail -3
else
    echo "  MISSING $REQ — cannot pin the environment" >&2; fail=1
fi
note "python" "$("$VENV/bin/python" -V 2>&1) at $VENV"
for mod in inspect_ai inspect_evals transformers sklearn openai torch; do
    v=$("$VENV/bin/python" -c "import importlib.metadata as m,sys;sys.stdout.write(m.version('$mod'.replace('sklearn','scikit-learn')))" 2>/dev/null) \
        && note "$mod" "$v" || { note "$mod" "MISSING"; fail=1; }
done

# --- HF cache must be on /workspace, or the 52 GB re-downloads ---
sz=$(du -sh "$HF_HOME" 2>/dev/null | cut -f1)
note "HF_HOME" "${HF_HOME} (${sz:-EMPTY})"
[ -d "$HF_HOME" ] || { echo "  HF cache missing — the model will re-download (~1 h)" >&2; fail=1; }

# --- the judge key: AgentHarm builds gpt-4o judges at task-BUILD time, so a run dies at import ---
if [ -f "$REPO/.env" ]; then
    set -a; source "$REPO/.env"; set +a
fi
[ -n "${OPENAI_API_KEY:-}" ] && note "OPENAI_API_KEY" "set (${#OPENAI_API_KEY} chars)" \
    || { note "OPENAI_API_KEY" "MISSING — AgentHarm cannot even build its task"; fail=1; }

# --- graded logs supply the refusal labels for the join ---
L05="$REPO/experiments/results/05-qwen35-agentic-dissociation/inspect_logs"
for d in logs_qwen35v_harmful logs_qwen35v_benign; do
    # shellcheck disable=SC2086
    compgen -G "$L05/$d/*.eval" > /dev/null && note "$d" "present" \
        || { note "$d" "MISSING (git pull?)"; fail=1; }
done

# --- the frozen probe, so a run can skip Pass A ---
P="$REPO/experiments/results/06-qwen35-inscorer-probe/probe_canonical/qwen35"
[ -f "${P}_svm.npz" ] && [ -f "${P}_svm.json" ] && note "frozen probe" "$P (--probe-in)" \
    || note "frozen probe" "absent — runs will need Pass A"

mkdir -p /workspace/acts_cache
echo
if [ "$fail" -eq 0 ]; then
    echo "READY. Next, under tmux so SSH cannot kill a long run:"
    echo "  tmux new -s run -d 'bash $REPO/experiments/run_qwen35_test_acts_cache.sh 2>&1 | tee /workspace/run.log'"
    echo "  tail -20 /workspace/run.log"
else
    echo "NOT READY — see the MISSING lines above." >&2
fi
exit "$fail"
