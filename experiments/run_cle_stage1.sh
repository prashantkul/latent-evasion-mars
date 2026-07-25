#!/usr/bin/env bash
# Stage 1 of experiments/cle_p_agentic_plan.md: clean foundation + band comparison, on val, GREEDY.
#
#   tmux new -s cle -d 'bash /workspace/latent-evasion-mars/experiments/run_cle_stage1.sh'
#   tail -f /workspace/cle_stage1.log
#
# Two arms, chained so the GPU never idles between them:
#
#   A  L20-30 beta=1  WITH control  (~60 min)  the band 08 used, re-run greedy
#   B  L53-63 beta=1  no control    (~30 min)  the refusal band
#
# Why A carries the control and B does not: beta=0 is an exact no-op, so the control depends on
# neither LAYERS nor BETA. One control pair per split serves every arm on it.
#
# Why greedy matters here: the 08 pilot ran with Inspect's hf default do_sample=True, so its control
# and attacked arms differ by sampling noise on top of the intervention. agentic_cle.py now passes
# do_sample=False, which is why A re-runs a band we have already "measured".
#
# Why a separate OUT: /workspace/cle_pilot holds the sampling-era pilot. Keeping the greedy runs
# apart means no json is overwritten and no one has to date-check a file to know which is which.
set -euo pipefail

REPO=${REPO:-/workspace/latent-evasion-mars}
OUT=${OUT:-/workspace/cle_stage1}
RUN="$REPO/experiments/run_agentic_cle_pilot.sh"

mkdir -p "$OUT"
exec > >(tee -a /workspace/cle_stage1.log) 2>&1

echo "############ STAGE 1A — L20-30 beta=1.0 + control (val, greedy) ############"
OUT="$OUT" SPLIT=val LAYERS=20-30 BETA=1.0 CONTROL=1 \
    TAG=val_L20-30_beta1.0_greedy bash "$RUN"

echo "############ STAGE 1B — L53-63 beta=1.0, reusing 1A's control ############"
OUT="$OUT" SPLIT=val LAYERS=53-63 BETA=1.0 CONTROL=0 \
    TAG=val_L53-63_beta1.0_greedy bash "$RUN"

echo "############ STAGE 1 COMPLETE ############"
ls -la "$OUT"/*.json
