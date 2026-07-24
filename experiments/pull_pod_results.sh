#!/usr/bin/env bash
# Pull run outputs off the RunPod box onto the Mac.
#
# RunPod's SSH gateway has no scp/sftp, so files come back base64-encoded over the command
# channel, one at a time, with an md5 check on each. Slow but verifiable -- and a corrupted eval
# log that still parses is exactly the kind of thing we do not want to discover later.
#
#   experiments/pull_pod_results.sh                       # default: the CLE pilot
#   REMOTE=/workspace/acts_cache DEST=... experiments/pull_pod_results.sh
#
# Idempotent: files already present with a matching md5 are skipped, so it is safe to run while a
# job is still writing and again once it finishes.
set -uo pipefail

POD=${POD:-75lieo8gutf3zf-64411fde@ssh.runpod.io}
KEY=${KEY:-$HOME/.ssh/id_ed25519}
REMOTE=${REMOTE:-/workspace/cle_pilot}
EXTRA=${EXTRA:-/workspace/cle_pilot.log}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST=${DEST:-$HERE/results/08-agentic-cle-pilot}
MAXMB=${MAXMB:-40}

pod() { printf '%s\nexit\n' "$1" | ssh -tt -o StrictHostKeyChecking=no -o ConnectTimeout=30 \
        "$POD" -i "$KEY" 2>/dev/null | sed 's/\x1b\[[0-9;?]*[a-zA-Z]//g; s/\r//g'; }

mkdir -p "$DEST"
echo "pod    $REMOTE (+ $EXTRA)"
echo "dest   $DEST"

# One listing pass: path|size|md5, so we can skip and verify without extra round trips.
# Single-quoted remote snippet -- the local shell must not expand $1 before it is sent.
FIND_CMD="find $REMOTE $EXTRA -type f -exec sh -c "
FIND_CMD+=\''printf "%s|%s|%s\\n" "$1" "$(stat -c%s "$1")" "$(md5sum "$1"|cut -c1-32)"'\'
FIND_CMD+=" _ {} \\;"
list=$(pod "$FIND_CMD" | grep -aE '^/workspace/.*\|[0-9]+\|[0-9a-f]{32}$')

[ -z "$list" ] && { echo "nothing found under $REMOTE" >&2; exit 1; }
n=0; got=0; skipped=0
while IFS="|" read -r rf size md5; do
    [ -z "$rf" ] && continue
    n=$((n + 1))
    rel="${rf#/workspace/}"
    out="$DEST/$rel"
    mkdir -p "$(dirname "$out")"
    if [ -f "$out" ] && [ "$(md5 -q "$out" 2>/dev/null)" = "$md5" ]; then
        skipped=$((skipped + 1)); continue
    fi
    mb=$((size / 1048576))
    if [ "$mb" -gt "$MAXMB" ]; then
        echo "  SKIP  $rel (${mb} MB > MAXMB=${MAXMB}; route large files via GCS instead)"
        continue
    fi
    printf '  pull  %-58s %6s KB ... ' "$rel" "$((size / 1024))"
    # Wrapped base64 between ANCHORED markers. -w0 would make one huge line that the pty's
    # canonical mode mangles, and unanchored markers also match the shell's echo of the command
    # itself (which the terminal wraps, so the marker text recurs several times).
    pod "echo __B64_BEGIN__; base64 $rf; echo __B64_END__" \
        | awk '/^__B64_BEGIN__$/{f=1;next} /^__B64_END__$/{f=0} f' \
        | tr -d '\n' | base64 -d > "$out" 2>/dev/null
    if [ "$(md5 -q "$out" 2>/dev/null)" = "$md5" ]; then
        echo "ok"; got=$((got + 1))
    else
        echo "MD5 MISMATCH — removing"; rm -f "$out"
    fi
done <<< "$list"

echo
echo "$got pulled, $skipped already current, of $n remote file(s)"
[ "$got" -gt 0 ] && find "$DEST" -type f -newermt '-1 minute' | sed "s|$DEST/|  new: |"
exit 0
