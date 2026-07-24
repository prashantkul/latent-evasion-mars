#!/usr/bin/env bash
# Push cached activations from the pod straight to GCS with a SHORT-LIVED access token.
#
# The token is minted on the Mac (`gcloud auth print-access-token --account=...`) and expires in
# ~1 hour. Deliberately NOT a service-account key or refresh token: the pod is rented hardware, so
# nothing long-lived should ever land on it. Re-mint and re-run if it expires mid-upload.
#
#   # on the Mac
#   TOK=$(gcloud auth print-access-token --account=kulkarniprashants@gmail.com)
#   # on the pod
#   GCS_TOKEN=$TOK bash experiments/upload_acts_gcs.sh
set -euo pipefail

BUCKET=${BUCKET:-probe-activations-123213e}
PREFIX=${PREFIX:-qwen35-27b/incontext}
SRC=${SRC:-/workspace/acts_cache}
: "${GCS_TOKEN:?GCS_TOKEN not set — mint one on the Mac with gcloud auth print-access-token}"

upload() {   # upload <local-path> <object-name>
    local f=$1 obj=$2 sz sess
    [ -f "$f" ] || { echo "skip (missing): $f"; return 0; }
    sz=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f")
    echo "→ $obj  ($(numfmt --to=iec "$sz" 2>/dev/null || echo "$sz B"))"

    # Resumable upload: a single POST would have to buffer ~460 MB and cannot be retried.
    sess=$(curl -sS -X POST \
        -H "Authorization: Bearer $GCS_TOKEN" \
        -H "Content-Type: application/json; charset=UTF-8" \
        -H "X-Upload-Content-Type: application/octet-stream" \
        -H "X-Upload-Content-Length: $sz" \
        -d '{}' \
        -D - -o /dev/null \
        "https://storage.googleapis.com/upload/storage/v1/b/$BUCKET/o?uploadType=resumable&name=$PREFIX/$obj" \
        | tr -d '\r' | awk '/^[Ll]ocation: /{print $2}')
    [ -n "$sess" ] || { echo "FAILED to open upload session for $obj (token expired?)" >&2; return 1; }

    curl -sS -X PUT -H "Content-Length: $sz" --data-binary "@$f" "$sess" \
        | python3 -c "import json,sys; d=json.load(sys.stdin); print('   ok', d.get('name'), d.get('size'), 'md5', d.get('md5Hash'))"
}

cd "$SRC"
# Manifest first: the activations are meaningless without the read convention that produced them.
python3 - "$PREFIX" <<'PY' > MANIFEST.json
import hashlib, json, os, sys
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()
files = {f: {"bytes": os.path.getsize(f), "sha256": sha(f)}
         for f in sorted(os.listdir(".")) if f.endswith((".npz", ".json")) and f != "MANIFEST.json"}
json.dump({
    "prefix": sys.argv[1],
    "model": "Qwen/Qwen3.5-27B",
    "dataset": "AgentHarm test_public (176 harmful + 176 benign) / val (32+32)",
    "read_position": "post-instruction token (-1), IN-CONTEXT: read inside the Inspect scorer from "
                     "the model's real agentic context (AgentHarm agent system prompt + tools)",
    "layer_index": "0-based over transformer blocks; embedding layer skipped (hidden_states[l+1])",
    "shape": "X (N, L=64, H=5120) float32; y (N,) 1=harmful 0=benign; ids (N,)",
    "enable_thinking": False,
    "dtype": "bf16 model, float32 activations",
    "produced_by": "experiments/results/06-qwen35-inscorer-probe/qwen35_inscorer_experiment.py (Pass B)",
    "verify_with": "experiments/verify_test_acts_cache.py --acts qwen35_inscorer_test_acts.npz",
    "warning": "NOT interchangeable with 05's qwen35_test_acts.npz, which is from the INVALID "
               "offline path (missing agent system prompt).",
    "files": files,
}, sys.stdout, indent=2)
PY

upload MANIFEST.json                     MANIFEST.json
upload qwen35_inscorer_test_acts.npz     test_acts.npz
upload qwen35_inscorer_val_acts.npz      val_acts.npz
upload qwen35_inscorer_experiment.json   inscorer_experiment.json
echo "done → gs://$BUCKET/$PREFIX/"
