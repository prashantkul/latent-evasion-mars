"""Canonical probe artifact schema for re-evaluation.

One artifact = one probe type, keyed exactly like the training source of truth
(`classifier/train_latent.py` -> `utils/probes.py`):

    score = x @ w + b        fires when score > 0

    <stem>.npz   w (L, H) float32 | b (L,) float32 | layers (L,) int32
    <stem>.json  metadata, REQUIRED (see REQUIRED_META)

Why the metadata is required: a probe with the wrong `read_position` or `layer_index`
convention produces plausible-looking numbers rather than an error. Validating it on
load is the only thing standing between a teammate's handover and a silent garbage run.

Naming note: earlier artifacts spelled the bias three different ways -- `svm_b`,
`svm_bias`, `svm_b_bias` -- and `svm_b` collided with "SVM scores on benign samples"
in the scores npz. Canonical is `w`/`b`, matching `train_latent.py` and `utils.probes`.

Run directly to migrate the Qwen3.5 in-context probes from `06` into canonical form:
    uv run python experiments/probe_io.py
"""
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

PROBE_TYPES = ("svm", "single_direction")

REQUIRED_META = (
    "probe_type",      # svm | single_direction
    "model",           # HF id the activations came from
    "hidden",          # hidden dim, cross-checked against w
    "read_position",   # where in the sequence the activation is read
    "layer_index",     # indexing convention (the classic silent-mismatch footgun)
    "score",           # scoring + firing convention
    "trained_on",      # split the probe was fit on
)


def save_probe(stem: str, w: np.ndarray, b: np.ndarray, meta: Dict,
               layers: Optional[np.ndarray] = None) -> Tuple[str, str]:
    """Write <stem>.npz + <stem>.json in canonical form. Validates before writing."""
    w = np.asarray(w, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    layers = np.arange(len(b), dtype=np.int32) if layers is None else np.asarray(layers, dtype=np.int32)
    _validate(w, b, layers, meta, stem)

    os.makedirs(os.path.dirname(os.path.abspath(stem)) or ".", exist_ok=True)
    np.savez_compressed(f"{stem}.npz", w=w, b=b, layers=layers)
    json.dump(meta, open(f"{stem}.json", "w"), indent=2)
    return f"{stem}.npz", f"{stem}.json"


def load_probe(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Load a canonical probe. -> (w, b, layers, meta). Raises on any schema violation."""
    stem = path[:-4] if path.endswith((".npz", ".json")) else path
    npz_path, json_path = f"{stem}.npz", f"{stem}.json"
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"probe weights not found: {npz_path}")
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"probe metadata sidecar not found: {json_path}\n"
            "The sidecar is required -- without it we cannot check that the probe's "
            "read position and layer indexing match the evaluation."
        )

    z = np.load(npz_path)
    missing = [k for k in ("w", "b", "layers") if k not in z.files]
    if missing:
        raise ValueError(f"{npz_path}: missing canonical key(s) {missing}; found {z.files}. "
                         "Legacy artifacts use svm_w/svm_b/svm_bias/dirs -- migrate them first.")
    w, b, layers = z["w"].astype(np.float32), z["b"].astype(np.float32), z["layers"].astype(np.int32)
    meta = json.load(open(json_path))
    _validate(w, b, layers, meta, stem)
    return w, b, layers, meta


def score(acts: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """acts (N, L, H) -> scores (N, L). Fires when > 0."""
    if acts.ndim != 3:
        raise ValueError(f"expected activations (N, L, H), got {acts.shape}")
    if acts.shape[1] != w.shape[0] or acts.shape[2] != w.shape[1]:
        raise ValueError(f"activation/probe mismatch: acts {acts.shape} vs w {w.shape}")
    return np.einsum("nlh,lh->nl", acts.astype(np.float32), w) + b


def _validate(w, b, layers, meta, stem) -> None:
    if w.ndim != 2:
        raise ValueError(f"{stem}: w must be (L, H), got {w.shape}")
    if b.shape[0] != w.shape[0] or layers.shape[0] != w.shape[0]:
        raise ValueError(f"{stem}: length mismatch w {w.shape[0]} / b {b.shape[0]} / layers {layers.shape[0]}")
    if not np.isfinite(w).all() or not np.isfinite(b).all():
        raise ValueError(f"{stem}: non-finite values in probe weights")
    if len(set(layers.tolist())) != len(layers):
        raise ValueError(f"{stem}: duplicate layer indices")

    absent = [k for k in REQUIRED_META if k not in meta]
    if absent:
        raise ValueError(f"{stem}: metadata missing required key(s) {absent}")
    if meta["probe_type"] not in PROBE_TYPES:
        raise ValueError(f"{stem}: probe_type must be one of {PROBE_TYPES}, got {meta['probe_type']!r}")
    if int(meta["hidden"]) != w.shape[1]:
        raise ValueError(f"{stem}: metadata hidden={meta['hidden']} but w has {w.shape[1]} dims")


# --------------------------------------------------------------------------------------
# Migration: the valid Qwen3.5 in-context probes (06) -> canonical form.
#
# Source of truth is `qwen35_inscorer_experiment_scores.npz`, which embeds the exact
# probe used inside the Inspect scorer. NOT `06/probe/`, which is a byte-identical copy
# of 05's probe and is therefore INVALID (05 omitted the agent system prompt).
# --------------------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
R06 = os.path.join(HERE, "results", "06-qwen35-inscorer-probe")

_BASE_META = {
    "model": "Qwen/Qwen3.5-27B",
    "hidden": 5120,
    "read_position": "post-instruction token (-1), in-context: read inside the Inspect scorer "
                     "from the model's real agentic context (AgentHarm agent system prompt + tools)",
    "layer_index": "0-based over transformer blocks; embedding layer skipped (hidden_states[l+1])",
    "score": "x @ w + b; fires when > 0",
    "trained_on": "AgentHarm val split, 32 harmful (1) / 32 benign (0), in-context",
    "enable_thinking": False,
    "dtype": "bf16 model, float32 probe",
    "provenance": "experiments/results/06-qwen35-inscorer-probe/qwen35_inscorer_experiment.py (Pass A)",
}


def migrate_06(out_dir: Optional[str] = None, scores: Optional[str] = None,
               val_acts: Optional[str] = None, extra_meta: Optional[Dict] = None) -> List[str]:
    """Canonicalize an in-scorer run's probe. Defaults to the committed 06 artifacts; pass
    `scores`/`val_acts` to canonicalize a fresh run (e.g. a re-baselined activation cache)."""
    out_dir = out_dir or os.path.join(R06, "probe_canonical")
    z = np.load(scores or os.path.join(R06, "qwen35_inscorer_experiment_scores.npz"), allow_pickle=True)
    v = np.load(val_acts or os.path.join(R06, "qwen35_inscorer_val_acts.npz"), allow_pickle=True)
    Xv, yv = v["X"], v["y"]
    base = {**_BASE_META, **(extra_meta or {})}

    written = []
    stem = os.path.join(out_dir, "qwen35_svm")
    written += list(save_probe(stem, z["svm_w"], z["svm_bias"],
                               {**base, "probe_type": "svm",
                                "fit": "LinearSVC C=0.1, StandardScaler folded into raw-space (w, b)"}))

    # Mean-difference: 06 stored only the unit direction and applied the threshold ad hoc
    # downstream. Canonical form carries the bias, matching load_single_direction_probes:
    #   b = -0.5 * w . (mu_harmful + mu_harmless)   for unit w
    dirs = z["dirs"].astype(np.float32)
    mu1 = Xv[yv == 1].mean(0).astype(np.float32)      # (L, H)
    mu0 = Xv[yv == 0].mean(0).astype(np.float32)
    sd_b = -0.5 * np.einsum("lh,lh->l", dirs, mu1 + mu0)
    stem = os.path.join(out_dir, "qwen35_single_direction")
    written += list(save_probe(stem, dirs, sd_b,
                               {**base, "probe_type": "single_direction",
                                "fit": "unit-norm centroid difference; bias = midpoint between val centroids"}))
    return written


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Canonicalize an in-scorer run's probe artifacts.")
    ap.add_argument("--scores", help="<run>_scores.npz (svm_w / svm_bias / dirs). Default: committed 06.")
    ap.add_argument("--val-acts", help="val activations npz the probe was fit on. Default: committed 06.")
    ap.add_argument("--out-dir", help="destination directory for the canonical artifacts")
    ap.add_argument("--note", help="free-text provenance appended to the metadata")
    a = ap.parse_args()
    extra = {"provenance_note": a.note} if a.note else None
    for f in migrate_06(a.out_dir, a.scores, a.val_acts, extra):
        print("wrote", f)
