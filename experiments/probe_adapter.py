"""Bridge between the canonical eval artifact and CLE's per-layer checkpoints.

Closes the format gap so one probe can be evaluated by `eval_probe.py` and then handed to
`cle-a.py` / `cle-p.py` unchanged. Key names already agree (`w`, `b`) -- this is a container
shim, not a semantic conversion.

Three directions:
  to_probe_dict  canonical -> in-memory ProbeDict, what utils/hooks.py wants (no files)
  export_cle     canonical -> svm_layerXX.pt files that utils/probes.py loads
  import_cle     CLE checkpoint dir -> canonical (w, b) + metadata sidecar

WHY EXPORT ALWAYS USES THE `svm` CONTAINER, even for a mean-difference probe:
`load_svms` reads `w` and `b` straight from the file, so (w, b) round-trip exactly.
`load_single_direction_probes` deliberately IGNORES any stored bias and RECONSTRUCTS it at load
time from class centroids in `HFx_train.pt` / `HLx_train.pt`. Writing an sd_layerXX.pt without
shipping matching centroid files would silently produce a different boundary than the one we
evaluated. So we export the container that preserves the numbers and record the probe's real
type in the metadata. Run CLE with `--probe_type svm` regardless of how the probe was fit.

    uv run python experiments/probe_adapter.py export --probe <stem> --out-dir <cle dir>
    uv run python experiments/probe_adapter.py import --svm-dir <cle dir> --out <stem> \
        --model <hf id> --read-position "..." --trained-on "..."
"""
import argparse
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_io import load_probe, save_probe


def to_probe_dict(stem: str, layers: Optional[List[int]] = None,
                  device: str = "cpu") -> Dict[int, Dict[str, torch.Tensor]]:
    """Canonical probe -> the ProbeDict shape utils/hooks.py expects: {layer: {w, b}}.

    In-memory, so an Inspect run can install CLE hooks without writing checkpoints to disk.
    """
    w, b, idx, _ = load_probe(stem)
    want = set(layers) if layers is not None else set(int(i) for i in idx)
    pos = {int(l): n for n, l in enumerate(idx)}
    missing = sorted(want - set(pos))
    if missing:
        raise ValueError(f"{stem}: probe has no weights for layer(s) {missing}")
    return {l: {"w": torch.from_numpy(w[pos[l]]).float().to(device),
                "b": torch.tensor(float(b[pos[l]])).float().to(device)}
            for l in sorted(want)}


def export_cle(stem: str, out_dir: str, layers: Optional[List[int]] = None) -> List[str]:
    """Canonical probe -> svm_layerXX.pt checkpoints loadable by utils/probes.load_svms."""
    w, b, idx, meta = load_probe(stem)
    os.makedirs(out_dir, exist_ok=True)
    pos = {int(l): n for n, l in enumerate(idx)}
    sel = sorted(set(layers) if layers is not None else set(pos))
    written = []
    for l in sel:
        n = pos[l]
        path = os.path.join(out_dir, f"svm_layer{l:02d}.pt")
        torch.save({"w": torch.from_numpy(w[n]).float(),
                    "b": torch.tensor(float(b[n])).float(),
                    "layer_idx": l,
                    "model_name": meta.get("model"),
                    "hidden_dim": int(w.shape[1]),
                    # The probe's real type, so an sd probe exported in the svm container is not
                    # mistaken for an SVC later.
                    "probe_type_actual": meta.get("probe_type"),
                    "read_position": meta.get("read_position"),
                    "layer_index_convention": meta.get("layer_index"),
                    "source": os.path.basename(stem)}, path)
        written.append(path)
    return written


def import_cle(svm_dir: str, out_stem: str, meta: Dict, probe_type: str = "svm",
               layers: Optional[List[int]] = None) -> List[str]:
    """CLE svm_layerXX.pt checkpoints -> a canonical artifact eval_probe.py can score.

    single_direction checkpoints are not supported here on purpose: their bias lives in the
    centroid files, not the checkpoint, so importing one without those files would invent a
    boundary. Convert those with probe_io.migrate_06 from the run that produced them.
    """
    if probe_type != "svm":
        raise ValueError("only the svm container round-trips (w, b); see the module docstring")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from utils.probes import discover_available_layers

    sel = layers if layers is not None else discover_available_layers(svm_dir, "svm")
    ws, bs = [], []
    for l in sel:
        obj = torch.load(os.path.join(svm_dir, f"svm_layer{l:02d}.pt"), map_location="cpu")
        if not (isinstance(obj, dict) and "w" in obj):
            raise ValueError(f"unexpected checkpoint format at layer {l}")
        ws.append(obj["w"].float().view(-1).numpy())
        bs.append(float(torch.as_tensor(obj.get("b", 0.0)).view(())))
    w = np.stack(ws)
    meta = {**meta, "hidden": int(w.shape[1]),
            "probe_type": meta.get("probe_type", "svm"),
            "score": meta.get("score", "x @ w + b; fires when > 0")}
    return list(save_probe(out_stem, w, np.array(bs), meta, layers=np.array(sel)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("export", help="canonical -> CLE svm_layerXX.pt")
    e.add_argument("--probe", required=True)
    e.add_argument("--out-dir", required=True)
    e.add_argument("--layers", help="comma-separated subset, e.g. 22,24,26")

    i = sub.add_parser("import", help="CLE svm_layerXX.pt -> canonical")
    i.add_argument("--svm-dir", required=True)
    i.add_argument("--out", required=True, help="canonical stem to write")
    i.add_argument("--model", required=True)
    i.add_argument("--read-position", required=True)
    i.add_argument("--trained-on", required=True)
    i.add_argument("--layer-index", default="0-based over transformer blocks; embedding layer skipped")

    a = ap.parse_args()
    if a.cmd == "export":
        layers = [int(x) for x in a.layers.split(",")] if a.layers else None
        for p in export_cle(a.probe, a.out_dir, layers):
            print("wrote", p)
        print("\nRun CLE with --probe_type svm (the container preserves (w, b) exactly).")
    else:
        for p in import_cle(a.svm_dir, a.out, {
                "model": a.model, "read_position": a.read_position,
                "trained_on": a.trained_on, "layer_index": a.layer_index}):
            print("wrote", p)


if __name__ == "__main__":
    main()
