"""Verify Qwen3.5-27B download completeness + read config (layers/hidden for probe sweep)."""
import glob
import json
import os

D = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B")
snap = sorted(glob.glob(D + "/snapshots/*/"))[-1]
cfg = json.load(open(snap + "config.json"))
print("arch:", cfg.get("architectures"), "layers:", cfg.get("num_hidden_layers"),
      "hidden:", cfg.get("hidden_size"))

idx = glob.glob(snap + "model.safetensors.index.json")
if idx:
    wm = json.load(open(idx[0]))["weight_map"]
    shards = sorted(set(wm.values()))
    missing = [s for s in shards if not os.path.exists(snap + s)]
    print("shards:", len(shards), "missing:", missing or "none")
else:
    print("safetensors:", [os.path.basename(f) for f in glob.glob(snap + "*.safetensors")])

incomplete = glob.glob(D + "/**/*.incomplete", recursive=True)
print("incomplete files:", len(incomplete))
print("downloader running:", bool(os.popen('pgrep -f "hf download Qwen/Qwen3.5-27B"').read().strip()))
