"""Smoke test: Gemma-4-31B-it QAT w4a16 on the GB10 Spark.
Validates: loads, memory fit, text-backbone path, hidden_states shape, tok/s."""
import time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "google/gemma-4-31B-it-qat-w4a16-ct"

t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, device_map="cuda:0", dtype=torch.bfloat16,
)
model.eval()
print(f"[load] {time.time()-t0:.1f}s")
print(f"[mem] allocated {torch.cuda.memory_allocated()/1e9:.1f}GB  "
      f"reserved {torch.cuda.memory_reserved()/1e9:.1f}GB")

# discover structure: where is the text backbone / decoder layers?
top = [n for n, _ in model.named_children()]
print(f"[struct] top-level children: {top}")
cfg = model.config
tcfg = getattr(cfg, "text_config", None)
n_layers = getattr(cfg, "num_hidden_layers", None) or getattr(tcfg, "num_hidden_layers", None)
h_size = getattr(cfg, "hidden_size", None) or getattr(tcfg, "hidden_size", None)
print(f"[cfg] num_hidden_layers={n_layers} hidden_size={h_size} "
      f"arch={cfg.architectures}")
# find the .layers ModuleList (the decoder stack)
for name, mod in model.named_modules():
    if name.endswith("layers") and hasattr(mod, "__len__") and len(mod) == (n_layers or -1):
        print(f"[backbone] decoder stack at: model.{name}  ({len(mod)} layers)")
        break

msgs = [{"role": "user", "content": "Book a table for two at 7pm and email me the confirmation."}]
enc = tok.apply_chat_template(
    msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
).to("cuda:0")
ids = enc["input_ids"]

with torch.no_grad():
    out = model(**enc, output_hidden_states=True)
hs = out.hidden_states
mid = len(hs) // 2
print(f"[hidden_states] tuple_len={len(hs)} (embed + {len(hs)-1} layers)  "
      f"per-layer shape={tuple(hs[mid].shape)}")
last = hs[mid][0, -1, :]
print(f"[act] mid-layer last-token vec: dim={last.shape[0]} "
      f"norm={last.float().norm().item():.2f}")

N = 64
t0 = time.time()
with torch.no_grad():
    gen = model.generate(**enc, max_new_tokens=N, do_sample=False)
dt = time.time() - t0
n_new = gen.shape[1] - ids.shape[1]
print(f"[gen] {n_new} tok in {dt:.1f}s = {n_new/dt:.2f} tok/s")
print("[decode]", tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)[:200].replace("\n", " "))
print("OK")
