# INVALID — do not use for evaluation

These files are a **byte-identical copy of the probe from `05-qwen35-agentic-dissociation`**
(verified by md5 against `05/qwen35_svm_probe_scores.npz`). `05` is INVALID: its activations were
reconstructed offline and omitted AgentHarm's agent system prompt, so they are not the model's real
agentic context.

The sidecar here reads convincingly (`best_layer: 54`) and sat inside the *valid* experiment's
folder, which made it an easy thing to pick up by mistake. Kept for provenance only.

**Use instead:** `../probe_canonical/` — the probe actually used inside the Inspect scorer in `06`,
migrated to the canonical schema by `experiments/probe_io.py`.
