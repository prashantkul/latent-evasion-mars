"""M0a: confirm Inspect's hf provider returns per-layer hidden states for our model,
and that the shape aligns with the trained Llama-3.2-3B probe (28 layers x 3072)."""
import asyncio
import os

from inspect_ai.model import get_model


async def main():
    model_id = os.environ.get("LLAMA32_3B_MODEL", "unsloth/Llama-3.2-3B-Instruct")
    model = get_model(f"hf/{model_id}", hidden_states=True, device="cuda:0")

    out = await model.generate(input="Write a short poem about the sea.")
    print("completion (first 80 chars):", repr(out.completion[:80]))

    meta = out.metadata or {}
    print("metadata keys:", list(meta.keys()))
    hs = meta.get("hidden_states")
    if hs is None:
        print("NO hidden_states in metadata")
        return

    # transformers GenerateDecoderOnlyOutput.hidden_states:
    # tuple over generated steps; each step is a tuple over layers of (batch, seq, dim)
    print("type(hidden_states):", type(hs).__name__)
    try:
        print("num generation steps:", len(hs))
        step0 = hs[0]
        print("num layer-tensors in step0 (incl. embedding):", len(step0))
        import torch
        t = step0[-1]
        print("last-layer tensor type/shape:", type(t).__name__,
              tuple(t.shape) if hasattr(t, "shape") else "n/a")
        # dim check against probe
        dim = t.shape[-1]
        print(f"hidden dim = {dim}  (probe expects 3072: {'OK' if dim == 3072 else 'MISMATCH'})")
        print(f"layer count (excl embedding) = {len(step0) - 1}  "
              f"(probe expects 28: {'OK' if len(step0) - 1 == 28 else 'MISMATCH'})")
    except Exception as e:
        print("inspection error:", repr(e))


if __name__ == "__main__":
    asyncio.run(main())
