"""M0b: confirm the AgentHarm dataset loads (no gating) and the chat_dataset toggle
swaps agentic vs chat versions of the same behaviors. No model/judge calls."""
from inspect_evals.agentharm.agentharm import agentharm


def summarize(chat):
    t = agentharm(split="test_public", chat_dataset=chat)
    ds = t.dataset
    n = len(ds)
    s0 = ds[0]
    tools = s0.metadata.get("target_functions") if s0.metadata else None
    print(f"\n=== chat_dataset={chat} ===")
    print("samples:", n)
    print("sample0 id:", getattr(s0, "id", None))
    print("sample0 input (first 120):", repr(str(s0.input)[:120]))
    print("sample0 target_functions:", tools)
    md = s0.metadata or {}
    print("sample0 metadata keys:", list(md.keys()))
    print("sample0 category:", md.get("category"))


if __name__ == "__main__":
    for chat in (False, True):
        try:
            summarize(chat)
        except Exception as e:
            print(f"chat_dataset={chat} FAILED:", repr(e))
