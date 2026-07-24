"""Per-token x per-layer probe scores for one harmful and one harmless agentic prompt.

Produces the data behind the tokens-by-layers heat map. GPU, but tiny: one forward pass per
prompt, because hidden_states[0][layer] already has shape [batch, seq_len, hidden] -- EVERY
position is there. The rest of the repo reads [0, -1, :] and discards the sequence axis; here we
keep it.

Run it twice via --beta to get the pre- and post-steering grids:
    --beta 0   no-op hooks, the untouched model
    --beta 1   CLE-P active on --layers, so the recorded hidden states are POST-intervention
               (a forward hook that returns a value replaces the layer output, and transformers
               collects hidden_states after the hook)

WHY IT GOES THROUGH INSPECT rather than tokenizing a prompt ourselves: the tokens must be exactly
the ones the agent is driven with -- AgentHarm's 558-char system prompt plus rendered tool schemas.
Re-rendering that context offline is precisely what invalidated experiment 05. We capture the real
input_ids with a forward pre-hook on the model and read hidden states in the scorer, so both come
from the same forward pass.

  python3 experiments/extract_token_scores.py --probe <stem> --beta 0 --out tok_beta0.npz
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "results", "06-qwen35-inscorer-probe"))

import qwen35_inspect_patch                                        # noqa: E402
from agentic_cle import geometric_margins, hf_layers, parse_layers  # noqa: E402
from probe_adapter import to_probe_dict                            # noqa: E402
from probe_io import load_probe                                    # noqa: E402
from utils.hooks import projection_hook, remove_hooks              # noqa: E402

from inspect_ai import Task, eval as inspect_eval                  # noqa: E402
from inspect_ai.model import GenerateConfig, get_model             # noqa: E402
from inspect_ai.scorer import Score, Target, mean, scorer          # noqa: E402
from inspect_ai.solver import TaskState                            # noqa: E402
from inspect_evals.agentharm.agents.agent import get_agent         # noqa: E402
from inspect_evals.agentharm.utils import (                        # noqa: E402
    filter_dataset, get_benchmark_tools, load_dataset, setup_tools_from_metadata,
)

GRID = {}      # tag -> (L, T) scores
IDS = {}       # tag -> input_ids of the captured forward


def capture_ids_hook(store):
    """The exact token ids the model is driven with -- not a re-render of the chat template."""
    def hook(module, args, kwargs):
        ids = kwargs.get("input_ids")
        if ids is None and args:
            ids = args[0]
        if ids is not None and ids.ndim == 2 and ids.shape[1] > 1:   # prefill, not a decode step
            store["ids"] = ids[0].detach().cpu().numpy()
    return hook


@scorer(metrics=[mean()])
def grid_scorer(tag, w, b):
    """Score EVERY position at EVERY layer, instead of the usual post-instruction token only."""
    async def score(state: TaskState, target: Target) -> Score:
        hs = ((state.output.metadata or {}) if state.output else {}).get("hidden_states")
        if hs is None:
            return Score(value=0.0, metadata={"error": "no hidden_states"})
        step0 = hs[0]
        # step0[l] is [batch, seq_len, hidden]; skip the embedding layer as everywhere else.
        H = np.stack([step0[l][0].float().cpu().numpy() for l in range(1, len(step0))])  # (L,T,H)
        GRID[tag] = np.einsum("lth,lh->lt", H, w) + b[:, None]                           # (L,T)
        return Score(value=0.0)
    return score


def task(name, split, scorer_fn, limit_id=None):
    ds = filter_dataset(load_dataset(name, split, False), [], None, None)
    tools = get_benchmark_tools(name)
    plan = [setup_tools_from_metadata(name, tools, 0), get_agent(agent="default", tool_choice="none")]
    return Task(dataset=ds, solver=plan, scorer=scorer_fn,
                config=GenerateConfig(max_tokens=1, max_connections=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("QWEN35_27B_MODEL", "Qwen/Qwen3.5-27B"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--probe", required=True)
    ap.add_argument("--layers", default="20-30")
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--margin-quantile", type=float, default=0.5)
    ap.add_argument("--val-acts", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--log-dir", default="./logs_token_scores")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    w, b, _idx, _meta = load_probe(args.probe)
    qwen35_inspect_patch.install()
    model = get_model(f"hf/{args.model}", hidden_states=True, device=args.device,
                      enable_thinking=False, do_sample=False)
    layers_mod = hf_layers(model)
    sel = parse_layers(args.layers, len(layers_mod))
    margins = geometric_margins(args.val_acts, args.probe, sel, args.margin_quantile)

    store = {}
    base = getattr(model.api, "model", None)
    id_handle = base.register_forward_pre_hook(capture_ids_hook(store), with_kwargs=True)
    probes = to_probe_dict(args.probe, sel, device=args.device)
    hooks = [layers_mod[l].register_forward_hook(
        projection_hook(probes[l]["w"], probes[l]["b"], args.beta, margins[l])) for l in sel]

    try:
        with torch.no_grad():
            for tag, name in (("harmful", "harmful"), ("harmless", "benign")):
                inspect_eval(task(name, args.split, grid_scorer(tag, w, b)),
                             model=model, log_dir=f"{args.log_dir}/{tag}", limit=1)
                IDS[tag] = store.get("ids")
                print(f"  {tag}: grid {GRID[tag].shape}  ids {None if IDS[tag] is None else IDS[tag].shape}",
                      flush=True)
    finally:
        remove_hooks(hooks); id_handle.remove()

    tok = model.api.tokenizer
    out = {"beta": args.beta, "layers": np.array(sel)}
    for tag in ("harmful", "harmless"):
        out[f"{tag}_scores"] = GRID[tag].astype(np.float32)
        ids = IDS[tag]
        out[f"{tag}_tokens"] = np.array(
            tok.convert_ids_to_tokens(ids.tolist()) if ids is not None else [], dtype=object)
    np.savez(args.out, **out, allow_pickle=True)
    print(f"saved {args.out}", flush=True)
    for tag in ("harmful", "harmless"):
        n_tok = out[f"{tag}_scores"].shape[1]
        if len(out[f"{tag}_tokens"]) != n_tok:
            print(f"  WARNING {tag}: {len(out[f'{tag}_tokens'])} token strings for {n_tok} "
                  "score columns — the id capture did not line up with the scored forward")


if __name__ == "__main__":
    main()
