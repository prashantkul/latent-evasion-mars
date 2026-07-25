"""The chat-vs-agentic ablation: same harmful behaviours, with tools and without.

AgentHarm ships both framings of the *same* 44 behaviours. `chat_dataset=True` gives 44 plain chat
prompts with no tools; `chat_dataset=False` gives 176 agentic samples -- the same 44 behaviours,
each augmented 4 ways, with tools in `target_functions`. Holding the request fixed and varying only
the framing isolates how much of the refusal rate is about the *agentic setting* rather than the
content of the ask.

No probes, no steering, no intervention. This is a behavioural baseline.

THE JOIN IS THE WHOLE DESIGN. Agentic is 4x augmented and chat is base-only, so comparing 44 rows
against 176 rows as two independent rates confounds the framing with whatever the augmentation did.
The samples carry `id_original`, so we aggregate agentic to the behaviour level and compare
**paired over the 44 behaviours**. `id` will not do -- see the M0 gotcha in
experiments/agentharm-refusal-tracker.md.

Note `chat_dataset` exists only on `agentharm()`, not `agentharm_benign()`, so this ablation is
harmful-only. There is no benign chat arm to run.

  # on the pod: run the chat arm (44 samples, no tools -- fast)
  python3 experiments/chat_vs_agentic_ablation.py run --log-dir /workspace/ablation

  # anywhere: pair it against an already-graded agentic run
  python3 experiments/chat_vs_agentic_ablation.py join \
      --chat '<chat>/*.eval' --agentic '<agentic>/*.eval' --out ablation.json
"""
import argparse
import glob
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "results", "06-qwen35-inscorer-probe"))


def rows_from_log(pattern):
    """Per-sample (behaviour, refusal, score) from a graded AgentHarm .eval.

    The combined scorer emits value={"score": float, "refusal": 1.0|0.0}; the behaviour key is
    metadata["id_original"], which is what makes chat and agentic comparable.
    """
    from inspect_ai.log import read_eval_log

    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no .eval matched {pattern}")
    out = []
    for p in paths:
        log = read_eval_log(p)
        if log.status != "success":
            print(f"  skipping {os.path.basename(p)} (status={log.status})", file=sys.stderr)
            continue
        for s in (log.samples or []):
            md = s.metadata or {}
            key = md.get("id_original") or md.get("name")
            val = None
            for sc in (s.scores or {}).values():
                if isinstance(getattr(sc, "value", None), dict) and "refusal" in sc.value:
                    val = sc.value
                    break
            if key is None or val is None:
                continue
            out.append({"behaviour": str(key), "category": md.get("category"),
                        "refusal": float(val["refusal"]), "score": float(val.get("score", 0.0))})
    if not out:
        raise SystemExit(f"{pattern} produced no scored samples — was it graded?")
    return out


def cmd_run(args):
    import qwen35_inspect_patch                                       # noqa: E402
    from inspect_ai import eval as inspect_eval                       # noqa: E402
    from inspect_ai.model import GenerateConfig, get_model            # noqa: E402
    from inspect_evals.agentharm.agentharm import agentharm           # noqa: E402

    qwen35_inspect_patch.install()
    # Greedy, for the same reason every other arm is greedy: Inspect's hf provider defaults
    # do_sample to True, and a sampled run is not comparable to the frozen baseline.
    model = get_model(f"hf/{args.model}", device=args.device, enable_thinking=False,
                      do_sample=False)
    cfg = GenerateConfig(max_tokens=args.max_tokens, max_connections=1)

    for chat in ([True, False] if args.with_agentic else [True]):
        tag = "chat" if chat else "agentic"
        task = agentharm(split=args.split, chat_dataset=chat)
        task.config = cfg
        print(f"\n>>> {tag} (chat_dataset={chat}) split={args.split}", flush=True)
        log = inspect_eval(task, model=model, log_dir=f"{args.log_dir}/{tag}",
                           limit=args.limit or None)[0]
        print(f"=== {tag}: {log.status} ===", flush=True)
        if log.status == "success":
            m = {k: float(v.value)
                 for s in (log.results.scores if log.results else [])
                 for k, v in (s.metrics or {}).items()
                 if k in ("avg_score", "avg_refusals", "avg_score_non_refusals")}
            print("  " + "  ".join(f"{k} {v:.4f}" for k, v in m.items()), flush=True)


def cmd_join(args):
    chat = rows_from_log(args.chat)
    agentic = rows_from_log(args.agentic)

    def by_behaviour(rows):
        agg = {}
        for r in rows:
            agg.setdefault(r["behaviour"], []).append(r)
        return agg

    c_by, a_by = by_behaviour(chat), by_behaviour(agentic)
    shared = sorted(set(c_by) & set(a_by))
    only_c, only_a = sorted(set(c_by) - set(a_by)), sorted(set(a_by) - set(c_by))

    print(f"chat rows {len(chat)} over {len(c_by)} behaviours | "
          f"agentic rows {len(agentic)} over {len(a_by)} behaviours")
    print(f"paired on {len(shared)} behaviours"
          + (f" | chat-only {len(only_c)} | agentic-only {len(only_a)}" if only_c or only_a else ""))
    if not shared:
        raise SystemExit("no shared behaviours — check the join key")

    # Paired at the behaviour level. Agentic contributes the mean over its augmentations, so each
    # behaviour carries equal weight in both arms regardless of how many variants it has.
    pairs = []
    for b in shared:
        cr = statistics.fmean(r["refusal"] for r in c_by[b])
        ar = statistics.fmean(r["refusal"] for r in a_by[b])
        pairs.append({"behaviour": b, "category": c_by[b][0]["category"],
                      "chat_refusal": cr, "agentic_refusal": ar, "delta": ar - cr})

    chat_mean = statistics.fmean(p["chat_refusal"] for p in pairs)
    agen_mean = statistics.fmean(p["agentic_refusal"] for p in pairs)
    deltas = [p["delta"] for p in pairs]
    mean_delta = statistics.fmean(deltas)

    # Bootstrap over behaviours -- the unit of resampling is the behaviour, since the 4 agentic
    # augmentations of one behaviour are not independent observations.
    lo = hi = None
    try:
        import random
        rng = random.Random(0)
        boots = []
        for _ in range(10000):
            samp = [deltas[rng.randrange(len(deltas))] for _ in deltas]
            boots.append(statistics.fmean(samp))
        boots.sort()
        lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]
    except Exception as e:                                    # never let CI failure lose the point
        print(f"  (bootstrap skipped: {e})", file=sys.stderr)

    # Discordant behaviours: refused in one framing, not the other (agentic by majority of its 4).
    b_ca = sum(1 for p in pairs if p["chat_refusal"] > 0.5 >= p["agentic_refusal"])
    b_ac = sum(1 for p in pairs if p["agentic_refusal"] > 0.5 >= p["chat_refusal"])

    print(f"\n=== chat vs agentic — refusal, paired over {len(pairs)} behaviours ===")
    print(f"  chat     {chat_mean:.4f}")
    print(f"  agentic  {agen_mean:.4f}")
    ci = f"  95% CI [{lo:+.4f}, {hi:+.4f}]" if lo is not None else ""
    print(f"  agentic - chat  {mean_delta:+.4f}{ci}")
    print(f"  discordant: refused in chat only {b_ca}  |  refused agentically only {b_ac}")

    cats = {}
    for p in pairs:
        cats.setdefault(p["category"] or "?", []).append(p)
    print("\n  by category:")
    for c, ps in sorted(cats.items()):
        print(f"    {c:16s} n={len(ps):3d}  chat {statistics.fmean(x['chat_refusal'] for x in ps):.3f}"
              f"  agentic {statistics.fmean(x['agentic_refusal'] for x in ps):.3f}")

    res = {"n_behaviours": len(pairs), "chat_refusal": chat_mean, "agentic_refusal": agen_mean,
           "delta_agentic_minus_chat": mean_delta, "ci95": [lo, hi],
           "discordant_chat_only": b_ca, "discordant_agentic_only": b_ac,
           "unmatched_chat": only_c, "unmatched_agentic": only_a, "pairs": pairs}
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)
        print(f"\nsaved {args.out}")
    return res


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the chat arm (and optionally the agentic arm)")
    r.add_argument("--model", default=os.environ.get("QWEN35_27B_MODEL", "Qwen/Qwen3.5-27B"))
    r.add_argument("--device", default="cuda:0")
    r.add_argument("--split", default="test_public", choices=["val", "test_public"])
    r.add_argument("--max-tokens", type=int, default=4096)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--log-dir", default="./logs_ablation")
    r.add_argument("--with-agentic", action="store_true",
                   help="also re-run the agentic arm; omit to pair against an existing graded run")
    r.set_defaults(fn=cmd_run)

    j = sub.add_parser("join", help="pair a chat run against an agentic run by behaviour")
    j.add_argument("--chat", required=True, help="glob for the chat .eval")
    j.add_argument("--agentic", required=True, help="glob for the agentic .eval")
    j.add_argument("--out", default=None)
    j.set_defaults(fn=cmd_join)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
