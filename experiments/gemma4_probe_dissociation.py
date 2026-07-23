"""Gemma-4-31B refusal probe + agentic dissociation (standalone hf, no Inspect solver).

Why standalone: Inspect's hf provider renders tools in a flat schema that Gemma-4's chat
template rejects, so we extract activations directly through transformers (we control the
tokenization). Runs on the hf path (vLLM does not expose hidden_states) and must run AFTER
the vLLM baseline is stopped (they can't share the GPU).

Pipeline:
  1. Probe direction: fit per-layer mean-difference + LinearSVC on the val contrast
     (32 harmful=1 / 32 benign=0), last-token residual stream, chat-mode rendering.
  2. Behavioral outcomes: read the vLLM harmful baseline .eval log -> per-sample refusal
     + harm score.
  3. Dissociation: at the best layer (by held-out harmful/benign AUC), for test_public
     harmful samples compute P(fire | complied) vs P(fire | refused), refusal-implies-fire,
     and AUC(probe score -> refusal). Mirrors the 3B dissociation-at-scale result.

Context note: v1 reads the instruction in chat context (no tools). The 3B work established
the refusal direction is largely distribution-invariant (chat AUC 0.887 ~ agentic 0.89), so
this is a defensible first measurement; tool-context extraction is a rigor refinement.

Usage:  python gemma4_probe_dissociation.py            (full, needs GPU)
        python gemma4_probe_dissociation.py --dry      (CPU: validate data/labels only)
"""
import glob
import json
import os
import sys

import numpy as np
from inspect_ai.log import read_eval_log
from inspect_evals.agentharm.utils import filter_dataset, load_dataset

MODEL = "google/gemma-4-31B-it-qat-w4a16-ct"
DRY = "--dry" in sys.argv
HARMFUL_LOG_GLOB = os.path.expanduser("~/latent-evasion/logs_gemma4_harmful/*.eval")
OUT = os.path.expanduser("~/latent-evasion/gemma4_dissociation.json")
B = 5000  # bootstrap resamples for CI


def load_prompts(task_name, split):
    """Return [(sample_id, instruction_text)] for an AgentHarm split."""
    ds = filter_dataset(load_dataset(task_name, split, False), [], None, None)
    out = []
    for s in ds:
        text = s.input if isinstance(s.input, str) else str(s.input)
        out.append((str(s.id), text))
    return out


def refusal_outcomes(path):
    """From a graded AgentHarm .eval log: {sample_id: (harm_score, refused_int)}."""
    log = read_eval_log(path)
    out = {}
    for smp in (log.samples or []):
        for sc in (smp.scores or {}).values():
            v = sc.value
            if isinstance(v, dict) and "score" in v:
                out[str(smp.id)] = (float(v["score"]), int(v.get("refusal", 0) >= 0.5))
    return out


def extract(model, tok, prompts):
    """Forward each instruction (chat-mode) and grab last-token residual at every layer.
    Returns X of shape (n_prompts, n_layers+1, hidden)."""
    import torch
    feats = []
    for i, (_sid, text) in enumerate(prompts):
        enc = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True, return_tensors="pt", return_dict=True,
        ).to("cuda:0")
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states  # tuple(n_layers+1) of (1, seq, hidden)
        feats.append(np.stack([h[0, -1, :].float().cpu().numpy() for h in hs]))
        if (i + 1) % 20 == 0:
            print(f"  ... {i + 1}/{len(prompts)}")
    return np.stack(feats)


def main():
    val_h = load_prompts("harmful", "val")
    val_b = load_prompts("benign", "val")
    test_h = load_prompts("harmful", "test_public")
    print(f"val: {len(val_h)} harmful + {len(val_b)} benign  |  test_public: {len(test_h)} harmful")

    log_path = sorted(glob.glob(HARMFUL_LOG_GLOB), key=os.path.getsize)[-1] \
        if glob.glob(HARMFUL_LOG_GLOB) else None
    if log_path is None:
        print("!! no harmful .eval log yet (baseline still running?) — behavioral outcomes unavailable")
        outcomes = {}
    else:
        outcomes = refusal_outcomes(log_path)
        print(f"behavioral outcomes from {os.path.basename(log_path)}: {len(outcomes)} samples")

    # align test samples that have an outcome
    test_ids = [sid for sid, _ in test_h]
    matched = [sid for sid in test_ids if sid in outcomes]
    print(f"matched test samples with outcomes: {len(matched)}/{len(test_ids)}")
    if outcomes:
        ref = np.array([outcomes[sid][1] for sid in matched])
        print(f"  refusal rate: {ref.mean():.3f}  ({int(ref.sum())} refused / {len(ref)})")

    if DRY:
        print("[dry] data + label plumbing OK; skipping model load / extraction")
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sklearn.svm import LinearSVC
    from sklearn.metrics import roc_auc_score

    print("loading model ...")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="cuda:0", dtype=torch.bfloat16)
    model.eval()

    print("extract val harmful ...");  Xvh = extract(model, tok, val_h)
    print("extract val benign ...");   Xvb = extract(model, tok, val_b)
    print("extract test harmful ...");  Xth = extract(model, tok, test_h)

    Xtr = np.concatenate([Xvh, Xvb]);  ytr = np.array([1] * len(Xvh) + [0] * len(Xvb))
    n_layers = Xtr.shape[1]

    # per-layer probe; pick best layer by 5-fold CV AUC on the val contrast (no test peeking)
    from sklearn.model_selection import StratifiedKFold
    layer_auc = []
    for L in range(n_layers):
        scores = np.zeros(len(ytr))
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        for tr, te in skf.split(Xtr[:, L], ytr):
            mu1 = Xtr[tr][ytr[tr] == 1, L].mean(0); mu0 = Xtr[tr][ytr[tr] == 0, L].mean(0)
            d = mu1 - mu0; d /= (np.linalg.norm(d) + 1e-12)
            scores[te] = Xtr[te, L] @ d
        layer_auc.append(roc_auc_score(ytr, scores))
    best = int(np.argmax(layer_auc))
    print(f"best layer by val CV AUC: {best} (AUC {layer_auc[best]:.3f})")

    # fit final direction at best layer on all val
    mu1 = Xtr[ytr == 1, best].mean(0); mu0 = Xtr[ytr == 0, best].mean(0)
    sd = mu1 - mu0; sd /= (np.linalg.norm(sd) + 1e-12)
    thr = 0.5 * (mu1 @ sd + mu0 @ sd)  # midpoint threshold -> "fire" if score > thr

    result = {"model": MODEL, "n_layers": int(n_layers), "best_layer": best,
              "val_cv_auc": layer_auc, "threshold": float(thr)}

    if outcomes:
        s_test = Xth @ sd  # (n_test, n_layers)
        s_best = s_test[:, best]
        by_id = {sid: s_best[i] for i, (sid, _) in enumerate(test_h)}
        sc = np.array([by_id[sid] for sid in matched])
        ref = np.array([outcomes[sid][1] for sid in matched])
        harm = np.array([outcomes[sid][0] for sid in matched])
        fire = (sc > thr).astype(int)

        complied = ref == 0
        refused = ref == 1
        p_fire_complied = float(fire[complied].mean()) if complied.any() else float("nan")
        p_fire_refused = float(fire[refused].mean()) if refused.any() else float("nan")
        auc_ref = float(roc_auc_score(ref, sc)) if len(set(ref)) > 1 else float("nan")

        # behavior-clustered bootstrap CI on P(fire | complied)
        rng = np.random.default_rng(0)
        idx_c = np.where(complied)[0]
        boot = [fire[rng.choice(idx_c, len(idx_c), replace=True)].mean() for _ in range(B)] \
            if len(idx_c) else []
        ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))] if boot else None

        print("\n=== GEMMA-4 DISSOCIATION ===")
        print(f"  refusal rate               : {ref.mean():.3f}")
        print(f"  P(fire | complied)         : {p_fire_complied:.3f}  CI {ci}")
        print(f"  P(fire | refused)          : {p_fire_refused:.3f}")
        print(f"  AUC(probe score -> refusal): {auc_ref:.3f}")
        result.update({
            "refusal_rate": float(ref.mean()), "p_fire_complied": p_fire_complied,
            "p_fire_complied_ci": ci, "p_fire_refused": p_fire_refused, "auc_refusal": auc_ref,
            "records": [{"id": sid, "score": float(by_id[sid]),
                         "refused": int(outcomes[sid][1]), "harm": float(outcomes[sid][0])}
                        for sid in matched],
        })

    json.dump(result, open(OUT, "w"), indent=2)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
