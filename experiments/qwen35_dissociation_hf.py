"""Qwen3.5-27B dissociation via standalone transformers (no Inspect hf provider).

Behavioral outcomes (refused/complied) come from the vLLM AgentHarm run's .eval log.
Representation comes from an hf forward pass: fit the refusal direction on the val
harmful/benign contrast, read it on test_public harmful, correlate.

  P(fire | complied) vs the 3B's 0.81 == does the dissociation generalize to a capable
  Mamba-hybrid model.

Runs on the H100 AFTER vLLM is stopped (needs the GPU). Chat-mode reads, thinking off
(enable_thinking=False) to match the baseline. All layers swept; best chosen by val CV AUC.
"""
import glob
import json
import os
import sys

import numpy as np
from inspect_ai.log import read_eval_log
from inspect_evals.agentharm.utils import filter_dataset, load_dataset

MODEL = os.environ.get("QWEN35_MODEL", "Qwen/Qwen3.5-27B")
HARMFUL_LOG_GLOB = os.environ.get("HARMFUL_LOG", "/opt/latent-evasion-mars/logs_qwen35v_harmful/*.eval")
OUT = os.environ.get("OUT", "/opt/qwen35_dissociation.json")
DRY = "--dry" in sys.argv
B = 5000


def load_prompts(task_name, split):
    ds = filter_dataset(load_dataset(task_name, split, False), [], None, None)
    return [(str(s.id), s.input if isinstance(s.input, str) else str(s.input)) for s in ds]


def refusal_outcomes(path):
    log = read_eval_log(path)
    out = {}
    for smp in (log.samples or []):
        for sc in (smp.scores or {}).values():
            v = sc.value
            if isinstance(v, dict) and "score" in v:
                out[str(smp.id)] = (float(v["score"]), int(v.get("refusal", 0) >= 0.5))
    return out


def extract(model, tok, prompts):
    import torch
    feats = []
    for i, (_sid, text) in enumerate(prompts):
        enc = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True, return_tensors="pt", return_dict=True,
            enable_thinking=False,
        ).to("cuda:0")
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states  # (n_layers+1) x (1, seq, hidden)
        feats.append(np.stack([h[0, -1, :].float().cpu().numpy() for h in hs[1:]]))  # skip embed
        if (i + 1) % 20 == 0:
            print(f"  ... {i+1}/{len(prompts)}", flush=True)
    return np.stack(feats)


def main():
    vh = load_prompts("harmful", "val"); vb = load_prompts("benign", "val")
    th = load_prompts("harmful", "test_public")
    print(f"val {len(vh)}h/{len(vb)}b  test {len(th)}h", flush=True)

    logp = sorted(glob.glob(HARMFUL_LOG_GLOB), key=os.path.getsize)[-1]
    outc = refusal_outcomes(logp)
    print(f"outcomes from {os.path.basename(logp)}: {len(outc)}", flush=True)
    matched = [sid for sid, _ in th if sid in outc]
    ref = np.array([outc[sid][1] for sid in matched])
    print(f"matched {len(matched)}  refusal_rate {ref.mean():.3f}", flush=True)
    if DRY:
        print("[dry] ok"); return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    print("loading model ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="cuda:0", dtype=torch.bfloat16)
    model.eval()

    print("extract val harmful", flush=True); Xvh = extract(model, tok, vh)
    print("extract val benign", flush=True);  Xvb = extract(model, tok, vb)
    print("extract test harmful", flush=True); Xth = extract(model, tok, th)

    Xtr = np.concatenate([Xvh, Xvb]); ytr = np.array([1]*len(Xvh) + [0]*len(Xvb))
    by_id = {sid: Xth[i] for i, (sid, _) in enumerate(th)}
    Xm = np.stack([by_id[sid] for sid in matched])
    L = Xtr.shape[1]

    cv = []
    for li in range(L):
        s = np.zeros(len(ytr))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xtr[:, li], ytr):
            mu1 = Xtr[tr][ytr[tr]==1, li].mean(0); mu0 = Xtr[tr][ytr[tr]==0, li].mean(0)
            d = mu1-mu0; d/=(np.linalg.norm(d)+1e-12); s[te] = Xtr[te, li] @ d
        cv.append(roc_auc_score(ytr, s))
    best = int(np.argmax(cv)); bl = best+1
    print(f"best layer L{bl} val-CV AUC {cv[best]:.3f}", flush=True)

    mu1 = Xtr[ytr==1, best].mean(0); mu0 = Xtr[ytr==0, best].mean(0)
    d = mu1-mu0; d/=(np.linalg.norm(d)+1e-12); thr = 0.5*(mu1@d + mu0@d)
    sc = Xm[:, best] @ d; fire = (sc > thr).astype(int)
    comp = ref == 0; refd = ref == 1
    pfc = float(fire[comp].mean()); pfr = float(fire[refd].mean())
    auc_ref = float(roc_auc_score(ref, sc)) if len(set(ref))>1 else float("nan")
    rng = np.random.default_rng(0)
    boot = [fire[comp][rng.integers(0, comp.sum(), comp.sum())].mean() for _ in range(B)]
    ci = [float(np.percentile(boot,2.5)), float(np.percentile(boot,97.5))]

    print("\n=== QWEN3.5-27B DISSOCIATION ===", flush=True)
    print(f"  best layer L{bl} (val harmful/benign AUC {cv[best]:.3f})")
    print(f"  refusal rate            {ref.mean():.3f}")
    print(f"  P(fire | complied)      {pfc:.3f}  CI {ci}")
    print(f"  P(fire | refused)       {pfr:.3f}")
    print(f"  refusal=>fired          {int(fire[refd].sum())}/{int(refd.sum())}")
    print(f"  AUC(score -> refusal)   {auc_ref:.3f}")
    json.dump({"model": MODEL, "best_layer": bl, "val_cv_auc": cv, "refusal_rate": float(ref.mean()),
               "p_fire_complied": pfc, "p_fire_complied_ci": ci, "p_fire_refused": pfr,
               "auc_refusal": auc_ref,
               "records": [{"id": s, "score": float(sc[i]), "refused": int(ref[i]), "harm": outc[s][0]}
                           for i, s in enumerate(matched)]}, open(OUT, "w"), indent=2)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
