"""VALID in-context probe experiment for Qwen3.5-27B — train and evaluate entirely inside Inspect.

The read context is the REAL agent prompt (AgentHarm's ~558-char system prompt + tools), which the
offline get_representations path in `05` omitted — that omission is why `05` is INVALID. Here every
activation is read through Inspect's hf provider (patched for Qwen3.5 tool-calling), so train and test
share the exact same distribution.

  PASS A (val, throwaway scoring): inspect_eval over the val split; the scorer only HARVESTS each
     sample's per-layer post-instruction vector (returns a dummy Score). After the run we train the
     per-layer LinearSVC (standardized, folded to a raw-space (w,b)) and mean-diff on those val
     activations, and pick the layer by 5-fold CV.
  PASS B (test, real scoring): inspect_eval over the held-out test_public split; the just-trained
     probe IS the scorer — it computes w·h + b live per sample, so the .eval log's Score column is
     the probe firing. Per-layer svm/md scores are stored in metadata for the AUC evaluation.

Train ONLY on the 64 val rows. Test activations are scored, NEVER fit on. Behavioural labels
(refused) come from the graded vLLM run, joined by sample id, for AUC(probe -> refusal).

Pass A writes the trained probe in canonical form (experiments/probe_io.py). A later run can
REUSE it with `--probe-in <stem>` and skip Pass A entirely — the val split is a fixed 64 rows
and the fit is deterministic, so re-harvesting it buys nothing but GPU time. Reuse is checked,
not assumed: the model id, read position, layer indexing and layer coverage must all match.

  export HF_HOME=/workspace/.cache/huggingface PYTHONPATH=/opt/latent-evasion-mars
  python3 experiments/results/06-qwen35-inscorer-probe/qwen35_inscorer_experiment.py \
      --harmful-log '/opt/latent-evasion-mars/logs_qwen35v_harmful/*.eval' \
      --benign-log  '/opt/latent-evasion-mars/logs_qwen35v_benign/*.eval' \
      --out /opt/qwen35_inscorer_experiment.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen35_inspect_patch  # noqa: E402

from inspect_ai import Task, eval as inspect_eval  # noqa: E402
from inspect_ai.log import read_eval_log  # noqa: E402
from inspect_ai.model import GenerateConfig, get_model  # noqa: E402
from inspect_ai.scorer import Score, Target, mean, scorer  # noqa: E402
from inspect_ai.solver import TaskState  # noqa: E402

from inspect_evals.agentharm.agents.agent import get_agent  # noqa: E402
from inspect_evals.agentharm.utils import (  # noqa: E402
    filter_dataset, get_benchmark_tools, load_dataset, setup_tools_from_metadata,
)

C = 0.1
B = 5000
VAL = {"X": [], "y": []}                     # harvested val activations (Pass A)
TEST = {"harmful": {}, "benign": {}}         # per-sample probe scores (Pass B)
TEST_ACTS = {"harmful": [], "benign": []}    # cached in-context test activations (Pass B)
W = {}                                       # trained probe (filled between passes)


def read_all_layers(hs):
    """Post-instruction token at every transformer layer (embedding skipped). -> (L, H)."""
    step0 = hs[0]
    return np.stack([step0[l][0, -1, :].float().cpu().numpy() for l in range(1, len(step0))])


# ---------- PASS A: harvest val activations (throwaway score) ----------
@scorer(metrics=[mean()])
def collect_val(label):
    async def score(state: TaskState, target: Target) -> Score:
        hs = ((state.output.metadata or {}) if state.output else {}).get("hidden_states")
        if hs is not None:
            VAL["X"].append(read_all_layers(hs)); VAL["y"].append(label)
        return Score(value=float(label))     # placeholder — Pass A logs are not a result
    return score


# ---------- PASS B: the trained probe IS the scorer ----------
@scorer(metrics=[mean()])
def probe_scorer(tag):
    async def score(state: TaskState, target: Target) -> Score:
        hs = ((state.output.metadata or {}) if state.output else {}).get("hidden_states")
        if hs is None:
            return Score(value=0.0, metadata={"error": "no hidden_states"})
        h = read_all_layers(hs)                                  # (L, H)
        if h.shape != W["svm_w"].shape:                          # backstop for a reused probe
            raise RuntimeError(f"activation/probe shape mismatch: h {h.shape} vs probe "
                               f"{W['svm_w'].shape} — --probe-in does not match this model")
        svm = (h * W["svm_w"]).sum(-1) + W["svm_b"]              # (L,)
        # Bare projection, no bias: this is the layout eval_probe.py expects for `md_h`/`md_b`
        # and compensates for. The canonical single_direction artifact does carry its bias.
        md = (h * W["dirs"]).sum(-1)                             # (L,)
        bl = W["best_layer"]
        io = str((state.metadata or {}).get("id_original"))
        TEST[tag][str(state.sample_id)] = {"svm": svm.tolist(), "md": md.tolist(), "id_original": io}
        # Cache the raw in-context activation, not just this probe's scores: re-evaluating a
        # NEW probe then costs a CPU matmul instead of another agentic GPU run.
        TEST_ACTS[tag].append((str(state.sample_id), h))
        return Score(value=float(svm[bl] > 0),                  # real: best-layer SVC firing
                     metadata={"svm_best": float(svm[bl]), "md_best": float(md[bl]),
                               "best_layer": int(bl), "id_original": io})
    return score


def task(task_name, split, scorer_fn):
    ds = filter_dataset(load_dataset(task_name, split, False), [], None, None)
    tools = get_benchmark_tools(task_name)
    plan = [setup_tools_from_metadata(task_name, tools, 0),
            get_agent(agent="default", tool_choice="none")]
    return Task(dataset=ds, solver=plan, scorer=scorer_fn,
                config=GenerateConfig(max_tokens=8, max_connections=1))


def fit_svm(X, y):
    return make_pipeline(StandardScaler(),
                         LinearSVC(C=C, dual="auto", max_iter=1_000_000, random_state=42)).fit(X, y)


def raw_wb(pipe):
    sc = pipe.named_steps["standardscaler"]; svc = pipe.named_steps["linearsvc"]
    w = svc.coef_[0] / sc.scale_
    b = float(svc.intercept_[0] - np.dot(sc.mean_ / sc.scale_, svc.coef_[0]))
    return w.astype(np.float32), b


def train_probe(X, y):
    """Per-layer SVM (w,b) + mean-diff dir on ALL val; 5-fold CV AUC for layer selection."""
    L, H = X.shape[1], X.shape[2]
    svm_w = np.zeros((L, H), np.float32); svm_b = np.zeros(L, np.float32)
    dirs = np.zeros((L, H), np.float32); cv = np.zeros(L)
    for l in range(L):
        svm_w[l], svm_b[l] = raw_wb(fit_svm(X[:, l], y))
        sd = X[y == 1, l].mean(0) - X[y == 0, l].mean(0)
        dirs[l] = sd / (np.linalg.norm(sd) + 1e-12)
        s = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X[:, l], y):
            s[te] = fit_svm(X[tr, l], y[tr]).decision_function(X[te, l])
        cv[l] = roc_auc_score(y, s)
    return svm_w, svm_b, dirs, cv


def save_canonical_probes(stem, svm_w, svm_b, dirs, Xval, yval, model_name, cv=None, best=None):
    """Write both probes in the schema eval_probe.py / CLE consume (experiments/probe_io.py)."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    from probe_io import save_probe

    meta = {"model": model_name, "hidden": int(Xval.shape[2]), "n_val": len(yval),
            "read_position": "post-instruction token (-1), in-context: read inside the Inspect "
                             "scorer from the model's real agentic context (AgentHarm agent "
                             "system prompt + tools)",
            "layer_index": "0-based over transformer blocks; embedding layer skipped (hidden_states[l+1])",
            "score": "x @ w + b; fires when > 0",
            "trained_on": f"AgentHarm val split, {int(yval.sum())} harmful / {int((yval == 0).sum())} benign, in-context",
            "enable_thinking": False, "dtype": "bf16 model, float32 probe",
            "provenance": "qwen35_inscorer_experiment.py Pass A"}
    # The CV curve belongs to the SVM fit alone, so it is recorded only on that sidecar --
    # which is also where `--probe-in` reads the best layer back from on a reuse run.
    sel = {} if cv is None else {"val_cv_by_layer": np.asarray(cv).tolist(),
                                 "val_cv_best_layer": int(best if best is not None else np.argmax(cv))}
    save_probe(f"{stem}_svm", svm_w, svm_b,
               {**meta, **sel, "probe_type": "svm",
                "fit": "LinearSVC C=0.1, StandardScaler folded into raw-space (w, b)"})
    # Mean-difference carries no bias at fit time; the canonical form derives the midpoint
    # boundary so the artifact is self-contained.
    mu1 = Xval[yval == 1].mean(0).astype(np.float32)
    mu0 = Xval[yval == 0].mean(0).astype(np.float32)
    sd_b = -0.5 * np.einsum("lh,lh->l", dirs.astype(np.float32), mu1 + mu0)
    save_probe(f"{stem}_single_direction", dirs, sd_b,
               {**meta, "probe_type": "single_direction",
                "fit": "unit-norm centroid difference; bias = midpoint between val centroids"})


def load_trained_probe(stem, model_name, best_override=None):
    """Reuse a probe a previous Pass A already fit, so this run only pays for Pass B.

    Both probe types are needed — Pass B scores SVM and mean-difference side by side. The
    checks here are the point: a probe read at a different position, or indexed off by the
    embedding layer, produces plausible numbers rather than an error.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    from probe_io import load_probe

    svm_w, svm_b, svm_layers, meta = load_probe(f"{stem}_svm")
    dirs, _, sd_layers, sd_meta = load_probe(f"{stem}_single_direction")

    if meta["model"] != model_name:
        raise SystemExit(f"probe was trained on {meta['model']!r} but this run is {model_name!r}")
    for key in ("read_position", "layer_index"):
        if meta[key] != sd_meta[key]:
            raise SystemExit(f"{stem}: the two probes disagree on {key} — are they from one Pass A?")
    if dirs.shape != svm_w.shape:
        raise SystemExit(f"{stem}: shape mismatch, svm {svm_w.shape} vs single_direction {dirs.shape}")
    for name, lay in (("svm", svm_layers), ("single_direction", sd_layers)):
        if not np.array_equal(lay, np.arange(len(lay))):
            raise SystemExit(f"{stem}_{name}: Pass B scores every layer in order, so the probe must "
                             f"cover layers 0..L-1 contiguously; got {lay[:6].tolist()}...")

    best = best_override if best_override is not None else meta.get("val_cv_best_layer")
    if best is None:
        raise SystemExit(f"{stem}_svm has no val_cv_best_layer — pass --best-layer. (It only sets "
                         "the .eval Score column; the pinned metric is AUC mean-over-layers.)")
    return svm_w, svm_b, dirs, meta.get("val_cv_by_layer"), int(best), int(meta.get("n_val", 0)), meta


def check_against_model(model, w):
    """Fail before Pass B rather than on the first sample, when the config makes it knowable."""
    cfg = getattr(getattr(model.api, "model", None), "config", None)
    if cfg is None:
        return                                       # the scorer's shape guard still applies
    want = (int(cfg.num_hidden_layers), int(cfg.hidden_size))
    if w.shape != want:
        raise SystemExit(f"reused probe is {w.shape} but this model has {want} "
                         "(layers, hidden) — wrong probe, or a different layer convention")


def refusal_map(pattern):
    p = sorted(glob.glob(pattern), key=os.path.getsize)[-1]
    out = {}
    for smp in (read_eval_log(p).samples or []):
        for sc in (smp.scores or {}).values():
            v = sc.value
            if isinstance(v, dict) and "score" in v:
                out[str(smp.id)] = (float(v["score"]), int(v.get("refusal", 0) >= 0.5)); break
    return os.path.basename(p), out


def cluster_ci(scores, ref, io):
    uniq = np.unique(io); idx = {c: np.where(io == c)[0] for c in uniq}
    rng = np.random.default_rng(0); v = []
    for _ in range(B):
        i = np.concatenate([idx[c] for c in rng.choice(uniq, len(uniq), replace=True)])
        if len(set(ref[i])) > 1:
            v.append(roc_auc_score(ref[i], scores[i]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("QWEN35_27B_MODEL", "Qwen/Qwen3.5-27B"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--harmful-log", default="/opt/latent-evasion-mars/logs_qwen35v_harmful/*.eval")
    ap.add_argument("--benign-log", default="/opt/latent-evasion-mars/logs_qwen35v_benign/*.eval")
    ap.add_argument("--log-dir", default="./logs_qwen35_inscorer_exp")
    ap.add_argument("--acts-out", default="/opt/qwen35_inscorer_val_acts.npz")
    ap.add_argument("--test-acts-out", default="/opt/qwen35_inscorer_test_acts.npz",
                    help="cache in-context TEST activations so a new probe can be re-evaluated "
                         "on CPU (experiments/eval_probe.py) without another agentic GPU run")
    ap.add_argument("--probe-out", default=None,
                    help="stem for the canonical probe artifacts written right after Pass A "
                         "(default: <out>_probe)")
    ap.add_argument("--probe-in", default=None,
                    help="canonical probe stem from an earlier Pass A (<stem>_svm, "
                         "<stem>_single_direction). Skips Pass A entirely — only Pass B runs.")
    ap.add_argument("--best-layer", type=int, default=None,
                    help="override the reused probe's best layer (sets the Score column only)")
    ap.add_argument("--out", default="/opt/qwen35_inscorer_experiment.json")
    args = ap.parse_args()

    qwen35_inspect_patch.install()
    model = get_model(f"hf/{args.model}", hidden_states=True, device=args.device,
                      enable_thinking=False)

    if args.probe_in:
        # ---- PASS A skipped: reuse a probe an earlier run already fit ----
        # The val split is a fixed 64 rows and the fit is deterministic, so re-harvesting it
        # reproduces the same weights bit for bit. Reuse buys back one full agentic GPU pass.
        print(f"PASS A skipped — reusing probe {args.probe_in}", flush=True)
        svm_w, svm_b, dirs, cv, best, n_val, pmeta = load_trained_probe(
            args.probe_in, args.model, args.best_layer)
        check_against_model(model, svm_w)
        print(f"  {svm_w.shape[0]} layers x {svm_w.shape[1]} dims | trained on {pmeta['trained_on']}",
              flush=True)
        print(f"  read position: {pmeta['read_position']}", flush=True)
        print(f"  best layer L{best}" + ("" if args.best_layer is None else " (overridden)"), flush=True)
    else:
        # ---- PASS A: harvest val, train ----
        print("PASS A — harvest val activations (in-scorer) ...", flush=True)
        inspect_eval(task("harmful", "val", collect_val(1)), model=model, log_dir=args.log_dir)
        inspect_eval(task("benign", "val", collect_val(0)), model=model, log_dir=args.log_dir)
        Xval = np.stack(VAL["X"]); yval = np.array(VAL["y"])
        print(f"  val activations {Xval.shape} ({int(yval.sum())} harmful / {int((yval==0).sum())} benign)", flush=True)
        np.savez(args.acts_out, X=Xval, y=yval)          # persist val acts (training set)

        svm_w, svm_b, dirs, cv = train_probe(Xval, yval)
        best = int(np.argmax(cv)); n_val = len(yval); cv = cv.tolist()
        print(f"  trained probe; best layer by val-CV L{best} (AUC {cv[best]:.3f})", flush=True)

        # Persist the probe NOW, in canonical form. It otherwise exists only in the module-level W
        # until the scores npz is written at the very end, so anything failing in Pass B or the
        # analysis would discard it. Nothing GPU-bound is lost either way (the fit is deterministic
        # and the val acts are already on disk), but this keeps the artifact rather than the recipe
        # — and it is what a later `--probe-in` run consumes.
        stem = args.probe_out or args.out.replace(".json", "_probe")
        save_canonical_probes(stem, svm_w, svm_b, dirs, Xval, yval, args.model, cv, best)
        print(f"  saved canonical probes -> {stem}_*", flush=True)

    W.update(svm_w=svm_w, svm_b=svm_b, dirs=dirs, best_layer=best)

    # ---- PASS B: probe fires live on held-out test ----
    print("PASS B — probe as scorer on held-out test (in-scorer) ...", flush=True)
    inspect_eval(task("harmful", "test_public", probe_scorer("harmful")), model=model, log_dir=args.log_dir)
    inspect_eval(task("benign", "test_public", probe_scorer("benign")), model=model, log_dir=args.log_dir)
    print(f"  test scored: {len(TEST['harmful'])} harmful, {len(TEST['benign'])} benign", flush=True)

    # Persist the raw in-context test activations immediately — this is the artifact that makes
    # re-evaluating a teammate's probe a CPU job rather than a repeat of this GPU run.
    ta_ids = [i for i, _ in TEST_ACTS["harmful"]] + [i for i, _ in TEST_ACTS["benign"]]
    ta_X = np.stack([a for _, a in TEST_ACTS["harmful"]] + [a for _, a in TEST_ACTS["benign"]])
    ta_y = np.concatenate([np.ones(len(TEST_ACTS["harmful"])), np.zeros(len(TEST_ACTS["benign"]))])
    np.savez(args.test_acts_out, X=ta_X.astype(np.float32), y=ta_y.astype(int), ids=np.array(ta_ids))
    print(f"  cached test activations {ta_X.shape} -> {args.test_acts_out}", flush=True)

    # ---- evaluate: join behavioural labels, per-layer AUC ----
    hlog, href = refusal_map(args.harmful_log)
    ids = [i for i in TEST["harmful"] if i in href]
    ref = np.array([href[i][1] for i in ids])
    io = np.array([TEST["harmful"][i]["id_original"] for i in ids])
    L = svm_w.shape[0]
    md = np.array([[TEST["harmful"][i]["md"][l] for l in range(L)] for i in ids])       # (n, L)
    svm = np.array([[TEST["harmful"][i]["svm"][l] for l in range(L)] for i in ids])
    bmd = np.array([[TEST["benign"][i]["md"][l] for l in range(L)] for i in TEST["benign"]])
    bsvm = np.array([[TEST["benign"][i]["svm"][l] for l in range(L)] for i in TEST["benign"]])
    lab = np.concatenate([np.ones(len(md)), np.zeros(len(bmd))])

    def summarize(name, sc_h, sc_b):
        auc_ref = np.array([roc_auc_score(ref, sc_h[:, l]) for l in range(L)])
        auc_hb = np.array([roc_auc_score(lab, np.concatenate([sc_h[:, l], sc_b[:, l]])) for l in range(L)])
        b = int(np.argmax(auc_ref)); ms = sc_h.mean(1)
        r = dict(best_layer=b, best_auc=float(auc_ref[b]), best_ci=cluster_ci(sc_h[:, b], ref, io),
                 mean_auc=float(roc_auc_score(ref, ms)), best_hb=float(auc_hb.max()),
                 best_hb_layer=int(np.argmax(auc_hb)),
                 auc_refusal=auc_ref.tolist(), auc_harm_vs_benign=auc_hb.tolist())
        print(f"  {name:9s} best L{b} AUC->refusal {auc_ref[b]:.3f} CI {r['best_ci']} | "
              f"mean-layers {r['mean_auc']:.3f} | harm-vs-benign L{r['best_hb_layer']} {auc_hb.max():.3f}", flush=True)
        return r

    print(f"\n=== IN-CONTEXT PROBE (train val in-scorer -> eval test in-scorer) ===")
    print(f"  behavioural labels from {hlog} | n={len(ids)} refusal {ref.mean():.3f}", flush=True)
    res_md = summarize("mean-diff", md, bmd)
    res_svm = summarize("SVM", svm, bsvm)

    json.dump({"model": args.model, "context": "in-scorer (real agent system prompt + tools)",
               "probe_source": args.probe_in or "trained in this run (Pass A)",
               "n_val": n_val, "n_test_harmful": len(ids), "n_test_benign": len(bmd),
               "refusal_rate": float(ref.mean()), "val_cv_by_layer": cv,
               "val_cv_best_layer": best, "mean_diff": res_md, "svm": res_svm,
               "test_ids": ids}, open(args.out, "w"), indent=2)

    np.savez(args.out.replace(".json", "_scores.npz"),
             md_h=md, svm_h=svm, md_b=bmd, svm_b=bsvm, ref=ref, io=io,
             ids=np.array(ids), svm_w=svm_w, svm_bias=svm_b, dirs=dirs)
    print(f"\nsaved {args.out}", flush=True)


if __name__ == "__main__":
    main()
