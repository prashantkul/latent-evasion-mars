"""Sanity check: read the saved mini-run .eval logs and print full-precision per-sample
probe metadata, to confirm within-setting variation is real (not a reused-activation bug)."""
import glob
from inspect_ai.log import read_eval_log

for path in sorted(glob.glob("logs_mini/*.eval")):
    log = read_eval_log(path)
    ds = log.eval.dataset.name if log.eval and log.eval.dataset else "?"
    print(f"\n=== {path.split('/')[-1]}  dataset={ds}  status={log.status} ===")
    for smp in (log.samples or []):
        sc = smp.scores.get("probe_scorer") if smp.scores else None
        m = (sc.metadata if sc else None) or {}
        print(f"  id={smp.id} id_orig={m.get('id_original')} cat={m.get('category')} "
              f"refused={m.get('refused_text')}")
        print(f"     firing_frac={m.get('probe_firing_fraction')!r}  "
              f"L14={m.get('probe_score_layer14')!r}  "
              f"argmaxL={m.get('probe_argmax_layer')} max={m.get('probe_score_max')!r}")
