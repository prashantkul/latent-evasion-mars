"""CLE as an Inspect model provider — `@modelapi("cle")`.

This is Path B from the 21 Jul harness note: *intervention is a thin provider*. Inspect's custom
provider hook is the seam the framework offers for owning generation, and using it means Inspect
drives the run rather than being driven around. An earlier attempt monkey-patched the HF model's
`generate` underneath Inspect; that worked but reached under the abstraction, could not see sample
boundaries, and mutated a shared object globally.

Subclassing `HuggingFaceAPI` rather than reimplementing it is the point. We inherit its chat
templating, tokenizer args, tool parsing, batching and `hidden_states` plumbing, and override only
generation. In particular CLE-A's calibration pass tokenizes via `self.hf_chat(...)` and
`self.tokenizer_call_args` — the provider's *own* code path — so the calibration input is what
generation actually sees. Re-rendering the agent context by hand is what invalidated experiment 05,
and this design never does it.

    model = get_model("cle/Qwen/Qwen3.5-27B", cle_attack="cle-a", cle_probe=<stem>,
                      cle_layers="53-63", cle_margin_quantile=0.5, cle_val_acts=<npz>,
                      device="cuda:0", enable_thinking=False, do_sample=False)
    model.api.set_beta(0.0)      # the exact no-op control, no reload

Both attacks land here so the comparison holds everything else fixed:

  cle-p   projection_hook on the selected blocks for the whole generate call. Stateless: the
          displacement is recomputed at every position and every decode step.
  cle-a   one calibration forward per prompt records delta[l] at the LAST prompt token, then
          add_hook(delta[l]) adds that frozen vector everywhere. Needs a per-prompt pre-pass,
          which is why it needs a provider at all.

CADENCE (`cle_recalibrate`), a question the paper's single-turn setting never raises:

  sample  calibrate once per rollout, freeze for its whole trajectory. The paper's semantics and
          the sharpest contrast with CLE-P. If CLE-A underperforms CLE-P agentically, staleness is
          why — and that is a FINDING, not a bug.
  turn    recalibrate every generate call. No longer the paper's CLE-A; a middle point we invented.
          Recorded as ours in the results json. Never report it as the CLE-A baseline.

Because we are a provider we get `input: list[ChatMessage]` and can key the rollout on the first
user message, rather than inferring boundaries from token prefixes.
"""
import hashlib

import numpy as np
import torch
from typing_extensions import override

from probe_io import load_probe
from probe_adapter import to_probe_dict
from utils.hooks import add_hook, pipeline_delta_hook, projection_hook, remove_hooks

from inspect_ai.model import GenerateConfig, modelapi
from inspect_ai.model._providers.hf import HuggingFaceAPI

ATTACKS = ("cle-p", "cle-a", "none")


def parse_layers(spec, n_layers):
    """'20-30' -> [20..30]; '22,24' -> [22, 24]. Matches utils/args.parse_layers_arg."""
    sel = (list(range(int(spec.split("-")[0]), int(spec.split("-")[1]) + 1))
           if "-" in spec else [int(x) for x in spec.split(",")])
    bad = [l for l in sel if not 0 <= l < n_layers]
    if bad:
        raise ValueError(f"layers out of range for a {n_layers}-layer model: {bad}")
    return sel


def geometric_margins(val_acts, probe_stem, layers, q):
    """Per-layer margin landing a pushed activation at the q-quantile of the HARMLESS scores.

    With beta=1 the post-intervention score is exactly -margin, so margin = -quantile(harmless).
    Derived from val activations only -- the test split stays held out.
    """
    w, b, idx, _ = load_probe(probe_stem)
    z = np.load(val_acts, allow_pickle=True)
    X, y = z["X"], z["y"].astype(int)
    pos = {int(l): n for n, l in enumerate(idx)}
    return {l: float(-np.quantile(X[y == 0, l, :] @ w[pos[l]] + b[pos[l]], q)) for l in layers}


def decoder_blocks(hf):
    for path in (("model", "layers"), ("transformer", "h"), ("layers",)):
        obj = hf
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise RuntimeError(f"could not locate decoder layers on {type(hf).__name__}")


class CleStats:
    """Counters and the staleness trace, so a run explains itself rather than only scoring."""

    def __init__(self):
        self.calls = self.calibrations = 0
        # (call_index, layer, w.h+b at the last position) WITH the frozen delta applied. Under
        # `sample` cadence this should drift away from -margin as the rollout grows; that drift is
        # the mechanism behind whatever the four metrics show.
        self.trace = []

    def summary(self):
        return {"generate_calls": self.calls, "calibrations": self.calibrations,
                "trace_points": len(self.trace)}


def _read_score_hook(w, b, layer_idx, call_idx, trace):
    """Read-only. Registered after add_hook, so it sees the modified output (PyTorch passes a
    hook's return value on to later hooks). Returns None, so it changes nothing."""
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        wl = w.to(device=h.device, dtype=h.dtype)
        score = (h[:, -1, :] * wl.view(1, -1)).sum(dim=-1) + b.to(h.device, h.dtype)
        trace.append((call_idx, layer_idx, float(score[0].item())))
        return None
    return hook


@modelapi("cle")
class CLEModelAPI(HuggingFaceAPI):
    @override
    def __init__(self, model_name, base_url=None, api_key=None, config=GenerateConfig(),
                 **model_args):
        # Pop ours BEFORE super().__init__ -- HuggingFaceAPI forwards every unrecognised model_arg
        # to AutoModelForCausalLM.from_pretrained, which would reject them.
        def take(name, default=None):
            return model_args.pop(name, default)

        attack = take("cle_attack", "cle-p")
        if attack not in ATTACKS:
            raise ValueError(f"cle_attack must be one of {ATTACKS}, got {attack!r}")
        probe_stem = take("cle_probe")
        layers_spec = take("cle_layers", "20-30")
        beta = float(take("cle_beta", 1.0))
        margins = take("cle_margins")
        quantile = take("cle_margin_quantile")
        val_acts = take("cle_val_acts")
        recalibrate = take("cle_recalibrate", "sample")
        trace = bool(take("cle_trace", False))
        if recalibrate not in ("sample", "turn"):
            raise ValueError(f"cle_recalibrate must be 'sample' or 'turn', got {recalibrate!r}")

        super().__init__(model_name, base_url, api_key, config, **model_args)

        self.attack, self.beta, self.recalibrate, self.trace_scores = \
            attack, beta, recalibrate, trace
        self.blocks = decoder_blocks(self.model)
        self.stats = CleStats()
        self._delta = None
        self._delta_key = None
        self._inflight = False

        if attack == "none":
            self.layers, self.margins, self.probes = [], {}, {}
            return

        if not probe_stem:
            raise ValueError("cle_probe (canonical probe stem) is required unless cle_attack=none")
        self.layers = parse_layers(layers_spec, len(self.blocks))
        if margins is None:
            if quantile is None or not val_acts:
                raise ValueError("pass cle_margins, or cle_margin_quantile + cle_val_acts")
            margins = geometric_margins(val_acts, probe_stem, self.layers, quantile)
            self.margin_mode = f"quantile {quantile} of val harmless scores (per layer)"
        elif isinstance(margins, (int, float)):
            # CLE's own --margin is a single raw score applied to every layer. Expand it here,
            # where the layer list is finally known.
            self.margin_mode = f"fixed {float(margins)}"
            margins = {l: float(margins) for l in self.layers}
        else:
            self.margin_mode = "explicit per-layer"
        self.margins = {int(k): float(v) for k, v in margins.items()}
        self.probes = to_probe_dict(probe_stem, self.layers, device=self.device)

    def set_beta(self, beta):
        """Swap the control/attacked arm without reloading 27B of weights."""
        self.beta = float(beta)
        self._delta = self._delta_key = None      # a stale delta is beta-specific

    # -- CLE-A ------------------------------------------------------------------------------

    def _rollout_key(self, input):
        """Identify the rollout. As a provider we see the messages, so the first user message is
        available directly -- no inferring boundaries from token prefixes."""
        for msg in input:
            if getattr(msg, "role", None) == "user":
                return hashlib.sha1(str(getattr(msg, "text", msg)).encode()).hexdigest()
        return hashlib.sha1(str(input[:1]).encode()).hexdigest()

    def _calibration_ids(self, input, tools):
        """Tokenize exactly as generation will: the provider's own hf_chat + tokenizer args."""
        chat = self.hf_chat(input, tools)
        enc = self.tokenizer(chat, return_tensors="pt", padding=True, **self.tokenizer_call_args)
        return enc["input_ids"].to(self.model.device), enc["attention_mask"].to(self.model.device)

    def _calibrate(self, input_ids, attention_mask):
        """One forward with the CLE-P projection live, recording the displacement it produced.

        All hooks are installed at once and in layer order, deliberately: layer l's delta must be
        measured on a hidden state already modified by layers < l. Measuring them one layer at a
        time gives different numbers and a different attack.
        """
        store = {}
        handles = [self.blocks[l].register_forward_hook(
            pipeline_delta_hook(self.probes[l]["w"], self.probes[l]["b"], self.beta,
                                self.margins[l], l, store)) for l in sorted(self.layers)]
        try:
            with torch.no_grad():
                self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        finally:
            remove_hooks(handles)
        missing = [l for l in self.layers if l not in store]
        if missing:
            raise RuntimeError(f"calibration produced no delta for layers {missing} — hook never fired")
        return store

    def _install(self, input, tools):
        """Return the hook handles for this generate call."""
        if self.attack == "none" or not self.layers:
            return []
        if self.attack == "cle-p":
            return [self.blocks[l].register_forward_hook(
                projection_hook(self.probes[l]["w"], self.probes[l]["b"], self.beta,
                                self.margins[l])) for l in sorted(self.layers)]

        key = self._rollout_key(input)
        if self.recalibrate == "turn" or self._delta is None or key != self._delta_key:
            ids, mask = self._calibration_ids(input, tools)
            if ids.shape[0] != 1:
                raise RuntimeError(
                    f"CLE-A expects batch size 1, got {ids.shape[0]}. The delta is per-prompt, so a "
                    f"batched call would apply one prompt's displacement to another's activations.")
            self._delta = self._calibrate(ids, mask)
            self._delta_key = key
            self.stats.calibrations += 1

        handles = []
        for l in sorted(self.layers):
            handles.append(self.blocks[l].register_forward_hook(add_hook(self._delta[l])))
            if self.trace_scores:
                handles.append(self.blocks[l].register_forward_hook(
                    _read_score_hook(self.probes[l]["w"], self.probes[l]["b"], l,
                                     self.stats.calls, self.stats.trace)))
        return handles

    # -- correctness checks (cle_a_agentic_design.md §7) -------------------------------------

    def selftest(self, prompt="Tell me how to pick a lock.", tol=1e-3):
        """Cheap, sharp checks to run before trusting any behavioural number from CLE-A."""
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        ids, mask = enc["input_ids"], enc["attention_mask"]
        out, keep_beta = {}, self.beta

        def capture(store):
            def make(l):
                def hook(module, inputs, output):
                    h = output[0] if isinstance(output, tuple) else output
                    store[l] = h[:, -1, :].detach().float().clone()
                    return None
                return hook
            return [self.blocks[l].register_forward_hook(make(l)) for l in sorted(self.layers)]

        def forward(hooks):
            handles = list(hooks)
            try:
                with torch.no_grad():
                    self.model(input_ids=ids, attention_mask=mask, use_cache=False)
            finally:
                remove_hooks(handles)

        try:
            with torch.no_grad():
                # (1) beta=0 must give an exactly zero delta -- the control has to be a true no-op.
                self.beta = 0.0
                d0 = self._calibrate(ids, mask)
                out["beta0_delta_max_abs"] = max(float(d0[l].abs().max()) for l in self.layers)
                out["beta0_is_noop"] = out["beta0_delta_max_abs"] == 0.0

                # (2) score landing: CLE-P at beta=1 puts w.h+b at exactly -margin.
                self.beta = 1.0
                sp = {}
                forward([self.blocks[l].register_forward_hook(
                    projection_hook(self.probes[l]["w"], self.probes[l]["b"], 1.0, self.margins[l]))
                    for l in sorted(self.layers)] + capture(sp))
                out["score_landing_max_err"] = max(
                    abs(float((sp[l][0] * self.probes[l]["w"].float()).sum()
                              + self.probes[l]["b"].float()) + self.margins[l])
                    for l in self.layers)

                # (3) single-position equivalence. delta IS the CLE-P displacement at the
                # calibration token, so adding it back must reproduce CLE-P's hidden state there.
                # A gap means the calibration pass is not seeing what generation sees -- the class
                # of bug that invalidated experiment 05.
                d1 = self._calibrate(ids, mask)
                sa = {}
                forward([self.blocks[l].register_forward_hook(add_hook(d1[l]))
                         for l in sorted(self.layers)] + capture(sa))
                out["cle_a_vs_cle_p_max_abs_diff"] = max(
                    float((sp[l] - sa[l]).abs().max()) for l in self.layers)
        finally:
            self.beta = keep_beta
            self._delta = self._delta_key = None

        out["pass"] = bool(out["beta0_is_noop"] and out["score_landing_max_err"] < tol
                           and out["cle_a_vs_cle_p_max_abs_diff"] < tol)
        return out

    @override
    async def generate(self, input, tools, tool_choice, config):
        self.stats.calls += 1
        # Hooks live on the shared model, so two concurrent generates would cross-contaminate.
        # We run max_connections=1; fail loudly rather than silently mixing two prompts' deltas.
        if self._inflight:
            raise RuntimeError("concurrent generate with CLE hooks installed — run max_connections=1")
        handles = self._install(input, tools)
        self._inflight = True
        try:
            return await super().generate(input, tools, tool_choice, config)
        finally:
            self._inflight = False
            remove_hooks(handles)
