"""Answer-probe EPF prototype (no repo files modified; classes subclassed here).

Idea: the shipped EPF weights particles by step-text token confidence, which is
provably uninformative (pairwise AUC ~0.5) and rewards degeneration. Instead,
after each step we PROBE the model: continue the trajectory with "Answer:" for a
few tokens (temp 0, logprobs on) and read the probability mass the model puts on
each choice letter. The particle weight becomes a function of that letter
distribution (its own forced-decision confidence — same model, no judge), and the
final answer can be a weighted MARGINAL over letters instead of an argmax particle.

Arms:
  base_epf       mean_logprob weights, resample every step (shipped EPF control)
  epf_probe      probe weights, resample every step (EPF machinery, better signal)
  probe_adaptive probe weights, resample only when ESS/N < 0.5
  sc_probe       probe weights, never resample (pure SC + probe used for voting)
  base_adaptive  mean_logprob weights, resample only when ESS/N < 0.5

Usage (compute node, vLLM with --enable-prefix-caching serving the model):
  python probe_epf.py --endpoint http://localhost:8500/v1 --model-name qwen-omni-3b \
    --bench mmar --limit 200 --budgets 8 --arms base_epf,epf_probe,probe_adaptive,sc_probe \
    --jsonl probe_out/mmar_3b.jsonl
"""

import argparse
import asyncio
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, "/work/hdd/bcey/awaheed/its-for-audio-reasoning/entropic-particle-filter")

from benchmarking.mmau_pro.prompt import build  # noqa: E402
from benchmarking.mmau_pro.scoring import LETTERS, predicted_index  # noqa: E402
from its_hub import (  # noqa: E402
    EntropicParticleFiltering,
    OpenAICompatibleLanguageModel,
    StepGeneration,
)
from its_hub.core.algorithms.particle_filtering import _inv_sigmoid, _softmax  # noqa: E402

_PUNCT = ")>].,:;!?*'\"( "


def letter_masses(logprobs_obj, valid: str) -> dict | None:
    """From an OpenAI logprobs object, find the first token position that carries
    choice-letter mass and return {letter: prob} summed over token variants."""
    content = (logprobs_obj or {}).get("content") or []
    for pos in content:
        tops = pos.get("top_logprobs") or []
        masses = defaultdict(float)
        for e in tops:
            tok = (e.get("token") or "").strip(_PUNCT)
            if len(tok) == 1 and tok.upper() in valid:
                masses[tok.upper()] += math.exp(e.get("logprob", -100.0))
        if masses:
            return dict(masses)
    return None


class HotStartStepGeneration(StepGeneration):
    """First reasoning step sampled hotter than the rest.

    Oracle misses are dominated by a shared root misperception: every particle
    inherits the same step-1 reading of the audio. Sampling step 1 hot spreads
    the roots; the probe-guided filter then prunes bad ones. Step index is
    inferred from the presence of a trailing assistant turn (steps so far)."""

    def __init__(self, *args, first_step_temp=1.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_step_temp = first_step_temp

    def _get_temperature(self, messages_or_messages_lst):
        if not messages_or_messages_lst:
            return self.temperature
        first = messages_or_messages_lst[0]
        if isinstance(first, list):  # batch: recurse per conversation
            return [self._get_temperature(m) for m in messages_or_messages_lst]
        last = messages_or_messages_lst[-1]
        is_step1 = not (getattr(last, "role", None) == "assistant"
                       and getattr(last, "content", None))
        return self.first_step_temp if is_step1 else self.temperature


LENSES = [
    "Focusing first on any speech and what is said:",
    "Focusing first on the background sounds and acoustic events:",
    "Focusing first on the music and its elements:",
    "Focusing first on how the audio changes over time:",
]


class LensStepGeneration(StepGeneration):
    """Seeds step 1 of each particle with a rotating perception-lens prefix
    (percept-angle diversity: the swarm serializes DIFFERENT aspects of the
    audio instead of 32 retellings of the same first impression)."""

    def __init__(self, *args, lenses=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lenses = lenses or LENSES

    async def aforward(self, lm, prompt_or_prompts, steps_so_far=None, **kw):
        if isinstance(prompt_or_prompts, list) and steps_so_far is not None:
            lens_for, seeded, k = {}, [], 0
            for i, st in enumerate(steps_so_far):
                if not st:  # step 1 of this particle: seed a lens as partial step
                    lens = self.lenses[k % len(self.lenses)]
                    k += 1
                    lens_for[i] = lens
                    seeded.append([lens])
                else:
                    seeded.append(st)
            if lens_for:
                res = await super().aforward(lm, prompt_or_prompts, seeded, **kw)
                # fold the lens into the stored step so future context reproduces
                # what the model actually saw
                return [
                    ((lens_for[i] + "\n" + r[0], *r[1:]) if i in lens_for else r)
                    for i, r in enumerate(res)
                ]
        return await super().aforward(lm, prompt_or_prompts, steps_so_far, **kw)


class AnswerProbeEPF(EntropicParticleFiltering):
    """EPF whose per-step particle weight comes from an answer probe.

    adaptive_ess: None = resample every step (shipped behavior);
                  float t = resample only when ESS/N < t (0.0 = never resample).
    use_probe:    False = keep the base self-certainty weight (control arms).
    weight_style: 'logit' = log-weight logit(pmax)/probe_temp (softmax sharpens);
                  'raw'   = log-weight log(pmax)/probe_temp, i.e. resampling
                  probability directly proportional to probe mass (bounded ratios).
    retire_stopped: stopped particles keep their slot and their vote but leave the
                  resampling pool — resampling happens only among still-active
                  particles (untempered softmax over active weights; the ESS gate,
                  when set, is evaluated over the active subset).
    """

    def __init__(self, *args, use_probe=True, adaptive_ess=None, probe_max_tokens=3,
                 n_choices=4, probe_temp=1.0, probe_suffix="Answer:",
                 weight_style="logit", retire_stopped=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_probe = use_probe
        self.adaptive_ess = adaptive_ess
        self.probe_max_tokens = probe_max_tokens
        self.probe_temp = probe_temp  # divide probe log-odds: >1 flattens the weight
        self.probe_suffix = probe_suffix  # e.g. 'Final Answer: \\boxed{' for P9
        self.weight_style = weight_style
        self.retire_stopped = retire_stopped
        self.valid = LETTERS[: max(1, n_choices)]

    # ---- probe ----
    async def _probe(self, lm, particles, base_messages, skip=None):
        """Probe particles that have at least one step; return letter dists."""
        idxs, msgs_lst = [], []
        for i, p in enumerate(particles):
            if not p.steps or (skip and i in skip):
                continue
            text = self.sg._post_process(p.steps, stopped=p.is_stopped)
            sfx = self.probe_suffix
            cut = text.find(sfx)
            if cut < 0 and sfx != "Answer:" and "Answer:" in text:
                # stopped trajectory wrote e.g. 'Final Answer: C' without \boxed —
                # cut at the generic marker and re-append the full probe suffix
                text = text[: text.find("Answer:")]
                cut = -1
            prefix = text[: cut + len(sfx)] if cut >= 0 else (
                text + ("" if text.endswith("\n") else "\n") + sfx)
            msgs = list(base_messages) + [
                {"role": "assistant", "content": prefix}
            ]
            idxs.append(i)
            msgs_lst.append(msgs)
        if not idxs:
            return {}
        resps = await lm.agenerate(
            msgs_lst, max_tokens=self.probe_max_tokens, temperature=0.0,
            logprobs=True, top_logprobs=20,
        )
        out = {}
        for i, r in zip(idxs, resps):
            lp = r.get("_logprobs") if isinstance(r, dict) else None
            out[i] = letter_masses(lp, self.valid)
        return out

    async def _apropagate(self, lm, particles, prompt, tools=None, tool_choice=None,
                          base_messages=None):
        was_stopped = [p.is_stopped for p in particles]
        particles = await super()._apropagate(
            lm, particles, prompt, tools=tools, tool_choice=tool_choice,
            base_messages=base_messages,
        )
        if self.use_probe:
            skip = {i for i, w in enumerate(was_stopped) if w}
            dists = await self._probe(lm, particles, base_messages or [], skip=skip)
            for i, p in enumerate(particles):
                if was_stopped[i]:
                    continue  # keep the weight from the round it stopped
                d = dists.get(i)
                if not hasattr(p, "probe_dists"):
                    p.probe_dists = []
                p.probe_dists.append(d)
                if d:
                    pmax = max(d.values())
                    if self.weight_style == "raw":
                        # softmax(log p) = p / sum(p): resample prob linear in pmax
                        w = math.log(max(pmax, 1e-7)) / self.probe_temp
                    else:
                        w = _inv_sigmoid(min(max(pmax, 1e-6), 1 - 1e-6)) / self.probe_temp
                elif self.weight_style == "raw":
                    # neutral for raw = uniform-over-letters pmax (log 0 would be MAX)
                    w = math.log(1.0 / max(2, len(self.valid))) / self.probe_temp
                else:
                    w = 0.0  # neutral when the probe found no letter mass
                p.partial_log_weights[-1] = float(w)
        return particles

    # deepcopy in resampling drops ad-hoc attrs; carry probe_dists through
    # (Particle.deepcopy only copies known fields, so patch after resample)
    def _resampling_indices(self, probabilities, num_particles):
        if self.adaptive_ess is not None:
            p = np.asarray(probabilities)
            ess = 1.0 / float(np.sum(np.clip(p, 1e-12, 1) ** 2))
            if ess / num_particles >= self.adaptive_ess:
                return list(range(num_particles))
        return super()._resampling_indices(probabilities, num_particles)

    def _resampling(self, particles, probabilities, num_particles):
        if self.retire_stopped:
            active = [i for i, p in enumerate(particles) if not p.is_stopped]
            if len(active) <= 1:
                return particles
            # untempered softmax over the ACTIVE particles only (the passed-in
            # `probabilities` were computed over the full pool incl. frozen echoes)
            probs = _softmax(np.array([particles[i].log_weight for i in active]))
            if self.adaptive_ess is not None:
                ess = 1.0 / float(np.sum(np.clip(probs, 1e-12, 1) ** 2))
                if ess / len(active) >= self.adaptive_ess:
                    return particles
            # grandparent resampler: bypass this class's gated _resampling_indices
            chosen = super()._resampling_indices(list(probs), len(active))
            out = list(particles)
            for slot, parent in zip(active, chosen):
                out[slot] = particles[active[parent]]
            return out
        idx = self._resampling_indices(probabilities, num_particles)
        return [particles[i] for i in idx]


def patched_deepcopy(self):
    import copy as _copy
    new = type(self)(
        steps=_copy.deepcopy(self.steps), is_stopped=self.is_stopped,
        partial_log_weights=_copy.deepcopy(self.partial_log_weights),
        partial_signals=_copy.deepcopy(self.partial_signals),
    )
    if hasattr(self, "probe_dists"):
        new.probe_dists = _copy.deepcopy(self.probe_dists)
    return new


def build_arm(arm, temp, max_steps, n_choices, step_token="\n\n",
              probe_suffix="Answer:", probe_max_tokens=3):
    if arm.startswith("hot_"):
        sg = HotStartStepGeneration(step_token=step_token, stop_token="Answer:",
                                    max_steps=max_steps, temperature=temp,
                                    first_step_temp=1.2)
        arm = arm[len("hot_"):]
    elif arm.startswith("lens_"):
        sg = LensStepGeneration(step_token=step_token, stop_token="Answer:",
                                max_steps=max_steps, temperature=temp)
        arm = arm[len("lens_"):]
    else:
        sg = StepGeneration(step_token=step_token, stop_token="Answer:",
                            max_steps=max_steps, temperature=temp)
    common = dict(sg=sg, resampling_method="systematic", temperature_method="ess",
                  ess_threshold=0.6, early_phase=0.7, self_certainty_signal="mean_logprob",
                  self_certainty_style="logit", n_choices=n_choices,
                  probe_suffix=probe_suffix, probe_max_tokens=probe_max_tokens)
    if arm == "base_epf":
        return AnswerProbeEPF(use_probe=False, adaptive_ess=None, **common)
    if arm == "epf_probe":
        return AnswerProbeEPF(use_probe=True, adaptive_ess=None, **common)
    if arm == "probe_adaptive":
        return AnswerProbeEPF(use_probe=True, adaptive_ess=0.5, **common)
    if arm.startswith("probe_adaptive_t"):
        return AnswerProbeEPF(use_probe=True, adaptive_ess=0.5,
                              probe_temp=float(arm.rsplit("t", 1)[1]), **common)
    if arm == "sc_probe":
        return AnswerProbeEPF(use_probe=True, adaptive_ess=0.0, **common)
    if arm == "base_adaptive":
        return AnswerProbeEPF(use_probe=False, adaptive_ess=0.5, **common)
    # --- round-4 arms: raw weights and stopped-particle retirement ---
    if arm == "raw_probe":  # prob ~ pmax, resample every step, no retirement
        return AnswerProbeEPF(use_probe=True, adaptive_ess=None,
                              weight_style="raw", **common)
    if arm == "retire_every_raw":  # raw weights, every step, active pool only
        return AnswerProbeEPF(use_probe=True, adaptive_ess=None,
                              weight_style="raw", retire_stopped=True, **common)
    if arm.startswith("retire_every_t"):  # logit/T, every step, active pool only
        return AnswerProbeEPF(use_probe=True, adaptive_ess=None, retire_stopped=True,
                              probe_temp=float(arm.rsplit("t", 1)[1]), **common)
    if arm.startswith("retire_t"):  # logit/T, ESS-gated (over actives), retirement
        return AnswerProbeEPF(use_probe=True, adaptive_ess=0.5, retire_stopped=True,
                              probe_temp=float(arm.rsplit("t", 1)[1]), **common)
    raise ValueError(arm)


def load_records(bench, limit, audio_root=None):
    if bench == "mmar":
        from benchmarking.mmar.loader import load_mmar_mcq
        recs = load_mmar_mcq("/u/awaheed/epf_data/mmar", subset="full",
                             audio_root=audio_root)
    elif bench == "mmau_pro_d1k":
        from benchmarking.mmau_pro.loader import load_mmau_mcq
        recs = load_mmau_mcq("/u/awaheed/epf_data/mmau_pro_d1k", subset="d1k")
    elif bench == "mmau":
        # ORIGINAL MMAU test-mini (1,000 single-audio MCQ, all gradeable)
        from benchmarking.mmau.loader import load_mmau_mcq as load_mmau_tm
        recs = load_mmau_tm("/u/awaheed/epf_data/mmau", subset="full",
                            audio_root=audio_root or "/u/awaheed/epf_data/mmau")
    else:
        raise ValueError(bench)
    recs = [r for r in recs if r.answer_index is not None]
    return recs[:limit]


def item_report(rec, res, gold):
    n = len(res.responses)
    texts = [p.get("content", "") for p in res.responses]
    preds = [predicted_index(t, rec.choices) for t in texts]
    letts = [LETTERS[p] if p is not None else "" for p in preds]
    fw = list(res.log_weights_lst)
    parts = res.particles or []
    # full per-step probe dists per particle (final one is usually an "echo" of the
    # trajectory's own committed answer; earlier ones are forced-early decisions)
    all_dists, dists = [], []
    for p in parts:
        pd = getattr(p, "probe_dists", None) or []
        pd = [({k: round(v, 4) for k, v in d.items()} if d else None) for d in pd]
        all_dists.append(pd)
        dists.append(next((d for d in reversed(pd) if d), None))
    row = {
        "unique_id": rec.unique_id, "gold": gold, "n": n,
        "pred_letters": letts, "final_weights": [round(w, 4) for w in fw],
        "steps_used": list(res.steps_used_lst),
        "probe_dists": dists,
        "probe_dists_all": all_dists,
        "texts_hash": [hash(t) % 10**8 for t in texts],
    }
    return row


def selectors_from_row(r):
    n = r["n"]; letts = r["pred_letters"]; fw = r["final_weights"]
    dists = r["probe_dists"]; th = r["texts_hash"]
    out = {}
    i_sel = max(range(n), key=lambda i: fw[i])
    out["argmax_w"] = letts[i_sel]
    tally = defaultdict(float)
    for L in letts:
        if L:
            tally[L] += 1
    out["majority"] = max(tally, key=tally.get) if tally else ""
    seen, uniq = set(), []
    for i in range(n):
        if th[i] not in seen:
            seen.add(th[i]); uniq.append(i)
    t2 = defaultdict(float)
    for i in uniq:
        if letts[i]:
            t2[letts[i]] += 1
    out["dedup_majority"] = max(t2, key=t2.get) if t2 else ""
    # marginal probe vote: sum letter masses over unique particles
    t3 = defaultdict(float)
    for i in uniq:
        if dists[i]:
            for L, m in dists[i].items():
                t3[L] += m
    out["probe_marginal"] = max(t3, key=t3.get) if t3 else out["dedup_majority"]
    # marginal over ALL particles (resampling-weighted marginal)
    t4 = defaultdict(float)
    for i in range(n):
        if dists[i]:
            for L, m in dists[i].items():
                t4[L] += m
    out["probe_marginal_all"] = max(t4, key=t4.get) if t4 else out["majority"]
    # pre-stop and mean-over-steps marginal votes (dedup'd), using the full lists
    ad = r.get("probe_dists_all") or [[] for _ in range(n)]
    t5, t6 = defaultdict(float), defaultdict(float)
    for i in uniq:
        pd = [d for d in (ad[i] if i < len(ad) else []) if d]
        if not pd:
            continue
        pre = pd[-2] if len(pd) >= 2 else pd[-1]
        for L, m in pre.items():
            t5[L] += m
        for d in pd:
            for L, m in d.items():
                t6[L] += m / len(pd)
    out["probe_marg_prestop"] = max(t5, key=t5.get) if t5 else out["dedup_majority"]
    out["probe_marg_mean"] = max(t6, key=t6.get) if t6 else out["dedup_majority"]
    # entropy-certainty marginal (user idea, 2026-08-27): normalize each unique
    # particle's final probe dist, damp by certainty c = 1 - H/log(K) — a
    # confused particle (mass spread over letters) counts less in the vote
    t7 = defaultdict(float)
    for i in uniq:
        pd = [d for d in (ad[i] if i < len(ad) else []) if d and sum(d.values()) > 0]
        if not pd:
            if letts[i]:
                t7[letts[i]] += 1.0
            continue
        d = pd[-1]; tot = sum(d.values())
        p = {k: v / tot for k, v in d.items()}
        H = -sum(v * math.log(v) for v in p.values() if v > 1e-12)
        c = 1.0 - H / math.log(max(2, len(p)))
        for L, v in p.items():
            t7[L] += c * v
    out["ent_marg"] = max(t7, key=t7.get) if t7 else out["dedup_majority"]
    # argmax probe-confidence particle -> its text letter
    conf = [max(d.values()) if d else -1 for d in dists]
    if max(conf) > -1:
        j = max(range(n), key=lambda i: conf[i])
        out["probe_argmax_text"] = letts[j]
        out["probe_argmax_letter"] = max(dists[j], key=dists[j].get)
    else:
        out["probe_argmax_text"] = out["argmax_w"]
        out["probe_argmax_letter"] = out["argmax_w"]
    return out


async def run(a):
    from its_hub.core.algorithms.particle_filtering import Particle
    Particle.deepcopy = patched_deepcopy

    random.seed(a.seed); np.random.seed(a.seed % 2**32)
    recs = load_records(a.bench, a.limit, audio_root=a.audio_root)
    print(f"{len(recs)} records ({a.bench})", flush=True)
    lm = OpenAICompatibleLanguageModel(
        endpoint=a.endpoint, api_key="NO_API_KEY", model_name=a.model_name,
        max_tokens=300, max_concurrency=-1)

    os.makedirs(os.path.dirname(os.path.abspath(a.jsonl)), exist_ok=True)
    done = set()
    if os.path.exists(a.jsonl):
        with open(a.jsonl) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if not r.get("error"):
                        done.add((r["unique_id"], r["arm"], r["budget"]))
                except Exception:
                    pass
    out = open(a.jsonl, "a")
    lock = asyncio.Lock()

    async def one(arm, budget, rec, sem):
        async with sem:
            t0 = time.time()
            try:
                # per-item algorithm instance: the probe's valid-letter set depends
                # on this record's choice count, and instances are cheap
                alg = build_arm(arm, a.temp, a.max_steps, len(rec.choices),
                                step_token=a.step_token.replace("\\n", "\n"),
                                probe_suffix=a.probe_suffix,
                                probe_max_tokens=a.probe_max_tokens)
                msgs, _ = build(a.prompt_method, rec, audio_mode="local-path")
                res = await alg.ainfer(lm, msgs, budget, return_response_only=False,
                                       record_trace=a.store_texts)
                gold = LETTERS[rec.answer_index]
                row = item_report(rec, res, gold)
                if a.store_texts:
                    # full final texts + the per-iteration trace: every particle's
                    # newly generated step BEFORE each resample — pruned lineages
                    # included — plus resample parents/probabilities
                    row["texts"] = [p.get("content", "") for p in res.responses]
                    row["trace"] = res.trace
                row.update({"arm": arm, "budget": budget, "error": None,
                            "latency_s": round(time.time() - t0, 1)})
            except Exception as e:
                row = {"unique_id": rec.unique_id, "arm": arm, "budget": budget,
                       "error": f"{type(e).__name__}: {e}"}
        async with lock:
            out.write(json.dumps(row) + "\n"); out.flush()
        return row

    for arm in a.arms.split(","):
        for budget in [int(b) for b in a.budgets.split(",")]:
            todo = [r for r in recs if (r.unique_id, arm, budget) not in done]
            per = max(1, a.max_inflight // budget)
            sem = asyncio.Semaphore(per)
            print(f"[{arm} b{budget}] {len(todo)} to do, {per} concurrent items", flush=True)
            t0 = time.time()
            rows = await asyncio.gather(*(one(arm, budget, r, sem) for r in todo))
            errs = sum(1 for r in rows if r.get("error"))
            print(f"  done in {time.time()-t0:.0f}s ({errs} errors)", flush=True)
    out.close()
    await lm.close()

    # report
    rows = []
    with open(a.jsonl) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if not r.get("error"):
                rows.append(r)
    by = {}
    for r in rows:
        by[(r["unique_id"], r["arm"], r["budget"])] = r
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in by.values():
        sels = selectors_from_row(r)
        for k, v in sels.items():
            c = agg[(r["arm"], r["budget"])][k]
            c[1] += 1
            if v == r["gold"]:
                c[0] += 1
    cols = ["argmax_w", "majority", "dedup_majority", "probe_marginal",
            "probe_marginal_all", "probe_marg_prestop", "probe_marg_mean",
            "probe_argmax_text", "probe_argmax_letter"]
    print(f"\n{'arm':16s} {'bud':>4} " + " ".join(f"{k:>19}" for k in cols))
    for (arm, bud), kv in sorted(agg.items()):
        line = f"{arm:16s} {bud:>4} "
        for k in cols:
            c, nn = kv.get(k, [0, 0])
            line += f"{(c/nn if nn else 0):>19.4f}"
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:8500/v1")
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--bench", default="mmar")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--arms", default="base_epf,epf_probe,probe_adaptive,sc_probe")
    ap.add_argument("--budgets", default="8")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--prompt-method", type=int, default=4,
                    help="CoT prompt id from benchmarking.mmau_pro.prompt.METHODS (4=plan-and-solve, 5=least-to-most, 9=evidence-grounded boxed)")
    ap.add_argument("--step-token", default="\\n\\n",
                    help="step boundary; pass \\n for line-per-step prompts like P9")
    ap.add_argument("--probe-suffix", default="Answer:",
                    help=r"assistant-prefill suffix for the answer probe; for P9 use 'Final Answer: \boxed{'")
    ap.add_argument("--probe-max-tokens", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--audio-root", default=None,
                    help="re-root relative audio paths (TTA views), e.g. "
                         ".../epf_data/deepdive_views/noise")
    ap.add_argument("--store-texts", action="store_true",
                    help="store full particle texts + per-iteration trace "
                         "(pruned lineages included) in each row")
    ap.add_argument("--max-inflight", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--jsonl", required=True)
    a = ap.parse_args()
    asyncio.run(run(a))


if __name__ == "__main__":
    main()
