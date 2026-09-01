"""Answer-probe particle filtering — the validated EPF repair for audio MCQ.

Three changes versus :class:`EntropicParticleFiltering`, each validated on
MMAR/MMAU-Pro full sets (Qwen2.5-Omni 3B/7B, Qwen2-Audio):

1. **Answer-probe weight** (replaces self-certainty). After every generation
   step, each still-active particle is probed: its trajectory-so-far plus the
   probe suffix (default ``"Answer:"``) is sent as an assistant prefill and the
   model's next-token distribution is read (temperature 0, top-20 logprobs).
   The probability mass on valid choice letters gives ``pmax``; the particle's
   log-weight becomes ``logit(pmax) / probe_temp``. ``probe_temp=3`` is
   essential: untempered probe logits let finished particles' near-1.0 echo
   dominate resampling and collapse swarm diversity.

2. **ESS-gated resampling** (replaces resample-every-step). Systematic
   resampling fires only when ``ESS / n_particles < adaptive_ess`` (default
   0.5); otherwise the step is a no-op. Fires on ~4-12% of rounds in practice,
   preserving self-consistency-like diversity.

3. **Marginal vote selection** (replaces argmax-weight). The answer is a
   marginal vote over deduplicated final texts using the stored probe
   distributions — see :func:`answer_vote`, which returns the three committed
   selectors (``probe_marg_mean``, ``probe_marginal``, ``ent_marg``). Report
   all three; regime guide: ``probe_marg_mean`` for budgets <= 16,
   ``ent_marg`` for large budgets (7B, Qwen2-Audio), ``probe_marginal`` for
   large budgets (3B, smallest worst-case deficit).

Because the vote is computed from the returned particles, callers should use
``ainfer(..., return_response_only=False)`` and pass the result to
:func:`answer_vote`. Instantiate per item when ``n_choices`` varies.
"""

import math
import string
from collections import defaultdict

import numpy as np

from its_hub.core.algorithms.particle_filtering import (
    EntropicParticleFiltering,
    ParticleFilteringResult,
    _inv_sigmoid,
)

LETTERS = string.ascii_uppercase
_PROBE_TOKEN_STRIP = ")>].,:;!?*'\"( "


def letter_masses(logprobs_obj: dict | None, valid: str) -> dict | None:
    """From an OpenAI-style logprobs object, find the first token position that
    carries choice-letter mass and return ``{letter: prob}`` summed over token
    variants (``"A"``, ``" A"``, ``"(A"`` ... all fold into ``A``)."""
    content = (logprobs_obj or {}).get("content") or []
    for pos in content:
        tops = pos.get("top_logprobs") or []
        masses: dict[str, float] = defaultdict(float)
        for e in tops:
            tok = (e.get("token") or "").strip(_PROBE_TOKEN_STRIP)
            if len(tok) == 1 and tok.upper() in valid:
                masses[tok.upper()] += math.exp(e.get("logprob", -100.0))
        if masses:
            return dict(masses)
    return None


class AnswerProbeParticleFiltering(EntropicParticleFiltering):
    """EPF whose per-step particle weight comes from an answer probe.

    Args (beyond :class:`EntropicParticleFiltering`):
        n_choices: number of MCQ options for THIS item (valid letters A..).
        adaptive_ess: resample only when ``ESS/N`` falls below this fraction
            (``None`` restores shipped resample-every-step; 0.0 never resamples).
        probe_temp: divisor on the probe log-odds weight; >1 flattens.
        probe_suffix: assistant prefill that forces an early answer.
        probe_max_tokens: decode length of each probe call.
    """

    def __init__(self, *args, n_choices: int = 4, adaptive_ess: float | None = 0.5,
                 probe_temp: float = 3.0, probe_suffix: str = "Answer:",
                 probe_max_tokens: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.valid = LETTERS[: max(1, n_choices)]
        self.adaptive_ess = adaptive_ess
        self.probe_temp = probe_temp
        self.probe_suffix = probe_suffix
        self.probe_max_tokens = probe_max_tokens

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
                # trajectory committed via the generic marker without the custom
                # suffix — cut there and re-append the full probe suffix
                text = text[: text.find("Answer:")]
                cut = -1
            prefix = text[: cut + len(sfx)] if cut >= 0 else (
                text + ("" if text.endswith("\n") else "\n") + sfx)
            msgs = [*base_messages, {"role": "assistant", "content": prefix}]
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
        dists = await self._probe(
            lm, particles, base_messages or [],
            skip={i for i, w in enumerate(was_stopped) if w},
        )
        for i, p in enumerate(particles):
            if was_stopped[i]:
                continue  # keep the weight from the round it stopped
            d = dists.get(i)
            p.probe_dists.append(d)
            if d:
                pmax = max(d.values())
                w = _inv_sigmoid(min(max(pmax, 1e-6), 1 - 1e-6)) / self.probe_temp
            else:
                w = 0.0  # neutral when the probe found no letter mass
            p.partial_log_weights[-1] = float(w)
        return particles

    def _resampling_indices(self, probabilities, num_particles):
        if self.adaptive_ess is not None:
            p = np.asarray(probabilities)
            ess = 1.0 / float(np.sum(np.clip(p, 1e-12, 1) ** 2))
            if ess / num_particles >= self.adaptive_ess:
                return list(range(num_particles))
        return super()._resampling_indices(probabilities, num_particles)

    def _resampling(self, particles, probabilities, num_particles):
        idx = self._resampling_indices(probabilities, num_particles)
        return [particles[i] for i in idx]


def answer_vote(result: ParticleFilteringResult,
                parsed_letters: list[str] | None = None) -> dict[str, str]:
    """The three committed selectors over a finished run's particles.

    Returns ``{"probe_marg_mean": L, "probe_marginal": L, "ent_marg": L}``.
    ALWAYS report all three accuracies side by side (project convention).

    Args:
        result: from ``ainfer(..., return_response_only=False)``.
        parsed_letters: optional per-particle letter parsed from the final text,
            used as a 1.0-vote fallback for particles without any probe dist.
    """
    parts = result.particles or []
    texts = [r.get("content", "") for r in result.responses]
    n = len(parts)
    letts = parsed_letters or [""] * n
    seen, uniq = set(), []
    for i in range(n):
        h = hash(texts[i])
        if h not in seen:
            seen.add(h)
            uniq.append(i)

    t_mean, t_marg, t_ent = (defaultdict(float) for _ in range(3))
    fallback = defaultdict(float)
    for i in uniq:
        pd = [d for d in parts[i].probe_dists if d and sum(d.values()) > 0]
        if not pd:
            if letts[i]:
                for t in (t_mean, t_marg, t_ent):
                    t[letts[i]] += 1.0
            continue
        # probe_marg_mean: mean of ALL per-step dists (raw masses)
        for d in pd:
            for letter, m in d.items():
                t_mean[letter] += m / len(pd)
        # probe_marginal: final dist, raw masses
        for letter, m in pd[-1].items():
            t_marg[letter] += m
        # ent_marg: final dist normalized, damped by certainty 1 - H/log(K)
        tot = sum(pd[-1].values())
        p = {k: v / tot for k, v in pd[-1].items()}
        ent = -sum(v * math.log(v) for v in p.values() if v > 1e-12)
        c = 1.0 - ent / math.log(max(2, len(p)))
        for letter, v in p.items():
            t_ent[letter] += c * v
        if letts[i]:
            fallback[letts[i]] += 1.0

    def pick(t):
        if t:
            return max(t, key=t.get)
        return max(fallback, key=fallback.get) if fallback else ""

    return {"probe_marg_mean": pick(t_mean), "probe_marginal": pick(t_marg),
            "ent_marg": pick(t_ent)}
