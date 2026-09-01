# EPF Mechanics — exactly what the code computes, with line references

All paths are relative to `entropic-particle-filter/`. Line numbers are for the current
working tree (checked 2026-08-27). The configuration described is the one the epf2/epf3
sweeps actually ran (set in `benchmarking/mmau_pro/diversity_probe.py::build_epf`,
lines 200–242, and `mmar_epf/run_node.sh` env): P4 plan-and-solve prompt, temp 0.8,
step token `"\n\n"`, stop substring `"Answer:"`, `max_steps=6`, 300 tokens/step,
signal `mean_logprob` or `entropy`, style `logit`, **systematic** resampling,
temperature method **ESS** with `ess_threshold=0.6`, `early_phase=0.7`,
final selection **ARGMAX**.

The pipeline per round: generate step → summarize logprobs → transform to weight →
softmax (+ annealing) → resample → repeat → select. Each stage below.

---

## 1. Where the raw numbers come from (the API layer)

**Request.** When a step is generated with weighting enabled, the client adds
`logprobs: true` (and `top_logprobs: 20` only for the entropy signal) to the
chat-completions request — `its_hub/core/lms/openai_lm.py:250-254`. The knob arrives
there from `StepGeneration.aforward(..., return_logprobs=True, top_logprobs=...)`
(`its_hub/core/lms/step_generation.py:176-181`, batch call at `:242-251`), which is
invoked by the algorithm at `its_hub/core/algorithms/particle_filtering.py:223-232`
(`_apropagate`). `top_logprobs` defaults to `None` for the `mean_logprob` signal and
is forced to 20 for `entropy` (`particle_filtering.py:126-128`) — this is why
`entropy` is `null` in all `mean_logprob`-signal rows.

**Response.** vLLM returns OpenAI-shaped logprobs: for each generated token, its
logprob and (optionally) the top-20 alternatives. The client stashes the raw object on
the message as `message["_logprobs"]` (`openai_lm.py:320`).

**Important:** these are the logprobs of the tokens under the model's *final
distribution as served* — sampling temperature (0.8) affects *which* tokens got
sampled, but the recorded logprob values are the model's scores for those tokens as
returned by vLLM. No top-k/top-p is set by this client anywhere in the pipeline.

## 2. Raw logprobs → per-step summary

`summarize_step_logprobs` — `its_hub/core/utils.py:67-107`. For the ~N tokens of one
step (one paragraph):

- **`mean_logprob`** (`utils.py:92-97`): the arithmetic mean of the chosen tokens'
  logprobs, `c = (1/N) Σᵢ log p(tokᵢ)`. This is the "length-normalized" part — it is
  normalized **within the step only**, never across the whole trajectory. `c ≤ 0`;
  `exp(c)` is the geometric-mean per-token probability.
- **`entropy`** (`utils.py:99-106`): at each token position, take the returned top-20
  `(token, logprob)` pairs and compute `Hₜ = −Σ_top20 p·log p` (a truncated-support
  approximation of the full-vocabulary entropy); then average over positions,
  `H = (1/N) Σₜ Hₜ`. Units: nats. `None` when `top_logprobs` wasn't requested.
- **`num_tokens`**: N. If N = 0 the summary is `{mean_logprob: 0.0, entropy: None,
  num_tokens: 0}` (`utils.py:88-95`) — see the neutral-weight guard below.

So per step, per particle, the stored *raw* signals are exactly:
`{mean_logprob, entropy, num_tokens}` (persisted verbatim into
`Particle.partial_signals` at `particle_filtering.py:242`, and into the sweep JSONLs'
`particles[*].signals` by `diversity_probe.py:497-506`).

## 3. Summary → particle log-weight (we do NOT use raw numbers)

`_self_certainty_logweight` — `particle_filtering.py:145-194`. Two mappings on top of
the raw summary:

**(a) signal → confidence `c` in log-space** (`:181-189`):
- `mean_logprob` signal: `c = mean_logprob` (≤ 0).
- `entropy` signal: `c = −entropy` (≤ 0; falls back to `mean_logprob` if entropy is
  `None`). I.e. high entropy → low confidence.

**(b) confidence → log-weight, style `logit`** (`:191-194`):
```
s = exp(min(c, 0))            # geometric-mean per-token prob, in (0, 1]
w = _inv_sigmoid(s) = log(s / (1 − s))   # particle_filtering.py:64-68, clipped to [1e-7, 1−1e-7]
```
So the log-weight is the **log-odds of the per-token confidence**, not the logprob
itself. Worked values (mean_logprob signal):

| step mean_logprob `c` | `s = eᶜ` | weight `w = logit(s)` |
|---|---|---|
| −2.0  | 0.135 | −1.85 |
| −1.0  | 0.368 | −0.54 |
| −0.5  | 0.607 | +0.43 |
| −0.2  | 0.819 | +1.51 |
| −0.05 | 0.951 | +2.97 |
| −0.005| 0.995 | +5.30 |

Note the steepness near `c → 0`: the transform *amplifies* differences among very
confident steps. This is why 4-token `Answer: X` paragraphs (`c ≈ −0.03`) and
verbatim-repeat degenerate steps (`c ≈ −0.005`) dominate the weight scale, while
genuine reasoning paragraphs (`c ≈ −0.6…−1.2`) sit far below. (`style="raw"` would use
`c` directly — `:191-192` — but every sweep ran `logit`.)

**Guard rails:** if the endpoint returned no logprobs (`num_tokens == 0`) the weight is
forced to **0.0** (neutral), because the summarizer's `mean_logprob = 0.0` fallback
would otherwise read as *perfect* confidence and that particle would win every round
(`:170-179`). `uniform_weights=True` (ablation arms only) forces 0.0 always
(`:168-169`).

**Weights are per-step, not accumulated.** Each round appends the new step's `w` to
`Particle.partial_log_weights` (`:241`), and everything downstream reads
`Particle.log_weight` = **the most recent entry only** (`:47-52`; used at `:373` and
`:409`). There is no summation over the trajectory. (This is standard SMC bookkeeping —
resampling resets weights — but it means the *final selection* is decided by the last
step alone; see §7.)

**Stopped particles keep their final weight.** `_apropagate` skips particles whose
`is_stopped` is set (`:211-219`), so a particle that emitted `Answer:` at round 1
carries that round's weight into every later resampling round (comment at `:371-372`)
and can keep being duplicated.

## 4. Log-weights → resampling distribution

Every round (`ainfer` loop, `:358-376`): collect `log_weights = [p.log_weight for p]`
(`:373`) and convert to probabilities:

- **Plain PF** (`_weights_to_probabilities`, `:319-328`): untempered
  `p = softmax(w)` (`_softmax` at `:71-74`), and `_last_temperature = 1.0`.
- **EPF** overrides it (`:584-599`):
  ```
  p₀ = softmax(w)                                   # :587
  T  = _temperature_annealing(p₀, step, N)          # :588-590  (see §5)
  p  = softmax(w · (1/T))                           # :599  ← what the resampler uses
  ```
  Dividing log-weights by `T` is identical to `p₀^(1/T)` renormalized: `T > 1`
  flattens the distribution toward uniform; `T = 1` leaves it unchanged. `T` is
  recorded per round as `_last_temperature` (`:592`) and lands in the trace as
  `resample.temperature` (`:394`).

## 5. Annealing — exactly what it does to the temperature

`_temperature_annealing` — `particle_filtering.py:557-582`:

```
progress  = current_step / max_steps                 # :568   (max_steps = 6)
ess       = 1 / Σ pᵢ²                                # :505-520  (_effective_sample_size)
ess_ratio = ess / N                                  # :572
T = 1.0                                              # default: no tempering (:575)
if ess_ratio < ess_threshold and progress < early_phase:      # gate, :576  (0.6, 0.7)
    T = _temperature_ess(ess_ratio, progress)        # :577-579 (sweeps use ESS method)
```

`_temperature_ess` — `:541-555`:
```
T = max(1.0, (1 / ess_ratio) · (1 − progress))
```
Properties, with the sweep's `max_steps = 6`:
- **Severity term** `1/ess_ratio`: worse collapse → more flattening (ESS ratio 0.1 →
  raw 10× at progress 0).
- **Decay term** `(1 − progress)`: the same collapse gets a weaker response as the run
  advances — round 1 (progress ⅙) multiplies by 0.83, round 4 by 0.33.
- **Silent zone**: at `ess_ratio = 0.5, progress = 0.5`, `T = 2·0.5 = 1.0` → *no
  tempering even though the gate condition passed*. Moderate mid-run collapse is
  never treated.
- **Hard cutoff**: `progress ≥ 0.7` (round ≥ 5 of 6) → gate closed regardless.
- Alternatives not used by the sweeps: `_temperature_entropy` (`:531-539`,
  `T = max(1, 1/(H_norm + (1−H_norm)·progress))` with normalized weight-entropy from
  `_entropy_n`, `:483-503`) and `_temperature_base` (`:522-529`, `T = max(1,
  2 − progress)` — always-on schedule).

Annealing changes **only the resampling draw for that round**. It does not modify any
particle's stored weight, does not skip the resample, and the **final selection is
explicitly untempered** (`:408-410` recomputes plain `softmax(log_weights)`).

## 6. Resampling — the sampling-with-replacement step

Every round, unconditionally (`:400-406`): draw N children from the N parents
proportionally to `p`, then `deepcopy` each survivor (`Particle.deepcopy`, `:54-61`)
so duplicates evolve independently afterwards.

- **Systematic** (used by all EPF sweeps): `_resampling_systematic_indices`,
  `:267-286`. One uniform offset `u ~ U(0,1)`; N evenly spaced pointers
  `(i + u)/N`; walk the cumulative sum of `p` and take the parent whose cumsum bucket
  each pointer lands in. Consequences: a parent with `pᵢ ≥ k/N` is guaranteed
  ⌊N·pᵢ⌋ copies (low variance); a parent with `pᵢ ≈ 1/N` gets 0 or 1; **exactly
  uniform `p` reproduces every parent exactly once (identity — no deaths)**.
- **Multinomial** (PF default, ablations only): `random.choices` with replacement,
  `:288-291` — even uniform `p` kills ≈ 1/e of lineages per round by collision.
- RNG: numpy global for systematic, stdlib `random` for multinomial — seeded by the
  probe CLI per shard (`diversity_probe.py:439-444`).

The loop runs until **every** particle is stopped (`:358`), where "stopped" means the
step text contained the stop substring `"Answer:"` or the 6-step cap was hit
(`step_generation.py:83-93` for the regex/substring/repeat logic and `:211-212` for
the max-steps check; the substring test is why `"Final Answer:"` also stops a
trajectory, and why non-canonical terminators like `答案:` do not).

## 7. Final selection

`ainfer`, `:408-417`. From the **final, untempered** last-step weights:
- `ARGMAX` (all sweeps): `selected_index = argmax(log_weights)` (`:416-417`) — i.e.
  the particle whose **final step** (almost always the short `Answer: X` paragraph)
  had the highest logit-transformed per-token confidence. `ties` go to the first index
  (`np.argmax` behavior).
- `SAMPLE` (ablations): draw one particle ∝ softmax weights (`:412-415`).

The returned object (`ParticleFilteringResult`, `:19-35`) carries every particle's
final text (`responses`), `log_weights_lst` (final per-particle last-step weights),
`selected_index`, `steps_used_lst`, the `Particle` objects (with `partial_signals` =
raw summaries and `partial_log_weights` = transformed weights per step), and, with
`record_trace=True`, the per-round trace: pre-resample snapshots
(`new_text`, `stopped`, `cum_log_weight` — despite the name, this is the *current*
i.e. most-recent log-weight, `:381-388`), post-tempering `probabilities`,
`temperature`, and resample `parents` (`:389-399`). The `majority` and `oracle`
numbers in the sweep CSVs are computed *outside* the algorithm by the probe
(`diversity_probe.py::compute_metrics`, `:76-136`), by parsing each particle's text
with `benchmarking/mmau_pro/scoring.py::predicted_index` (`:89-106`).

---

## 8. One round, end to end (numeric example, N = 4, mean_logprob/logit)

Step texts come back with mean logprobs `c = (−0.90, −0.55, −0.30, −0.04)`
(a long grounded paragraph → two mid paragraphs → a terse `Answer: C`):

1. `s = eᶜ = (0.407, 0.577, 0.741, 0.961)`
2. `w = logit(s) = (−0.38, +0.31, +1.05, +3.20)`
3. `p₀ = softmax(w) = (0.023, 0.046, 0.096, 0.835)`; `ESS = 1/Σp² = 1.39`,
   `ess_ratio = 0.35`
4. Round 1 of 6 → `progress = 1/6 = 0.167 < 0.7` and `0.35 < 0.6` → annealing fires:
   `T = max(1, (1/0.35)·(1−0.167)) = 2.38`
5. Resampling distribution `softmax(w/2.38) = (0.13, 0.17, 0.23, 0.47)` — the terse
   particle's expected copies drop from 3.3 to 1.9
6. Systematic draw with one uniform offset → e.g. parents `(1, 2, 3, 3)`: particle 0
   (the most-grounded, least "confident" one) is deleted; particle 3 is duplicated.
7. Copies are deep-copied and generate round 2 independently; each new step's weight
   **replaces** the old one for the next resample.

That single worked round contains every measured pathology in miniature: the logit
transform's steep end hands the terse step a 0.84 softmax share, annealing softens but
does not cancel it, and the deletion of the grounded particle is irreversible.
