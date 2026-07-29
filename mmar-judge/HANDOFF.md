# HANDOFF — what was done on this branch and why

This document explains the `mmar-judge` branch end-to-end for someone seeing it
for the first time. Companion docs: [README.md](README.md) (how to run),
[results.md](results.md) (numbers and analysis).

## 1. Goal

Run two inference-time-scaling baselines from the its_hub library — **Best-of-N**
and **Beam Search** — on the **MMAR audio benchmark** (1,000 multiple-choice
questions about audio clips; 996 gradeable). Both algorithms need a reward model
to score candidates, and the library's stock reward model is a **math** PRM
(Qwen2.5-Math-PRM-7B) that cannot hear audio. We replaced it with
**Qwen2.5-Omni-7B served via vLLM as an audio-capable LLM judge**: for every
scoring call the judge receives the item's actual audio clip, the question, the
lettered options, and the candidate text, and returns a score. For Beam Search
the judge scores **partial reasoning chains at every step** (with the audio),
not just final answers.

Prior MMAR runs (1–6, see `benchmarking/mmar/results/`) used entropic particle
filtering with the model's own self-certainty as the weight. This branch is the
first MMAR experiment with an **external** reward signal.

## 2. What was built

### Branch strategy
`v1` had BestOfN/BeamSearch/LLMJudge but no MMAR harness; the `prm-judge`
branch had the harness but had stripped those algorithms out of its_hub. This
branch (`mmar-judge`, cut from `v1`) combines them: the harness files
(loaders, scoring, audio helpers, serve scripts) were restored file-by-file
from `prm-judge`; the algorithms come from `v1` unchanged except for the audio
patches below.

### its_hub changes (all backward-compatible; full test suite passes)
- `its_hub/api/types.py` — non-text content parts (audio) no longer raise;
  added `has_nontext_content()` / `base_user_messages()` helpers.
- `its_hub/core/lms/step_generation.py` — new optional `base_messages`
  parameter so an audio-carrying user turn reaches the model at every
  step-generation call (previously the prompt was flattened to text).
- `its_hub/core/algorithms/beam_search.py` — when the prompt contains audio,
  the original messages are carried to both the generator and the reward
  model instead of the flattened string.
- `its_hub/core/lms/openai_lm.py` + `orchestrator.py` — optional
  `logprobs`/`top_logprobs` passthrough (needed by the probability-based judge
  scorers); responses carry `_logprobs`.

### New code
- `benchmarking/mmar_judge/judge.py` — the judge. `MMARAudioJudge` is bound to
  one MMAR item and builds judge requests as [system rubric, user(audio +
  question + options + candidate)]. Two adapters plug it into the algorithms:
  `AudioJudgeORM` (Best-of-N, scores final answers) and `AudioJudgePRM`
  (Beam Search, scores partial chains). **Three scoring modes**:
  - `rubric` — judge writes JSON `{"score": 0-100, "reasoning": ...}`
  - `pyes` — judge answers Yes/No; score = P(Yes)/(P(Yes)+P(No)) from token
    logprobs (GenRM-style)
  - `geval` — judge writes a digit 0–9; score = expected digit Σp(s)·s from
    token logprobs (G-Eval-style)
- `benchmarking/mmar_judge/run_judge.py` — resumable runner (JSONL source of
  truth keyed (item, alg, budget); error rows retried on rerun; CSV compatible
  with the official-scorer crosscheck; trend report in the .log).
- `benchmarking/mmar_judge/rejudge.py` — offline re-scorer: re-judges the
  stored candidates of a previous run under a different scoring mode, so scorer
  comparisons are candidate-controlled.
- `mmar-judge/` — configs (budgets are knobs: local {1,4,8}; collab up to 128),
  serve scripts (judge GPU1:8110, policy GPU0:8100), per-model configs for all
  five policy models (Omni, Qwen2-Audio, Phi-4-MM speech-LoRA, Kimi via the
  required run19 wrapper, Mellow via its shim), and `collab_run.sh` — the
  one-command driver.

## 3. Validation so far (all on Qwen2.5-Omni-7B as both policy and judge)

Gates passed in order: data gate (1000/996), capacity audit (longest clip ≈
1.4K tokens vs 32K context), a manual judge call proving the judge really hears
the clip (it named the audio content unprompted and zeroed a fabricated claim),
a 10-item plumbing smoke, and a 100-item stratified smoke. Zero errors and 100%
judge-JSON parse rate throughout (~6K judge calls). An adversarial code review
confirmed one real bug (unbounded Best-of-N request fan-out at high budgets)
which is fixed.

### Headline numbers (100 stratified items, prompt P4, policy temp 0.8)

| alg  | budget | selected | oracle | majority |
|------|--------|----------|--------|----------|
| BoN  | 1      | 0.54     | 0.54   | 0.54     |
| BoN  | 4      | 0.53     | 0.76   | 0.56     |
| BoN  | 8      | 0.54     | 0.88   | 0.54     |
| Beam | 8      | 0.52     | 0.66   | 0.53     |

### The three findings that matter

1. **The judge selects no better than greedy.** Oracle accuracy scales
   beautifully with budget (a correct answer usually exists among the
   candidates) but judge-argmax selection is flat at ≈0.53 — the
   "selector is the bottleneck" wall from the earlier EPF runs, now
   replicated with an external audio judge.
2. **Score clustering was a symptom, not the disease.** The 0-100 verbalized
   scores piled up at 85/90/95, causing argmax ties on ~72% of items (23–28%
   with tied candidates disagreeing → order-based picks). We implemented the
   two literature fixes — P(Yes) and G-Eval expected score from logprobs — and
   re-judged the *same* stored candidates: ties collapsed to 0–1 per 100, but
   **accuracy did not move** (0.51–0.56 across all scorers). Pooled AUROC of
   the judge's scores vs candidate correctness is only **0.56–0.59**: the
   judge's ranking signal itself is barely above chance. See results.md
   "Scorer ablation".
3. **Beam Search demonstrates the early-pruning failure.** At the same budget
   8, beam's oracle is 0.66 vs BoN's 0.88: the judge's noisy step scores prune
   correct paths mid-search and cloning collapses diversity (only 3.0 distinct
   final candidates of 8, and the final argmax was a full tie on 100/100
   items). This is a live demonstration of the failure mode the Rollout
   Roulette paper (arXiv:2502.01618) uses to motivate particle filtering.

## 4. Current status and open work

- Done: all plumbing, validation, and the scorer ablation above; results in
  `results.md`; smoke artifacts in `results/smoke10_qwen-omni/` and
  `results/smoke100_qwen-omni/`.
- NOT yet run: the **full 996-item runs** (any model) and the other four
  policy models. `collab_run.sh <model config>` does a full model end-to-end.
- Known caveats: self-judging bias for the Omni cell (judge = policy weights);
  Mellow is a 167M model and won't follow the CoT format well; beam at b=128
  costs ~770 audio judge calls per item (see config_collab.sh warning).
- Candidate next steps discussed: listwise/pairwise comparative judging
  (changes the judgment task, not the score format), a larger judge
  (Qwen3-Omni-30B), judge ensembling, or locking in the negative result with
  full-set runs.
