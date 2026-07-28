# mmar-judge results

BoN and Beam Search on MMAR with **Qwen2.5-Omni-7B as an audio LLM judge**
(replacing the paper's math PRM, which cannot hear audio). All results,
inferences, and conclusions for this experiment line live here.

## Protocol

- **Dataset**: MMAR, 1000 single-audio MCQ items; 996 gradeable under the letter
  scorer (`full` subset). `le30s` subset (983 items) for Qwen2-Audio and Kimi-Audio.
  Data root: `/home/exx/inference-time-scaling/mmar`.
- **Judge**: Qwen2.5-Omni-7B (`ae9e1690`, vLLM 0.22.1, GPU1 :8110, bf16,
  enforce-eager, max-model-len 32768). Judge input per call: the item's audio clip
  (file:// content part) + question + lettered options + candidate text. Strict-JSON
  output `{"score": 0-10, "reasoning": ...}` at temperature 0; scores normalized to
  [0,1]; parse failures fall back to 5/10 (tracked as `parse_ok=false`).
- **Rubrics** (verbatim in `benchmarking/mmar_judge/judge.py`):
  - FINAL (BoN): "score how likely the candidate's final choice is the CORRECT
    option ... judge only against the audio evidence".
  - STEP (Beam, every level): "score how promising the PARTIAL reasoning is so far
    ... do not penalize it for being incomplete".
- **Algorithms** (its_hub, `mmar-judge` branch): BestOfN (budget = N candidates,
  temperature 0.8, prompt P4 plan-and-solve, judge-argmax selection; identical
  candidates dedupe before judging) and BeamSearch (beam width 4, step token
  `\n\n`, stop `Answer:`, max 6 steps; judge scores every partial chain with
  audio). Beam b=4 is skipped by design: with width 4 it is a single non-branching
  chain (num_beams = 1) — BoN b=1 is the shared greedy baseline.
- **Budgets**: local BoN {1,4,8}, Beam {8}; collaborator grid up to 128
  (`configs/config_collab.sh`).
- **Metrics**: `sel_acc` (algorithm's pick), `oracle` (any candidate correct),
  `major` (majority vote), judge parse rate, mean selected-judge-score,
  per-modality (sound/music/speech/mix) breakdown. Denominator = gradeable items.
- **Known caveats**: (1) self-judging bias for the Omni policy run — judge and
  policy share weights; cross-model runs are the mitigation. (2) Mellow (167M)
  is expected to follow P4/`Answer:` poorly. (3) Judge score ties break by
  candidate order in BestOfN argmax — tie rate tracked from the JSONL traces.

## Runs

<!-- One section per run; tables generated from results/<run>/{bon,beam}.log -->

### smoke10_qwen-omni (plumbing gate)

_pending_

### smoke100_qwen-omni (sanity gate vs Run 1 EPF anchors)

_pending_

### full_qwen-omni

_pending_

## Judge-quality analysis

_pending — from JSONL traces: score distribution, parse rate, tie rate,
mean judge score of correct vs incorrect candidates, step-score trajectories._

## Inferences

_pending_

## Conclusions

_pending_
