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

### smoke10_qwen-omni (plumbing gate — PASS, 2026-07-28)

BoN b4 + Beam b8 on 10 stratified items, 0 errors, 350 judge calls, 100% JSON
parse rate. Manual sanity call: the judge's reasoning named the audio content
unprompted ("the audio contains a bird call"), ranked a correct candidate above
a wrong one, and gave 0/10 to a fabricated "I hear music" step on a bird clip —
the judge verifiably hears the clip. Capacity audit (audit_judge.json): longest
30 clips ≤ 1443 prompt tokens vs 32768 context, 25.1 audio tok/s — AUDIT PASS.

**Rubric escalation**: the original 0-10 rubric produced argmax ties on 8/10
items (7/10 with tied-top candidates disagreeing on the letter → order-based
pick). Switched to 0-100 with anti-round-number instructions: ties 6/10,
harmful 5/10 on the same items. All later runs use 0-100.

### smoke100_qwen-omni (sanity gate — PASS, 2026-07-28)

100 stratified items (seed 0), policy = judge = Qwen2.5-Omni-7B (self-judging,
disclosed), P4, temp 0.8 policy / 0.0 judge, 0-100 rubric. 400 rows, 0 errors,
3,466 judge calls, 100% parse.

| alg  | b | sel_acc | sel_tb | oracle | major | notes |
|------|---|---------|--------|--------|-------|-------|
| bon  | 1 | 0.540 | 0.540 | 0.540 | 0.540 | greedy baseline (no judge calls) |
| bon  | 4 | 0.530 | 0.560 | 0.760 | 0.560 | |
| bon  | 8 | 0.540 | 0.540 | 0.880 | 0.540 | |
| beam | 8 | 0.520 | 0.530 | 0.660 | 0.530 | distinct 0.381 (beam collapses diversity) |

- b1 = 0.540 is consistent with the Run 1 EPF anchor (b1 sel ≈ 0.55-0.60 on the
  full set). Sanity band met; judge selection is NOT anti-correlated (gate: pass).
- **Oracle scales (0.54 → 0.76 → 0.88) but judge selection is flat (~0.53-0.54)**
  — the external audio judge replicates the selector-is-the-bottleneck verdict
  from the EPF self-certainty runs, on its first outing.
- Judge discrimination exists but is weak: mean score of correct candidates
  0.846 vs wrong 0.774 (b4); 0.838 vs 0.789 (b8).
- Ties at n=100: 72-74% raw, but only 23-28% harmful (tied-top disagreeing);
  majority-among-tied (sel_tb) recovers +3pp at b4 (0.56, = majority voting).
- Beam vs BoN at the same b=8: beam oracle 0.66 vs BoN 0.88 — step-level
  pruning discards diversity (distinct 0.381 vs 1.0), the classic early-pruning
  failure the Rollout Roulette paper attributes to deterministic search.

### Scorer ablation: rubric vs P(Yes) vs G-Eval (offline re-judge, 2026-07-28)

Controlled comparison on the SAME stored smoke100 candidates (b4: 400, b8: 800;
`rejudge.py`): only the judge's scoring rule varies. pyes = P(Yes)/(P(Yes)+P(No))
from first-token logprobs (GenRM-style); geval = E[digit]/9 over 0-9 token
probabilities (G-Eval-style). Both 1 generated token, 100% parse, ~98%+ answer mass.

| scorer | b | sel_acc | sel_tb | major | oracle | ties/100 | harmful | AUROC (pooled) |
|--------|---|---------|--------|-------|--------|----------|---------|-------|
| rubric | 4 | 0.53 | 0.56 | 0.56 | 0.76 | 72 | 23 | 0.573 |
| rubric | 8 | 0.54 | 0.54 | 0.54 | 0.88 | 74 | 28 | |
| pyes   | 4 | 0.56 | 0.56 | 0.56 | 0.76 | 5  | 1  | 0.559 |
| pyes   | 8 | 0.52 | 0.52 | 0.54 | 0.88 | 11 | 1  | |
| geval  | 4 | 0.53 | 0.53 | 0.56 | 0.76 | 0  | 0  | 0.591 |
| geval  | 8 | 0.51 | 0.51 | 0.54 | 0.88 | 0  | 0  | |

- **Clustering/ties: solved.** Distinct score values 15 → 208 (pyes) → 1039
  (geval); harmful ties 23-28/100 → 0-1/100. The mechanism worked exactly as the
  literature predicts.
- **Accuracy: unchanged** (0.51-0.56 across all scorers ≈ greedy 0.54, far below
  oracle). Removing discretization noise exposed the real bottleneck: the 7B
  Omni judge's per-candidate discrimination is barely above chance —
  **AUROC 0.56-0.59** (correct-vs-wrong candidates, n=1200). No re-scaling of a
  ~0.57-AUROC signal can select well; the score FORMAT was never the binding
  constraint.
- Escalation paths if selection must improve: comparative judging
  (listwise/pairwise — different judgment task, not a rescaling), a larger judge
  (Qwen3-Omni-30B is cached on this machine), or ensembling.

### full_qwen-omni

_pending_

## Judge-quality analysis

From smoke100 traces: 100% strict-JSON parse over 3,466 calls; final-score
distribution clusters high (85/90/95 favored) with full 0-95 range used on beam
steps; step scores do flag bad paths (0/20 tails). Judge temp 0.0. Remaining
weakness: top-cluster ties (see above) — candidate fixes if it matters at full
scale: pairwise/tournament judging, or ranking prompts instead of absolute scores.

## Inferences

_pending (full run)_

## Conclusions

_pending (full run)_
