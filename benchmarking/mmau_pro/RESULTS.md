# MMAU‑Pro × Inference‑Time Scaling (PF/EPF with generator self‑certainty) — Run Log & Results

**Question.** On a *fixed* Qwen2.5‑Omni‑7B, can Particle Filtering (PF) / Entropic PF (EPF) —
with particle weights taken from the **generator's own token log‑probabilities / entropy
("self‑certainty"), no external reward model** — improve MMAU‑Pro multiple‑choice accuracy
(the RL‑free alternative to GRPO)?

**TL;DR (full 957‑MCQ result).** With a good chain‑of‑thought prompt (#4 plan‑and‑solve),
PF@4 / EPF@4 reach **0.576 / 0.575 vs a 0.534 baseline — about +4 pp, borderline‑significant
(z≈1.8, p≈0.07)**. With the other prompt (#2) ITS is neutral. The gain is **small,
prompt‑dependent, concentrated in hard perceptual categories** (spatial audio, sound, speech),
and **does not scale with budget** — consistent with self‑certainty being a weak (fluency)
reward signal. There is **no GRPO result on MMAU‑Pro** to compare against (see caveats).

Last updated: 2026‑07‑10. (Runs 1–11: Qwen2.5‑Omni‑7B. **Runs 12–13 (§18–19): Audio Flamingo 3** — porting, adaptation sub‑analyses, and the EPF grid replication on a second model. **Run 14 (§20): Gemma 4 E2B IT** on the ≤30 s subset — in progress.)

---

## 1. Setup

- **Model / serving:** Qwen2.5‑Omni‑7B (thinker, text‑out) via vLLM 0.22.1, OpenAI‑compatible, port 8100.
  - Required env/flags discovered: **`VLLM_USE_FLASHINFER_SAMPLER=0`** (flashinfer JIT sampler fails to build on Blackwell sm_120); installed server‑side audio decoders **`librosa soundfile av resampy`** (`av`/PyAV is what vLLM falls back to; without it → "install vllm[audio]" / "Invalid audio file"); `--allowed-local-media-path <data>` + `--limit-mm-per-prompt '{"audio":3}'`; `HF_HOME` on the 3.4 TB volume.
  - Audio input: both `input_audio` (base64) and `audio_url` (`file://`, "local‑path") work; **local‑path used for the 957 run** (avoids re‑sending ~26 MB base64 each PF step).
- **Particle weight:** generator self‑certainty (now the only weight source in its_hub — the former `weight_source=` kwarg is gone). PF uses signal `mean_logprob`, style `logit`; EPF uses signal `entropy`. Reasoning is chunked into PF/EPF steps on `\n\n` (`StepGeneration(step_token="\n\n", stop_token="Answer:", max_steps=6)`), `max_tokens_per_step=300`.
- **Data:** `mmau_pro_testmini` — 957 MCQ items (non‑empty `choices`; `answer` is the choice *text*). `le30s` split = 411 MCQ (≤30 s audio) used for dev. Scoring: lettered choices A–K, parse `Answer: <letter>`, normalized/fuzzy match to gold (5 items ungradeable → excluded).
- **Arms:** `baseline` = budget 1 (single trajectory, no resampling); `pf` / `epf` = budgets as noted.
- **Harness:** `benchmarking/mmau_pro/` (`run_mmau.py`, `prompt.py`, `scoring.py`, `loader.py`, `audio.py`, `cot_compare.py`, `phase0_gate.py`, `ab_causality.py`). Raw outputs in `benchmarking/mmau_pro/results/`. Client‑side deps: `pip install its_hub[benchmark]` (click, pandas, pyarrow on top of the `[lm]` extra).

## 2. Validation (does the pipeline actually work on audio?)

- **Phase‑0 gate 1 — logprobs WITH audio input: PASS.** vLLM returns generated‑token `logprobs`/`top_logprobs` even with audio in the prompt → self‑certainty is viable, no fallback needed.
- **Phase‑0 gate 2 — `continue_final_message` WITH an audio user turn: PASS.** The model continues a prefilled partial assistant turn → the step‑by‑step audio carry works.
- **A/B causality (does the model *hear* it?): PASS.** 15 items, greedy, audio‑present vs audio‑removed: **0.533 vs 0.400**, and **6/15 answers changed** — multi‑audio "which clip contains X" items are correct only with audio. The audio reaches the model.
- **Pipeline smoke (n=8):** baseline 0.500, PF@4 0.625, EPF@4 0.375 — 0 errors. (Also surfaced that the *original* terse prompt made the model answer in one letter; fixed by the CoT prompts below.)

## 3. Run 1 — 8‑prompt CoT comparison (which prompt makes the model reason, chunkably?)

20 single‑audio le30s items, one greedy generation each. `reasoned` = fraction with ≥15 words of reasoning; `chunks` = mean `\n\n`‑segments (PF step granularity).

| # | prompt | acc | reasoned | avg_words | avg_chunks |
|---|--------|----:|---------:|----------:|-----------:|
| 1 | assistant‑prefill CoT | 0.350 | 1.00 | 112 | 4.1 |
| 2 | zero‑shot CoT (user trigger) | 0.400 | 0.95 | 110 | 4.0 |
| 3 | few‑shot CoT | 0.300 | 0.90 | 28 | **1.0** |
| **4** | **plan‑and‑solve** | **0.500** | 1.00 | **138** | **5.2** |
| 5 | least‑to‑most | 0.450 | 1.00 | 42 | 2.1 |
| 6 | describe‑then‑reason (audio) | 0.350 | 1.00 | 93 | 3.0 |
| 7 | format‑forcing (## Step) | 0.400 | 1.00 | 104 | 3.6 |
| 8 | anti‑shortcut (≥3 steps) | 0.250 | 1.00 | 65 | 2.1 |

**Takeaways:** 7/8 prompts produce real, chunkable reasoning (so PF/EPF have steps to resample). Few‑shot (#3) collapsed to terse answers (1 chunk) → dropped. Carried forward: **#4 (best), #2, #7**.

### Run 1b — n=40 stratified re‑run + new prompt #9

Re‑run of the screen on **40 single‑audio le30s items stratified across `category`** (even round‑robin coverage — this is a *fresh* sample, **not** the smallest‑40 above, so it is not a direct extension of the n=20) plus a **9th prompt**: evidence‑grounded steps ending in `Final Answer: \boxed{LETTER}` (supplied verbatim; mapped as a system‑prompt CoT). Same metric defs, greedy, one generation/item. (Raw CSV/log removed as redundant after the full 957 bake-off — see Run 5.)

Category mix (40): `sound 5, voice_chat 5, spatial_audio 5, speech 5, open 5, music 5, sound_speech 5, sound_music 4, music_speech 1`.

| # | prompt | acc | reasoned | avg_words | avg_chunks |
|---|--------|----:|---------:|----------:|-----------:|
| 1 | assistant‑prefill CoT | 0.550 | 1.00 | 78.5 | 2.9 |
| 2 | zero‑shot CoT (user trigger) | 0.525 | 1.00 | 97.4 | 3.4 |
| 3 | few‑shot CoT | 0.475 | 0.88 | 26.3 | **1.0** |
| **4** | **plan‑and‑solve** | 0.575 | 1.00 | 101.6 | **4.3** |
| 5 | least‑to‑most | **0.625** | 0.90 | 31.9 | 2.0 |
| 6 | describe‑then‑reason (audio) | 0.600 | 1.00 | 70.5 | 2.5 |
| 7 | format‑forcing (## Step) | 0.550 | 1.00 | 92.7 | 3.6 |
| 8 | anti‑shortcut (≥3 steps) | 0.550 | 1.00 | 60.6 | 2.0 |
| 9 | evidence‑grounded (boxed) | 0.500 | 1.00 | 96.2 | **1.9** |

**Takeaways:**
- **Accuracy is within noise.** With ~40 gradeable items/cell the 95% CI is ≈±15pp, so the 0.475–0.625 spread is *not* significant — this is a chunkability screen, not a ranking. The auto‑"BEST" (least‑to‑most, 0.625) is statistically indistinguishable from #4/#6.
- **#4 plan‑and‑solve stays the most chunkable** (4.3 `\n\n` chunks, 100% reasoned) — still the right pick for PF/EPF (which need many resamplable steps), consistent with the downstream runs below using it.
- **#3 few‑shot** again collapses to 1 chunk (terse) — confirmed unsuitable.
- **New #9 (boxed)** reasons plenty (96 words, 100% reasoned) but chunks **lowest (1.9)** — exactly as flagged: its single‑newline `Step N:` format doesn't split on `\n\n`, so *as written* it is the least PF‑friendly prompt despite heavy reasoning. Its `\boxed{}` answers parsed 40/40 (the scorer was extended with a top‑priority `\boxed{LETTER}` rule for this).

## 4. Run 2 — CoT × ITS ablation (n=30, budget 4)

30 smallest single‑audio le30s items; `{2,4,7} × {baseline, PF@4, EPF@4}`; 270 runs, 0 errors.

| prompt | baseline | PF@4 | EPF@4 |
|--------|---------:|-----:|------:|
| #2 zero‑shot | 0.367 | 0.400 | **0.500** |
| #4 plan‑and‑solve | **0.567** | 0.533 | 0.500 |
| #7 format‑forcing | 0.500 | 0.333 | 0.367 |

**Takeaways:** mixed/noisy at n=30. #7 ITS *hurts* → dropped. Carried forward **#2 and #4** to a budget sweep. (Note: #4's 0.567 baseline here did **not** hold at larger n — see below.)

## 5. Run 3 — 150‑item budget sweep (the scaling test)

150 single‑audio le30s items; `{2,4} × {baseline@1, PF/EPF @4,8,16}`; 2100 runs, 0 errors; n≈148/cell (95% CI ≈ ±8 pp).

| prompt | base | PF@4 | PF@8 | PF@16 | EPF@4 | EPF@8 | EPF@16 |
|--------|----:|----:|----:|-----:|-----:|-----:|------:|
| #2 zero‑shot | 0.426 | 0.432 | 0.459 | 0.439 | **0.480** | 0.459 | 0.432 |
| #4 plan‑and‑solve | 0.412 | **0.527** | 0.439 | 0.453 | 0.419 | 0.439 | 0.419 |

**Takeaways:** **no clean budget scaling** — accuracy does not rise 4→8→16 (often peaks at 4, then drops). Apparent best cells (#4 PF@4 +11.5 pp, #2 EPF@4 +5.4 pp) are at/near the ±8 pp noise band and don't replicate at higher budget. This motivated the full‑957 run to resolve signal vs noise.

## 6. Run 4 — FULL 957 MCQ (the decisive result)

All 957 MCQ; `{2,4} × {baseline@1, PF@4, EPF@4}`; **5742 runs, 0 errors**; n=952/cell (5 ungradeable); 95% CI ≈ ±3.3 pp; `local-path` audio; item‑concurrency 12.

| prompt | base@1 | PF@4 | EPF@4 | PF Δ | EPF Δ |
|--------|-------:|-----:|------:|-----:|------:|
| #2 zero‑shot | 0.561 | 0.548 | 0.558 | −1.3 | −0.3 |
| **#4 plan‑and‑solve** | 0.534 | **0.576** | **0.575** | **+4.2** | **+4.1** |

Significance (2‑proportion z vs baseline): **#4 PF z=+1.84, EPF z=+1.80 (p≈0.07, borderline)**; #2 PF z=−0.55, EPF z=−0.14 (n.s.).

**Regression to the mean:** the big n=148 effects shrank at n=952 — #4 PF@4 **+11.5 → +4.2 pp**, #2 EPF@4 **+5.4 → −0.3 pp** — confirming the small‑n bumps were largely noise. A real but modest ~+4 pp residual remains on #4.

### #4 plan‑and‑solve — accuracy by category (base / PF@4 / EPF@4)

| category | n | base | PF@4 | EPF@4 |
|----------|--:|----:|----:|-----:|
| spatial_audio (hardest) | 69 | 0.203 | **0.319** | 0.246 |
| sound | 199 | 0.412 | 0.447 | **0.487** |
| speech | 171 | 0.556 | **0.632** | 0.602 |
| voice_chat | 64 | 0.484 | **0.562** | 0.531 |
| open | 105 | 0.886 | 0.933 | **0.943** |
| multi | 89 | 0.438 | 0.393 | 0.472 |
| music (already strong) | 220 | 0.655 | 0.659 | 0.641 |
| sound_speech | 18 | 0.111 | 0.389 | 0.389 |

By `length_type` (#4): EPF helps the **long / ultra‑long** clips (long 0.500→0.586, ultra‑long 0.500→0.625); PF helps short/medium more.

**Interpretation:** ITS helps where the model is *uncertain about perception* (spatial audio, environmental sound, speech) and adds nothing where it's already strong (music). This is the most interesting, interpretable part of the result.

## 7. Run 5 — full 957, 9-prompt greedy CoT bake-off (pick the best base prompt)

**Goal:** with prompts #1–#9 fixed, run **every prompt over all 957 MCQ, one greedy generation each (t=0)** — no PF/EPF — to choose the best base prompt. (Different decoding setup from Run 4: a single full greedy completion per item, *not* PF's step-chunked `base@1`, so absolute numbers differ slightly.) All-audio incl. **89 multi-audio**, `local-path`, concurrency 24 → **8,613 generations, 0 errors, ~58 min**. n=952 gradeable/prompt (5 ungradeable); **846** after also excluding the **106 single-choice** trivial items — `acc (excl-1)` is the fair comparator (single-choice items are correct for every prompt).

| # | prompt | acc (all, 952) | **acc (excl-1, 846)** | avg_words | avg_chunks |
|---|--------|----:|----:|----:|----:|
| 1 | assistant‑prefill CoT | 0.593 | 0.563 | 87 | 3.0 |
| 2 | zero‑shot CoT | 0.550 | 0.517 | 85 | 2.8 |
| 3 | few‑shot CoT | 0.538 | 0.502 | 29 | 1.0 |
| 4 | plan‑and‑solve | 0.559 | 0.513 | 128 | 5.0 |
| **5** | **least‑to‑most** | **0.613** | **0.571** | 42 | 2.3 |
| 6 | describe‑then‑reason | 0.582 | 0.537 | 96 | 2.7 |
| 7 | format‑forcing (## Step) | 0.597 | 0.564 | 92 | 3.5 |
| 8 | anti‑shortcut (≥3 steps) | 0.589 | 0.544 | 69 | 2.1 |
| 9 | evidence‑grounded (boxed) | 0.550 | 0.505 | 106 | 2.0 |

**Best prompt — a 3-way top tier (paired McNemar on the 846 common items):**
- **#5 least‑to‑most (0.571)** leads but is **statistically tied** with **#7 format‑forcing (0.564, p=0.67)** and **#1 assistant‑prefill (0.563, p=0.63)**; #8 anti‑shortcut (0.544) borderline (p=0.10).
- #5 is **significantly better** than the verbose/boxed/few‑shot prompts: vs #4 plan‑and‑solve p=0.001, vs #9 boxed p<0.001, vs #2 p=0.001, vs #3 p<0.001, vs #6 p=0.03.

**Key finding — chunkability ≠ accuracy.** The prompts that were *best for PF chunking* (#4 plan‑and‑solve 5.0 chunks, #9 boxed) are the **worst for greedy accuracy** (0.513, 0.505). The most accurate prompt, **#5 least‑to‑most, is concise** (42 words, 2.3 chunks). A longer, more‑segmented trace did not buy correctness here — it slightly hurt it. (So the prompt that's best to *run greedily* and the prompt that's best to *feed PF* may differ — #4 still chunks best for ITS; #5/#7/#1 score best as one-shot prompts.)

**By category (acc_all):** least‑to‑most wins on sound (0.55) & music (0.70); assistant‑prefill/format‑forcing win on speech (0.66–0.67); `open` ≈ 0.9 across the board; **spatial_audio is hard for all (0.20–0.32)**. Full matrix in `cot957_html/index.html`.

**Artifacts:** `results/run05_cot957/cot957.log` (score tables), `results/run05_cot957/cot957.csv` (8,613 per-response rows), `results/run05_cot957/cot957.jsonl` (resumable raw), `results/run05_cot957/cot957_html/index.html` (summary + accuracy matrix + per-category side-by-side pages).

## 8. Run 6 — EPF diversity sweep (is scaling worth it, or is the weight the bottleneck?)

**Why:** Run 5 ranked prompts by *greedy* accuracy, but EPF needs **exploration**. We ran **Entropic
Particle Filtering** on 100 stratified single-audio items × 4 prompts {**#4** plan-and-solve, **#5**
least-to-most, **#7** format-forcing, **#9** evidence-grounded/boxed} × 2 self-certainty weights
{`mean_logprob`, `entropy`} × budgets {1, 8, 16, 32}, instrumented with new SMC metrics
([diversity_probe.py](benchmarking/mmau_pro/diversity_probe.py)). Config: temp **0.8**, **systematic**
resampling, **ess_threshold 0.6**, **early_phase 0.7**. **3,200 EPF runs, 0 errors, ~35 min on both GPUs**
(GPU0:8100 + GPU1:8101, items round-robined). n=99 gradeable (1 ungradeable).

**The result is the same in all 8 (prompt × signal) cells.** Representative numbers (`mean_logprob`):

*Selected accuracy (what EPF returns) — barely scales with N:*

| prompt | b1 | b8 | b16 | b32 |
|--------|---:|---:|---:|----:|
| #4 plan-and-solve | 0.374 | 0.505 | **0.586** | 0.525 |
| #5 least-to-most | 0.556 | 0.586 | 0.566 | 0.566 |
| #7 format-forcing | 0.455 | 0.505 | 0.556 | 0.576 |
| #9 evidence-grounded | 0.475 | 0.475 | 0.535 | 0.576 |

*Oracle accuracy (is the correct answer in ANY particle?) — climbs strongly with N:*

| prompt | b1 | b8 | b16 | b32 | gap (oracle−selected) @b32 |
|--------|---:|---:|---:|----:|----:|
| #4 plan-and-solve | 0.374 | 0.626 | 0.717 | 0.848 | **+0.32** |
| #5 least-to-most | 0.556 | 0.747 | 0.747 | 0.828 | +0.26 |
| #7 format-forcing | 0.455 | 0.667 | 0.717 | 0.798 | +0.22 |
| #9 evidence-grounded | 0.475 | 0.778 | 0.838 | **0.929** | **+0.35** |

**Same run, `entropy` signal** — selected accuracy (b1→b32):

| prompt | b1 | b8 | b16 | b32 |
|--------|---:|---:|---:|----:|
| #4 plan-and-solve | 0.434 | 0.525 | **0.556** | 0.434 |
| #5 least-to-most | 0.495 | 0.566 | 0.525 | 0.535 |
| #7 format-forcing | 0.475 | 0.535 | 0.576 | **0.616** |
| #9 evidence-grounded | 0.455 | 0.545 | 0.556 | 0.576 |

`entropy` oracle accuracy (b1→b32) + gap:

| prompt | b1 | b8 | b16 | b32 | gap @b32 |
|--------|---:|---:|---:|----:|----:|
| #4 plan-and-solve | 0.434 | 0.677 | 0.758 | 0.798 | +0.36 |
| #5 least-to-most | 0.495 | 0.828 | 0.778 | 0.828 | +0.29 |
| #7 format-forcing | 0.475 | 0.677 | 0.717 | 0.869 | +0.25 |
| #9 evidence-grounded | 0.455 | 0.788 | 0.899 | **0.949** | **+0.37** |

**Both signals behave identically** — oracle climbs to 0.80–0.95, selected stalls at ~0.55, gap +0.22 to
+0.37. `entropy` edges `mean_logprob` on a couple of cells (#7 selected 0.616 @b32; #9 oracle 0.949) but
not meaningfully — **neither weight converts oracle coverage into selected accuracy.**

**Findings:**
1. **Oracle coverage climbs hard with budget** (≈0.45 → 0.85–0.95) — more particles *do* surface the
   correct answer. Exploration is real and scales.
2. **Selected accuracy does NOT scale** — it plateaus at ~0.52–0.62 from b8→b32 (sometimes *drops*, e.g.
   #4 peaks at b16 then falls). **Plurality-vote (majority) is no better** (~0.50–0.60).
3. **Huge oracle − selected gap (+0.22 to +0.37 @b32):** the right answer is in the swarm 80–95% of the
   time but EPF returns the wrong particle. On **~40%** of items the correct answer is even a swarm
   *minority* the self-certainty weight votes against (P9 entropy b32: 37/100 present-but-not-selected).
4. **Diversity is real but the distinct-answer *ratio* falls with N** (0.97→0.06) — partly mechanical
   (ratio = #unique/N), since the absolute #distinct answers actually *grows* (≈1.4 → 2.5 among 32). The
   swarm converges its plurality onto a confident, often-wrong answer while the tail increasingly holds
   the correct one.

**Verdict → the self-certainty WEIGHT is the bottleneck, not the particle count.** This is exactly the
"weak fluency reward, doesn't scale" prediction. Scaling N raises *oracle* but not *selected* because
self-certainty (both `mean_logprob` and `entropy`) cannot identify the correct particle — it favors a
confident-but-wrong plurality. **No amount of extra particles fixes selected accuracy; only a better
weight does** → the answer-choice-confidence reward (§12).

**Higher temperature?** *Not the priority.* At temp 0.8 the right answer is already present 80–95% of the
time (oracle ≫ selected); a higher temp would mostly push oracle *higher* while selected stays stuck,
widening the gap — reinforcing (not fixing) the weight diagnosis. Worth a small probe later, but the
decisive lever is the weight, not exploration.

**Prompt note (vindicates not picking by greedy acc):** **#9 evidence-grounded** — the *worst* greedy
prompt (Run 5: 0.505) — has the **richest swarm** (highest oracle, 0.93–0.95 @b32; most distinct answers).
With a competent weight, #9 + EPF has the most headroom (~0.95 oracle ceiling). #4 plan-and-solve is
similar. So the best EPF base prompt ≠ the best greedy prompt.

## 9. Run 7 — terminal answer-confidence re-rank (can a better selector recover the oracle gap?)

**Why:** Run 6's gap is a *selection* failure (right answer in the swarm, EPF picks wrong). This tests the
cheapest fix — re-rank the finished EPF swarm by **answer confidence** instead of self-certainty. Three
scorers: **L-audio** (letter read-out, chat, *with* audio), **L-text** (letter read-out, text-only),
**O-text** (option likelihood via `/v1/completions` echo, text-only, length-normalized) — each with two
selection rules (**argmax-particle**, **conf-vote**), vs baselines **epf** (status quo) / **majority** /
**oracle**. 100 stratified single-audio items × prompts {4,5,7,9} × budgets {8,16,32}; EPF mean_logprob
swarm. **1,200 runs, 0 errors.** n=99/cell → 95% CI ≈ ±10 pp.
(Server constraint: chat returns no `prompt_logprobs` here, so option-likelihood must use the text-only
`/completions` echo path — hence the L-text control, to compare letter-vs-option fairly.)

**Headline @budget 32 — nothing recovers the gap:**

| prompt | oracle | epf (status quo) | majority | best re-rank |
|--------|---:|---:|---:|---:|
| #4 plan-and-solve | 0.828 | 0.505 | 0.566 | 0.556 (L-audio vote) |
| #5 least-to-most | 0.818 | 0.586 | 0.616 | 0.616 (L-audio vote) |
| #7 format-forcing | 0.798 | 0.566 | 0.576 | 0.576 (L-text vote) |
| #9 evidence-grounded | 0.889 | 0.535 | 0.525 | 0.535 (L-audio vote) |

Every selector — self-certainty, majority, and all six confidence re-ranks — clusters at **0.50–0.62**,
while oracle is **0.80–0.89**. The unrecovered gap (+0.23 to +0.35) is essentially untouched; all deltas
vs epf are within the ±10 pp noise band.

**Why it fails — it reshuffles, it doesn't improve.** Decomposing the most-aggressive rule at b32
(*recovered* = oracle-right-but-EPF-wrong items it rescues; *broken* = EPF-right items it newly loses):

| prompt | rule | recovered | broken | net acc vs epf |
|--------|------|---:|---:|---:|
| #9 evidence-grounded | L-audio argmax | 14/35 | 16 | −0.020 |
| #4 plan-and-solve | O-text argmax | 13/32 | 12 | +0.010 |

The re-rank *does* find ~10–14 correct minority particles — but discards an equal number EPF had right.
**One-for-one trade → net zero.** Answer-confidence is no better calibrated than self-certainty at telling
the model's own right answers from its wrong ones.

**Cross-cuts:**
- **L-audio ≈ L-text** — re-attending the audio in the probe gives no consistent edge → the audio-less
  constraint on option-likelihood was *not* the limiting factor (the comparison is fair).
- **O-text ≤ L-text**, and **`otext_vote` is the worst rule** (net negative, e.g. P5/P9 −0.05 to −0.10) →
  full-option-text likelihood did not beat single-letter; the surface-form de-bias hypothesis didn't pan out.
- **majority** is quietly the best baseline (small positive, ≈ best re-rank) — reaffirming Run 6.

**Verdict → the ceiling is calibration, and NO self-generated signal fixes it.** Fluency self-certainty
(Runs 4/6), answer-letter confidence, and option-text likelihood all fail identically: the model is
confidently wrong on the items it gets wrong, so any self-derived weight reshuffles rather than improves.
Closing the oracle gap requires an **independent/external verifier** (a different judge model, or a trained
verifier/PRM) — not a reweighting of the generator's own confidence. This is a stronger, more decisive
negative than Run 6 (it rules out the "just pick a better self-signal" hope).

## 10. Run 8 — where does the (lack of) diversity come from: generation or resampling?

**Why:** even at budget 32 the EPF swarm reaches ~80% consensus (~4/5 particles agree) — is that because
the particles were *born similar* (low generation diversity) or because resampling *collapsed* an initially
diverse swarm? We ran two arms on the same 100 items, holding generation fixed (temp 0.8, mean_logprob) and
toggling only resampling: **EPF** (systematic resampling ON, as deployed) vs **INDEP** (resampling OFF → N
independent step-chunked trajectories = the generator's intrinsic diversity). Prompts {4,5,7,9} × budgets
{8,16,32}, **2,400 runs, 0 errors**, plus a per-step ESS curve (logged free inside the loop).

**Result @budget 32 (mean across the 4 prompts):**

| arm | distinct | consensus | **oracle** | selected |
|-----|---:|---:|---:|---:|
| EPF (resample **ON**) | 0.065 | 0.813 | **0.838** | 0.558 |
| INDEP (resample **OFF**) | 0.089 | 0.697 | **0.957** | 0.558 |

Three findings, in order of importance:

1. **Resampling actively *culls the correct answer* (the big one).** Turning resampling off lifts **oracle
   +0.12 (0.838 → 0.957)** — independent N=32 sampling contains the right answer **~96%** of the time, but
   EPF's resampling concentrates on high-self-certainty particles and discards the (lower-fluency, often
   correct) minority *before the end*, dropping oracle to ~0.84. The penalty is **worst at low budget**
   (P4: +0.22 @b8, +0.19 @b16, +0.14 @b32) — fewer particles, more gets culled.
2. **Selection is still the sole binding bottleneck.** `selected` is **0.558 in *both* arms** despite INDEP
   oracle 0.957 — preserving diversity doesn't help if the selector can't pick the right particle (Run 7).
3. **"Born similar" is real but not the limiter.** Even INDEP is low-diversity in absolute terms (~0.089
   distinct ≈ **~3 distinct answers among 32**, ~0.70 consensus) — the model is fairly deterministic at temp
   0.8. But that's still *enough to contain the correct answer 96% of the time*, so low distinct-count is not
   what's capping accuracy. (Raising temperature would add diversity, but coverage isn't the problem.)

**Per-step ESS explains the mechanism.** EPF's weights peak hard mid-trajectory — ESS dips to ~0.25–0.35
around step 2 (e.g. P7 b32: `0.83 → 0.24 → 0.33 → 0.40 → 0.46`) — i.e. strong concentration pressure that
culls the minority — then *recovers* near the end as duplicates equalize, so the **final** ESS looks benign
(~0.7–0.9, matching Run 6). INDEP shows the same early dip but **stays low** (`0.84 → 0.19 → 0.14 → 0.30 …`),
revealing the underlying self-certainty weights are genuinely peaked; resampling is what acts on that peak to
remove particles. So Run 6's healthy-looking final ESS was masking a sharp mid-trajectory cull.

**Verdict:** the swarm is *born* low-diversity but still covers the answer (~96% under independent sampling);
**EPF's resampling is net-harmful here** — it lowers the oracle ceiling by ~12 pp without improving selected
accuracy. Within the RL-free framing, the implied pipeline is **plain best-of-N sampling (no resampling) +
an external selector**, not particle filtering. The binding constraint is unchanged and reconfirmed: the
**selector**, not particle count, diversity, or temperature.

### Full‑957 replication (all 957 MCQ, temp 0.8)

Re‑ran the identical EPF‑vs‑INDEP comparison on the **entire 957‑MCQ set** (incl. 89 multi‑audio),
**22,968 runs, 0 errors**, n=952/cell (5 ungradeable), 95% CI ≈ ±3 pp — so this is the decisive‑scale
version of the n=100 result above. Mean across the 4 prompts:

| budget | arm | distinct | consensus | oracle | selected |
|---|---|---:|---:|---:|---:|
| 8 | EPF | 0.184 | 0.872 | 0.731 | 0.581 |
| 8 | INDEP | 0.265 | 0.726 | **0.880** | 0.592 |
| 16 | EPF | 0.107 | 0.854 | 0.803 | 0.597 |
| 16 | INDEP | 0.155 | 0.715 | **0.936** | 0.587 |
| 32 | EPF | 0.061 | 0.833 | 0.850 | 0.595 |
| 32 | INDEP | 0.087 | 0.705 | **0.966** | 0.589 |

**Both findings hold at full scale:** (1) turning resampling **off lifts oracle +0.149 @b8 / +0.133 @b16 /
+0.116 @b32** — EPF resampling culls the correct minority, worst at low budget; (2) **selected accuracy is
unchanged (~0.59 in both arms, every budget)** despite INDEP oracle reaching **0.966** — the selector is the
sole bottleneck. The ±3 pp CIs make the oracle gap unambiguous (it's ~12–15 pp). Raw: `divsource_full.*`.

## 11. Run 9 — does a higher temperature (1.0) add diversity?

**Why:** Run 8 showed the swarm is *born* low-diversity (even independent N=32 → ~3 distinct answers). Is
the lever simply **temperature**? Same divsource setup (EPF vs INDEP arms, per-step ESS), 100 stratified
items, prompts {4,5,7,9} × budgets {8,16,32}, **only the sampling temperature changed: 0.8 → 1.0** (all else
identical). **2,400 runs, 0 errors.**

**Result — temperature barely moves anything (mean across 4 prompts, `0.8 → 1.0`):**

| budget | arm | distinct | consensus | oracle | selected |
|---|---|---|---|---|---|
| 32 | EPF | 0.065 → 0.065 | 0.81 → 0.82 | 0.838 → 0.813 | 0.558 → 0.540 |
| 32 | INDEP | 0.089 → 0.096 | 0.70 → 0.66 | 0.957 → 0.980 | 0.558 → 0.525 |
| 8 | EPF | 0.192 → 0.191 | 0.86 → 0.85 | 0.692 → 0.702 | 0.563 → 0.503 |
| 8 | INDEP | 0.273 → 0.293 | 0.73 → 0.69 | 0.848 → 0.866 | 0.556 → 0.558 |

(Budget 16 is the same story.) Raising the temperature **does not add meaningful diversity** — distinct/
consensus move by ~0.01–0.04, within noise — and **selected accuracy is unchanged** (~0.55 at both temps).
INDEP oracle nudges up ~+0.02 (slightly more spread surfaces the answer a touch more often), but the swarm
was already covering the answer ~96%, so it's moot.

**Verdict:** the model is **intrinsically low-entropy on MCQ answers** — "born similar" persists even at temp
1.0, so temperature is **not** the lever. Combined with Runs 7–8, the bottleneck is now triply confirmed to
be the **selector**, not particle count, not diversity, and not temperature. (Higher temps than 1.0 would
trade coherence for spread; not worth it given oracle is already ~0.96 and selection is the wall.)

## 12. Run 10 — Run 6's EPF budget sweep on the FULL 957 (4 prompts × 2 signals × {1,8,16,32})

**Why:** Run 6 (the EPF diversity/scaling sweep) was only n=100. This re-runs the **entire grid on all 957
MCQ**: prompts {4,5,7,9} × signals {mean_logprob, entropy} × budgets {1,8,16,32}, EPF only. To avoid
recompute, the **mean_logprob {8,16,32}** cells were carried over from Run 8's full‑957 EPF arm (identical
config, same `compute_metrics`); only **mean_logprob{1} + entropy{1,8,16,32}** were newly run. **30,624 rows,
0 errors**, n=952 gradeable/cell (±3 pp CI).

**Mean across the 4 prompts:**

| signal | budget | **selected** | oracle | majority | distinct | consensus |
|--------|---:|---:|---:|---:|---:|---:|
| mean_logprob | 1 | 0.557 | 0.557 | 0.557 | (1 particle) | — |
| mean_logprob | 8 | 0.581 | 0.731 | 0.589 | 0.184 | 0.872 |
| mean_logprob | 16 | **0.597** | 0.803 | 0.614 | 0.107 | 0.854 |
| mean_logprob | 32 | 0.595 | 0.850 | 0.614 | 0.061 | 0.833 |
| entropy | 1 | 0.567 | 0.567 | 0.567 | (1 particle) | — |
| entropy | 8 | 0.593 | 0.756 | 0.605 | 0.195 | 0.853 |
| entropy | 16 | 0.582 | 0.814 | 0.605 | 0.113 | 0.835 |
| entropy | 32 | 0.586 | 0.866 | 0.607 | 0.064 | 0.823 |

**Findings (now at decisive scale):**
1. **Selected accuracy barely scales with budget.** From the single‑trajectory anchor (b1 ≈ 0.557/0.567) it
   gains only **~+0.03–0.04** and **saturates by budget 8–16** (mean_logprob peaks 0.597 @b16; entropy 0.593
   @b8), then is flat/dips at 32. With ±3 pp CIs the ITS lift over b1 is marginal‑to‑borderline — **no real
   scaling.** (Matches Run 6's noisy n=100 read, now tight.)
2. **Oracle climbs steeply** (0.56 → 0.85–0.87 @b32) — exploration keeps surfacing the answer — but selected
   doesn't follow → the oracle−selected gap *widens* with budget. **majority ≈ selected** (~0.59–0.61), both
   far below oracle.
3. **The two weight signals are equivalent.** entropy is a hair better on a couple cells (selected@b8,
   oracle@b32) but within noise; neither escapes the plateau. So Run 6's "weak fluency reward" conclusion is
   not specific to mean_logprob — entropy behaves the same.

**Verdict:** confirms Run 6 at full scale — **EPF buys ~+0.03–0.04 selected accuracy over a single sample and
plateaus by ~8 particles; neither self‑certainty signal scales; the oracle gap only grows.** Consistent with
Runs 7–9: the selector is the wall. Raw: `run6_full.*` (mean_logprob{8,16,32} provenance = Run 8).

## 13. Run 11 — the Run 6/10 EPF grid on the FULL MMAU-Pro test set (5,090 MCQ)

**Why:** every prior run used the 957-MCQ testmini. This scales the canonical EPF grid — 4 prompts
{4,5,7,9} × 2 signals {mean_logprob, entropy} × budgets {1,8,16,32} — to the **full MMAU-Pro test
set**: `test` parquet, 5,305 rows → **5,090 MCQ** (incl. 497 single-choice trivial items, same
convention as the 957 which had 106; 24 ungradeable → n=5,066/cell, **95% CI ≈ ±1.4 pp, bootstrap
SE ≈ ±0.7 pp**). Audio from `mmau_pro_audio/` (5,787 files, max clip 600 s; 430 two-audio + 26
three-audio items). Config identical to Run 10 (temp 0.8, systematic, ess 0.6/0.7, style logit).
The 957 overlap was **seeded from `run6_full.jsonl`** (testmini ⊂ test; same mechanism as Run 10's
Run-8 seeding) — the 957-slice of this run reproduces Run 10's table exactly, by construction.
**132,256 fresh EPF runs (162,880 rows total), 0 errors, ~50 h on both GPUs**, budget-staged
(b1 → b8 → b16 → b32, resumable JSONL).

**Serving notes (required at this scale):** `--gpu-memory-utilization 0.85` (not 0.9) +
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and `--max-inflight 24` for the budget-1 stage
— at b1 the probe otherwise encodes 64 distinct audios per GPU simultaneously and the O(n²) audio
encoder OOM-killed an engine (one cell had to be re-swept; final data has 0 errors). Details in
`SETUP_GUIDE.md` §3/§10.

**Mean across the 4 prompts** (bootstrap in `epf_full5090_bootstrap.html`; SE ≈ ±0.007/cell):

| signal | budget | **selected** | oracle | majority |
|--------|---:|---:|---:|---:|
| mean_logprob | 1 | 0.557 | 0.557 | 0.557 |
| mean_logprob | 8 | 0.585 | 0.732 | 0.596 |
| mean_logprob | 16 | **0.590** | 0.792 | 0.605 |
| mean_logprob | 32 | **0.590** | 0.845 | 0.613 |
| entropy | 1 | 0.558 | 0.558 | 0.558 |
| entropy | 8 | 0.587 | 0.754 | 0.601 |
| entropy | 16 | 0.583 | 0.818 | 0.607 |
| entropy | 32 | 0.586 | 0.865 | 0.608 |

**Findings (the Run 6/10 story at decisive scale, now with sub-pp error bars):**
1. **Selected accuracy saturates by budget 8.** +2.8–3.3 pp over b1, then flat: every b8→b16→b32
   step is within ±1 pp per cell. The apparent b8→b16 bumps in Run 10 (e.g. P7 mean_logprob
   +4.7 pp) collapse to ≈0 at full scale (−0.1 pp) — they were noise, as suspected.
2. **Oracle keeps climbing** — 0.56 → 0.845/0.865 @b32 (mean); **P9 evidence-grounded reaches
   0.914 with BOTH signals** — the right answer is in the swarm 91% of the time while P9's
   selected accuracy is the *lowest* (0.565–0.577). The oracle−selected gap at b32 is +0.26 on
   average and **+0.34 for P9**.
3. **majority ≈ selected + ~0.02** at every budget — plurality voting doesn't close the gap either.
4. **The two signals are equivalent** (differences within ~±0.5 pp mean; entropy slightly higher
   oracle, mean_logprob slightly higher selected — both within noise).

**Bootstrap headline** (`epf_full5090_bootstrap.html`): a 100-question eval wobbles ±0.049 on these
cells; the reported full-set numbers are precise to ±0.0069 (~7× tighter) — and 2.3× tighter than
the 957-run's ±0.016.

**Verdict:** unchanged and now essentially beyond statistical doubt at this benchmark's full scale —
**EPF's selected accuracy does not scale with particle count past ~8; exploration (oracle) scales
beautifully; no self-certainty signal converts it; the selector is the wall.** This is the
strongest-n version of the §14 verdict; the next lever remains an external/independent selector
(§17).

## 14. Honest verdict

- **Best ITS number: 0.576 (#4 PF@4), ≈ +4 pp over our own baseline, no RL used** — borderline‑significant (p≈0.07), prompt‑dependent, category‑localized, and **non‑scaling** with budget.
- It is **not** a "ITS matches GRPO for free" result. The gain is small and self‑certainty behaves like a weak, fluency‑based reward — it can't reliably steer resampling toward *correct* trajectories.
- Leaderboard context (reported numbers, full MMAU‑Pro test set, NOT our measurement): base Qwen2.5‑Omni ≈ 52%, AF3 51.7%, Gemini‑2.5‑Flash 59.2%, human 77.9%. Our testmini‑MCQ base (0.53–0.56) is in range; best ITS (0.576) is a few points above base.

### Caveats / what NOT to claim
- **No GRPO/RL result exists on MMAU‑Pro.** Published GRPO numbers (R1‑AQA, SARI, **Omni‑R1: 65.9 → 71.3**) are on the **original MMAU**, a *different and easier* benchmark — not comparable to these MMAU‑Pro numbers. The "+4 pp" here is **over our own no‑ITS baseline**, *not* over any RL result.
- n=952 → ±3.3 pp CI; the #4 effect is borderline (p≈0.07), not conclusively significant.
- `baseline` = PF at budget 1 (single self‑certainty trajectory), i.e. the same CoT prompt with no resampling.

## 15. Files (`benchmarking/mmau_pro/results/`)

Organized **one folder per run** (see `results/README.md` for the full index); cross-run figures in
`results/plots/`. Each run folder holds `<experiment>.jsonl` (raw, resumable), `.csv`, `.log`, plus
one-file HTML reports.

| folder / file | what |
|------|------|
| `run01_cot_screen/cot_compare.log` | **Run 1**: 8‑prompt CoT comparison (n=20, smallest) |
| `run02_ablation/mmau_ablation.jsonl` | **Run 2**: n=30 CoT×ITS ablation, 270 rows |
| `run03_sweep150/mmau_150_sweep.jsonl` | **Run 3**: 150‑item budget sweep, 2100 rows |
| `run04_full957/mmau_957_results.jsonl` (+`.log`) | **Run 4**: full 957 run, 5742 rows (the headline result) |
| `run05_cot957/cot957.{jsonl,csv,log}` | **Run 5**: full 957 × 9 greedy bake-off (8,613 responses) |
| `run05_cot957/cot957_html/`, `cot957_all.html` | **Run 5**: summary+matrix pages; single-file side-by-side (~6 MB) |
| `run06_epf_div/epf_div.{jsonl,csv,log}` | **Run 6**: EPF diversity sweep n=100 (per-item SMC metrics) |
| `run07_rerank/rerank.{jsonl,csv,log}` | **Run 7**: terminal answer-confidence re-rank, 1,200 rows |
| `run08_divsource/divsource.*` | **Run 8**: EPF vs INDEP + per-step ESS (n=100, temp 0.8) |
| `run08_divsource/divsource_full.*` | **Run 8 (full)**: same on all 957 MCQ — appended to §10 |
| `run09_divsource_t1/divsource_t1.*` | **Run 9**: EPF vs INDEP at temp 1.0 (n=100) |
| `run10_run6_full/run6_full.*` | **Run 10**: Run 6 grid on full 957; 30,624 rows. mean_logprob{8,16,32} seeded from Run 8 |
| `run10_run6_full/epf_div.html` | one-file report (trend heatmaps + appendix), **generated from `run6_full.csv`** |
| `run10_run6_full/epf_div_bootstrap.html` | **Run 10**: bootstrap error bars (std100 + SE957) |
| `run11_epf_full5090/epf_full5090.{jsonl,csv,log}` | **Run 11**: the grid on the FULL 5,090-MCQ test set; 162,880 rows (957-slice seeded from `run6_full.jsonl`) |
| `run11_epf_full5090/epf_full5090_bootstrap.html` | **Run 11**: bootstrap error bars + interactive acc-vs-budget plot |
| `run12_af3_full5090/af3_cot_screen.{jsonl,csv,log}` | **Run 12**: AF3 9-prompt chunkability screen, 40 stratified items (folder named before the scope narrowed to single-audio) |
| `run12_af3_full5090/af3_chunk_probe100.jsonl` | **Run 12**: AF3 n=100 random chunk/sentence probe, prompts {4,5,7,9} |
| `run12_af3_full5090/af3_param_check200.jsonl` | **Run 12**: base-AF3 200-item step-trajectory check (tok30/max8, t=0.8, b=1) |
| `run12_af3_full5090/af3think_param_check{200,1000}.jsonl` | **Run 12**: AF3-Think(+trigger) trajectory checks, same items/config |
| `run12_af3_full5090/af3base_cotuser_check1000.jsonl`, `af3think_notrigger_check1000.jsonl` | **Run 12**: the 2×2 weights×trigger decomposition cells (n=1000) |
| `af3_chat_template.jinja` (this dir) | **Run 12**: multi-audio chat-template fix (one `<sound>` per clip), passed via `vllm serve --chat-template` |
| `run12_af3_full5090/epf_p7think.{jsonl,csv,log}` | **Run 13**: the AF3-Think P7 EPF grid (stages 1+2, one resumable JSONL; 37,280 rows) |
| `run12_af3_full5090/epf_p7think_bootstrap.html` | **Run 13**: bootstrap error bars (std100 + SE4660) |
| `run12_af3_full5090/stage1_ids_1000.txt`, `stage2_ids_remaining.txt`, `smoke_ids_16.txt` | **Run 13**: item lists (seed-11 length-balanced 1,000; remaining 3,660 single-audio; smoke) |
| `run14_gemma4e2b/durations_test5090.csv`, `le30s_*_ids.txt`, `probe_ids_100.txt`, `smoke_ids_16.txt` | **Run 14**: measured clip durations + the ≤30 s ids files (1,947 single + 243 multi) — `make_le30s_ids.py` |
| `run14_gemma4e2b/probe100_t0{0,8}.*`, `step_probe_*.*`, `smoke_epf16.*` | **Run 14**: 4-prompt × 2-temp probe, step-trajectory probe (`step_probe.py`), EPF smoke |
| `run14_gemma4e2b/epf_gemma_le30s.{jsonl,csv,log}` (+`_bootstrap.html`) | **Run 14**: the Gemma 4 E2B EPF grid at max_steps=6 (`scripts/run14_sweep.sh`); 31,152 rows |
| `run14_gemma4e2b/epf_gemma_le30s_max12.{jsonl,csv,log}` (+`_bootstrap.html`) | **Run 14b**: max_steps=12 ablation (`scripts/run14_sweep_max12.sh`); 31,152 rows |
| `run14_gemma4e2b/epf_gemma_le30s_max12_b64128.*` | **Run 14c** (collaborator): budgets {64,128} extension (`scripts/run14_collab_b64_128.sh`) |
| `run15_gemma4e4b/smoke_e4b16.*`, `anchors_e4b.csv` | **Run 15**: E4B preflight artifacts (smoke + greedy anchors) |
| `run15_gemma4e4b/epf_gemma4e4b_max12.*` | **Run 15** (collaborator): E4B full grid b1→128 (`scripts/run15_e4b_b1_128.sh`) |
| `plots/` | cross-run figures: `acc_vs_budget.{png,html}` (Run 6), `epf_acc_vs_budget.{png,html}` (Run 10), `acc_vs_budget_combined.html`, `epf_temp_*.html` (EPF × self-consistency overlays) |
| `smoke/mmau_smoke.jsonl` | initial 8‑item pipeline smoke |

Each `run_mmau` row: `{unique_id, method, arm, budget, category, length_type, correct, latency_s, error, content}`.

## 16. Reproduce

```bash
# serve (Blackwell)
HF_HOME=$BV/hf_cache CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 \
  /home/exx/miniconda3/envs/epf/bin/vllm serve Qwen/Qwen2.5-Omni-7B \
  --served-model-name qwen-omni --port 8100 --trust-remote-code --dtype bfloat16 \
  --max-model-len 32768 --enforce-eager --gpu-memory-utilization 0.9 \
  --allowed-local-media-path /home/exx/inference-time-scaling/mmau_pro_testmini \
  --limit-mm-per-prompt '{"audio":3}'

# full 957 (resumable, concurrent)
conda run -n epf python -m benchmarking.mmau_pro.run_mmau \
  --endpoint http://localhost:8100/v1 --model-name qwen-omni \
  --data-root /home/exx/inference-time-scaling/mmau_pro_testmini --subset full \
  --prompt-methods 2,4 --arms baseline,pf,epf --budgets 4 \
  --audio-mode local-path --item-concurrency 12 \
  --output benchmarking/mmau_pro/results/run04_full957/mmau_957_results.jsonl

# Run 5: full 957 x 9 greedy bake-off (resumable) + paginated HTML
uv run python -m benchmarking.mmau_pro.cot_compare \
  --endpoint http://localhost:8100/v1 --model-name qwen-omni \
  --data-root /home/exx/inference-time-scaling/mmau_pro_testmini \
  --subset full --select all --audio-mode local-path --concurrency 24 \
  --jsonl benchmarking/mmau_pro/results/run05_cot957/cot957.jsonl \
  --csv   benchmarking/mmau_pro/results/run05_cot957/cot957.csv \
  --log   benchmarking/mmau_pro/results/run05_cot957/cot957.log
uv run python -m benchmarking.mmau_pro.make_report \
  --in benchmarking/mmau_pro/results/run05_cot957/cot957.csv \
  --out-dir benchmarking/mmau_pro/results/run05_cot957/cot957_html --paginate category

# Run 6: EPF diversity sweep on both GPUs (start a 2nd replica on GPU1 first)
CUDA_VISIBLE_DEVICES=1 ... vllm serve ... --port 8101   # 2nd replica (same launch cmd as :8100)
uv run python -m benchmarking.mmau_pro.diversity_probe \
  --endpoints http://localhost:8100/v1,http://localhost:8101/v1 --model-name qwen-omni \
  --data-root /home/exx/inference-time-scaling/mmau_pro_testmini \
  --prompts 4,5,7,9 --signals mean_logprob,entropy --budgets 1,8,16,32 \
  --temp 0.8 --ess-threshold 0.6 --early-phase 0.7 --limit 100 --max-inflight 64 \
  --jsonl benchmarking/mmau_pro/results/run06_epf_div/epf_div.jsonl \
  --csv   benchmarking/mmau_pro/results/run06_epf_div/epf_div.csv \
  --log   benchmarking/mmau_pro/results/run06_epf_div/epf_div.log

# Run 7: terminal answer-confidence re-rank (3 scorers x 2 rules vs baselines, both GPUs)
uv run python -m benchmarking.mmau_pro.rerank_probe \
  --endpoints http://localhost:8100/v1,http://localhost:8101/v1 --model-name qwen-omni \
  --data-root /home/exx/inference-time-scaling/mmau_pro_testmini \
  --prompts 4,5,7,9 --budgets 8,16,32 --limit 100 --max-inflight 48 \
  --jsonl benchmarking/mmau_pro/results/run07_rerank/rerank.jsonl \
  --csv   benchmarking/mmau_pro/results/run07_rerank/rerank.csv \
  --log   benchmarking/mmau_pro/results/run07_rerank/rerank.log

# Run 8: diversity source — EPF (resample ON) vs INDEP (resample OFF) + per-step ESS
uv run python -m benchmarking.mmau_pro.divsource_probe \
  --endpoints http://localhost:8100/v1,http://localhost:8101/v1 --model-name qwen-omni \
  --data-root /home/exx/inference-time-scaling/mmau_pro_testmini \
  --prompts 4,5,7,9 --budgets 8,16,32 --limit 100 --max-inflight 64 \
  --jsonl benchmarking/mmau_pro/results/run08_divsource/divsource.jsonl \
  --csv   benchmarking/mmau_pro/results/run08_divsource/divsource.csv \
  --log   benchmarking/mmau_pro/results/run08_divsource/divsource.log

# Run 10: Run 6's EPF budget sweep on the FULL 957 (both signals, budgets 1/8/16/32).
# (mean_logprob{8,16,32} were seeded from Run 8's full-957 EPF arm to avoid recompute.)
uv run python -m benchmarking.mmau_pro.diversity_probe \
  --endpoints http://localhost:8100/v1,http://localhost:8101/v1 --model-name qwen-omni \
  --data-root /home/exx/inference-time-scaling/mmau_pro_testmini \
  --prompts 4,5,7,9 --signals mean_logprob,entropy --budgets 1,8,16,32 \
  --select all --limit 2000 --temp 0.8 --max-inflight 64 \
  --jsonl benchmarking/mmau_pro/results/run10_run6_full/run6_full.jsonl \
  --csv   benchmarking/mmau_pro/results/run10_run6_full/run6_full.csv \
  --log   benchmarking/mmau_pro/results/run10_run6_full/run6_full.log

# Run 11: the grid on the FULL 5,090-MCQ test set (budget-staged; b1 needs low inflight,
# and serve with --gpu-memory-utilization 0.85 + PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# — see SETUP_GUIDE.md). Seed first: cp results/run10_run6_full/run6_full.jsonl results/run11_epf_full5090/epf_full5090.jsonl
for B in 1 8 16 32; do INFLIGHT=64; [ "$B" -eq 1 ] && INFLIGHT=24
  conda run -n epf python -m benchmarking.mmau_pro.diversity_probe \
    --endpoints http://localhost:8100/v1,http://localhost:8101/v1 --model-name qwen-omni \
    --data-root /home/exx/inference-time-scaling/mmau_pro_testmini \
    --subset test --audio-root /home/exx/inference-time-scaling/mmau_pro_audio \
    --prompts 4,5,7,9 --signals mean_logprob,entropy --budgets $B \
    --select all --limit 6000 --temp 0.8 --ess-threshold 0.6 --early-phase 0.7 \
    --max-inflight $INFLIGHT \
    --jsonl benchmarking/mmau_pro/results/run11_epf_full5090/epf_full5090.jsonl \
    --csv   benchmarking/mmau_pro/results/run11_epf_full5090/epf_full5090.csv \
    --log   benchmarking/mmau_pro/results/run11_epf_full5090/epf_full5090.log
done
conda run -n epf python -m benchmarking.mmau_pro.epf_bootstrap --n 10000 \
  --in  benchmarking/mmau_pro/results/run11_epf_full5090/epf_full5090.csv \
  --out benchmarking/mmau_pro/results/run11_epf_full5090/epf_full5090_bootstrap.html

# ---- Runs 12–13 (Audio Flamingo 3; see §18–19). Serving env `af3serve` =
# vllm 0.22.1 + transformers>=5.5 (AF3 processor needs transformers 5.x; the pinned
# epf env stays the client). Serve per GPU (0.85 util, Blackwell flags as §3 of
# SETUP_GUIDE) — note the REQUIRED chat-template override for multi-audio and
# VLLM_ALLOW_LONG_MAX_MODEL_LEN for the 20k window:
HF_HOME=~/hf_cache CUDA_VISIBLE_DEVICES=<g> VLLM_USE_FLASHINFER_SAMPLER=0 \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  <af3serve>/bin/vllm serve <model> --served-model-name <name> --port 810<g> \
  --dtype bfloat16 --max-model-len 20000 --enforce-eager --gpu-memory-utilization 0.85 \
  --allowed-local-media-path <repo>/data --limit-mm-per-prompt '{"audio":3}' \
  --chat-template benchmarking/mmau_pro/af3_chat_template.jinja
# <model> = nvidia/audio-flamingo-3-hf (base, name af3) or the merged AF-Think
# checkpoint (name af3think) — built by scratch merge script: remap think/ adapter keys
# (old pre-5.x naming!) onto the -hf model, apply non_lora embed + 196 LoRA pairs.

# Run 13: the P7×think EPF grid, budget-staged over one resumable JSONL.
# Stage 1 = stage1_ids_1000.txt; stage 2 = stage2_ids_remaining.txt (same command).
for B in 1 8 16 32; do INFLIGHT=64; [ "$B" -eq 1 ] && INFLIGHT=24
  conda run -n epf python -m benchmarking.mmau_pro.diversity_probe \
    --endpoints http://localhost:8100/v1,http://localhost:8101/v1 --model-name af3think \
    --data-root <repo>/data/mmau_pro --subset test \
    --prompts 7 --signals mean_logprob,entropy --budgets $B \
    --ids-file benchmarking/mmau_pro/results/run12_af3_full5090/stage1_ids_1000.txt \
    --temp 0.8 --ess-threshold 0.6 --early-phase 0.7 --max-steps 6 --tokens-per-step 30 \
    --think-trigger --max-inflight $INFLIGHT \
    --jsonl benchmarking/mmau_pro/results/run12_af3_full5090/epf_p7think.jsonl \
    --csv   benchmarking/mmau_pro/results/run12_af3_full5090/epf_p7think.csv \
    --log   benchmarking/mmau_pro/results/run12_af3_full5090/epf_p7think.log
done
```

## 17. Next lever

**Run 7 ruled out the obvious self-signal fix.** We tested the answer-choice-confidence reward as a terminal
re-rank (answer-letter confidence and option-text likelihood, with and without audio) and it does **not**
recover the oracle gap — it reshuffles correct↔incorrect roughly one-for-one (net ≈ 0). Combined with Runs
4/6, this rules out *any* self-generated confidence (fluency, answer-letter, option-text) as the weight: the
generator is confidently wrong on the items it misses, so no reweighting of its own signal can separate its
right answers from its wrong ones.

**Run 8 added a second structural finding:** EPF's *resampling is net-harmful* on this task — it culls the
correct (lower-fluency) minority and lowers the oracle ceiling ~12 pp vs plain independent sampling, with no
gain in selected accuracy. So the EPF/PF machinery isn't earning its keep here.

**The implied pipeline within the RL-free framing:** drop particle filtering for **plain best-of-N sampling**
(no resampling → oracle ~0.96 @N=32) and spend the budget on a **better *independent* selector** — a signal
that doesn't share the generator's blind spots:
- a **different judge model** scoring/ranking the N finished candidates (cross-model verification), or
- a **trained verifier / PRM** (supervised on correctness) — but that re-introduces training, leaving the
  RL-free framing.

Cheaper diagnostics worth a look before committing: (a) re-rank with a *stronger* model as judge on the
oracle-but-EPF-wrong items to confirm an external signal *can* recover them (upper-bound check); (b) a
self-critique/debate pass (still self-signal, likely same ceiling, but cheap to falsify). The honest read:
within the fixed-single-model, RL-free constraint, **inference-time scaling on MMAU-Pro is selection-limited
and that limit is the model's own calibration** — not solvable by more particles or a cleverer self-weight.

## 18. Run 12 — porting to Audio Flamingo 3: gates, adaptation, and what makes AF3 reason

**Why:** replicate the Run-11 EPF grid on a second audio LM, **NVIDIA Audio Flamingo 3** (8.4B =
Qwen2.5-7B LLM + AF-Whisper encoder). Everything below was needed to make that experiment
*meaningful* on AF3, and several sub-results are findings in their own right. New box, same
hardware as the reference (2× RTX PRO 6000 Blackwell); data = official `gamma-lab-umd/MMAU-Pro`
HF dump (loader verified: 5,090 MCQ / 24 ungradeable / 0 missing audio).

**Serving.** Only `nvidia/audio-flamingo-3-hf` is vLLM-loadable (the `-3`/`-3-chat` repos are
NVILA custom code). AF3's HF processor needs **transformers ≥5** while the pinned client env
freezes 4.57.3 → a dedicated serving env `af3serve` (vllm 0.22.1 + transformers 5.13); the `epf`
env stays the client. Serve: bf16, `--max-model-len 20000` + `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`
(AF3 ≈ 25 audio tok/s, 600 s clip ≈ 15k tokens; trained text window ~16k), 0.85 util, Blackwell
flags per SETUP_GUIDE §3.

**Gates (phase0 + causality): PASS.** Gate 1 (token logprobs + top20 WITH audio) and Gate 2
(`continue_final_message` WITH audio) pass on both endpoints with real MMAU-Pro audio. A/B
causality (15 items): **0.733 with audio vs 0.533 without, 6/15 answers changed** — AF3 hears it.

**Multi-audio (excluded from Run 13 by decision).** Two independent defects found:
1. NVIDIA's shipped chat template renders **one `<sound>` per message** regardless of clip count →
   HTTP 500 on the 430 multi-audio MCQ. Fixed with [`af3_chat_template.jinja`](af3_chat_template.jinja)
   (one `<sound>` per clip — the format vLLM's own AF3 code expects); verified: 2- and 3-clip
   prompts accepted, and a 440 Hz-vs-880 Hz two-clip pitch question answered correctly.
2. Deeper: transformers' `AudioFlamingo3Processor.validate_inputs` enforces **1 text : 1 audio**,
   so multi-audio only works when clips are already in vLLM's LRU processor cache (fresh pairs
   400) — unusable for a long run. Relaxing the check was verified exactly correct (joint 2-audio
   call → 200 `<sound>` tokens = 75+125 from the known-good separate path), but we opted to run
   **single-audio only** (4,660 MCQ) rather than patch site-packages.

**Chunkability: AF3 never emits `\n\n` — the Run-11 step token has nothing to grip.** 9-prompt
screen (40 stratified items, greedy): `avg_chunks = 1.0 for all 9 prompts`; not truncation
(max_tokens 700, responses 30–45 words, clean EOS). AF3 writes steps *inline as prose*
("Step 1: … Step 2: …" separated by ". "), and n=100 shows reasoning length tracks difficulty
(open 1.1 sentences → spatial_audio 8.3; P4 wrong answers 4.3 vs correct 2.5 sentences) — the
reasoning exists, only the delimiter differs. Prompts #2/#7/#9 collapse to ~1-word answers.

**Why #7/#9 collapse — AF3 ignores the system turn (causal, both directions).** 15 items × 6
variants, greedy: identical instruction text scores what its *placement* dictates —

| variant | acc | avg words |
|---|---:|---:|
| P7 instructions in system (orig) | 0.87 | 1.1 |
| P7 instructions in user | 0.73 | 1.0 (markdown format unfollowable even user-side) |
| P9 instructions in system (orig) | 0.80 | 1.1 |
| **P9 instructions in user** | 0.53 | **30.7** |
| P4 instruction in user (orig) | 0.60 | 35.7 |
| **P4 instruction in system** | **0.87** | **1.1** |

Two lessons: (a) behavioral instructions must be in the **user turn** for AF3 (its MCQA-heavy SFT
answers a bare Question/Options with a bare letter); (b) **making base AF3 reason costs ~25 pp**
(0.87 terse vs 0.60 reasoning, same items/instruction) — CoT-as-noise on a perception task.

**Step machinery adaptation (its_hub/probe changes).** Sentence-delimiter (`". "`) stepping was
prototyped and rejected: numbered lists make single-digit "steps" (`"3"`) whose near-perfect
logprob (**w≈+4.9**) hijacked resampling (observed full swarm collapse), plus max-steps
truncation. **`tokens_per_step=30`** (native `StepGeneration` mode, exposed as
`diversity_probe --tokens-per-step`) has none of these pathologies. Second fix: AF3 almost never
emits the `"Answer:"` stop token (0–15% of trajectories), so finished particles were re-asked to
continue until max_steps (verified: continuations come back empty) — added **EOS detection** to
`StepGeneration` (via vLLM's `stop_reason`; `openai_lm` now carries `_finish_reason`/`_stop_reason`)
→ particles stop at natural end-of-turn; **3.6× fewer requests**, no behavior change for endpoints
without `stop_reason` (95-test suite unchanged).

**Base-AF3 trajectory check** (200 length-balanced items, tok30/max8, t=0.8, b=1, acc counts
unparsed-as-wrong): P4 **0.410** (16% unparsed!), P5 **0.295** (21.5%), P7 0.515, P9 **0.590** —
reasoning prompts often end WITHOUT any extractable answer, and terse direct answering wins big.

**AF-Think: the reasoning is a shipped adapter, not a prompt.** The `-hf` repo's `think/` folder =
PEFT LoRA (r=64, α=16; 196 A/B pairs over q/k/v/o+gate/up/down × 28 layers) **plus** a required
1.09 GB non-LoRA embed matrix, activated by a **user-turn trigger sentence** ("Please think and
reason about the input sound/music/speech before you respond."). ⚠ The model card's own loading
recipe **silently no-ops on transformers 5.x** — the think weights use pre-5.x key naming and the
recipe's `strict=False` hides the mismatch. We merged manually (explicit key remap; every tensor
verified applied) → full checkpoint served as `af3think`.

**The 2×2 that decided Run 13's substrate** (same 1,000 seed-11 single-audio items, tok30/max8,
t=0.8, b=1; acc = unparsed-as-wrong):

| weights \ prompt | no trigger | + trigger |
|---|---|---|
| base | terse (n=200 proxies 0.52–0.59) | **0.516**, *no reasoning* (1.03 steps, 7.5 tok) |
| think | **0.527**, terse (1.01 steps, 3.2 tok) | **0.485–0.537** by prompt, real reasoning (3.6–6.9 steps) |

Per-prompt think+trigger (n=1000): P4 0.485 (4.49 steps), P5 0.484 (4.65), **P7 0.537 (3.75)**,
P9 0.495 (6.92 steps; 54% hit the 8-step cap — P9 needs max_steps≈12). Unparsed collapses to
0.7–3.2% (the adapter trained answer-commitment in).

**Takeaways.**
1. Reasoning requires **both** the adapter weights and the trigger — the trigger does nothing to
   base weights (7.5 tok) and the weights change nothing without it (3.2 tok, acc unchanged).
2. **Budget-1 accuracy is flat (~0.52±0.02) across every mode** — think-reasoning is
   accuracy-neutral (vs actively harmful base CoT); what it buys is *structure*: multi-step,
   answer-committing, resamplable trajectories — i.e. the substrate EPF needs.
3. Grid decision: **think + trigger, prompt #7 only** (best think cell, most length-robust),
   `tokens_per_step=30`, `max_steps=6` (93% of P7-think trajectories fit), single-audio.

**Artifacts:** `results/run12_af3_full5090/af3_cot_screen.*`, `af3_chunk_probe100.jsonl`,
`af3_param_check200.jsonl`, `af3think_param_check{200,1000}.jsonl`,
`af3base_cotuser_check1000.jsonl`, `af3think_notrigger_check1000.jsonl`;
`af3_chat_template.jinja`; harness changes: `diversity_probe`
`--tokens-per-step/--think-trigger/--ids-file`, gate scripts `--subset/--audio-root`,
`its_hub` EOS-stop. Merged checkpoint: `/home/tariqvrh4/af3-think-merged`.

## 19. Run 13 — the EPF grid on AF3-Think (P7, single-audio MMAU-Pro)

**Why:** the Run-11 question on a second model, on the only AF3 configuration where step-wise
resampling is meaningful (Run 12). Config: **P7 + think + trigger**, `tokens_per_step=30`,
`max_steps=6`, temp 0.8, ess 0.6 / early 0.7, systematic, style logit, signals
{mean_logprob, entropy}, budgets {1,8,16,32}; single-audio only. Run in two ids-file stages over
one resumable JSONL: **stage 1** = the seed-11 length-balanced 1,000 (same items as every Run-12
probe), **stage 2** = the remaining 3,660 single-audio MCQ. Smoke (16 items, b8): 0 errors,
0.87 s/item.

**Full run: 37,280 EPF rows, 0 errors, ~10 h wall on both GPUs** (stage 1 4.4 h + stage 2 5.6 h;
stage-2 s/item ≈ half of stage 1's — the seed-11 sample was deliberately ultra-long-heavy).
Stage-1 cell s/item (2 GPUs): b1 0.28 / b8 1.08 / b16 2.16 / b32 4.4 — linear in budget, on the
Qwen anchors (§9 of SETUP_GUIDE); natural-mix stage-2: 0.15 / 0.40 / 0.75 / 1.4.

**FULL single-audio set — n=4,638 gradeable/cell, 95% CI ≈ ±1.4 pp:**

| signal | b1 | b8 | b16 | b32 | oracle@32 | majority@32 | gap@32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| mean_logprob | 0.549 | 0.573 | **0.576** | 0.569 | **0.822** | 0.588 | **+0.25** |
| entropy | 0.560 | 0.580 | **0.582** | 0.582 | **0.843** | 0.604 | **+0.26** |

distinct-answer ratio 0.98→0.06 across budgets; consensus ≈0.85; final ESS 0.79–0.87; parse
0.98–0.99 (think-mode answer commitment survives the step machinery — resampling genuinely ran
on multi-step trajectories, unlike anything base AF3 could offer). The n=1,000 stage-1 slice
told the same story with numbers ~2–3 pp lower across the board (harder length mix).

**Findings (full scale):**
1. **Selected accuracy saturates by budget 8–16** at ~0.57–0.58 (+2.2–2.4 pp over the
   single-trajectory anchor); every b8→b16→b32 step is within ±1 pp.
2. **Oracle climbs to 0.82–0.84**; the oracle−selected gap is **+0.25/+0.26 — statistically
   identical to Qwen's +0.26** (Run 11). Majority ≈ selected +2 pp. The two self-certainty
   signals are equivalent (entropy ≈ +1 pp, borderline at this CI).
3. → **The Run-11 verdict replicates cross-model**, including on a model whose reasoning was
   explicitly trained-in via an adapter: exploration scales beautifully, no self-certainty signal
   converts it into selected accuracy, the selector is the wall.

**Caveats:** single-audio only (430 multi-audio MCQ excluded — Run 12's processor limitation);
prompt #7 only (grid comparability to Run 11's 4-prompt mean is qualitative); AF3-Think trigger
sentence appended to the user turn is part of the prompt definition for this run.

**Bootstrap** (`epf_p7think_bootstrap.html`): a 100-question eval wobbles ±0.049 on these cells;
the full-set numbers are precise to ±0.0073 (closed-form cross-check matches). Note:
`epf_bootstrap.py` got a small fix so its cross-check picks cells present in the CSV
(single-prompt grids used to KeyError).

**Artifacts:** `results/run12_af3_full5090/epf_p7think.{jsonl,csv,log}`,
`epf_p7think_bootstrap.html`, `stage{1,2}_ids*.txt`; reproduce block in §16.

## 20. Run 14 — Gemma 4 E2B IT on the ≤30 s MMAU-Pro subset (IN PROGRESS)

**Why:** the Run-11/13 question on a third audio LM, `google/gemma-4-E2B-it` (vLLM
`Gemma4ForConditionalGeneration`, served from the `af3serve` env — vllm 0.22.1 +
transformers 5.13 have native gemma4 audio support; no new env, no chat-template override:
the native template takes a system turn and renders one `<|audio|>` per clip).

**The defining constraint — 30 s audio window.** Gemma 4 E2B encodes at most
**750 audio tokens × 40 ms = 30 s per clip** (`processor_config.json`); vLLM
*warns and truncates* longer clips (`gemma4_mm.py`), it does not error. MMAU-Pro's
clip p50 is ~50 s (max 600 s), so most items would be silently half-heard.
**Decision (2026-07-10): run only items where EVERY clip ≤ 30 s** — measured with
[`make_le30s_ids.py`](make_le30s_ids.py) from the audio headers (the `le30s` parquet
only exists for testmini): 1,947 single-audio + 243 multi-audio = 2,190 eligible.
**Preflight verified the truncation empirically**: 1-token generation on a 600 s clip →
770 prompt tokens (the 750 cap + text) vs 270 for a 10 s clip; server warns, no HTTP error.

**Multi-audio: DROPPED after preflight (2026-07-10) → final scope = 1,947 single-audio.**
Unlike AF3 (whose multi-audio failure was template/processor *plumbing*), Gemma 4's
plumbing is fine — 2- and 3-clip requests are accepted and `usage.prompt_tokens` adds up
exactly (95 tone + 270 natural → 347 joint) — but **comprehension is unreliable**: with
2 clips the model answers "3" to "how many clips?", describes four nonexistent clips,
and picks the same option letter regardless of clip order (440/880 Hz pitch pairs and
tone-vs-natural discrimination both order-insensitive). Consistent with the Gemma-3n
lineage's single-audio-per-prompt training. Single-clip grounding is solid (A/B causality
0.400 with audio vs 0.200 without, 10/15 answers changed).

**Serving bug found (and fixed): vLLM gemma4 batched-audio crash.** The first concurrent
probe killed BOTH engines instantly (`AttributeError: 'list' object has no attribute
'squeeze'` in `gemma4_mm._process_audio_input`): vLLM caches Gemma-4 audio features
*unpadded per item*, and a batch of different-length audios can't be stacked — it reaches
the audio tower as a **list**, which the code doesn't handle. Single requests (batch 1)
work, which is why all gates passed first. **Upstream vLLM main has the identical code as
of 2026-07-10**, so this is not fixed by upgrading. Fix (user decision: patch a *clone*,
keep `af3serve` pristine): env **`gemmaserve`** = clone of `af3serve` + a ~20-line local
patch that re-pads the list to the batch max mel length and rebuilds the validity mask;
original backed up at `envs/gemmaserve/.../gemma4_mm.py.orig`. `scripts/serve_gemma4_e2b.sh`
serves from the clone. Verified against the exact failing workload (concurrency-16 probe).

**Config (parity with Runs 11/13):** temp 0.8, ess 0.6 / early 0.7, systematic, style
logit, signals {mean_logprob, entropy}, budgets {1,8,16,32}, budget-staged over one
resumable JSONL (`scripts/run14_sweep.sh`). Prompt + step mode + max_steps chosen by
the Run-14 probes (see below) — **no `--think-trigger`** (that is AF3's adapter
activation, not applicable here). **Final config (2026-07-14): prompts {4,7},
`tokens_per_step=30`, `max_steps=6`** — user decision for exact Run-13 step-config
parity (tok30 × 6 = 180-token budget), superseding the initial coverage-based pick of
12; a max-12 partial run (1,514 b1 rows) is archived as
`epf_gemma_le30s.jsonl.max12_partial`.

**Prompt/step selection protocol (replaces Run 12's 9-prompt screen by user decision):**
1. gates (`phase0_gate`) + A/B causality + 30 s-truncation check + 2-clip pitch check;
2. **100-item probe** (`probe_ids_100.txt`, seed-14 category round-robin over the ≤30 s
   single-audio pool), prompts {4,5,7,9}, budget-1 plain generation at **t=0 AND t=0.8**
   (`cot_compare --prompts 4,5,7,9 --temperature ...`; separate JSONLs per temp — the
   resume key has no temperature) → user picks the sweep prompt;
3. [`step_probe.py`](step_probe.py) (new; reimplements Run 12's lost param-check):
   budget-1 EPF-style trajectories in both step modes (`\n\n` max6 vs tok30 max8) →
   step mode + max_steps by the Run-12 criteria (real steps, no near-empty
   high-logprob steps, ≤~10% max_steps truncation, p95 coverage);
4. 16-item EPF smoke (all single-audio after the multi drop) → 0 errors, sane metrics,
   s/item anchor.

**Probe results (2026-07-13, seed-14 100-item sample, 0 errors everywhere):**

*100-item 4-prompt × 2-temp budget-1 probe* (`probe100_t00/t08.*`; acc_all / acc_excl-1,
n=100 gradeable, ±~10 pp):

| # | prompt | t=0 | t=0.8 | reasoned | avg words | avg `\n\n`-chunks |
|---|---|---|---|---|---:|---:|
| 4 | plan-and-solve | **0.520 / 0.461** | 0.460 / 0.393 | 1.00 | 205–212 | **6.4–6.8** |
| 5 | least-to-most | 0.420 / 0.348 | 0.460 / 0.404 | 0.72–0.74 | 57–59 | 4.6–4.7 |
| 7 | format-forcing | 0.470 / 0.416 | **0.500 / 0.449** | 1.00 | 108–111 | 3.9 |
| 9 | evidence-grounded | 0.490 / 0.427 | 0.490 / 0.427 | 1.00 | 117–125 | 2.3 |

Unlike AF3, Gemma reasons under every prompt and emits `\n\n` freely. All accuracy
differences are within noise; **user pick: sweep P4 + P7** (best greedy + most chunkable;
best t=0.8, and the Run-13 prompt).

*Step probe* (`step_probe.py`, new — reimplements Run 12's lost param-check; budget-1
EPF-style trajectories, t=0.8, n=100 × 4 prompts per config):
- **`\n\n` stepping is DEGENERATE on Gemma** (`step_probe_dnl_max6.*`): median
  **1 token/step** (the stop-string fires inside/right after the first tokens), 67–78%
  unparsed on P4/5/7, 60–97% of trajectories truncated at max_steps, acc collapses to
  ~0.11. The rich `\n\n` chunking of *plain* generations does not survive as a stepping
  delimiter — the Run-12 tiny-step/logprob-hijack pathology in a new form. Cross-model
  lesson: chunk counts in free generation say nothing about `stop`-based stepping.
- **`tokens_per_step=30` is healthy** (`step_probe_tok30_max8/12.*`): median 30 tok/step,
  unparsed P4 7% / P7 1% (at max12), acc back at plain-gen levels (P4 0.43, P7 0.50),
  `Answer:` stop fires 89–98%, EOS detection works. Coverage: P7 ends naturally 99% by
  step 9 (mean 5.2); P4 is verbose (mean 263 tok, mean 9.2 steps, p95 = 13). Within a
  6-step budget: **P7 fits 83%, P4 fits 14%** of trajectories — the user chose
  **max_steps = 6 anyway** (Run-13 parity; see config note above), so P4 cells are
  pre-registered as a truncated-CoT regime.

*EPF smoke* (`smoke_epf16.*`: 16 items × {P4,P7} × both signals, b8): **64 rows, 0
errors**, parse 0.97–1.00, final ESS 0.86–0.96, distinct 0.15–0.18 — the full machinery
(logprob weights, systematic resampling, EOS stops) runs end-to-end. Cell speed 0.34–0.64
s/item @b8 (2 GPUs) → **full-sweep estimate ~5–8 h wall** (1,947 × 2 prompts × 2 signals
× {1,8,16,32}).

**Sweep results — max_steps=6 (the Run-13-parity config): 31,152 rows, 0 errors,
n=1,934 gradeable/cell (95% CI ≈ ±2.2 pp).** Selected / oracle / majority by budget:

| cell | b1 | b8 | b16 | b32 | oracle@32 | majority@32 | gap@32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| P4 mean_logprob | 0.249 | 0.271 | 0.299 | 0.307 | 0.560 | 0.357 | +0.25 |
| P4 entropy | 0.248 | 0.305 | 0.309 | 0.323 | 0.608 | 0.359 | +0.29 |
| P7 mean_logprob | 0.428 | **0.437** | 0.425 | 0.414 | 0.616 | 0.428 | +0.20 |
| P7 entropy | 0.422 | **0.427** | 0.423 | 0.402 | 0.664 | 0.429 | +0.26 |

(P4 parse 0.71–0.80 across budgets — truncation regime as pre-registered; P7 parse
0.93–0.96. distinct-ratio 0.95→0.05, consensus 0.65–0.86, final ESS 0.81–1.0.)

**Findings (third model):** (1) **P7 selected accuracy peaks at b8 (0.437/0.427) and
then *declines* to b32** — no budget scaling; (2) **oracle climbs monotonically** to
0.62–0.66 (P7) — the oracle−selected gap is **+0.20–0.29 @b32**; (3) majority ≈
selected +0.02; (4) the two signals are equivalent. **The Run-11/13 verdict replicates
on Gemma 4 E2B**: exploration scales, no self-certainty signal converts it, the
selector is the wall. (P4's upward crawl with budget is a truncation-parse lottery —
more particles = more chances one finishes; see the max12 ablation below.)

**max_steps=12 ablation (Run 14b, `scripts/run14_sweep_max12.sh`,
`epf_gemma_le30s_max12.*`; seeded from the archived max12 partial; launched
concurrently with the max6 b32 stage — timings not cost anchors): 31,152 rows,
0 errors.**

| cell (max12) | b1 | b8 | b16 | b32 | oracle@32 | majority@32 | gap@32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| P4 mean_logprob | 0.430 | 0.439 | 0.452 | **0.454** | 0.598 | 0.466 | +0.14 |
| P4 entropy | 0.407 | 0.447 | 0.448 | 0.446 | 0.603 | 0.461 | +0.16 |
| P7 mean_logprob | 0.433 | 0.454 | **0.458** | 0.449 | 0.617 | 0.450 | +0.17 |
| P7 entropy | 0.444 | **0.460** | **0.460** | 0.447 | 0.643 | 0.456 | +0.20 |

(parse 0.95–0.98 everywhere — the truncation artifact is gone.)

**6-vs-12 verdict:** (1) **P4 is +13–18 pp better with 12 steps at every budget** —
the max6 P4 numbers were pure truncation artifacts; (2) **P7 gains only +2–4.5 pp**,
mostly at b32 where max6 had *declined* (resampling under truncation is actively
harmful at high budget); (3) **the scaling story is identical in both configs**:
selected saturates by b8–16 (best cell 0.460 = +1.6–4 pp over b1) and dips at b32,
oracle climbs to 0.60–0.64, gap +0.14–0.20, majority ≈ selected +0.01, signals
equivalent. So the step budget moves the *level* (via completion rate) but not the
*shape* — the selector remains the wall on the third model. Bootstraps:
`epf_gemma_le30s_bootstrap.html` (max6) and `epf_gemma_le30s_max12_bootstrap.html`.

**Run 14c (planned, collaborator hardware): budgets {64, 128}** — same max12 config
(P4+P7 × both signals × the same committed 1,947 ids), fresh JSONL
`epf_gemma_le30s_max12_b64128.*` merged with ours at analysis time (no key overlap).
Runner: `scripts/run14_collab_b64_128.sh`; full setup/smoke runbook: `RUN14_COLLAB.md`
(env pins, model sha256, dataset layout, the mandatory gemma4 vLLM patch
`scripts/patch_vllm_gemma4.py`, greedy anchors, 16-item EPF smoke). Caveat when
reporting: b64/128 come from a different machine than b1–32 (same pins/GPU class).
Note for analysis: `epf_bootstrap.py`'s budget axis is hardcoded to {1,8,16,32} —
extend it before bootstrapping the merged grid.

## 21. Run 15 — Gemma 4 **E4B** IT, the full grid b1→128 (planned, collaborator hardware)

**Why:** the E2B story (§20) on the bigger sibling — does model scale change the
selected-saturation / oracle-climb shape, and where does the E4B level sit? Config =
Run 14's max12 arm exactly: prompts {4,7}, signals {mean_logprob, entropy},
`tokens_per_step=30`, `max_steps=12`, temp 0.8, ess 0.6/0.7, systematic+logit,
**budgets {1,8,16,32,64,128}**, the same committed 1,947-item ≤30 s single-audio ids
(**E4B has the identical 30 s/clip cap**: `audio_seq_length=750 × 40 ms` — verified
from its processor config). Model: `google/gemma-4-E4B-it` (16.0 GB, revision
`fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd`, `model.safetensors` sha256
`cfbd3d2f1cd71bd471c37fe2bf8546d5028d41e5736f64e1ca6c6b8893125503`). The gemma4
vLLM batched-audio patch (§20) is architecture-level and covers E4B unchanged.
Runner: `scripts/run15_e4b_b1_128.sh` (E4B servers on ports 820x via
`scripts/serve_gemma4_e4b.sh`, so it can run on a disjoint GPU set concurrently with
Run 14c). Local preflight (gates/truncation/anchors/smoke): *results below.*

**Preflight (this box, 2026-07-15): ALL PASS.** Weights sha256 verified against HF LFS
metadata; gates PASS ×2 endpoints (Gate-1 anchor: first token `'Kn'`
logprob=`-4.268167495727539`, top_logprobs=20); 30 s truncation identical to E2B
(600 s clip → 770 prompt tokens, warn-not-fail); greedy anchors (t=0, local-path):
item `22211743…` P4→D (gold B) / P7→B ✓, item `aceca2ce…` P4→C ✓ / P7→C ✓ (differs
from E2B's P7→A on the second item — a genuine checkpoint-identity check); 16-item
EPF smoke in the exact Run-15 config: **64 rows, 0 errors**, parse 0.95–1.00, ESS
0.84–0.93 (`results/run15_gemma4e4b/smoke_e4b16.*`). Smoke s/item @b8 (2 GPUs):
P4 0.75, P7 0.37–0.44 (~1.3–2× E2B) → **b1→128 estimate ≈ 35–45 h per 2 GPUs**,
scaling ≈ linearly with endpoint count.

**400-item b8 check + the P4 drop (2026-07-15).** A 400-item seeded sample (prefix-
compatible with `probe_ids_100`; `results/run15_gemma4e4b/probe_ids_400.txt`,
`smoke_e4b400.*`; 1,600 rows, 0 errors) compared E4B vs E2B at the identical config
and items: **P7: E4B ≥ E2B** (+1.5–2.7 pp, within ±5 pp CI); **P4: E4B −10–13 pp
BELOW E2B, beyond CI** — E4B's plan-and-solve writes ~300 words and truncates at the
shared tok30×12 budget (parse dips to 0.93–0.96); the pre-registered carry-over
caveat materialized. **Decision (user, 2026-07-16): Run 15 = P7 only.**

**Local results (this box) — P7 × both signals × budgets {1,8,16}, full 1,947 items
(n=1,934, ±2.2 pp; 11,682 rows, 0 errors; b8 seeded from the 400-item check):**

| signal | b1 | b8 | b16 | oracle@16 | E2B sel (b1→b16) |
|---|---:|---:|---:|---:|---|
| mean_logprob | 0.465 | 0.464 | **0.479** | 0.574 | 0.433 → 0.458 |
| entropy | 0.464 | **0.480** | 0.476 | 0.584 | 0.444 → 0.460 |

E4B sits **+2–3 pp above E2B** at every point (best single-trajectory anchor yet:
0.465); selected is already flat b1→b16 (+0–1.6 pp) while oracle climbs +11–12 pp —
the saturation shape, fourth configuration. Notably **E4B's oracle climbs slower
than E2B's** (0.574/0.584 vs 0.601/0.628 @b16): the bigger model is more
deterministic, so exploration surfaces fewer alternatives and the oracle−selected
gap is smaller (+0.10 vs +0.14–0.17).

**Remaining (collaborator): P7 × both signals × budgets {32,64,128}** — the committed
JSONL carries b1–16, so `scripts/run15_e4b_b1_128.sh` resumes past them (≈12 h per
2 GPUs). Artifacts: `results/run15_gemma4e4b/epf_gemma4e4b_max12.{jsonl,csv,log}`.
Caveats: step params carried from E2B (no E4B step-probe; P4's truncation is why it
was dropped); E4B b32–128 from a different machine than b1–16 (same pins/GPU class).

**Caveats (pre-registered):** ≤30 s single-audio subset only (1,947 of 5,090 MCQ →
comparisons with Runs 11/13 are qualitative, not item-matched); two prompts {4,7};
**P4 at tok30×6 truncates 86% of trajectories** (user-accepted, Run-13 parity — P4
numbers reflect truncated CoT, expect elevated unparsed); `epf_bootstrap.py` renders
prompts {4,5,7,9} only (fine — picks restricted to that set).

**Artifacts:** `results/run14_gemma4e2b/` — `durations_test5090.csv`,
`le30s_{single,multi,all}_ids.txt`, `probe_ids_100.txt`, `smoke_ids_16.txt`,
`probe100_t00.*`, `probe100_t08.*`, `step_probe_*.{jsonl,csv,log}`, `smoke_epf16.*`,
`epf_gemma_le30s.{jsonl,csv,log}` + `epf_gemma_le30s_bootstrap.html`.
Scripts: `scripts/serve_gemma4_e2b.sh`, `scripts/run14_sweep.sh`.
