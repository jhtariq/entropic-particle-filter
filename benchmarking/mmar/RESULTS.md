# MMAR — results log

Experiments on the MMAR benchmark (1,000 single-audio MCQ reasoning items).
**Protocol lineage:** Runs 1–3 replicate `benchmarking/mmau_pro/RESULTS.md`
§13 (Run 11, Qwen2.5-Omni-7B EPF grid), §14 (Run 12, Omni-3B) and §16 (Run 15,
Qwen2-Audio-7B-Instruct) — identical prompts/signals/budgets/EPF knobs/serving —
on MMAR instead of MMAU-Pro. Run numbering restarts at 1 for this benchmark;
mmau_pro numbering is untouched.

Scoring convention (mirrors mmau_pro): gold answers are fuzzy-matched to a choice
at load time (`match_answer_index`, threshold 0.85); the 4/1,000 items whose gold
matches no single choice run but score `null` and are **excluded from accuracy
denominators** (n = 996 per cell).

## TL;DR

Runs 1–2 (July 22, 2026) replicate mmau_pro Runs 11/12 (Qwen2.5-Omni-7B/3B EPF
grids, P4+P5 × {mean_logprob, entropy} × b{1,8,16,32}) on the full 1,000-item MMAR
set — 32,000 rows total, 0 errors. **The MMAU-Pro findings transfer to MMAR almost
line for line:** selected accuracy saturates by b8–b16 (7B ~0.59–0.63, 3B
~0.52–0.55); oracle climbs monotonically (7B → 0.86–0.88, 3B → 0.85–0.91) and is
**size-invariant** while selection is not; majority ≈ selected + ~0.02; the two
self-certainty signals are equivalent; and the oracle−selected gap **widens as the
model shrinks** (7B +0.22…+0.30 → 3B +0.30…+0.41, peaking at +0.41 for P5×entropy:
oracle 0.910, selected 0.497). The selector is the wall on MMAR too. Every matched
cell lands within 1–3 pp of its MMAU-Pro counterpart despite MMAR being the
nominally harder reasoning benchmark. Our lettered scoring agrees with MMAR's
official token-overlap scorer to ≤ +0.007 per cell.

**Run 3** (same day) adds the cross-family check — Qwen2-Audio-7B-Instruct, P4+P9
on a pinned ≤30 s subset (983 items; its encoder truncates at 30 s), single GPU:
the shape replicates on a third model / second family (selected saturates ~0.48–0.50,
oracle climbs to 0.66–0.71, gap +0.19/+0.21), the oracle ceiling is
family-dependent (tracks base competence, ~0.2 below Omni-7B's), MMAR again sits
2–4 pp above the model's MMAU-Pro cells, and Run 15's "majority strictly beats
selection on P9" does NOT transfer (majority ≈ selected here). 15,728 rows, 0
errors, ~3.7 h on one GPU.

**b64/b128 extension (Runs 1+2, 2026-07-27/28):** oracle still climbing at 128
particles — 7B 0.90–0.95, 3B 0.88–0.94 (size-invariance holds at every budget) —
while selection stays flat from b8 through b128 on both sizes; the 3B's P5×entropy
gap reaches **+0.47** (oracle 0.936, selected 0.467), the campaign's widest.
Four doublings of compute bought ~0 selection. The selector is the whole game.

**Run 5** (July 22, 2026) adds the fourth model / third family and the first
LoRA-served audio path — Phi-4-multimodal-instruct (speech-LoRA), P4+P8 ×
{mean_logprob, entropy} × b{1,8,16,32}, full 1,000-item set, single GPU (mmau_pro
Run 18 replica). 16,000 rows, 0 errors, 8.4 h on one GPU. The shape replicates
(selected saturates by b8 ~0.44; oracle climbs to 0.81–0.86 @b32) with the
**widest oracle−selected gap in the series (+0.36/+0.40 means, peak +0.42 on
P8×entropy — rivalling Omni-3B's +0.41)**. Two firsts: (a) the "MMAR sits above
MMAU-Pro" regularity **flips** — Phi-4-MM scores 3–4.5 pp *below* its Run 18 cells,
so the offset is model-dependent, not benchmark-level; (b) its b1 selected (0.42)
is the lowest of the four models yet its b32 oracle reaches into Omni territory —
weak selector, strong exploration. Music weakest for the fourth model running;
official-scorer agreement ≤ +0.006 per cell.

**Run 6** (July 22–23, 2026) adds the fifth model / fourth family — Kimi-Audio-7B-Instruct
(Whisper-v3 front-end), P2+P4 × {mean_logprob, entropy} × b{1,8,16,32} on the ≤30 s `le30s`
subset (979 gradeable), single GPU, served through the Run 19 vLLM wrapper (`serve_kimi.py`,
which monkey-patches six Kimi bugs in vLLM 0.22.1). 15,728 rows, 0 errors, ~21 h on one GPU
(the slowest model in the series — batch-of-one audio encoding). The shape replicates
(selected saturates by b8 ~0.51–0.56; oracle climbs to 0.73–0.87 @b32; gap +0.27/+0.32 means,
peak +0.33 on P4×entropy — mid-pack). Crucially the **"MMAR above MMAU-Pro" regularity holds
for Kimi** (+2.3…+5.9 pp on selected at matched cells) — so Phi-4-MM's Run 5 flip does not
generalize; across five models MMAR-higher is the majority case (4 of 5), Phi-4-MM the lone
inversion. Music weakest for the fifth model running; 2-choice items sel 0.545 — the healthiest
binary performance yet; official-scorer agreement +0.003…+0.009 per cell.

## 1. Run 1 — Qwen2.5-Omni-7B EPF grid (mmau_pro Run 11 replica, P4+P5)

**Why.** Cross-benchmark generalization test of mmau_pro Run 11: same model, same EPF
knobs, on MMAR (harder single-audio reasoning). Does "selected saturates / oracle
climbs / the selector is the wall" transfer?

**Config.** Qwen2.5-Omni-7B (rev `ae9e1690543ffd5c0221dc27f79834d0294cba00`), served
`qwen-omni`, one bf16 vLLM replica per GPU on :8100/:8101 (`--max-model-len 32768
--enforce-eager --gpu-memory-utilization 0.85`, expandable_segments,
FLASHINFER_SAMPLER=0, `--allowed-local-media-path /home/exx/inference-time-scaling`).
Grid: prompts **{4,5}** × signals {mean_logprob, entropy} × budgets {1,8,16,32},
`--select all` over the full 1,000-item set; temp 0.8, ess-threshold 0.6, early-phase
0.7, step_token `\n\n`, stop `Answer:`, max_steps 6, 300 tok/step, systematic
resampling, style logit. Budget-staged detached driver, `--max-inflight` 24@b1 / 64
otherwise. **Scope note:** Run 11 ran P{4,5,7,9}; this run was narrowed to P4+P5 by
user decision (halves compute; the side-by-side below re-slices Run 11 to P4+P5 so
the comparison stays apples-to-apples). The b1 cells for P7/P9 that ran before the
narrowing are preserved in `mmar_run01_p79_b1_partial.jsonl` (excluded from all
analysis); P7/P9 can be added later by rerunning the same JSONL with `--prompts 7,9`.

**n / errors / cost.** 1,000 items, 996 gradeable per cell (4 ungradeable golds, see
header). 16 cells × 1,000 = **16,000 rows, 0 errors** (every cell exactly 1,000; no
resume sweeps needed). Wall: b8 15 min, b16 35 min, b32 84 min (+b1 ~5 min) ≈ **2.3 h
on the GPU pair**. Cell rates at b8: 0.30–0.32 s/item (P4), 0.12–0.14 s/item (P5).
Gates: phase0 PASS/PASS ×2 endpoints; A/B causality 0.80 with audio vs 0.27 without
(12/15 flips); capacity audit PASS (25.1 tok/s, max prompt 1,443 tok ≪ 32k). Screen
(40 stratified, P4,5,7,9 greedy): parse 92–100%, P4 avg 3.9 chunks.

**Results (per cell, n=996; sel / oracle / majority).**

| prompt | signal | b1 | b8 | b16 | b32 |
|---|---|---|---|---|---|
| P4 plan-and-solve | mean_logprob | .568 / .568 / .568 | .567 / .720 / .571 | .588 / .796 / .606 | .590 / .870 / .612 |
| P4 plan-and-solve | entropy | .545 / .545 / .545 | .577 / .748 / .592 | .586 / .820 / .606 | .575 / .880 / .599 |
| P5 least-to-most | mean_logprob | .582 / .582 / .582 | .608 / .777 / .617 | .626 / .812 / .637 | **.632** / .851 / .630 |
| P5 least-to-most | entropy | .568 / .568 / .568 | .611 / .796 / .619 | .597 / .849 / .617 | .613 / .881 / .620 |

**Side-by-side vs mmau_pro Run 11 (both re-sliced to P4+P5 means; sel / oracle / maj).**

| signal | b | MMAU-Pro Run 11 | MMAR Run 1 |
|---|---|---|---|
| mean_logprob | 1 | .558 / .558 / .558 | .575 / .575 / .575 |
| mean_logprob | 8 | .592 / .720 / .598 | .588 / .748 / .594 |
| mean_logprob | 16 | .597 / .775 / .606 | .607 / .804 / .621 |
| mean_logprob | 32 | .594 / .820 / .610 | .611 / .861 / .621 |
| entropy | 1 | .559 / .559 / .559 | .557 / .557 / .557 |
| entropy | 8 | .590 / .754 / .605 | .594 / .772 / .606 |
| entropy | 16 | .586 / .807 / .608 | .592 / .835 / .612 |
| entropy | 32 | .581 / .850 / .597 | .594 / .880 / .610 |

**Takeaways.**
1. **The Run 11 shape transfers to MMAR, almost line for line.** Selected gains
   +3.6/+3.7 pp from b1→b32 with most of it by b8–b16 (bootstrap SE ±0.016 → the
   selected gains are ~2σ, borderline exactly as on MMAU-Pro); oracle climbs
   monotonically to 0.861/0.880; majority ≈ selected + 0.01–0.02 at every budget;
   the two signals are equivalent within noise (mean_logprob a touch better on
   selected at b16/32, entropy a touch better on oracle).
2. **The selector is the wall here too:** oracle−selected = +0.25 (mean_logprob) /
   +0.29 (entropy) at b32, ~16–18σ. Swarm diagnostics match Run 11: distinct-answer
   ratio 0.97 → 0.06 as budget grows, ESS ratio drifts down (0.71–0.88 at b32).
3. **MMAR is not harder for Omni-7B than MMAU-Pro on the same cells** — MMAR sits
   +0.3…+1.7 pp above on selected and +1.7…+3.0 pp above on oracle. "Harder
   reasoning benchmark" does not translate into a lower EPF ceiling; the
   single-audio ≤56 s clips likely help.
4. **Where the selector fails most** (P4+P5 mean @b32): Signal Layer sel 0.488 vs
   oracle 0.866 (widest gap); music is the weakest modality (sel 0.517, oracle
   0.808) while mix-sound-speech is strongest (sel 0.669). By choice count:
   2-choice items score sel 0.528 — barely above the 0.50 random baseline, i.e.
   MMAR's binary items are genuinely hard, they do not inflate accuracy (the
   opposite of mmau_pro's single-choice quirk); 4-choice sel 0.619. Non-English
   items (125/cell) sel 0.620 — no penalty vs overall.
5. **Scoring cross-check:** official token-overlap scorer (mapping the selected
   letter back to choice text) agrees with our lettered scoring to +0.001…+0.007
   per cell (`crosscheck_b1.json`, `crosscheck_b32.json`).

**b64/b128 extension (2026-07-27, same serving/knobs, staged inflight 64).**
8,000 new rows, 0 errors (JSONL now 24,000 = 6 budgets × 4 cells × 1,000); b64
3.3 h, b128 4.7 h on the GPU pair — b128 ran *sub*-linearly vs b64 (1.45×): in the
one-item-per-endpoint regime vLLM absorbs the larger particle fan-out efficiently.

| cell | b64 sel / orc / maj | b128 sel / orc / maj |
|---|---|---|
| P4 × mean_logprob | .577 / .909 / .608 | .578 / .946 / .618 |
| P4 × entropy | .588 / .921 / .626 | .579 / .935 / .608 |
| P5 × mean_logprob | .614 / .876 / .630 | .597 / .896 / .628 |
| P5 × entropy | .612 / .895 / .630 | .614 / .922 / .627 |

**Extension takeaway:** oracle has NOT saturated by b128 — it reaches 0.90–0.95
and every doubling still buys coverage — while selected (and majority) have been
flat since b8 through four consecutive doublings (.58–.61). Gap at b128:
**+0.28…+0.37**. The selector wall stands at every budget measured.

**Artifacts** (`results/run01_epf_grid/`): `mmar_run01.{jsonl,csv,log}`,
`mmar_bootstrap.html` (10k-resample bootstrap; three sections — Runs 1+2+3),
`screen40_p4579.*`, `smoke16/`, `audit_capacity.json`, `crosscheck_b{1,32}.json`,
`mmar_run01_p79_b1_partial.jsonl`, `servers/` (vLLM + watchdog logs), `stages.log`.

**Reproduce.**
```bash
bash benchmarking/mmar/scripts/serve.sh    benchmarking/mmar/scripts/config_run01.sh
nohup setsid bash benchmarking/mmar/scripts/watchdog.sh benchmarking/mmar/scripts/config_run01.sh \
  > benchmarking/mmar/results/run01_epf_grid/servers/watchdog.log 2>&1 &
nohup setsid bash benchmarking/mmar/scripts/run_grid.sh benchmarking/mmar/scripts/config_run01.sh \
  > benchmarking/mmar/results/run01_epf_grid/stages.log 2>&1 &
# then: epf_bootstrap --in "Qwen2.5-Omni-7B — MMAR Run 1=<csv>" ; official_crosscheck --csv <csv>
```

## 2. Run 2 — Qwen2.5-Omni-3B EPF grid, P4+P5 (mmau_pro Run 12 replica)

**Why.** The size axis: Run 12 found oracle coverage size-invariant while selection
degrades — does that hold on MMAR? Run 2 is prompt-matched to Run 1 (P4+P5), so the
7B-vs-3B comparison is exact.

**Config.** Qwen2.5-Omni-3B (rev `f75b40e3da2003cdd6e1829b1f420ca70797c34e`), served
`qwen-omni-3b`, everything else identical to Run 1 (serve template, both GPUs,
:8100/:8101, same EPF knobs, budget-staged 24@b1/64). Prompts **{4,5} only**, per
Run 12: the 40-item screen confirmed P7/P9 break mechanically on the 3B on MMAR too
(P7: 8-word answers, 1.1 chunks, 5% reasoned; P9: single-`\n` steps, 1.0 chunks) —
nothing for `\n\n` resampling to work with.

**n / errors / cost.** 16 cells × 1,000 = **16,000 rows, 0 errors**, no resume
sweeps. Wall: b1 2.5 min, b8 11.5 min, b16 28.5 min, b32 72 min ≈ **1.9 h GPU-pair**
(b8 rate ≈ same as 7B — audio encoding dominates, not model size). Gates: phase0
PASS/PASS ×2; capacity PASS (identical tokenization); A/B causality 0.667 with vs
0.533 without audio, 6/15 flips — audio reaches the model, but coupling is visibly
weaker than the 7B's 0.80-vs-0.27, consistent with a smaller model leaning on priors.

**Results (per cell, n=996; sel / oracle / majority).**

| prompt | signal | b1 | b8 | b16 | b32 |
|---|---|---|---|---|---|
| P4 plan-and-solve | mean_logprob | .532 / .532 / .532 | .547 / .733 / .554 | .558 / .776 / .571 | .550 / .851 / .545 |
| P4 plan-and-solve | entropy | .526 / .526 / .526 | .524 / .738 / .532 | .538 / .803 / .561 | .541 / .858 / .562 |
| P5 least-to-most | mean_logprob | .461 / .461 / .461 | .523 / .786 / .547 | .543 / .838 / .566 | .540 / .874 / .565 |
| P5 least-to-most | entropy | .470 / .470 / .470 | .510 / .812 / .541 | .522 / .880 / .553 | .497 / **.910** / .540 |

**Side-by-side vs mmau_pro Run 12 (P4+P5 means; sel / oracle / maj).**

| signal | b | MMAU-Pro Run 12 | MMAR Run 2 |
|---|---|---|---|
| mean_logprob | 1 | .486 / .486 / .486 | .496 / .496 / .496 |
| mean_logprob | 8 | .531 / .729 / .552 | .535 / .760 / .551 |
| mean_logprob | 16 | .538 / .783 / .560 | .551 / .807 / .569 |
| mean_logprob | 32 | .538 / .838 / .563 | .545 / .863 / .555 |
| entropy | 1 | .481 / .481 / .481 | .498 / .498 / .498 |
| entropy | 8 | .504 / .756 / .552 | .517 / .775 / .537 |
| entropy | 16 | .504 / .817 / .552 | .530 / .841 / .557 |
| entropy | 32 | .509 / .856 / .555 | .519 / .884 / .551 |

**Takeaways.**
1. **Run 12 replicates on MMAR, cell by cell** — every matched cell within 1–3 pp,
   same saturation/climb shapes, signals equivalent.
2. **Oracle is size-invariant on MMAR; selection is not.** 3B b32 oracle
   .863/.884 vs 7B's .861/.880 — indistinguishable. Selected drops ~6 pp going
   7B→3B (.611/.594 → .545/.519). The oracle−selected gap widens from +0.25/+0.29
   (7B) to +0.32/+0.37 (3B means), peaking at **+0.41** on P5×entropy (oracle
   0.910, selected 0.497) — the widest gap measured on either benchmark at these
   prompts. Confirms Run 12's core claim: "the oracle−selected gap is the whole
   story and it widens as models shrink."
3. **P5 stop-token caveat leaves a milder trace than on MMAU-Pro:** P5 parse
   0.90–0.97 (vs 1.00 for the 7B) and lower b1 selected (.461/.470) — the
   `Answer:` substring occasionally fires on sub-answers, but nothing like Run
   12's 75%-truncation pathology; not re-tuned, per the replication constraint.
4. **By modality** (b32, P4+P5 mean, sel/oracle): music again weakest (.436/.817);
   sound has the biggest gap (.535/.911); mix-sound-speech strongest selected
   (.581/.893).
5. **Scoring cross-check** at b32: official token-overlap agrees to +0.002…+0.006
   (`crosscheck_b32.json`).

**b64/b128 extension (2026-07-27/28, same serving/knobs).** 8,000 new rows, 0
errors (JSONL now 24,000); b64 3.1 h, b128 4.25 h on the GPU pair.

| cell | b64 sel / orc / maj | b128 sel / orc / maj |
|---|---|---|
| P4 × mean_logprob | .544 / .896 / .551 | .537 / .927 / .560 |
| P4 × entropy | .550 / .913 / .568 | .563 / .928 / .574 |
| P5 × mean_logprob | .548 / .887 / .563 | .505 / .877 / .542 |
| P5 × entropy | .514 / .925 / .549 | .467 / .936 / .514 |

**Extension takeaway:** oracle size-invariance holds all the way to b128 — the 3B
reaches 0.88–0.94, matching the 7B's coverage — while its selection sits ~5–10 pp
below the 7B and even drifts DOWN on P5×entropy (.514 → .467 from b64 to b128,
oracle 0.936): gap **+0.47**, the widest measured anywhere in the campaign on
either benchmark. Exploration is model-size-free; selection is not, and more
particles can actively hurt a weak selector.

**Artifacts** (`results/run02_omni3b/`): `mmar_run02.{jsonl,csv,log}`,
`screen40_p4579.*` (P7/P9 breakage documentation), `smoke16/`,
`audit_capacity.json`, `crosscheck_b32.json`, `servers/`, `stages.log`. Combined
three-model bootstrap: `../run01_epf_grid/mmar_bootstrap.html`.

**Reproduce.** As Run 1 with `config_run02.sh` (model/rev/served-name/prompts/output
pinned there); bootstrap regenerated with both `--in` sections.

## 3. Run 3 — cross-family: Qwen2-Audio-7B-Instruct, ≤30 s subset, P4+P9 (mmau_pro Run 15 replica)

**Why.** Third model / second family on MMAR: does the shape hold across
architectures here as it did on MMAU-Pro (Run 15)?

**Config.** Qwen2-Audio-7B-Instruct (rev `0a095220c30b7b31434169c3086508ef3ea5bf0a`),
served `qwen2-audio`, **single GPU** (gpu 0 / :8100 only — gpu 1 reserved for other
work), `--max-model-len 8192` (its full context), otherwise the §1 serve template.
**New `le30s` subset** (983 items / **979 gradeable**; built by
`build_le30s_subset.py`, soundfile ≤30 s — Qwen2-Audio's Whisper-style encoder hard-
truncates at 30 s): drops 17 items, 15 of them music (music n 206→190). Prompts
**{4,9}** per Run 15's screens — and the MMAR 40-item screen reproduced the same
pathology exactly (P5: half the items unparseable, 1.1 chunks; P4 the only chunker
at 2.0; P9 92.5% parse). Signals × budgets × knobs identical to Runs 1/2; staged
24@b1 / 64.

**n / errors / cost.** 16 cells × 983 = **15,728 rows, 0 errors**, no resume sweeps.
Wall: b1 7.5 min, b8 28 min, b16 57 min, b32 127 min ≈ **3.7 h on ONE GPU**. Gates:
phase0 PASS/PASS; A/B causality 0.733 with vs 0.333 without audio (10/15 flips —
stronger coupling than this model showed on MMAU-Pro, 9/15); capacity PASS (25.4
tok/s; max prompt 834 ≪ 8,192−1,800).

**Results (per cell, n=979; sel / oracle / majority).**

| prompt | signal | b1 | b8 | b16 | b32 |
|---|---|---|---|---|---|
| P4 plan-and-solve | mean_logprob | .459 / .459 / .459 | .476 / .623 / .476 | .465 / .644 / .468 | .479 / .663 / .491 |
| P4 plan-and-solve | entropy | .457 / .457 / .457 | .476 / .623 / .483 | .471 / .662 / .482 | .491 / .693 / .497 |
| P9 evidence-grounded | mean_logprob | .462 / .462 / .462 | .494 / .627 / .490 | .489 / .652 / .492 | .499 / .692 / .496 |
| P9 evidence-grounded | entropy | .472 / .472 / .472 | .488 / .618 / .485 | .489 / .655 / .490 | .493 / .710 / .495 |

**Side-by-side vs mmau_pro Run 15 (P4+P9 means; sel / oracle / maj).**

| signal | b | MMAU-Pro Run 15 | MMAR Run 3 |
|---|---|---|---|
| mean_logprob | 1 | .435 / .435 / .435 | .460 / .460 / .460 |
| mean_logprob | 8 | .456 / .582 / .467 | .485 / .625 / .483 |
| mean_logprob | 16 | .454 / .620 / .469 | .477 / .648 / .480 |
| mean_logprob | 32 | .457 / .656 / .473 | .489 / .677 / .494 |
| entropy | 1 | .436 / .436 / .436 | .464 / .464 / .464 |
| entropy | 8 | .459 / .597 / .472 | .482 / .621 / .484 |
| entropy | 16 | .458 / .642 / .475 | .480 / .658 / .486 |
| entropy | 32 | .452 / .667 / .466 | .492 / .701 / .496 |

**Takeaways.**
1. **The shape replicates on a third model / second family on MMAR too** (now a
   3-model × 2-benchmark result): selected saturates by b8 (+2.1…+2.8 pp over b1,
   flat after; SE ≈ ±0.016 → not significant), oracle climbs to 0.68/0.70, gap
   +0.19/+0.21 at b32; signals equivalent (entropy holds a small oracle edge).
2. **Family-dependent oracle ceiling confirmed on MMAR:** 0.66–0.71 @b32 vs the
   Omni models' 0.85–0.91 on the same benchmark — coverage tracks base competence
   (b1 0.46 here vs 0.56–0.58 Omni-7B), not family-invariant.
3. **MMAR sits above MMAU-Pro for this model as well** (+2.1…+3.7 pp selected,
   +2.0…+3.4 pp oracle at matched cells) — the third model with the same offset.
4. **Run 15's "majority strictly beats selection on P9" does NOT transfer:** on
   MMAR, majority ≈ selected on every P9 cell (≤±0.5 pp, e.g. b32 .496 vs .499
   mean_logprob) vs mmau's +1.4–2.5 pp margins. The consensus-beats-argmax result
   looks benchmark-conditional, not a property of the model.
5. **Music is the weakest modality for the third model in a row** (.401 sel /
   .612 oracle @b32) — a benchmark-level regularity, not an Omni artifact.
6. Official-scorer cross-check @b32: +0.002…+0.008 (`crosscheck_b32.json`).

**b64/b128 extension (2026-07-28, dual-GPU, same knobs).** b64 COMPLETE (2.9 h,
3,932 rows, 0 errors). **b128 PARTIAL — stopped early by user decision** to free
the GPUs for the Run 5 extension: P4×both-signals and P9×mean_logprob completed
(983/983 clean each); P9×entropy stopped at 477/983. Partial-cell numbers are NOT
directly comparable (n differs); complete b128 later by rerunning
`run_ext_b64128.sh config_run03_b64128.sh` with `BUDGETS="128"` (resume-safe; the
1,518 error rows appended by the post-stop retry sweeps are auto-retried).

| cell | b64 sel / orc / maj (n=979) | b128 sel / orc / maj |
|---|---|---|
| P4 × mean_logprob | .474 / .701 / .479 | .477 / .731 / .479 |
| P4 × entropy | .467 / .724 / .483 | .483 / .752 / .490 |
| P9 × mean_logprob | .496 / .724 / .492 | .490 / .750 / .502 |
| P9 × entropy | .490 / .746 / .494 | (partial n=477: .508 / .792 / .519) |

**Extension takeaway:** same shape at the lower family ceiling — selected flat
(.47–.50 since b8), oracle still climbing at b128 (.73–.75 on complete cells,
+2.6…+3.0 pp over b64) but ~0.2 below the Omni models' coverage at every budget:
the family-dependent ceiling persists through b128.

**Artifacts** (`results/run03_qwen2audio_le30s/`): `mmar_run03.{jsonl,csv,log}`,
`screen40_p4579.*`, `smoke16/`, `audit_capacity.json`, `crosscheck_b32.json`,
`servers/`, `stages.log`; subset builder `../build_le30s_subset.py` →
`MMAR-meta-le30s.json` (983). Three-model bootstrap:
`../run01_epf_grid/mmar_bootstrap.html`.

**Reproduce.** `bash benchmarking/mmar/scripts/serve.sh
benchmarking/mmar/scripts/config_run03_q2a.sh` (single-GPU config: GPUS="0",
MAX_MODEL_LEN=8192, SUBSET=le30s, N_ITEMS=983, PROMPTS="4,9") → watchdog →
`run_grid.sh` with the same config; subset built once via
`python -m benchmarking.mmar.build_le30s_subset`.

## 5. Run 5 — cross-family: Phi-4-multimodal-instruct (5.6B, speech-LoRA-served), P4+P8 (mmau_pro Run 18 replica)

*(Run numbering skips 4 — reserved for the pending b64/128 budget extension of Runs 1–2.)*

**Why.** Fourth model / third family on MMAR, and the first whose audio path is a **LoRA
adapter** (Microsoft Conformer encoder → Phi-4-mini backbone). Two questions: does the
"selected saturates / oracle climbs / selector is the wall" shape hold on a fifth architecture
here as it did on MMAU-Pro (Run 18)? And does the cross-benchmark regularity from Runs 1–3 —
*MMAR sits a few pp above MMAU-Pro at matched cells* — hold for this model too?

**Config.** Phi-4-multimodal-instruct (rev `93f923e1a7727d1c4f446756212d9d3e8fcc5d81`), served
**base `phi4mm` + speech-LoRA adapter `speech`** (r=320); vLLM 0.22.1 does NOT auto-load the
adapter, so it is attached via `--enable-lora --max-lora-rank 320 --max-loras 1 --lora-modules
speech=<snapshot>/speech-lora` and **every request names `model="speech"`** — a base-named
request silently answers WITHOUT the adapter (Run 18's #1 gotcha). `serve.sh` grew a
backward-compatible LoRA block (guarded by `ENABLE_LORA`; Runs 1–3 unaffected); `SERVED_NAME=speech`
so the probe/watchdog/health-check all enforce the adapter is listed. **Single GPU** (gpu 0 /
:8100 only), `--max-model-len 32768`, otherwise the §1 serve template (bf16, enforce-eager,
util 0.85, expandable_segments, FLASHINFER_SAMPLER=0, media path
`/home/exx/inference-time-scaling`). HF cache pinned to the **default** `~/.cache/huggingface`
(the checkpoint lives there; not the big volume). Grid: prompts **{4,8}** × signals
{mean_logprob, entropy} × budgets {1,8,16,32}, `--select all` over the full 1,000-item set;
temp 0.8, ess-threshold 0.6, early-phase 0.7, step_token `\n\n`, no stop-regex (P4/P8 end
cleanly at `Answer:`), max_steps 6, 300 tok/step, systematic resampling, style logit — canonical,
zero deviations from Run 18. Budget-staged detached driver, `--max-inflight` 24@b1 / 64 otherwise.
**Prompts P4+P8 per Run 18's screen** (P8 best raw acc, P4 keeps cross-model comparability;
P9 — Run 15's winner — collapses on this model).

**n / errors / cost.** 1,000 items, 996 gradeable per cell (4 ungradeable golds, see header). 16
cells × 1,000 = **16,000 rows, 0 errors** (every cell exactly 1,000; no resume sweeps). Wall: b1
11 min, b8 56 min, b16 129 min, b32 306 min ≈ **8.4 h on ONE GPU** (b32 runs at conc=2 =
`64//32`, so the long-clip stragglers dominate its tail). Gates: phase0 PASS/PASS (logprobs +
continue_final_message, both audio-grounded); A/B causality **9/15 answers flip** without audio
(stronger coupling than Run 18's 6/15 on MMAU-Pro, in Qwen2-Audio territory); capacity audit PASS
(12.56 tok/s, max prompt **733 tok** on the 56 s longest clip ≪ 32k − 1,800 → full set fits, no
subset cap — MMAR clips are ≤56 s vs Run 18's 600 s). Screen (40 stratified, P4+P8 greedy): 40/40
parsed both prompts, reasoned 1.00, **chunks 3.2 (P4) / 2.9 (P8) — replicates Run 18's 3.2 / 2.8
line for line**. 16-item b8 smoke: 0 errors, 0.96 s/item.

**Results (per cell, n=996; sel / oracle / majority).**

| prompt | signal | b1 | b8 | b16 | b32 |
|---|---|---|---|---|---|
| P4 plan-and-solve | mean_logprob | .420 / .420 / .420 | .428 / .613 / .441 | .419 / .708 / .440 | **.464** / .806 / .470 |
| P4 plan-and-solve | entropy | .412 / .412 / .412 | .431 / .659 / .432 | .430 / .756 / .460 | .441 / .830 / .472 |
| P8 anti-shortcut (≥3 steps) | mean_logprob | .421 / .421 / .421 | **.466** / .669 / .448 | .442 / .750 / .461 | .437 / .808 / .461 |
| P8 anti-shortcut (≥3 steps) | entropy | .429 / .429 / .429 | .452 / .718 / .457 | .431 / .776 / .457 | .438 / **.855** / .449 |

**Side-by-side vs mmau_pro Run 18 (P4+P8 means; sel / oracle / maj).** Run 18 documents b1/8/16
locally (its b32–128 are pending collaborator GPUs); MMAR Run 5 additionally has b32.

| signal | b | MMAU-Pro Run 18 | MMAR Run 5 |
|---|---|---|---|
| mean_logprob | 1 | .454 / .454 / .454 | .420 / .420 / .420 |
| mean_logprob | 8 | .480 / .688 / .488 | .447 / .641 / .444 |
| mean_logprob | 16 | .474 / .763 / .488 | .430 / .729 / .450 |
| mean_logprob | 32 | *(pending collab)* | .450 / .807 / .465 |
| entropy | 1 | .454 / .454 / .454 | .420 / .420 / .420 |
| entropy | 8 | .478 / .719 / .489 | .441 / .688 / .444 |
| entropy | 16 | .467 / .798 / .492 | .430 / .766 / .458 |
| entropy | 32 | *(pending collab)* | .439 / .843 / .460 |

**Takeaways.**
1. **The shape replicates on a fourth MMAR model / LoRA-served architecture:** selected saturates
   by b8 (+2.1…+2.7 pp over b1, then flat — .447/.441 @b8 → .450/.439 @b32; bootstrap SE ±0.016
   → the selected gains are ~1–2σ, not significant, exactly as everywhere else), oracle climbs
   monotonically to **.807/.843 @b32**, majority ≈ selected +0.01–0.02 at every b≥8, and the two
   signals are equivalent within noise (entropy a touch better on oracle, mean_logprob a touch
   better on selected). distinct-ratio 0.97 → 0.07, final ESS 1.00 → 0.88 across b1→b32.
2. **The selector is the wall — the widest gap yet in the MMAR series:** oracle−selected @b32 =
   **+0.36 (mean_logprob) / +0.40 (entropy)** on the means, peaking at **+0.42** on P8×entropy
   (oracle .855, selected .438) — rivalling the Omni-3B's +0.41 record (Run 2). Diversity is
   present and exploration is strong; self-certainty selection simply cannot cash it in.
3. **The "MMAR sits above MMAU-Pro" regularity BREAKS for this model — it flips.** Where Omni-7B
   (+0.3…+1.7 pp), Omni-3B (+1.0…+2.6 pp) and Qwen2-Audio (+2.1…+3.7 pp) all scored *higher* on
   MMAR than on their MMAU-Pro cells, Phi-4-MM sits **3–4.5 pp BELOW** its Run 18 cells on
   selected (b1 .420 vs .454; b16 .430 vs .467–.474) and ~3–5 pp below on oracle. So the offset
   is **model-dependent, not a benchmark-level constant** — Phi-4-MM is the first model for which
   MMAR is genuinely harder than MMAU-Pro at matched cells.
4. **Oracle ceiling tracks base competence, reaching into Omni territory by b32:** .81–.86 @b32
   sits above Qwen2-Audio (.66–.71) and abuts the Omnis (.85–.91) — yet Phi-4-MM's b1 selected
   (.42) is the *lowest* of the four models (Omni .56, Qwen2-Audio .46). Weak selector + weak
   greedy baseline, but strong exploration: coverage catches up to the Omnis even when selection
   does not.
5. **Music is the weakest modality for the fourth model in a row** (b32, P4+P8 pooled: sel .334 /
   oracle .782), with sound next (.361 / .826) and speech strongest (.532 / .855) — a
   benchmark-level regularity, not an Omni artifact. By choice count, **2-choice items sel .472 —
   *below* the 0.50 random line** (MMAR's 171 binary items are adversarial for this model, they
   do not inflate accuracy), 4-choice sel .441.
6. **Scoring cross-check:** official token-overlap scorer (mapping the selected letter back to
   choice text) agrees with our lettered scoring to **+0.000…+0.006** per cell
   (`crosscheck_b1.json`, `crosscheck_b32.json`).

**Artifacts** (`results/run05_phi4mm/`): `mmar_run05.{jsonl,csv,log}`, `mmar_bootstrap.html`
(in `../run01_epf_grid/`; 10k-resample bootstrap, now four sections — Runs 1+2+3+5),
`screen40_p48.*`, `smoke16/`, `audit_capacity.json`, `ab_causality.log`,
`crosscheck_b{1,32}.json`, `servers/` (vLLM + watchdog logs), `stages.log`. Config:
`scripts/config_run05_phi4mm.sh` (single-GPU, LoRA-served).

**Reproduce.**
```bash
bash benchmarking/mmar/scripts/serve.sh    benchmarking/mmar/scripts/config_run05_phi4mm.sh
nohup setsid bash benchmarking/mmar/scripts/watchdog.sh benchmarking/mmar/scripts/config_run05_phi4mm.sh \
  > benchmarking/mmar/results/run05_phi4mm/servers/watchdog.log 2>&1 &
nohup setsid bash benchmarking/mmar/scripts/run_grid.sh benchmarking/mmar/scripts/config_run05_phi4mm.sh \
  > benchmarking/mmar/results/run05_phi4mm/stages.log 2>&1 &
# then: official_crosscheck --csv <csv> --budget {1,32}; bootstrap regenerated with all four --in sections
# (Qwen2.5-Omni-7B Run 1 / Omni-3B Run 2 / Qwen2-Audio Run 3 le30s / Phi-4-MM Run 5) -> run01_epf_grid/mmar_bootstrap.html
# NOTE: always rebuild with `--plotlyjs directory` (2026-07-23): keeps the HTML ~100 KB with a
# cached plotly.min.js sidecar next to it, instead of re-inlining the 4.8 MB library on every
# rebuild (which made the live-served page laggy). The sidecar must stay in the same folder.
```

## 6. Run 6 — cross-family: Kimi-Audio-7B-Instruct, ≤30 s subset, P2+P4 (mmau_pro Run 19 replica)

**Why.** Fifth model / fourth family on MMAR — Moonshot's Kimi-Audio (Whisper-v3 front-end →
Qwen2-based MoE-ish backbone, served through vLLM's native `MoonshotKimiaForCausalLM`). Two
questions, same as every cross-family run: does the "selected saturates / oracle climbs /
selector is the wall" shape hold on a fifth architecture here as it did on MMAU-Pro (Run 19)?
And does the cross-benchmark regularity — *MMAR sits a few pp above MMAU-Pro at matched cells*,
which held for the three Qwen models but **flipped** for Phi-4-MM (Run 5) — hold for this model?

**Config.** Kimi-Audio-7B-Instruct (rev `9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b`), served
`kimi-audio`, **single GPU** (gpu 1 / :8101 only — gpu 0 was running Run 5's phi4mm). Serving
goes through the **Run 19 wrapper `benchmarking/mmau_pro/run19/serve_kimi.py`** (NOT `serve.sh`):
vLLM 0.22.1 registers the class but its chat path is broken six ways — no auto chat template,
`apply_chat_template` renders empty, `continue_final_message` `KeyError(-1)`, concurrent-audio
`EngineDeadError`, model-emitted `[EOS]` 400s the next request, and `top_logprobs` decodes an
audio-vocab id and kills the server — all monkey-patched by the wrapper before the stock CLI
starts. A thin `scripts/serve_kimi.sh` (mirrors `serve.sh`: per-port conflict check, PID file,
health poll) launches it with Run 19's flags verbatim: `--chat-template
run19/template_kimi_audio_epf.jinja`, `--limit-mm-per-prompt '{"audio":1}'`, `--max-model-len
8192` (= `max_position_embeddings`; do NOT raise), util **0.85** (0.9 OOMs the audio encoder),
enforce-eager, `VLLM_USE_FLASHINFER_SAMPLER=0`, media path `/home/exx/inference-time-scaling`,
`HF_HOME` on the big volume (checkpoint pinned there from Run 19). **`le30s` subset** (983 items /
**979 gradeable**) — Kimi's Whisper front-end hard-truncates every clip to its first 30 s (same
window Qwen2-Audio used in Run 3), so `le30s` is exactly the eligible set; loss vs full = 17/1,000
= 1.7 % (well under the 5 % re-ask line). Grid: prompts **{2,4}** × signals {mean_logprob, entropy}
× budgets {1,8,16,32}, `--select all`; temp 0.8, ess-threshold 0.6, early-phase 0.7, step_token
`\n\n`, stop `Answer:`, max_steps 6, 300 tok/step, systematic resampling, style logit — canonical,
**zero deviations from Run 19** (these are `diversity_probe` defaults, so `run_grid.sh` replicates
Run 19's config exactly, differing only in loader/subset). Budget-staged detached driver,
`--max-inflight` 24@b1 / 64 otherwise. **Prompts P2+P4 per Run 19's screens** — P2 zero-shot CoT
(user trigger `Answer: Let's think step by step.`) and P4 plan-and-solve; both string-identical to
Run 19's survivors. `watchdog.sh`/`run_grid.sh` reused unchanged (model-agnostic).

**n / errors / cost.** 983 items, **979 gradeable** per cell (`le30s`; same denominator as Run 3).
16 cells × 983 = **15,728 rows, 0 errors** (every cell exactly 983; no resume sweeps). Wall: b1 20
min, b8 2 h 23 m, b16 5 h 30 m, b32 12 h 43 m ≈ **21 h on ONE GPU** (Kimi is the slowest model in
the series — the wrapper runs the audio encoder batch-of-one, no cross-item batching, and
enforce-eager; b32 at conc=2 = `64//32` lets the tail stragglers dominate). Gates: phase0 PASS/PASS
(logprobs + `continue_final_message`, both audio-grounded, through the wrapper); A/B causality
**10/15 answers flip** without audio (acc 0.467 with vs 0.133 without — the cleanest separation in
the series, above Run 19's own 11/15 on MMAU-Pro); capacity audit PASS (13.2 tok/s, max prompt
**461 tok** on the longest 30 s clips ≪ 8192 − 1,800 → `le30s` fits, no further cap — identical to
Run 19's 461). Screen (40 stratified, P2+P4 greedy): 40/40 parsed both prompts, reasoned 0.93 (P2)
/ 0.90 (P4), **chunks 5.0 (P2) / 10.7 (P4)** — matches Run 19's 5.1 / 7.7 (P4 chunks even more on
MMAR); MMAR's variable option format did not break parsing. 16-item b8 smoke: 64 rows, 0 errors.

**Results (per cell, n=979; sel / oracle / majority).**

| prompt | signal | b1 | b8 | b16 | b32 |
|---|---|---|---|---|---|
| P2 zero-shot CoT (user trigger) | mean_logprob | .495 / .495 / .495 | .505 / .635 / .507 | .528 / .693 / .525 | .507 / .731 / .516 |
| P2 zero-shot CoT (user trigger) | entropy | .510 / .510 / .510 | .550 / .713 / .567 | .540 / .756 / .566 | .513 / **.822** / .536 |
| P4 plan-and-solve | mean_logprob | .513 / .513 / .513 | **.558** / .714 / .569 | .557 / .792 / .579 | .553 / .861 / .592 |
| P4 plan-and-solve | entropy | .494 / .494 / .494 | .563 / .710 / .557 | .560 / .790 / .598 | .536 / **.867** / .577 |

**Side-by-side vs mmau_pro Run 19 (P2+P4 means; sel / oracle / maj).** Run 19 documents b1/8/16
locally (its b32 is pending collaborator GPUs); MMAR Run 6 additionally has b32.

| signal | b | MMAU-Pro Run 19 | MMAR Run 6 |
|---|---|---|---|
| mean_logprob | 1 | .474 / .474 / .474 | .504 / .504 / .504 |
| mean_logprob | 8 | .509 / .651 / .516 | .532 / .674 / .538 |
| mean_logprob | 16 | .501 / .703 / .526 | .542 / .742 / .552 |
| mean_logprob | 32 | *(pending collab)* | .530 / .796 / .554 |
| entropy | 1 | .457 / .457 / .457 | .502 / .502 / .502 |
| entropy | 8 | .498 / .666 / .521 | .556 / .712 / .562 |
| entropy | 16 | .505 / .734 / .526 | .550 / .773 / .582 |
| entropy | 32 | *(pending collab)* | .524 / .845 / .556 |

**Takeaways.**
1. **The shape replicates on a fifth MMAR model / fourth family:** selected saturates by b8
   (+3–7 pp over b1, then flat — P4 mean .513→.558→.557→.553; bootstrap SE ±0.016 → the selected
   gains are ~1–2σ, borderline, exactly as everywhere else), oracle climbs monotonically to
   **.73–.87 @b32**, majority ≈ selected + 0.01–0.04 at every b≥8, and the two signals are
   equivalent within noise (entropy a touch better on oracle — its b32 oracle .82–.87 beats
   mean_logprob's .73–.86 — mean_logprob a touch better on selected). distinct-ratio 0.97 → 0.06,
   final ESS 1.00 → 0.90 across b1→b32.
2. **The selector is the wall — a mid-pack gap in the series:** oracle−selected @b32 = **+0.27
   (mean_logprob) / +0.32 (entropy)** on the means, peaking at **+0.331** on P4×entropy (oracle
   .867, selected .536). Wider than Qwen2-Audio (+0.19/+0.21) and Omni-7B (+0.25/+0.29), narrower
   than Omni-3B (+0.41) and Phi-4-MM (+0.36/+0.40). Strong exploration (b32 oracle abuts the
   Omnis' .85–.91), selection can't cash it in.
3. **The "MMAR sits above MMAU-Pro" regularity HOLDS for Kimi — the Phi-4-MM flip does not
   generalize.** Kimi scores **+2.3…+5.9 pp above** its Run 19 cells on selected at every matched
   b1/8/16 cell (e.g. entropy b8 .556 vs .498; mean b16 .542 vs .501) and +2…+4 pp above on oracle.
   So across five models the offset is: Omni-7B/3B/Qwen2-Audio/Kimi all *higher* on MMAR,
   Phi-4-MM *lower* — **model-dependent, but MMAR-higher is the majority case (4 of 5)**; Phi-4-MM
   remains the lone inversion.
4. **Oracle ceiling reaches into Omni territory:** .73–.87 @b32 sits above Qwen2-Audio (.66–.71)
   and abuts the Omnis (.85–.91), while Kimi's b1 selected (.50–.51) is mid-pack (Omni .56,
   Qwen2-Audio/Phi-4-MM .42–.46) — competent greedy baseline, strong exploration.
5. **Music is the weakest modality for the fifth model in a row** (b32, P2+P4 pooled: sel .377 /
   oracle .731, the widest per-modality gap at +0.354), with sound next (.509 / .829) and
   mix-sound-speech (.599) / speech (.580 / .845) strongest — a benchmark-level regularity, not an
   architecture artifact. By choice count, **2-choice items sel .545 — *above* the 0.50 random
   line** (the healthiest binary performance in the series; contrast Omni-7B .528, Phi-4-MM .472
   below), 4-choice sel .526. Overall b32 pooled sel .527 / oracle .820.
6. **Scoring cross-check:** official token-overlap scorer (mapping the selected letter back to
   choice text) agrees with our lettered scoring to **+0.003…+0.009** per cell
   (`crosscheck_b1.json`, `crosscheck_b32.json`) — the official scorer is marginally more lenient,
   same pattern as Runs 1–3/5.

**Artifacts** (`results/run06_kimiaudio_le30s/`): `mmar_run06.{jsonl,csv,log}`, `mmar_bootstrap.html`
(in `../run01_epf_grid/`; 10k-resample bootstrap, now five sections — Runs 1+2+3+5+6),
`screen40_p24.*`, `smoke16/`, `audit_capacity.{json,log}`, `ab_causality.log`,
`crosscheck_b{1,32}.json`, `servers/` (vLLM + watchdog logs), `stages.log`. Configs:
`scripts/config_run06_kimi.sh` (single-GPU, `le30s`, `SERVED_NAME=kimi-audio`) and
`scripts/serve_kimi.sh` (wraps `run19/serve_kimi.py` — the run19 collaborator package itself is
untouched, read-only).

**Reproduce.**
```bash
bash benchmarking/mmar/scripts/serve_kimi.sh benchmarking/mmar/scripts/config_run06_kimi.sh
nohup setsid bash benchmarking/mmar/scripts/watchdog.sh benchmarking/mmar/scripts/config_run06_kimi.sh \
  > benchmarking/mmar/results/run06_kimiaudio_le30s/servers/watchdog.log 2>&1 &
nohup setsid bash benchmarking/mmar/scripts/run_grid.sh benchmarking/mmar/scripts/config_run06_kimi.sh \
  > benchmarking/mmar/results/run06_kimiaudio_le30s/stages.log 2>&1 &
# then: official_crosscheck --csv <csv> --budget {1,32}; bootstrap regenerated with all five --in
# sections (Omni-7B R1 / Omni-3B R2 / Qwen2-Audio R3 le30s / Phi-4-MM R5 / Kimi-Audio R6 le30s)
# -> run01_epf_grid/mmar_bootstrap.html, always with `--plotlyjs directory`.
```
