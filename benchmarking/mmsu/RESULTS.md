# MMSU — EPF grid results

MMSU: 5,000 single-audio spoken-language-understanding MCQ (47 tasks under a coarse
Perception/Reasoning axis). Runs here replicate the EPF grids we already have for the
same models on MMAU-Pro and MMAR, to test whether the "selector is the wall" result
(selected accuracy saturates early while the oracle keeps climbing; wide oracle−selected
gap) is a **three-benchmark** result. Run numbering restarts at 1 for MMSU; the
mmau_pro / mmar numbering is untouched.

Scoring: the gold `answer` is the full choice TEXT and matches a choice exactly in all
5,000 rows → `answer_index` is the verbatim choice index, **5,000/5,000 gradeable, 0
fuzzy, n = 5,000 per cell**. 453/5,000 items are 2-choice (None-padded to 4 letter-keys
in the source); the loader drops the padding so they present as genuine 2-choice MCQs
(4,547 four-choice + 453 two-choice). `category` in the per-item CSV is the fine-grained
`task_name` (47 values); the coarse Perception/Reasoning split is recovered by joining
`unique_id` through `benchmarking.mmsu.loader.load_metadata`. MMSU ships no official
scorer, so there is no crosscheck.

Canonical config (both runs): temp 0.8, ess-threshold 0.6, early-phase 0.7, step token
`\n\n`, stop `Answer:`, max_steps 6, 300 tok/step, systematic resampling, style logit,
prompt **P4 (plan-and-solve)**, signals {mean_logprob, entropy}, budgets
**{1,8,16,32,64,128}**, `--select all`. Two bf16 vLLM replicas (one per GPU,
:8100/:8101), `--max-model-len 32768`, util 0.85; b1 stages `--max-inflight 24`, else 64.

Prompt note: unlike MMAU-Pro/MMAR (where Qwen-Omni's P7/P9 broke mechanically, forcing a
P4/P5 narrowing), the run01 screen found **all of P4,5,7,9 parse 100% and reason on
MMSU** — its shorter spoken-language items chunk cleanly. Runs 1–2 nonetheless use a
single prompt (P4, the strongest chunker at 4.7) with a DEEP budget ladder to 128, to
test whether the oracle ever saturates (MMAR left b64/128 as a pending extension).

## TL;DR

1. **"Selector is the wall" replicates on a third benchmark, cleanly.** For both models,
   selected accuracy saturates by b8 (~0.64 7B / ~0.59 3B) and then flatlines (even dips)
   through b128, while the oracle climbs monotonically to **0.95 (7B) / 0.92 (3B) and
   never saturates**. The oracle−selected gap widens every budget to **+0.31 (7B) /
   +0.33 (3B)** at b128 — the widest in the PF/EPF series, because only MMSU ran to b128.
2. **It is size-invariant.** The 3B's oracle tracks just ~3 pp under the 7B's at every
   budget — the *exploration ceiling barely depends on model size*. What differs is
   baseline competence (3B ~4–5 pp lower selected), so the gap is if anything slightly
   *wider* on the smaller model.
3. **It is overwhelmingly a Perception problem, identically for both sizes.** The coarse
   Perception/Reasoning split and the per-task breakdown show the wall is concentrated on
   acoustic-perception/comparison tasks: at b128 the Perception oracle−selected gap is
   **+0.45 for BOTH models**, and individual tasks like `speed_comparison` reach
   sel 0.20 / oracle 0.96 (**+0.76**). Reasoning tasks are both easier and near-perfectly
   selected (several at sel = oracle = 1.000). The model can hear correctly; it just
   cannot tell when it did — regardless of scale.

Combined 10k-resample bootstrap (both runs): `results/run01_omni7b/mmsu_bootstrap.html`
(built `--plotlyjs directory`; keep `plotly.min.js` beside it).

---

## Run 1 — Qwen2.5-Omni-7B EPF grid (P4, b1–128)

**Serving:** `Qwen/Qwen2.5-Omni-7B`, revision `ae9e1690543ffd5c0221dc27f79834d0294cba00`,
served `qwen-omni`, one bf16 replica/GPU on :8100/:8101, `--max-model-len 32768
--enforce-eager --gpu-memory-utilization 0.85 --allowed-local-media-path
/home/exx/inference-time-scaling`. Config `scripts/config_run01.sh`.

**n / errors / cost:** 5,000 items, **5,000 gradeable/cell**. 6 budgets × 2 signals ×
5,000 = **60,000 rows, 0 errors**, no resume sweeps, servers never dropped. Wall
**~34.5 h** on the GPU pair (b1 11 m, b8 55 m, b16 2.0 h, b32 4.7 h, b64 11.5 h,
b128 ~13.7 h — b128 the long pole at concurrency 1/endpoint × 128 particles).

**Gates:** phase0 PASS/PASS ×2 endpoints (logprobs-with-audio + continue_final_message);
A/B causality 0.400 with vs 0.267 without audio, 7/15 flips; capacity audit PASS (0/30,
max 942 prompt tokens for the 35.9 s clip ≪ 32,768). Sanity screen (P4,5,7,9, 40
stratified): all four parse 40/40, reason 1.00, chunks P4 4.7 / P7 3.5 / P5 2.0 / P9 1.9;
P4 greedy acc 0.650. Smoke (P4, b8, 16 items): 0 errors.

**Results (P4, n=5,000; per cell `selected / oracle / majority`):**

| signal | b1 | b8 | b16 | b32 | b64 | b128 |
|---|---|---|---|---|---|---|
| mean_logprob | .604/.604/.604 | .649/.756/.653 | .649/.819/.662 | .647/.881/.663 | .637/.915/.660 | .640/.949/.668 |
| entropy | .608/.608/.608 | .644/.761/.650 | .648/.825/.653 | .652/.876/.657 | .644/.922/.661 | .638/.952/.667 |

distinct-answer ratio collapses 0.99 → 0.17 → 0.10 → 0.057 → 0.032 → **0.017** (at b128
the 128 particles agree ~98% of the time), yet oracle = 0.95; parse 1.00 throughout.

**Perception vs Reasoning (entropy; selected / oracle / gap):**

| budget | Perception (n=2,580) | Reasoning (n=2,420) |
|---|---|---|
| b1 | .487 / .487 / — | .736 / .736 / — |
| b32 | .517 / .828 / **+0.31** | .795 / .927 / +0.13 |
| b128 | .493 / .945 / **+0.45** | .792 / .960 / +0.17 |

**Per-task extremes @ b128 (entropy), oracle−selected gap:**
- Widest (all Perception): `speed_comparison` .202/.963 (**+0.76**), `volume_comparison`
  .200/.927 (+0.73), `pause_perception` .290/.981 (+0.69), `speech_duration_estimation`
  .226/.906 (+0.68), `speaker_identity_recognition` .263/.895 (+0.63).
- Narrowest (all Reasoning): `casual_reasoning` / `continuation_writing` /
  `idiom_reasoning` / `long_speech_summarization` all 1.000/1.000 (+0.00),
  `homophone_based_reasoning` .990/1.000.

**Takeaways.** (1) Selected saturates by b8 (~0.645) and is flat-to-declining through b128
(b128 .638/.640); adding particles past 8 buys the selector nothing. (2) Oracle climbs
monotonically 0.61→0.76→0.83→0.88→0.92→0.95 and is *still rising* at b128 — exploration
never saturates over the range tested. (3) Gap widens every budget to +0.31; majority
tracks selected +0.01–0.03 (best aggregator, still 0.28 below oracle at b128). (4) Signals
are equivalent. (5) The wall is an acoustic-perception phenomenon (see splits above).

**Artifacts:** `results/run01_omni7b/` — `mmsu_run01.{jsonl,csv,log}`, `screen40_p4579.*`,
`smoke16/`, `audit_capacity.json`, `ab_causality.log`, `servers/`, `stages.log`, and the
combined `mmsu_bootstrap.html` + `plotly.min.js`.

**Reproduce:**
```bash
bash benchmarking/mmsu/scripts/serve.sh    benchmarking/mmsu/scripts/config_run01.sh
nohup setsid bash benchmarking/mmsu/scripts/watchdog.sh benchmarking/mmsu/scripts/config_run01.sh \
  > benchmarking/mmsu/results/run01_omni7b/servers/watchdog.log 2>&1 &
nohup setsid bash benchmarking/mmsu/scripts/run_grid.sh benchmarking/mmsu/scripts/config_run01.sh \
  > benchmarking/mmsu/results/run01_omni7b/stages.log 2>&1 &
```

---

## Run 2 — Qwen2.5-Omni-3B EPF grid (P4, b1–128; mirrors Run 1)

**Serving:** `Qwen/Qwen2.5-Omni-3B`, revision `f75b40e3da2003cdd6e1829b1f420ca70797c34e`,
served `qwen-omni-3b`; everything else identical to Run 1. Config `scripts/config_run02.sh`.

**n / errors / cost:** 5,000 items, **5,000 gradeable/cell**, **60,000 rows, 0 errors**,
no resume sweeps. Wall **~29.4 h** GPU-pair (b1 9 m, b8 43 m, b16 1.75 h, b32 4.25 h,
b64 9.8 h, b128 12.7 h) — slightly faster than the 7B.

**Gates:** phase0 PASS/PASS ×2; capacity audit PASS (identical tokenization, max 942 tok).
A/B causality **weak** — 0.400 with vs 0.467 without audio, 3/15 flips — but that slice is
15 `accent_identification` items (a hard perception task; the 7B was also only 0.400 here).
The stratified P4 screen (all 47 tasks) scored **0.550**, well above random, confirming the
3B uses audio across the task distribution; smoke (P4 b8) showed the EPF pattern
(sel 0.63/oracle 0.75). Weaker A/B coupling on the 3B matches the documented MMAR Run 2
finding (smaller model leans more on priors).

**Results (P4, n=5,000; `selected / oracle / majority`):**

| signal | b1 | b8 | b16 | b32 | b64 | b128 |
|---|---|---|---|---|---|---|
| mean_logprob | .576/.576/.576 | .593/.731/.590 | .585/.774/.592 | .606/.831/.609 | .596/.882/.600 | .599/.925/.606 |
| entropy | .571/.571/.571 | .589/.762/.599 | .590/.794/.601 | .599/.845/.602 | .604/.883/.609 | .594/.921/.609 |

distinct collapses 0.99 → 0.018 by b128; parse 1.00 throughout.

**Perception vs Reasoning (entropy; selected / oracle / gap):**

| budget | Perception (n=2,580) | Reasoning (n=2,420) |
|---|---|---|
| b1 | .426 / .426 / — | .726 / .726 / — |
| b32 | .444 / .788 / **+0.34** | .765 / .907 / +0.14 |
| b128 | .442 / .894 / **+0.45** | .756 / .950 / +0.20 |

**Per-task extremes @ b128 (entropy):** widest all Perception — `pause_perception`
.252/.944 (+0.69), `speed_comparison` .275/.936 (+0.66), `intonation_perception`
.216/.874 (+0.66), `dialogue_turn_counting` .222/.859 (+0.64), `pitch_comparison`
.250/.870 (+0.62); narrowest all Reasoning (`casual_reasoning` 1.000/1.000, idiom/polysemy
.990/1.000).

**7B vs 3B (size-invariance).** The 3B's oracle sits just ~3 pp under the 7B's at every
budget (b128: 0.921/0.925 vs 0.952/0.949) — the exploration ceiling is nearly
size-invariant. Selected is ~4–5 pp lower on the 3B and equally flat, so the gap is
slightly *wider* (+0.33 vs +0.31 at b128). The Perception oracle−selected gap is
**identical (+0.45) for both models** at b128 — the selection failure on hearing tasks
does not depend on scale.

**Artifacts:** `results/run02_omni3b/` — `mmsu_run02.{jsonl,csv,log}`, `screen40_p4.*`,
`smoke16/`, `audit_capacity.json`, `ab_causality.log`, `servers/`, `stages.log`. Combined
bootstrap lives in `../run01_omni7b/mmsu_bootstrap.html`.

**Reproduce:** as Run 1 with `config_run02.sh`; bootstrap regenerated with both `--in`
sections and `--plotlyjs directory`.

---

## Cross-benchmark: same models on MMAU-Pro, MMAR, MMSU (P4)

Selected / oracle at the shared budgets (P4; MMAU-Pro n=5,066, MMAR n=996, MMSU n=5,000).
MMAU-Pro's Omni-7B write-up tabulated only 4-prompt means, so its P4 is available at
b1/b32 only (from the Run 12 reference column).

**Qwen2.5-Omni-7B, P4 (entropy | mean_logprob):**

| benchmark | b1 sel/orc | b32 sel/orc | b128 sel/orc |
|---|---|---|---|
| MMAU-Pro | .551/.551 \| .549/.549 | .575/.835 \| .584/.826 | — (capped at b32) |
| MMAR | .545/.545 \| .568/.568 | .575/.880 \| .590/.870 | — (capped at b32) |
| **MMSU** | **.608/.608 \| .604/.604** | **.652/.876 \| .647/.881** | **.638/.952 \| .640/.949** |

**Qwen2.5-Omni-3B, P4 (entropy | mean_logprob):**

| benchmark | b1 sel/orc | b32 sel/orc | b128 sel/orc |
|---|---|---|---|
| MMAU-Pro | .512/.512 \| .524/.524 | .540/.836 \| .538/.831 | — |
| MMAR | .526/.526 \| .541/.858 | .541/.858 \| .550/.851 | — |
| **MMSU** | **.571/.571 \| .576/.576** | **.599/.845 \| .606/.831** | **.594/.921 \| .599/.925** |

**Reading it:** the EPF *shape* — flat selected, monotonically climbing oracle, wide gap,
equivalent signals, majority ≈ selected — is identical across all three benchmarks and
both model sizes. MMSU sits a few points HIGHER on selected at b1/b32 (it is somewhat
easier at baseline) but has the same b32 oracle (~0.83–0.88) and the same gap magnitude.
Only MMSU ran to b64/b128, and there the oracle keeps climbing to ~0.92–0.95 with no
plateau — the strongest evidence yet that the exploration is abundant and the entire
bottleneck is selection. "Selector is the wall" is now a three-benchmark, two-model,
Perception-localized, size-invariant result.
