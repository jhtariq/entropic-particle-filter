# benchmarking/mmau — MMAU (original) audio-MCQ benchmark harness

MMAU (arXiv 2410.19168, **NOT MMAU-Pro**): the 1,000-item **test-mini** split of the
Massive Multi-Task Audio Understanding benchmark — single-audio multiple-choice items,
tasks sound/music/speech, 27 sub-category skills under a Reasoning/Information-Extraction
axis. This package is the fourth benchmark in the repo (after `benchmarking/mmau_pro/`,
`benchmarking/mmar/`, `benchmarking/mmsu/`) and reuses the mmau_pro machinery **by
import** — no forks. Data version: **MMAU-v05.15.25** (HF `gamma-lab-umd/MMAU-test-mini`
@ sha `ccd9696`, last modified 2025-08-19 — post-dates the May-2025 revision that changed
~25% of the QA pairs; published numbers from before it do NOT compare). The 9,000-item
`test` split ships with **empty answers** (leaderboard-only) and is archived unextracted.

## Data (outside the repo)

```
/home/exx/inference-time-scaling/data/mmau/
  MMAU-meta.json        # JSON array, exactly 1,000 items (materialized)
  audio/                # 1,000 .wav (~1.1 GB), referenced as "./audio/<id>.wav"
  code/evaluation.py    # official scorer, github.com/Sakshi113/MMAU @ 7468292
  raw/test-mini/        # source parquet (HF sha ccd9696) — audio embedded per row
  raw/test/             # blind 9,000-item split: parquet + test-audios.tar.gz (unused)
```

The parent `/home/exx/inference-time-scaling` is the `--allowed-local-media-path`
for vLLM (common parent of the mmau_pro / mmar / mmsu / mmau audio roots).
`materialize_test_mini.py` (idempotent) regenerates `MMAU-meta.json` + `audio/`
from the raw parquet.

Facts the loader/tests pin: 1,000 MCQ records; **1,000 gradeable** (gold `answer` is the
full choice text, raw-verbatim in every row — 0 fuzzy, 0 ungradeable). The shipped
choices/answer embed `(A) `-style letter prefixes; materialization **strips** them into
`choices`/`answer` (what the prompt shows — the shared prompt builder letters options
itself and would otherwise render "A. (A) Man") and keeps the shipped strings as
`choices_raw`/`answer_raw` (what the official token-overlap scorer must be fed).
Choice counts are NOT uniform: **4:948, 2:27, 5:24, 8:1** (LETTERS A–K covers the
8-choice item). Dataset quirks pinned by `verify_data`: **27** items carry textually
duplicate options after stripping and **16** of those duplicate the GOLD text under a
second letter — the twin letter grades wrong under BOTH our scorer and the official one
(a small unavoidable ceiling). Audio is heterogeneous as shipped: durations 1.5–**34.5 s**
(le10s:623, 10to30s:311, gt30s:66; total 3.98 h), sample rates 16k:411 / 32k:252 /
44.1k:226 / 48k:92 / 24k:19, channels mono:752 / stereo:179 / **6-channel:69**, formats
PCM_16:866 / float:73 / **MP3-in-.wav:46** / PCM_32:15 — `soundfile`/vLLM decode all of
them (MMSU precedent).

**Official scorer** — `code/evaluation.py` grades by lowercased token-set match
(`string_match`: all gold tokens present AND no distractor-unique tokens; prediction key
`model_output` — the repo README says `model_prediction`, the code wins).
`official_crosscheck.py` maps our `selected_letter` back to the RAW prefixed choice text
and reports per-cell agreement deltas (mmar precedent). **No `le30s` subset / builder** —
the 34.5 s max fits Qwen2.5-Omni's 32k window, so runs use the full set.

## How it plugs into mmau_pro

- `loader.py` emits mmau_pro's own `MCQRecord` dataclass: **`category` ← MMAU `task`**
  (sound/music/speech — the official leaderboard reporting axis), `length_type` ←
  duration bucket (soundfile-measured, `unknown` without soundfile), `answer_index` ←
  raw-verbatim choice index (`choices_raw.index(answer_raw)`, immune to the stripped
  duplicates; stripped-verbatim then fuzzy `match_answer_index` as parity fallbacks,
  dead code today). The Reasoning/Information-Extraction `category`, `difficulty`,
  `sub-category` (27), source `dataset` (15) and the raw choice strings stay out of the
  record — `load_metadata()` exposes them (join a CSV's `unique_id` through it).
- The four GPU runners here (`diversity_probe`, `cot_compare`, `phase0_gate`,
  `ab_causality`) are ~10-line wrappers over mmau_pro's `make_cli(loader_fn=…, …)`
  factories, instantiated with the MMAU loader.
- `scoring.py`, `prompt.py`, `audio.py`, `epf_bootstrap.py` are used from mmau_pro
  unchanged (lettered MCQ is benchmark-agnostic).

## Commands

```bash
# one-off data materialization from the raw parquet (idempotent)
python -m benchmarking.mmau.materialize_test_mini \
    --parquet /home/exx/inference-time-scaling/data/mmau/raw/test-mini/test_mini.parquet \
    --out-root /home/exx/inference-time-scaling/data/mmau

# data verification (exit-coded: 1000 records / 1000 gradeable / 0 missing audio)
python -m benchmarking.mmau.verify_data

# serving + watchdog + budget-staged grid (configs pin model/rev/prompts/output)
bash benchmarking/mmau/scripts/serve.sh    benchmarking/mmau/scripts/config_run01.sh
bash benchmarking/mmau/scripts/watchdog.sh benchmarking/mmau/scripts/config_run01.sh
bash benchmarking/mmau/scripts/run_grid.sh benchmarking/mmau/scripts/config_run01.sh

# gates against a served model
python -m benchmarking.mmau.phase0_gate --endpoint http://localhost:8100/v1 --model-name qwen-omni
python -m benchmarking.mmau.ab_causality --endpoint http://localhost:8100/v1 --model-name qwen-omni --limit 15
python -m benchmarking.mmau.audit_audio_capacity --endpoint http://localhost:8100/v1 --model-name qwen-omni

# offline crosscheck against the official scorer (per budget)
python -m benchmarking.mmau.official_crosscheck \
    --csv benchmarking/mmau/results/run01_omni7b/mmau_run01.csv --budget 1
```

Use the `epf` conda env's absolute python for anything that touches a GPU server
(`/home/exx/miniconda3/envs/epf/bin/python`); `uv run pytest tests/` for tests.

Published test-mini anchors (Omni-R1 paper, arXiv 2505.09439v3 Table I —
evaluated on the revised set): Qwen2.5-Omni-7B **65.9** avg (S/M/Sp 69.4/66.8/61.6),
Omni-R1 (GRPO) **71.3**, R1-AQA **65.6**, GPT-4o Voice **59.1**, Qwen2-Audio-7B
**52.7**, Human **82.2**. No published Qwen2.5-Omni-3B number is known.

Results: one folder per run under [results/](results/README.md); write-ups in
[RESULTS.md](RESULTS.md) (numbering starts at Run 1 for this benchmark; the
experimental protocol is inherited from MMSU Runs 1–2).
