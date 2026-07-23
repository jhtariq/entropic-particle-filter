# benchmarking/mmar — MMAR audio-MCQ benchmark harness

MMAR: 1,000 single-audio multiple-choice **reasoning** items across
speech/sound/music/mix modalities. This package is the second benchmark in the
repo (after `benchmarking/mmau_pro/`) and reuses the mmau_pro machinery **by
import** — no forks.

## Data (outside the repo)

```
/home/exx/inference-time-scaling/mmar/
  MMAR-meta.json        # JSON array, exactly 1,000 items
  audio/                # 1,000 .wav (3.5 GB), referenced as "./audio/<id>.wav"
  code/evaluation.py    # the official token-overlap scorer (used by official_crosscheck)
```

The parent `/home/exx/inference-time-scaling` is the `--allowed-local-media-path`
for vLLM (common parent of the mmau_pro and mmar audio roots).

Facts the loader/tests pin: 1,000 MCQ records; **996 gradeable** via
`match_answer_index` (994 verbatim + 1 normalized + 1 fuzzy), 4 ungradeable
(multi-answer / heavily-rephrased golds); choices are free text with variable
counts (4→815, 2→171, 3→10, 5→3, 6→1); durations 3–56 s (p50 20 s); 45/1,000
`timestamp` fields are malformed.

## How it plugs into mmau_pro

- `loader.py` emits mmau_pro's own `MCQRecord` dataclass: `category` ← MMAR
  `modality`, `length_type` ← duration bucket (le10s/10to30s/gt30s/unknown;
  soundfile-measured when available, `timestamp`-parsed otherwise),
  `answer_index` ← `mmau_pro.scoring.match_answer_index`. MMAR's layer /
  sub-category / language fields stay out of the record — `load_metadata()`
  exposes them for reporting cuts.
- The four GPU runners here (`diversity_probe`, `cot_compare`, `phase0_gate`,
  `ab_causality`) are ~10-line wrappers: mmau_pro's scripts expose a
  `make_cli(loader_fn=..., subset_choices=..., default_data_root=...,
  default_subset=...)` factory whose defaults reproduce the original MMAU-Pro
  CLIs byte-for-byte; the wrappers instantiate it with the MMAR loader.
- `scoring.py`, `prompt.py`, `audio.py`, `epf_bootstrap.py` are used from
  mmau_pro unchanged (lettered MCQ is benchmark-agnostic).

## Commands

```bash
# data verification (exit-coded: 1000 records / 996 gradeable / 0 missing audio)
python -m benchmarking.mmar.verify_data

# serving + watchdog + budget-staged grid (configs pin model/rev/prompts/output)
bash benchmarking/mmar/scripts/serve.sh    benchmarking/mmar/scripts/config_run01.sh
bash benchmarking/mmar/scripts/watchdog.sh benchmarking/mmar/scripts/config_run01.sh
bash benchmarking/mmar/scripts/run_grid.sh benchmarking/mmar/scripts/config_run01.sh

# gates against a served model
python -m benchmarking.mmar.phase0_gate --endpoint http://localhost:8100/v1 --model-name qwen-omni
python -m benchmarking.mmar.ab_causality --endpoint http://localhost:8100/v1 --model-name qwen-omni --limit 15
python -m benchmarking.mmar.audit_audio_capacity --endpoint http://localhost:8100/v1 --model-name qwen-omni

# official-scorer cross-check (offline, after a grid)
python -m benchmarking.mmar.official_crosscheck --csv benchmarking/mmar/results/run01_epf_grid/mmar_run01.csv
```

Use the `epf` conda env's absolute python for anything that touches a GPU server
(`/home/exx/miniconda3/envs/epf/bin/python`); `uv run pytest tests/` for tests.

Results: one folder per run under [results/](results/README.md); write-ups in
[RESULTS.md](RESULTS.md) (numbering starts at Run 1 for this benchmark; the
experimental protocol is inherited from mmau_pro Runs 11/12).
