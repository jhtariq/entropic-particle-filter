# benchmarking/mmsu — MMSU audio-MCQ benchmark harness

MMSU: 5,000 single-audio multiple-choice **spoken-language-understanding** items
across 47 fine-grained tasks, organised under a coarse Perception/Reasoning axis.
This package is the third benchmark in the repo (after `benchmarking/mmau_pro/` and
`benchmarking/mmar/`) and reuses the mmau_pro machinery **by import** — no forks.

## Data (outside the repo)

```
/home/exx/inference-time-scaling/data/mmsu/
  MMSU-meta.json        # JSON array, exactly 5,000 items
  audio/                # 5,000 .wav (1.7 GB), referenced as "./audio/<id>.wav"
```

The parent `/home/exx/inference-time-scaling` is the `--allowed-local-media-path`
for vLLM (common parent of the mmau_pro / mmar / mmsu audio roots).

Facts the loader/tests pin: 5,000 MCQ records; **5,000 gradeable** (the gold `answer`
is the full choice TEXT and matches a choice exactly in every row — 0 fuzzy, 0
ungradeable); `choices` is a letter-keyed **dict** `{"A":…,"B":…,"C":…,"D":…}`, but
**453 of the 5,000 are 2-choice items padded to 4 keys with `None` values** (the padding
can sit in ANY position, e.g. `{"A":None,"B":"male","C":None,"D":"female"}`). The loader
**drops the `None` padding** (mmar-style variable choice counts → 4,547 four-choice +
453 two-choice), so the model never sees blank options; the literal string `"None"` (19
items) is a real answer option and is kept. Durations 0.3–**35.9 s** (p50 ~4.5 s — 4 clips
exceed 30 s). The `.wav` files carry MP3/ID3 bytes upstream but `soundfile`/libsndfile and
vLLM decode them fine.

**No official scorer** — MMSU ships no `code/evaluation.py`, so there is no
`official_crosscheck` tool (grading is the loader's exact text→letter map, 0 fuzzy).
**No `le30s` subset / builder** — the 35.9 s max fits Qwen2.5-Omni's 32k window, so
Runs 1–2 use the full set. A future 30 s-encoder-window model (Qwen2-Audio / Kimi /
Phi-4-MM) would need a retargeted `build_le30s_subset.py` + an `le30s` entry in
`SUBSET_FILES`.

## How it plugs into mmau_pro

- `loader.py` emits mmau_pro's own `MCQRecord` dataclass: **`category` ← MMSU
  `task_name`** (the 47-way fine-grained stratifier / by-task tables), `length_type` ←
  duration bucket (le10s/10to30s/gt30s; soundfile-measured, `unknown` without soundfile),
  `answer_index` ← verbatim choice index (fuzzy `match_answer_index` fallback, dead code
  on MMSU today). MMSU's coarse Perception/Reasoning `category` field, sub-category,
  sub-sub-category and linguistics_sub_discipline stay out of the record —
  `load_metadata()` exposes them; each task_name maps 1-into-2 onto Perception/Reasoning,
  so both cuts stay retrievable (join a CSV's `unique_id` through `load_metadata`).
- The four GPU runners here (`diversity_probe`, `cot_compare`, `phase0_gate`,
  `ab_causality`) are ~10-line wrappers over mmau_pro's `make_cli(loader_fn=…, …)`
  factories, instantiated with the MMSU loader.
- `scoring.py`, `prompt.py`, `audio.py`, `epf_bootstrap.py` are used from mmau_pro
  unchanged (lettered MCQ is benchmark-agnostic).

## Commands

```bash
# data verification (exit-coded: 5000 records / 5000 gradeable / 0 missing audio)
python -m benchmarking.mmsu.verify_data

# serving + watchdog + budget-staged grid (configs pin model/rev/prompts/output)
bash benchmarking/mmsu/scripts/serve.sh    benchmarking/mmsu/scripts/config_run01.sh
bash benchmarking/mmsu/scripts/watchdog.sh benchmarking/mmsu/scripts/config_run01.sh
bash benchmarking/mmsu/scripts/run_grid.sh benchmarking/mmsu/scripts/config_run01.sh

# gates against a served model
python -m benchmarking.mmsu.phase0_gate --endpoint http://localhost:8100/v1 --model-name qwen-omni
python -m benchmarking.mmsu.ab_causality --endpoint http://localhost:8100/v1 --model-name qwen-omni --limit 15
python -m benchmarking.mmsu.audit_audio_capacity --endpoint http://localhost:8100/v1 --model-name qwen-omni
```

Use the `epf` conda env's absolute python for anything that touches a GPU server
(`/home/exx/miniconda3/envs/epf/bin/python`); `uv run pytest tests/` for tests.

Results: one folder per run under [results/](results/README.md); write-ups in
[RESULTS.md](RESULTS.md) (numbering starts at Run 1 for this benchmark; the
experimental protocol is inherited from mmau_pro Runs 11/12 and mmar Runs 1/2).
