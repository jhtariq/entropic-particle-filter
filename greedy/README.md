# Greedy Baseline — 6 models × 3 benchmarks (18 cells)

Greedy (temperature 0), **no CoT**, single-shot direct-answer inference for every
model on every benchmark. One plain chat completion per item — no particle
filtering, no step generation. This is the honest "greedy zero-shot without CoT"
reference row for all the scaling grids.

## The one command

```bash
git pull
nohup bash greedy/run_all.sh > greedy_run.log 2>&1 &
tail -f greedy_run.log
```

That single script does everything, in order, and **aborts immediately on the
first failure**:

1. **Phase 1a–1c — provision** (verify-first, idempotent): creates the conda env
   `epf`, downloads all six checkpoints at the exact pinned revisions, downloads
   all three datasets and **byte-verifies them against committed sha256 manifests**
   — your data will be bit-identical to the reference box or the script stops.
2. **Phase 1.5 — smoke**: every one of the 18 cells runs a 3-item mini-version
   (server up, health gates, real audio round-trip, 3 clean rows required), plus
   targeted stress items for the known capability limits. Only if all 18 pass
   does the real run start.
3. **Phase 2 — the 18 cells**, one at a time. Each cell script is self-contained:
   serve the model (one server per GPU, auto-detected) → health gates → greedy
   run → teardown. Each cell is resumable via its CSV; a failed run gets one
   automatic resume-retry before the master aborts.
4. **Phase 3 — summary**: a 6×3 accuracy table in `results/summary_table.txt`.

**Crashed / interrupted? Re-run the same command.** Nothing is recomputed:
provisioning skips verified items, cells skip completed rows.

## Requirements

- **Conda** installed (the script creates the `epf` env itself; python 3.11,
  vLLM 0.22.1 / torch 2.11.0 cu130 pinned in `requirements-epf.txt`).
- **ffmpeg 4–7** on the system (`sudo apt install ffmpeg`) — Mellow audio decode.
- **GPUs**: any count (auto-detected; one server per GPU). ≥32 GB VRAM per GPU is
  comfortable for the biggest model (Kimi ~22.6 GB bf16 at util 0.85); 24 GB is
  marginal. Reference box: 2× RTX PRO 6000 Blackwell (96 GB).
- **Disk: ~250 GB free** under `~/epf_data` (override via `greedy/config.local.sh`):
  models ~85 GB, MMAU-Pro audio 53 GB (+44 GiB transient zip), MMAR 3.5 GB,
  MMSU 1.7 GB (+1.5 GB transient parquets), parquets/misc ~2 GB.
- Network access to huggingface.co and github.com (Mellow code clone).

Non-default paths? Create `greedy/config.local.sh` (gitignored) with plain
`VAR=value` lines — e.g. `EPF_DATA_ROOT=/big/volume/epf_data`. Every knob in
`config.sh` can be overridden there or via the environment (`NUM_GPUS=8 bash …`).

## What exactly runs (the 18 cells)

| model | request `model=` | serve | audios sent | MMAU-Pro | MMAR | MMSU |
|---|---|---|---|---|---|---|
| Qwen2.5-Omni-7B | `qwen-omni` | vLLM | ≤3 | 5,090 | 1,000 | 5,000 |
| Qwen2.5-Omni-3B | `qwen-omni-3b` | vLLM | ≤3 | 5,090 | 1,000 | 5,000 |
| Qwen2-Audio-7B-Instruct | `qwen2-audio` | vLLM | ≤3 | 5,090 | 1,000 | 5,000 |
| Phi-4-multimodal-instruct | **`speech`** | vLLM + speech-LoRA | ≤3 | 5,090 | 1,000 | 5,000 |
| Kimi-Audio-7B-Instruct | `kimi-audio` | run19 wrapper | **1** | 5,090 | 1,000 | 5,000 |
| Mellow v0 | `mellow` | FastAPI shim | **≤2** | 5,090 | 1,000 | 5,000 |

FULL sets everywhere — by design. Model capability limits are handled by
**clamping, not subsetting** (unlike the earlier scaling runs, which used
per-model subsets):

- **Kimi-Audio** takes ONE audio per prompt → multi-audio items (~430 on MMAU-Pro)
  send only their FIRST clip. Its Whisper front-end hears only the first 30 s.
- **Mellow** has two audio slots → the 17 three-audio MMAU items send the first
  two clips; clips >10 s are cropped by the shim. The shim serves greedy with an
  extended 512-token text window (the faithful 129-token training window would
  truncate the MCQ prompt mid-question).
- **Qwen2-Audio** hears only the first 30 s of each clip (in-encoder truncation).

So a low number for these models on long/multi-audio items is *the model's
capability*, not a harness bug. Also expected: **Mellow's parse rate may be low**
(it was trained on inline `a) …` answers, not lettered options) — the parser
falls back to fuzzy choice-text matching; the summary reports parse rate per cell.

The prompt is identical for all 18 cells: the terse lettered-MCQ user turn plus a
system prompt that says answer with ONLY the letter, no reasoning. Temp 0,
max 64 tokens. Parsing/grading reuses the exact `predicted_index` machinery of
the scaling grids, so numbers are directly comparable.

Phi-4-MM note: the health gate requires the `speech` adapter to be exposed and
every request targets `model="speech"` — a base-model request would silently skip
the adapter and produce wrong-but-plausible numbers. The harness enforces this;
don't run manual requests against `phi4mm`.

## Rough cost

Per model: ~11,090 items × 1 short completion, split across your GPUs, plus 3
server startups (each of the 18 cells serves and tears down its own model).
Measured on the 2-GPU reference box: a vLLM cell sustains ~3–4 items/s (the
qwen-omni × MMAR cell — 1,000 items — took ~5 min including serving), so vLLM
MMAU/MMSU cells are ~25–30 min each; the Mellow shim serializes one request per
GPU at ~1.5 items/s, so Mellow × MMAU-Pro is ~1 h and all three Mellow cells ~2 h.
With the smoke phase (~35 min) expect **roughly 8 h end-to-end** on a 2-GPU box;
scales with GPU count. Kimi is the slowest to load (~3 min per startup).

## What to send back

```bash
cd greedy
tar czf greedy_results.tgz results
```

That's the 18 `results/<model>/<bench>/greedy.{csv,log}` pairs, the smoke outputs,
`results/summary.txt` (driver log) and `results/summary_table.txt` (the 6×3
table). If anything looked off, also mention which `results/servers/*.log` to
look at (they're inside the tarball already).

## If something fails

The master aborts at the first failing phase/cell and the log says which. Re-run
`bash greedy/run_all.sh` after fixing — everything resumes. To re-run a single
cell by hand:

```bash
bash greedy/cells/run_kimi_audio_mmar.sh     # any of the 18
```

Common causes: HF download hiccup (re-run), a stale server on ports 8100+
(`results/servers/*.pid` — kill those PIDs, never `pkill -f`), GPU OOM at serve
time (lower `GPU_MEM_UTIL`, e.g. `GPU_MEM_UTIL=0.8 bash greedy/run_all.sh`).
