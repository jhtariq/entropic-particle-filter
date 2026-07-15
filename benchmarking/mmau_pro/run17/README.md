# Run 17 — Mellow (167M) on the MMAU-Pro EPF grid, served by a custom shim

This package runs the FULL PF/EPF budget grid for **Mellow** (`soham97/mellow`, a
167M-param audio LM: HTSAT encoder → SmolLM2-135M decoder) — budgets **1→128** on the
5,073-item `test_no3a` subset of MMAU-Pro — on your own GPUs, with **one command**.
A 500-item pilot (+ the 326-item fully-heard slice) already ran at b1/8/16 on the
reference box; those rows ship as a gzipped seed in `run17/seeds/` and are reused
automatically, so your run completes b1/8/16 on the remaining items and then runs the
new b32/64/128 stages. Mellow is not servable by vLLM, so the package includes an
OpenAI-compatible serving shim (`benchmarking/mmau_pro/serve_mellow.py`, FastAPI +
transformers) that the harness talks to exactly as it talks to vLLM.

| run | model (pinned) | items | grid |
|---|---|---|---|
| 17 | `soham97/mellow` `83672db…` (v0.ckpt) + code `349f9b2…` + SmolLM2-135M `93efa2f…` | 5,073 MCQ (test minus 17 three-audio items) | P10+P11 × {mean_logprob, entropy} × budgets {1,8,16,32,64,128} — 4 curves per plot |

**Read this first — what this run is.** Mellow is tiny and was the first
weak-audio-coupling model in this campaign: its MCQ accuracy is near chance, its choices
respond only weakly to the audio (RESULTS.md §17), and it answers in ONE short burst (no
chain-of-thought is elicitable), so trajectories end at step 1 and the grid measures
**best-of-N sampling + self-certainty selection** rather than multi-step particle
filtering. That is the point of the run: the selection-vs-oracle scaling curve in the
tiny-model regime. Do not expect Qwen-like accuracy numbers.

## The shim (what replaces `vllm serve`)

`run_all.sh` starts one shim process per GPU (`serve_mellow.py`, ports 8100+i, via
`lib.sh:serve_one`). Each process loads the pinned checkpoint (fp32, ~1.5 GB VRAM) and
serves:

- `GET /v1/models` — health check (`lib.sh:wait_healthy` greps `"id"`);
- `POST /v1/chat/completions` — audio content parts (`file://` and base64), per-token
  `logprobs` + `top_logprobs=20`, vLLM-style `continue_final_message`, `stop` strings,
  `max_tokens`, `temperature`.

Generation is a KV-cache reimplementation of Mellow's reference loop, validated
**token-identical** to the reference in greedy mode (`validate_shim.py`, part of
`smoke.sh`). Two documented deviations from the reference wrapper: clips longer than the
10 s window get a deterministic FIRST-10s crop (the reference random-crops — that would
feed different audio to each PF step), and sampling is true multinomial at `temperature`
(the reference is effectively greedy).

**VRAM/concurrency notes**: the vLLM tuning in SETUP_GUIDE §3 (GPU_MEM_UTIL, flashinfer,
max-model-len) does NOT apply here. The shim decodes ONE request at a time per GPU
(`asyncio` lock; health endpoint stays responsive), so `MAX_INFLIGHT` is the only
throughput lever and any single ≥8 GB GPU works. More GPUs = linear speedup.

## Requirements

- Linux box with ≥1 NVIDIA GPU (≥8 GB), CUDA driver for the cu130 wheel stack (or your
  own torch wheels), conda, system **ffmpeg 4–7** (for torchcodec audio decode), and a
  HuggingFace account (`hf auth login`) for the dataset download.
- Disk: ~110 GB data (44 GiB zip transient + 53 GB audio + parquets) + ~2 GB models + env.
- GPU time at defaults: ≈ **TBD_ANCHOR GPU-hours** on the reference 2× RTX PRO 6000
  (Blackwell) box — see cost anchors below.

## The whole flow

```bash
git clone <repo-url> && cd entropic-particle-filter
git checkout self-log-probs

export EPF_DATA_ROOT=/big/volume/epf_data   # optional — defaults to ~/epf_data
export NUM_GPUS=8                           # optional — defaults to nvidia-smi count

bash benchmarking/mmau_pro/run17/setup_env.sh     # ONE conda env: 'epf'
bash benchmarking/mmau_pro/run17/fetch_data.sh    # parquets + audio (sha256-verified), exact-count checks
bash benchmarking/mmau_pro/run17/fetch_models.sh  # ckpts + SmolLM2 + code clone, all pinned
bash benchmarking/mmau_pro/run17/smoke.sh         # ~10-15 min: tests, data, parity, gates, tiny b8 cell

nohup bash benchmarking/mmau_pro/run17/run_all.sh > run17.log 2>&1 &
tail -f run17.log
```

Result: `benchmarking/mmau_pro/results/run17_mellow/epf_mellow_bootstrap.html` — **two
sections/plots**: FULL (5,073 `test_no3a`) and LE10S (the 326 fully-heard items), each
with the selected / oracle / majority accuracy-vs-budget curves for all four
prompt×signal lines, bootstrap error bars, and tables — plus
`run17/epf_mellow_no3a.{jsonl,csv,log}` and `run17/epf_mellow_le10s.csv`.

**Crashed / interrupted / machine rebooted?** Re-run the same `nohup … run_all.sh` line.
Every completed (item × prompt × signal × budget) cell is skipped via the append-only
JSONL resume; shims are restarted as needed. Nothing is ever recomputed or lost.

## Knobs (env vars or `config.local.sh` next to `config.sh`)

| knob | default | meaning |
|---|---|---|
| `EPF_DATA_ROOT` | `~/epf_data` | where datasets + HF model cache + code clone live |
| `NUM_GPUS` | auto | one shim per GPU, ports 8100+i |
| `PROMPTS_RUN17` | `10,11` | Mellow-native prompt methods (see RESULTS.md §17 screens) |
| `BUDGETS` | `1 8 16 32 64 128` | staged in order; b1/8/16 partially seeded (pilot rows reused) |
| `STOP_REGEX` | `[a-k]\)` | ends a trajectory at the first native-answer step (see below) |
| `WAVEFORM_CACHE` | `6000` | shim per-GPU audio LRU (~8 GB RAM; lower on small-RAM boxes) |
| `MAX_INFLIGHT` | `32` | per-endpoint item concurrency = `max(1, MAX_INFLIGHT // budget)` |
| `MAX_INFLIGHT_B1` | `24` | lower cap for the budget-1 stage (SETUP_GUIDE §7) |
| `MELLOW_VARIANT` | `v0` | `v0_s` = the "scaled" checkpoint (both are fetched) |
| `MELLOW_MAX_TEXT_TOKENS` | `0` | 0 = faithful 129-token window (KEEP — the extended mode breaks the model, §17) |
| `SINGLE_AUDIO_FILL` | `silence` | second audio slot for single-audio items |
| `DRY_RUN` | `0` | `1` prints the plan without executing |

## Things worth knowing

- **One env is enough.** The shim runs on the same torch 2.11.0 stack as run16 plus three
  small extras (`torchlibrosa`, `importlib_resources`, `torchcodec`) installed by
  `setup_env.sh`. vLLM is not used. `uv` is only for offline unit tests.
- **"Same model" is enforced four ways**: ckpt files are downloaded `--revision <sha>`
  and verified (`check_mellow_snapshot`), the SmolLM2 base is pinned
  (`check_smollm2_snapshot`), the GitHub code clone is pinned to a commit
  (`check_mellow_code`), and `smoke.sh` re-proves shim-vs-reference greedy parity on
  your GPUs before anything runs.
- **Expected smoke numbers**: 122+ unit tests pass; loader counts test=5090 (24
  ungradeable), test_no3a=5073 (24), test_le10s_flat=326; parity `RESULT: … PASS`;
  both phase0 gates PASS; the 4-item b8 cell ends with 0 errors.
- **Trajectories are single-step by design** (stop-regex ends them at the first native
  answer). If you see multi-chunk trajectories in the JSONL, something is off — check
  the prompts knob.
- **A block of identical connection errors** = a shim died (check
  `results/run17_mellow/servers/*.log`); `run_all.sh` retries once, and re-running
  sweeps up errored cells.

## Cost anchors (reference box: 2× RTX PRO 6000 Blackwell)

Measured on the 500-item pilot (P10+P11 × 2 signals per stage, `--waveform-cache 1024`):
~0.11 s per warm request at temp 0.8 (~6 tokens typical; ~57 tok/s single-request fp32);
cold-item cost is audio-decode-bound (~0.4–0.7 s for long clips — raise `--waveform-cache`
when the item working set fits in RAM). Stage wall-clock at ~826 items × 4 cells:
b8 = 43 min, b16 = 80 min → ≈ 10 min per (budget=8-equivalent) × 1,000 items × cell-pair.
Extrapolation on 5,073 items: completing b1/8/16 past the seeds ≈ 13 h, then
b32 ≈ 14 h / b64 ≈ 28 h / b128 ≈ 56 h on the 2-GPU reference box → **total ≈ 110
GPU-pair-hours**, scale ÷ (your GPU count / 2) — e.g. ≈ 28 h wall on 8 comparable GPUs.
Budget-128 dominates; stages are independent and resumable, so partial runs are useful.

## What to send back

Tarball of `benchmarking/mmau_pro/results/run17_mellow/`:
the HTML, `run17/epf_mellow_no3a.csv` + `run17/epf_mellow_le10s.csv` + `.log`,
`summary.txt`, `servers/*.log` if anything looked off, and (gzipped)
`run17/epf_mellow_no3a.jsonl`.

```bash
cd benchmarking/mmau_pro/results
gzip -k run17_mellow/run17/*.jsonl 2>/dev/null || true
tar czf run17_results.tgz run17_mellow --exclude='*.jsonl'
```

## Provenance notes

- Parquets: HF dataset `macabdul9/MMAU_Pro_Testmini` @ `81eb01fb…`. Audio: HF dataset
  `gamma-lab-umd/MMAU-Pro` `data.zip`, sha256-verified (same pins as run16).
- The two parquets in `run17/data/` are derived artifacts: `test_no3a` drops the 17
  three-audio MCQ rows (Mellow has exactly 2 audio slots; ids in
  `data/excluded_3audio_ids.txt`, which also lists 9 open-ended 3-audio rows), and
  `test_le10s_flat` keeps items Mellow FULLY hears (all clips ≤10 s with channels
  counted concatenated — the wrapper flattens stereo end-to-end). Regenerate/verify
  with `build_mellow_subset.py`.
- Prompt methods 10/11 (Mellow-native formats) live in `benchmarking/mmau_pro/prompt.py`;
  screens and decisions: RESULTS.md §17; shim details: `serve_mellow.py` docstring.
