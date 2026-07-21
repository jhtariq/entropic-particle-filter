# Run 19 — Kimi-Audio-7B-Instruct on the MMAU-Pro EPF grid (vLLM chat-path wrapper)

This package runs the PF/EPF budget grid for **Kimi-Audio-7B-Instruct**
(`moonshotai/Kimi-Audio-7B-Instruct`, 7B; Whisper-large-v3 front-end → Qwen2.5-family
backbone, tiktoken tokenizer) on the **single-audio ≤30 s MMAU-Pro test subset**
(`test_le30s_1a`, 1,947 MCQ) — on your own GPUs, with **one command**. The reference
box computed b1/8/16; those rows ship as a gzipped seed in `run19/seeds/` and are
reused automatically, so your run completes any b1/8/16 remainder and then runs the
remaining budget stages.

| run | model (pinned) | items | grid |
|---|---|---|---|
| 19 | `moonshotai/Kimi-Audio-7B-Instruct` `9a82a84…` | 1,947 MCQ (single-audio ∩ ≤30 s test subset) | P2+P4 × {mean_logprob, entropy} × budgets {1,8,16,32} — 4 curves per plot |

**Read this first — serving MUST go through `run19/serve_kimi.py`.** vLLM 0.22.1
registers this model natively, but six bugs make the stock path unusable for this
benchmark: no chat template is auto-wired, the tokenizer renders EMPTY prompts,
`continue_final_message` (the EPF step primitive) crashes, **any two concurrent
audio requests kill the whole engine** (EngineDeadError) once their clips differ in
length, the model's own generated text (it likes to end answers with a literal
"[EOS]") 400s the next continuation request through tiktoken's special-token check,
and **`top_logprobs` kills the server** once an audio-vocab token id cracks the
top-20 (its decode raises a KeyError that takes down the output handler — stochastic,
so short smokes can pass while long stages die). `lib.sh:serve_one` launches the
wrapper, which patches the four broken methods and passes the corrected
`template_kimi_audio_epf.jinja`. A plain `vllm serve` of this model 400s on every
chat request and dies under probe load — there is no silent-wrong-numbers mode here
(unlike run18's LoRA), but nothing works without the wrapper. `smoke.sh` Phase A
proves render + encode + decode offline; `lib.sh:sanity_prompt` re-proves the render
against every live server, and the concurrent smoke cell exercises the batching +
[EOS] fixes, before any experiment traffic.

## Requirements

- Linux box with NVIDIA GPUs, CUDA driver for the cu130 wheel stack (or your own
  torch/vLLM wheels), conda, and a HuggingFace account (`hf auth login`) for the
  dataset download.
- Disk: ~110 GB data (44 GiB zip transient + 53 GB audio + parquets) + ~23 GB model + env.
- **VRAM**: tuned on 2× RTX PRO 6000 Blackwell (96 GB) at `GPU_MEM_UTIL=0.85` and
  `RUN19_MAXLEN=8192`. Weights on-GPU ≈ 19 GB bf16 (7B LM with the TTS heads skipped
  + VQ adaptor + Whisper-large-v3 encoder) + KV cache + audio-encoder transients.
  All clips are ≤30 s on this subset, so encoder spikes are mild. **≥32 GB/GPU is
  comfortable; 24 GB is marginal (lower `MAX_INFLIGHT` first); 48 GB+ ideal.**
- GPU time at defaults: see cost anchors below. Stages are independent and
  resumable, so partial runs are useful.

## The whole flow

```bash
git clone <repo-url> && cd entropic-particle-filter
git checkout kimi-model

export EPF_DATA_ROOT=/big/volume/epf_data   # optional — defaults to ~/epf_data
export NUM_GPUS=8                           # optional — defaults to nvidia-smi count

bash benchmarking/mmau_pro/run19/setup_env.sh     # ONE conda env: 'epf' (vLLM 0.22.1 / torch 2.11.0)
bash benchmarking/mmau_pro/run19/fetch_data.sh    # parquets (incl. 1a subsets) + audio (sha256-verified)
bash benchmarking/mmau_pro/run19/fetch_models.sh  # checkpoint at the pinned revision (~23 GB)
bash benchmarking/mmau_pro/run19/smoke.sh         # tests, data, render asserts, serve+gates, tiny b8 cell

nohup bash benchmarking/mmau_pro/run19/run_all.sh > run19.log 2>&1 &
tail -f run19.log
```

Result: `benchmarking/mmau_pro/results/run19_kimiaudio/epf_kimi_bootstrap.html` — the
selected / oracle / majority accuracy-vs-budget curves for all four prompt×signal
lines, bootstrap error bars, and tables — plus `run19/epf_kimi_le30s1a.{jsonl,csv,log}`.

**Crashed / interrupted / machine rebooted?** Re-run the same `nohup … run_all.sh` line.
Every completed (item × prompt × signal × budget) cell is skipped via the append-only
JSONL resume; servers are restarted as needed. Nothing is ever recomputed or lost.

## Knobs (env vars or `config.local.sh` next to `config.sh`)

| knob | default | meaning |
|---|---|---|
| `EPF_DATA_ROOT` | `~/epf_data` | where datasets + HF model cache live |
| `NUM_GPUS` | auto | one vLLM replica per GPU, ports 8100+i |
| `PROMPTS_RUN19` | `2,4` | prompt methods: zero-shot CoT (best acc) + plan-and-solve (chunker, cross-run anchor); see RESULTS.md §19 screens |
| `BUDGETS` | `1 8 16 32` | staged in order; b1/8/16 seeded from the reference box |
| `RUN19_MAXLEN` | `8192` | = the model's max_position_embeddings — do NOT raise |
| `RUN19_SUBSET` | `test_le30s_1a` | 1,947 single-audio ≤30 s MCQ (see "Things worth knowing") |
| `GPU_MEM_UTIL` | `0.85` | do NOT raise to 0.9 — audio-encoder spikes OOM'd an engine (SETUP_GUIDE §10) |
| `MAX_INFLIGHT` | `64` | per-endpoint item concurrency = `max(1, MAX_INFLIGHT // budget)` |
| `MAX_INFLIGHT_B1` | `24` | lower cap for the budget-1 stage (SETUP_GUIDE §7) |
| `STOP_REGEX` | (empty) | optional trajectory stop; empty unless RESULTS.md §19 says otherwise |
| `DRY_RUN` | `0` | `1` prints the plan without executing |

## Things worth knowing

- **The six serving bugs.** vLLM 0.22.1's kimi path (a) auto-wires NO chat
  template, (b) renders EMPTY prompts through the stock tokenizer method,
  (c) crashes on `continue_final_message`, (d) dies (whole engine) when ≥2
  audio requests with different clip lengths are batched into one step — the
  stock code stacks per-item tensors that don't stack, (e) refuses to
  re-encode the model's own output when it contains special-token text like
  "[EOS]" (which this model emits often), 400ing the EPF continuation step,
  and (f) dies again under `top_logprobs` when an audio-vocab id (the LM head
  covers the audio vocabulary) reaches the top-20 and its per-candidate decode
  raises a tiktoken KeyError in the async output handler. `serve_kimi.py` +
  the corrected `template_kimi_audio_epf.jinja` fix all six (batch-of-one
  encoder path per item; encode gets `disallowed_special=()`; decode gets a
  per-token fallback rendering audio-range ids as ""; the corrected template
  also renders system messages, which the vLLM-bundled one silently drops).
  Never bypass the wrapper.
- **One audio per prompt, 30 seconds per audio.** vLLM's kimi implementation hard-caps
  requests at ONE audio (multi-audio 400s at request time; `--limit-mm-per-prompt`
  is silently clamped to 1), and the Whisper front-end truncates every clip to its
  first 30 s (~12.5 audio tokens/s → ~390 tokens/clip). The `test_le30s_1a` subset
  (single-audio ∩ all-clips-≤30 s; built by `build_1a_subset.py`, committed in
  `run19/data/`) makes both constraints lossless by construction. `test_1a` (4,660,
  length-unrestricted) exists for capacity audits and as a documented alternative —
  on it the model silently hears only the first 30 s of longer clips.
- **One env is enough.** `requirements-epf.txt` pins CUDA-13.0 builds of torch 2.11.0 /
  vLLM 0.22.1 (Blackwell reference box). On a different CUDA stack: install matching
  torch/vLLM wheels first, re-run `setup_env.sh` for the remaining pins. vLLM must
  register `MoonshotKimiaForCausalLM` (0.22.1 does).
- **"Same model" is enforced three ways**: the checkpoint is downloaded
  `--revision <sha>`, `check_snapshot` verifies the snapshot + every weight shard +
  the kimi tokenizer/Whisper-encoder assets, and the servers start with
  `--revision <sha>`. The TTS-only `audio_detokenizer/` (19 GB) and `vocoder/` (1 GB)
  are intentionally NOT downloaded — vLLM's text-out path never touches them.
- **Expected smoke numbers**: 122+ unit tests pass; loader counts test_le30s_1a=1947
  (13 ungradeable), test_1a=4660 (22), full=957; the offline kimi render assert prints
  "continuation clean"; `sanity_prompt` reports ~10 prompt tokens; both phase0 gates
  PASS (logprobs-with-audio, continue_final_message); the 4-item b8 cell ends with
  0 errors.
- **Blackwell note**: `VLLM_USE_FLASHINFER_SAMPLER=0` is set by `serve_one` (mandatory
  on sm_120; harmless elsewhere).
- **A block of identical connection errors** = a vLLM engine died (check
  `results/run19_kimiaudio/servers/*.log`); `run_all.sh` retries once, and re-running
  sweeps up errored cells.
- **Do not change temp/prompts/signals/subset after seeding** — the resume key is only
  (item × prompt × signal × budget), so a config drift silently forks the science
  between seeded and fresh rows.

## Cost anchors (reference box: 2× RTX PRO 6000 Blackwell)

Measured on the full 1,947-item b1/8/16 stages (P2+P4 × 2 signals = 4 cells per stage,
0 errors): **b1 = 16 min** (`--max-inflight 24`), **b8 = 132 min**, **b16 = 306 min**
→ 7.6 h total. Stage growth b8→b16 ≈ 2.3× (same as run18). Extrapolation for your
stages on the 2-GPU reference box: **b32 ≈ 11–12 h**; optional extensions b64 ≈ 23–27 h,
b128 ≈ 47–56 h. Scale ÷ (your GPU count / 2). Stages are independent and resumable, so
partial runs are useful. (16-item b8 smoke cell: ~7 min end-to-end incl. model load.)

## What to send back

Tarball of `benchmarking/mmau_pro/results/run19_kimiaudio/`:
the HTML, `run19/epf_kimi_le30s1a.csv` + `.log`, `summary.txt`, `servers/*.log` if
anything looked off, and (gzipped) `run19/epf_kimi_le30s1a.jsonl`.

```bash
cd benchmarking/mmau_pro/results
gzip -k run19_kimiaudio/run19/*.jsonl 2>/dev/null || true
tar czf run19_results.tgz run19_kimiaudio --exclude='*.jsonl'
```

## Provenance notes

- Parquets: HF dataset `macabdul9/MMAU_Pro_Testmini` @ `81eb01fb…`. Audio: HF dataset
  `gamma-lab-umd/MMAU-Pro` `data.zip`, sha256-verified (same pins as run16/17/18).
  Derived subsets: `test_le30s` from `run16/data/` (Run-15 provenance);
  `test_le30s_1a` + `test_1a` built by `run19/build_1a_subset.py` (single-audio filter)
  and committed in `run19/data/`.
- Model: single-snapshot revision `9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b`
  (= `refs/main` at package time). vLLM serves it natively (no trust-remote-code needed
  for the model class; the flag is passed anyway and is harmless); the wrapper +
  template exist because of the chat-path bugs above, not the model weights.
- Screens and decisions: RESULTS.md §19; serving template: SETUP_GUIDE §3 + the
  wrapper deltas documented in `serve_kimi.py`.
