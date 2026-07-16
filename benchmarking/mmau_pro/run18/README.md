# Run 18 — Phi-4-multimodal-instruct on the MMAU-Pro EPF grid (speech-LoRA served)

This package runs the FULL PF/EPF budget grid for **Phi-4-multimodal-instruct**
(`microsoft/Phi-4-multimodal-instruct`, 5.6B; Conformer audio encoder → Phi-4-mini
backbone) — budgets **1→128** on the FULL 5,090-MCQ MMAU-Pro test set — on your own
GPUs, with **one command**. The reference box already computed b1/8/16; those rows ship
as a gzipped seed in `run18/seeds/` and are reused automatically, so your run completes
any b1/8/16 remainder and then runs the new b32/64/128 stages.

| run | model (pinned) | items | grid |
|---|---|---|---|
| 18 | `microsoft/Phi-4-multimodal-instruct` `93f923e…` + its in-checkpoint `speech-lora/` | 5,090 MCQ (full test set) | P4+P8 × {mean_logprob, entropy} × budgets {1,8,16,32,64,128} — 4 curves per plot |

**Read this first — the speech-LoRA is the model.** The checkpoint ships LoRA adapters
that vLLM does NOT auto-load (base-weight loading skips all `lora` tensors). Microsoft's
intended audio path is base + `speech-lora`, so `lib.sh:serve_one` starts vLLM with
`--enable-lora --max-lora-rank 320 --lora-modules speech=<snapshot>/speech-lora`, and
**every request names the ADAPTER (`model="speech"`), not the base**. A request naming
the base (`phi4mm`) is accepted and silently answers WITHOUT the adapter — that is the
one way to publish wrong numbers with no error anywhere. All package scripts route
through `lib.sh:probe_model_name`; keep that discipline in any ad-hoc probing too.
`wait_healthy` fails hard if the adapter is missing from `/v1/models`.

## Requirements

- Linux box with NVIDIA GPUs, CUDA driver for the cu130 wheel stack (or your own
  torch/vLLM wheels), conda, and a HuggingFace account (`hf auth login`) for the
  dataset download.
- Disk: ~110 GB data (44 GiB zip transient + 53 GB audio + parquets) + ~12 GB model + env.
- **VRAM**: tuned on 2× RTX PRO 6000 Blackwell (96 GB) at `GPU_MEM_UTIL=0.85` and
  `RUN18_MAXLEN=32768`. Budget ≈ 11.2 GB bf16 weights + the rank-320 LoRA slot + KV
  cache + transient audio-encoder spikes (600 s clips → 60k-frame conformer inputs,
  several GB outside vLLM's budget — the reason for 0.85, see SETUP_GUIDE §10). On
  smaller cards (≥48 GB): lower `RUN18_MAXLEN` first (the longest real item needs
  15,056 prompt tokens + ~1,800 generation; 24576 is safe), then `MAX_INFLIGHT`.
- GPU time at defaults: ≈ **100–130 GPU-pair-hours** on the reference box for the
  b32/64/128 extension — see cost anchors below. Budget-128 dominates; stages are
  independent and resumable, so partial runs are useful.

## The whole flow

```bash
git clone <repo-url> && cd entropic-particle-filter
git checkout phi4mm

export EPF_DATA_ROOT=/big/volume/epf_data   # optional — defaults to ~/epf_data
export NUM_GPUS=8                           # optional — defaults to nvidia-smi count

bash benchmarking/mmau_pro/run18/setup_env.sh     # ONE conda env: 'epf' (vLLM 0.22.1 / torch 2.11.0)
bash benchmarking/mmau_pro/run18/fetch_data.sh    # parquets + audio (sha256-verified), exact-count checks
bash benchmarking/mmau_pro/run18/fetch_models.sh  # checkpoint incl. speech-lora, pinned revision
bash benchmarking/mmau_pro/run18/smoke.sh         # tests, data, snapshot, serve+gates, tiny b8 cell

nohup bash benchmarking/mmau_pro/run18/run_all.sh > run18.log 2>&1 &
tail -f run18.log
```

Result: `benchmarking/mmau_pro/results/run18_phi4mm/epf_phi4mm_bootstrap.html` — the
selected / oracle / majority accuracy-vs-budget curves for all four prompt×signal
lines, bootstrap error bars, and tables — plus `run18/epf_phi4mm_5090.{jsonl,csv,log}`.

**Crashed / interrupted / machine rebooted?** Re-run the same `nohup … run_all.sh` line.
Every completed (item × prompt × signal × budget) cell is skipped via the append-only
JSONL resume; servers are restarted as needed. Nothing is ever recomputed or lost.

## Knobs (env vars or `config.local.sh` next to `config.sh`)

| knob | default | meaning |
|---|---|---|
| `EPF_DATA_ROOT` | `~/epf_data` | where datasets + HF model cache live |
| `NUM_GPUS` | auto | one vLLM replica per GPU, ports 8100+i |
| `SERVE_MODE` | `lora` | `lora` (speech-lora attached — KEEP) \| `merged` \| `base` (diagnostics only) |
| `PROMPTS_RUN18` | `4,8` | prompt methods: plan-and-solve + anti-shortcut (see RESULTS.md §18 screens) |
| `BUDGETS` | `1 8 16 32 64 128` | staged in order; b1/8/16 seeded from the reference box |
| `RUN18_MAXLEN` | `32768` | max-model-len; longest real item = 15,056 prompt tokens (audited) |
| `GPU_MEM_UTIL` | `0.85` | do NOT raise to 0.9 — audio-encoder spikes OOM'd an engine (SETUP_GUIDE §10) |
| `MAX_INFLIGHT` | `64` | per-endpoint item concurrency = `max(1, MAX_INFLIGHT // budget)` |
| `MAX_INFLIGHT_B1` | `24` | lower cap for the budget-1 stage (SETUP_GUIDE §7) |
| `STOP_REGEX` | (empty) | optional trajectory stop; empty unless RESULTS.md §18 says otherwise |
| `DRY_RUN` | `0` | `1` prints the plan without executing |

## Things worth knowing

- **One env is enough.** `requirements-epf.txt` pins CUDA-13.0 builds of torch 2.11.0 /
  vLLM 0.22.1 (Blackwell reference box). On a different CUDA stack: install matching
  torch/vLLM wheels first, re-run `setup_env.sh` for the remaining pins. vLLM must
  register `Phi4MMForCausalLM` (0.22.1 does).
- **"Same model" is enforced three ways**: the checkpoint is downloaded `--revision <sha>`,
  `check_snapshot` verifies the snapshot + every weight shard + the `speech-lora/`
  adapter files, and the servers start with `--revision <sha>`. (`vision-lora/` may or
  may not be present in your snapshot — it is unused here and NOT required.)
- **Expected smoke numbers**: 122+ unit tests pass; loader counts test=5090 (24
  ungradeable), full=957; `/v1/models` lists BOTH `phi4mm` and `speech`; both phase0
  gates PASS (logprobs-with-audio, continue_final_message); the 4-item b8 cell ends
  with 0 errors.
- **Blackwell note**: `VLLM_USE_FLASHINFER_SAMPLER=0` is set by `serve_one` (mandatory
  on sm_120; harmless elsewhere). The LoRA path uses PunicaWrapperGPU — verified working
  on the reference Blackwell box.
- **A block of identical connection errors** = a vLLM engine died (check
  `results/run18_phi4mm/servers/*.log`); `run_all.sh` retries once, and re-running
  sweeps up errored cells.
- **Do not change temp/prompts/signals/subset after seeding** — the resume key is only
  (item × prompt × signal × budget), so a config drift silently forks the science
  between seeded and fresh rows.

## Cost anchors (reference box: 2× RTX PRO 6000 Blackwell)

Measured on the full 5,090-item b1/8/16 stages (P4+P8 × 2 signals = 4 cells per stage,
0 errors): **b1 = 44 min** (`--max-inflight 24`), **b8 = 197 min**, **b16 = 433 min**
→ 11.2 h total. Stage growth b8→b16 was ≈2.2× (audio prefill amortizes across steps;
generation dominates). Extrapolation for your stages on the 2-GPU reference box:
**b32 ≈ 14–16 h, b64 ≈ 29–35 h, b128 ≈ 58–77 h → total ≈ 100–130 GPU-pair-hours**,
scale ÷ (your GPU count / 2). Budget-128 dominates; stages are independent and
resumable, so partial runs are useful. (16-item b8 smoke cell: 0.79 s/item —
short-audio-biased; the full-set b8 rate is ≈0.58 s/item per cell.)

## What to send back

Tarball of `benchmarking/mmau_pro/results/run18_phi4mm/`:
the HTML, `run18/epf_phi4mm_5090.csv` + `.log`, `summary.txt`, `servers/*.log` if
anything looked off, and (gzipped) `run18/epf_phi4mm_5090.jsonl`.

```bash
cd benchmarking/mmau_pro/results
gzip -k run18_phi4mm/run18/*.jsonl 2>/dev/null || true
tar czf run18_results.tgz run18_phi4mm --exclude='*.jsonl'
```

## Provenance notes

- Parquets: HF dataset `macabdul9/MMAU_Pro_Testmini` @ `81eb01fb…` (ships the full-test
  parquet used here). Audio: HF dataset `gamma-lab-umd/MMAU-Pro` `data.zip`,
  sha256-verified (same pins as run16/17). No derived subsets — Run 18 uses the stock
  `test` parquet (5,090 MCQ; clips to 600 s; empirical rate 12.5 audio tokens/s).
- Model: single-snapshot revision `93f923e1a7727d1c4f446756212d9d3e8fcc5d81`
  (= `refs/main` at package time); custom code in-checkpoint (`--trust-remote-code`).
- Screens and decisions: RESULTS.md §18; serving template: SETUP_GUIDE §3.
