# MMSU Runs 4 (Mellow) + 5 (Phi-4-MM) — collaborator handoff

Audience: the collaborator who already ran the mmau_pro run16–19 extensions — you have the
`epf` env, the Mellow assets (ckpt + pinned code clone from `mmau_pro/run17/fetch_models.sh`),
and the Phi-4-multimodal snapshot (from `mmau_pro/run18/`). New here: the MMSU benchmark
harness (`benchmarking/mmsu/`) and two run configs. You supply your own MMSU dataset copy.

Both runs: **one prompt × {mean_logprob, entropy} × budgets {1, 8, 16, 32, 64, 128} × 5,000
items** (the MMSU house convention — deep ladder, single best prompt). Mellow runs P10 with
its native-answer stop; Phi-4 runs P4 through the speech-LoRA. Everything is resumable — if
anything crashes, re-run the same command; nothing is recomputed.

## 0. One-time setup

```bash
git fetch origin && git checkout mmsu-collab-mellow-phi4
```

Create `benchmarking/mmsu/scripts/local.sh` (gitignored; plain VAR=value lines — it wins
over every default in `common.sh`):

```bash
EPF_PY=<abs path to your epf python>            # e.g. $HOME/miniconda3/envs/epf/bin/python
EPF_VLLM=<abs path to your epf vllm>
HF_HOME_DIR=<your HF cache>                     # must contain the Phi-4 snapshot (run18 fetch)
DATA_ROOT=<your MMSU dir>                       # contains MMSU-meta.json + audio/ (5,000 wavs)
MEDIA_ROOT=<a PARENT dir of DATA_ROOT>          # served as --allowed-local-media-path
MELLOW_CODE_DIR=<your pinned soham97/mellow clone>   # from run17 fetch, commit 349f9b2
```

Verify the dataset copy before burning GPU-hours (expects 5,000 records, ~100% gradeable,
all wavs present):

```bash
$EPF_PY -m benchmarking.mmsu.verify_data --data-root $DATA_ROOT
```

## 1. Run 4 — Mellow (do this one first; it's light)

```bash
S=benchmarking/mmsu/scripts
bash $S/serve_mellow_shim.sh $S/config_run04_mellow.sh          # 1 shim/GPU on :8100/:8101, blocks until healthy
nohup setsid bash $S/watchdog.sh $S/config_run04_mellow.sh \
  > benchmarking/mmsu/results/run04_mellow/servers/watchdog.log 2>&1 &

# gates (both endpoints must PASS both gates):
$EPF_PY -m benchmarking.mmsu.phase0_gate --endpoint http://localhost:8100/v1 --model-name mellow
$EPF_PY -m benchmarking.mmsu.phase0_gate --endpoint http://localhost:8101/v1 --model-name mellow

nohup setsid bash $S/run_grid.sh $S/config_run04_mellow.sh \
  > benchmarking/mmsu/results/run04_mellow/stages.log 2>&1 &
tail -f benchmarking/mmsu/results/run04_mellow/stages.log      # watch for "stage bN ... 0 errors"
```

When done: kill the shims via their PID files
(`kill $(cat benchmarking/mmsu/results/run04_mellow/servers/*.pid)`) and the watchdog.
Never `pkill -f` by pattern — it can match your own shell.

## 2. Run 5 — Phi-4-multimodal

```bash
S=benchmarking/mmsu/scripts
bash $S/serve.sh $S/config_run05_phi4mm.sh                      # vLLM + speech-LoRA, 1 replica/GPU
nohup setsid bash $S/watchdog.sh $S/config_run05_phi4mm.sh \
  > benchmarking/mmsu/results/run05_phi4mm/servers/watchdog.log 2>&1 &

$EPF_PY -m benchmarking.mmsu.phase0_gate --endpoint http://localhost:8100/v1 --model-name speech
$EPF_PY -m benchmarking.mmsu.phase0_gate --endpoint http://localhost:8101/v1 --model-name speech

nohup setsid bash $S/run_grid.sh $S/config_run05_phi4mm.sh \
  > benchmarking/mmsu/results/run05_phi4mm/stages.log 2>&1 &
```

Note the request model name is `speech` (the LoRA adapter), not the base. If serving fails
to find the speech-lora dir, set `SPEECH_LORA_DIR` in `scripts/local.sh` to
`<your HF cache>/hub/models--microsoft--Phi-4-multimodal-instruct/snapshots/93f923e.../speech-lora`.

## 3. What to send back

The two results folders, whole:

```
benchmarking/mmsu/results/run04_mellow/   (mmsu_run04.{jsonl,csv,log} + stages.log + servers/)
benchmarking/mmsu/results/run05_phi4mm/   (mmsu_run05.{jsonl,csv,log} + stages.log + servers/)
```

(results/ is gitignored — tar them up or push to a scratch branch, whatever is easiest.)
The JSONLs are the source of truth; we regenerate reports on our side.

## Rough cost (same GPUs as the reference box)

- Run 4 (Mellow): the shim is the bottleneck, not the model; budget-128 stages dominate.
  Expect several hours end-to-end.
- Run 5 (Phi-4): MMAR's 1,000-item × 2-prompt × b≤32 grid took 8.4 h on ONE GPU; this is
  5,000 items × 1 prompt × b≤128 on two GPUs — expect roughly 30–40 GPU-pair-hours. The
  b64/b128 stages are most of it; partial results are useful, so feel free to send the
  folders after b32 completes and let the tail run.

## If something breaks

- Stage retries + resume are automatic (3 attempts, error sweep between). Re-running the
  same `run_grid.sh` line after ANY crash is always safe.
- An endpoint dying looks like a block of identical connection errors in stages.log — the
  watchdog log will show when it went down. Restart serving, re-run the same grid line.
- Anything unclear or off-pattern: stop after the current stage and ping us with
  stages.log + the servers/*.log tail rather than debugging blind.
