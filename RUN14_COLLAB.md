# RUN14_COLLAB — two Gemma 4 EPF runs: E2B budgets 64/128, and E4B budgets 1→128

You're running TWO sweeps (they share §§1–3 setup and can run **concurrently on
disjoint GPU sets** — E2B servers use ports 810x, E4B servers 820x):

- **Run 14c** (§4–6): Gemma 4 **E2B** — extend our local budgets {1,8,16,32} grid to
  **{64, 128}**. Deliverable: `epf_gemma_le30s_max12_b64128.{jsonl,csv,log}` from
  `benchmarking/mmau_pro/results/run14_gemma4e2b/`. ≈ 20–28 h per 2 GPUs.
- **Run 15** (§7): Gemma 4 **E4B** — **prompt P7 only**, budgets **{1,8,16,32,64,128}**,
  with **b1/8/16 already completed and committed** (the JSONL ships in git; those
  stages resume instantly, you effectively run b32/64/128). Deliverable:
  `epf_gemma4e4b_max12.{jsonl,csv,log}` from
  `benchmarking/mmau_pro/results/run15_gemma4e4b/`. ≈ **12 h per 2 GPUs**.

Shared config for both: signals {mean_logprob, entropy} × the committed
**1,947-item ≤30 s single-audio** MMAU-Pro subset, `tokens_per_step=30`,
`max_steps=12`, temp 0.8 (both models hear max 30 s/clip — same processor cap).
Prompts: Run 14c = {4,7}; Run 15 = {7} only (P4's verbose CoT truncates on E4B at
this step budget and scores a −10–13 pp artifact, so it was dropped there).

Everything below was verified end-to-end on 2× RTX PRO 6000 Blackwell (July 2026).
General background/lessons live in `SETUP_GUIDE.md`; this file is the exact path for
this run. Steps: env → model → data → serve(+patch!) → smoke → run.

## 1. Environments (two: client + serving)

```bash
# CLIENT env (runs the benchmark scripts)
conda create -n epf python=3.11 -y
~/miniconda3/envs/epf/bin/pip install -r requirements-epf.txt      # pinned (torch 2.11 cu130)
~/miniconda3/envs/epf/bin/pip install -e ".[dev,benchmark]"        # from the repo root

# SERVING env (runs vLLM; needs transformers 5.x, hence separate)
conda create -n gemmaserve python=3.11 -y
~/miniconda3/envs/gemmaserve/bin/pip install vllm==0.22.1 transformers==5.13.0 \
    librosa soundfile av resampy

# ⚠ MANDATORY PATCH — without it the engine CRASHES under any concurrency
# (vLLM gemma4 batched-audio bug, present in upstream main too; RESULTS.md §20):
~/miniconda3/envs/gemmaserve/bin/python scripts/patch_vllm_gemma4.py
# expect: "patched .../gemma4_mm.py"  (idempotent; aborts if your vllm differs)
```

**PATH gotcha:** `conda activate` may not win your PATH — always call the env's
absolute `bin/python` (SETUP_GUIDE §2).

Offline sanity: `~/miniconda3/envs/epf/bin/python -m pytest tests/ -q --ignore=tests/e2e`
→ 94–95 passed (one test skips if the dataset isn't in place yet).

## 2. Model

```bash
export HF_HOME=<big-volume>/hf_cache     # ~10 GB
~/miniconda3/envs/gemmaserve/bin/hf download google/gemma-4-E2B-it
```

**Identity check (make sure we're on the same checkpoint):**

```bash
SNAP=$(ls -d $HF_HOME/hub/models--google--gemma-4-E2B-it/snapshots/*/)
basename $SNAP                     # expect: 9dbdf8a839e4e9e0eb56ed80cc8886661d3817cf
sha256sum $SNAP/model.safetensors  # expect: 2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550
```

(FYI: this model hears at most **30 s per clip** — that's why the item list is the
≤30 s subset; nothing for you to configure.)

## 3. Dataset (MMAU-Pro, ~54 GB)

```bash
mkdir -p data/mmau_pro/data && cd data/mmau_pro
wget https://huggingface.co/datasets/gamma-lab-umd/MMAU-Pro/resolve/main/test.parquet
wget https://huggingface.co/datasets/gamma-lab-umd/MMAU-Pro/resolve/main/data.zip   # 47.5 GB
unzip data.zip            # audio files must land in data/mmau_pro/data/ (5,787 files)
ln -s ../test.parquet data/test-00000-of-00001.parquet   # loader expects this name
cd ../..
```

**Verification (both must pass):**

```bash
~/miniconda3/envs/epf/bin/python -c "
from benchmarking.mmau_pro.loader import load_mmau_mcq
recs = load_mmau_mcq('data/mmau_pro', subset='test')
print(len(recs), sum(1 for r in recs if r.answer_index is None))"
# expect: 5090 24

# regenerate the ids files and confirm they match the committed ones bit-for-bit:
~/miniconda3/envs/epf/bin/python -m benchmarking.mmau_pro.make_le30s_ids \
    --data-root data/mmau_pro --smoke-multi 0
git diff --stat benchmarking/mmau_pro/results/run14_gemma4e2b/   # expect: no changes
```

## 4. Serve (one vLLM server per GPU)

```bash
# per GPU g = 0,1,2,...  → port 8100+g   (run each detached)
GEMMA_VLLM=~/miniconda3/envs/gemmaserve/bin/vllm HF_HOME=<your-hf-home> \
  nohup setsid scripts/serve_gemma4_e2b.sh <g> > serve_gpu<g>.log 2>&1 &

curl -s http://localhost:810<g>/v1/models     # healthy = lists gemma4-e2b
```

The script pins the flags that matter (`VLLM_USE_FLASHINFER_SAMPLER=0` for Blackwell,
0.85 GPU util, `--max-model-len 16384`, local-media path). Don't kill servers with
`pkill -f "vllm serve"` — it matches your own shell; kill by PID (SETUP_GUIDE §10.3).

## 5. Smoke tests (~3 min total; run all three before the sweep)

```bash
# (a) gates — per endpoint; BOTH must PASS (repeat for every port):
~/miniconda3/envs/epf/bin/python -m benchmarking.mmau_pro.phase0_gate \
  --endpoint http://localhost:8100/v1 --model-name gemma4-e2b \
  --data-root data/mmau_pro --subset test
# On our stack Gate-1 prints exactly: first token: 'Kn' logprob=-3.9342823028564453
# (same GPU class + pins should reproduce it; PASS/FAIL is what must hold.)

# (b) greedy anchors — deterministic on the same stack; expect this letter table:
~/miniconda3/envs/epf/bin/python -m benchmarking.mmau_pro.cot_compare \
  --endpoint http://localhost:8100/v1 --model-name gemma4-e2b \
  --data-root data/mmau_pro --subset test --audio-mode local-path \
  --prompts 4,7 --temperature 0.0 \
  --ids 22211743-ee24-452c-9d94-03da86a1d90d,aceca2ce-7ee1-45b5-b222-0651983f4531
#   item 22211743… : P4 -> D (gold B, wrong)   P7 -> B (gold B, correct)
#   item aceca2ce… : P4 -> C (gold C, correct) P7 -> A (gold C, wrong)
# A different letter on a near-tie can happen on a different stack — re-check (a)
# and your env pins first if more than one letter differs.

# (c) EPF end-to-end — 16 committed smoke items, budget 8, THE exact sweep config:
~/miniconda3/envs/epf/bin/python -m benchmarking.mmau_pro.diversity_probe \
  --endpoints http://localhost:8100/v1,http://localhost:8101/v1 --model-name gemma4-e2b \
  --data-root data/mmau_pro --subset test \
  --prompts 4,7 --signals mean_logprob,entropy --budgets 8 \
  --temp 0.8 --ess-threshold 0.6 --early-phase 0.7 --max-steps 12 --tokens-per-step 30 \
  --ids-file benchmarking/mmau_pro/results/run14_gemma4e2b/smoke_ids_16.txt \
  --max-inflight 64 --jsonl /tmp/smoke16.jsonl --csv /tmp/smoke16.csv --log /tmp/smoke16.log
# PASS = "0 errors" on all 4 cells and parsed >= 0.95. Errors on cell 1 with
# 'ClientConnectorError'/'500' = the §1 patch is missing from your serving env.
```

## 6. The run

```bash
PY=~/miniconda3/envs/epf/bin/python \
ENDPOINTS=http://localhost:8100/v1,http://localhost:8101/v1,<...one per GPU...> \
  nohup setsid bash scripts/run14_collab_b64_128.sh \
  > benchmarking/mmau_pro/results/run14_gemma4e2b/sweep_b64128.log 2>&1 &
```

- 2 budget stages (64 then 128) × 4 cells each; **resumable** — if anything dies,
  rerun the same command (completed rows are skipped, errored rows retried).
- Cost anchor: ≈ **20–28 h on 2 GPUs**, scaling ≈ linearly with endpoint count
  (budget-128 items run 128 particles as one batch; many GPUs help a lot).
- Watch the log for the per-cell `done in Xs (Y s/item, N errors)` lines — N should
  be ~0. A sudden block of identical `ClientConnectorError` rows = an endpoint died:
  restart that server, rerun the command. Keep a 60 s health-curl on every endpoint.
- When it prints `DONE` — send back
  `benchmarking/mmau_pro/results/run14_gemma4e2b/epf_gemma_le30s_max12_b64128.{jsonl,csv,log}`.

## 7. Run 15 — Gemma 4 **E4B**, full grid b1→128

Same setup (§§1–3: envs incl. the patch, dataset) — only the model and scripts differ.
All expected values below were verified on our box (2026-07-15).

```bash
# model (~16 GB) + identity check:
~/miniconda3/envs/gemmaserve/bin/hf download google/gemma-4-E4B-it
SNAP=$(ls -d $HF_HOME/hub/models--google--gemma-4-E4B-it/snapshots/*/)
basename $SNAP                     # expect: fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd
sha256sum $SNAP/model.safetensors  # expect: cfbd3d2f1cd71bd471c37fe2bf8546d5028d41e5736f64e1ca6c6b8893125503

# serve — one per GPU in the E4B GPU set; port = 8200 + gpu:
GEMMA_VLLM=~/miniconda3/envs/gemmaserve/bin/vllm HF_HOME=<your-hf-home> \
  nohup setsid scripts/serve_gemma4_e4b.sh <g> > serve_e4b_gpu<g>.log 2>&1 &
curl -s http://localhost:820<g>/v1/models    # lists gemma4-e4b
```

**Smoke (mirror of §5, E4B expected values):**
- gates per endpoint (`--endpoint http://localhost:820x/v1 --model-name gemma4-e4b`):
  both PASS; our Gate-1 line: `first token: 'Kn' logprob=-4.268167495727539`.
- greedy anchors (same two `--ids` as §5b, `--model-name gemma4-e4b`, port 820x):
  `22211743…`: P4 → **D** (gold B), P7 → **B** ✓; `aceca2ce…`: P4 → **C** ✓, P7 → **C** ✓
  (note P7 on the second item is **C** for E4B vs **A** for E2B — if you get the E2B
  letters here, you're serving the wrong model).
- EPF smoke (§5c command with `--endpoints ...820x... --model-name gemma4-e4b`,
  outputs to /tmp): 0 errors on all 4 cells, parsed ≥ 0.95.

**The run:**

```bash
PY=~/miniconda3/envs/epf/bin/python \
ENDPOINTS=http://localhost:8200/v1,http://localhost:8201/v1,<...one per E4B GPU...> \
  nohup setsid bash scripts/run15_e4b_b1_128.sh \
  > benchmarking/mmau_pro/results/run15_gemma4e4b/sweep_e4b.log 2>&1 &
```

6 budget stages, but **b1/8/16 arrive pre-completed from git** (the committed JSONL;
expect `0 to do (1947 resumed)` for those stages — if you instead see `1947 to do`,
your checkout is missing the seed) → you effectively run **b32, b64, b128 for P7 ×
both signals**. Resumable, same monitoring rules as §6. ≈ **12 h on 2 GPUs**
(E4B is ~1.3–2× slower than E2B per request), ≈ linear in endpoint count. When it
prints `DONE` — send back
`benchmarking/mmau_pro/results/run15_gemma4e4b/epf_gemma4e4b_max12.{jsonl,csv,log}`.
