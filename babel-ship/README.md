# babel-ship — MMAU-Pro d1k judge-PRM sweeps on an external 4×A100-80GB node

Self-contained kit to reproduce the judge-weighted particle-filter
**process-reward** sweeps (`judge_epf_prm`, full budgets 1–128) on the
**MMAU-Pro d1k** benchmark for five audio LLMs, using the exact model
checkpoints from the original NCSA Delta campaign. No weights or data ship
with this folder — models are pulled from HF at pinned revisions, the
benchmark is expected to already exist on the target machine.

The method: an external judge (Qwen3-Omni-30B) listens to the real audio and
scores each particle's **latest reasoning step** with P(Yes) from top-20
logprobs; scores drive resample-every-step particle filtering with entropic
annealing. Three selectors are reported per cell:
`judge_marg_mean`, `judge_marginal`, `argmax_judge`.

## 0. Get the code

```bash
git clone git@github.com:jhtariq/entropic-particle-filter.git
cd entropic-particle-filter
git checkout babel-ship
```

## 1. Environment (python 3.11)

```bash
pip install -r babel-ship/requirements.txt
pip install -e . --no-deps          # its_hub, editable, from the repo root
```

## 2. Models — download + checkpoint sanity

```bash
# optional: export HF_HOME=/big/disk/hf_cache
python babel-ship/sanity_check_models.py
```

Downloads all six checkpoints (5 generators + judge, ~120GB total) at the
**pinned revisions** in `models_manifest.json` and fails loudly if any
snapshot resolves to a different commit. Do not run sweeps on a mismatch —
numbers would not be comparable with the campaign.

## 3. Data sanity

```bash
export BABEL_D1K_ROOT=/path/to/mmau_pro_d1k
python babel-ship/sanity_check_data.py
```

Must print `DATA_OK` (1,000 records / 991 gradeable, all audio present). The
loader expects the `macabdul9/MMAU-Pro-D1K` layout: parquet + `data/` audio
clips under one root.

## 4. Run everything

```bash
export BABEL_D1K_ROOT=/path/to/mmau_pro_d1k
nohup bash babel-ship/launch_all.sh > babel-ship/out/launch.log 2>&1 &
```

Layout (one judge feeds all three generator slots — the generator, not the
judge, is always the bottleneck):

| GPU | role |
|---|---|
| 0 | judge (single GPU, port 8801) |
| 1 | kimi-audio (~13h, the wall-clock bound) |
| 2 | qwen-omni (7B) → qwen-omni-3b |
| 3 | phi4mm → qwen2-audio |

Each model runs a 2-item **smoke gate** (real audio through gen + judge,
non-null per-step scores required) before its sweep starts. Everything is
**resume-safe**: rerun `launch_all.sh` (or a single `run_d1k_prm.sh MODEL`)
after any crash and completed (item, budget) cells are skipped.

Expect ~1 day wall clock; ~0.5–1.3M judge calls per model, gen-bound.

## 5. Score

```bash
bash babel-ship/score_all.sh
```

Prints the 3-selector accuracy table per budget for every finished model
(991 gradeable items per cell). Filing into a consolidated CSV is opt-in:
`EXTRA_ARGS=--fill-csv BABEL_CSV_DIR=... bash babel-ship/score_all.sh`.

## Gotchas (learned the hard way)

- **Judge serving**: `--mm-encoder-attn-backend TORCH_SDPA` is mandatory in
  vLLM 0.22.1 on every GPU generation. First judge start per GPU arch pays
  ~25 min of FlashInfer MoE JIT (then cached in `~/.cache/flashinfer`).
- **Phi-4-mm**: requests must target model name `speech` (the LoRA adapter);
  the base name silently skips the speech LoRA. `run_d1k_prm.sh` does this.
- **7B naming**: the server publishes `qwen-omni`, not `qwen-omni-7b`.
- **Kimi**: never plain `vllm serve` — the repo wrapper
  (`benchmarking/mmau_pro/run19/serve_kimi.py`) + chat template are required.
- **One writer per jsonl**: never run two sweeps of the same model at once.
- Audio paths are passed to vLLM as local files: the servers allow-list
  `realpath $BABEL_D1K_ROOT`, so keep the data on a real path (symlink roots
  get resolved).

## Outputs

`babel-ship/out/d1k_<model>_prmjudge.jsonl` — one row per item×budget with
full texts, per-step judge scores (`judge_scores_all`), weights, and
selector inputs; `out/sweep_<model>.log` / `out/serve_*.log` for progress.
