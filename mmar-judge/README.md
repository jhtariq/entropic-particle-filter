# mmar-judge — BoN + Beam Search on MMAR with an audio LLM judge

Replaces the math PRM with **Qwen2.5-Omni-7B served via vLLM as an audio-capable
judge**: for every scoring call the judge hears the item's audio clip and sees the
question, the lettered options, and the candidate text (final answers for Best-of-N;
partial reasoning chains at every step for Beam Search). Runner code lives in
`benchmarking/mmar_judge/`; this folder holds configs, launch scripts, and results.

## Layout

- `configs/` — `common.sh` (paths/ports), `config_local.sh` (budgets: BoN {1,4,8},
  Beam {8}), `config_collab.sh` (up to 128 — read its beam-cost warning),
  `judge_omni.sh`, `model_{omni,qwen2audio,phi4,kimi,mellow}.sh`
- `scripts/` — `serve_judge.sh`, `serve_policy.sh`, `run_smoke.sh`, `run_full.sh`,
  `stop_servers.sh`, `collab_run.sh` (end-to-end driver)
- `results/` — one folder per run: `{bon,beam}.{jsonl,csv,log}` + `servers/`
- `results.md` — protocol, tables, inferences, conclusions

## Quick start (collaborator)

```bash
cd <repo-root>
# everything in one command (serve judge + policy, run, stop servers):
bash mmar-judge/scripts/collab_run.sh mmar-judge/configs/model_omni.sh
```

Per model: `model_omni.sh` (full set), `model_qwen2audio.sh` (le30s subset),
`model_phi4.sh` (request name is `speech`, the LoRA adapter), `model_kimi.sh`
(le30s; served via the required run19 wrapper), `model_mellow.sh` (bespoke shim,
tiny model — expect format-following degradation).

## Manual staging

```bash
export OUT_DIR=mmar-judge/results/smoke100_qwen-omni
bash mmar-judge/scripts/serve_judge.sh
bash mmar-judge/scripts/serve_policy.sh mmar-judge/configs/model_omni.sh
bash mmar-judge/scripts/run_smoke.sh mmar-judge/configs/model_omni.sh   # 100 stratified items
bash mmar-judge/scripts/run_full.sh  mmar-judge/configs/model_omni.sh   # full set
bash mmar-judge/scripts/stop_servers.sh "$OUT_DIR"
```

Budget grids and concurrency are knobs in `configs/config_*.sh`; everything else
(judge rubrics, prompt method P4, scoring) is in `benchmarking/mmar_judge/run_judge.py`
flags. Runs resume from their JSONL (error rows are retried on rerun).

## Cost notes

- BoN: at most `budget` judge calls per item (fewer when candidates dedupe).
- Beam: ~`budget × levels` judge calls per item (levels ≤ max-steps 6). At b=128
  that is ~770 audio judge calls/item — see the warning in `config_collab.sh`.
