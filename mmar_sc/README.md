# MMAR — greedy (± CoT) and self-consistency

Twelve cells: **3 models × 4 arms** on the full MMAR set (1,000 items, **996 gradeable**).

| arm | prompt | temp | n | max tokens |
|---|---|---|---|---|
| `greedy_nocot` | direct answer (`NO_COT_SYS`) | 0.0 | 1 | 64 |
| `greedy_p4` | P4 plan-and-solve | 0.0 | 1 | 700 |
| `sc_nocot` | direct answer | 0.8 | 128 | 64 |
| `sc_p4` | P4 plan-and-solve | 0.8 | 128 | 700 |

Models, pinned to the exact revisions used everywhere else in this repo:

| key | HF id | revision | note |
|---|---|---|---|
| `qwen2_audio` | `Qwen/Qwen2-Audio-7B-Instruct` | `0a095220…` | encoder hears only the first 30 s |
| `phi4mm` | `microsoft/Phi-4-multimodal-instruct` | `93f923e1…` | requests **must** target the `speech` LoRA |
| `kimi_audio` | `moonshotai/Kimi-Audio-7B-Instruct` | `9a82a84c…` | run19 wrapper + `--stop "[EOS]"`, both mandatory |

### Kimi-Audio needs two non-obvious things

**1. The run19 wrapper, never plain `vllm serve`.** vLLM 0.22.1 *does* register
`MoonshotKimiaForCausalLM`, but its chat path is broken six ways (empty prompt rendering,
engine death on concurrent audio, `[EOS]` re-encode failures, audio-vocab decode crashes).
[`run19/serve_kimi.py`](../benchmarking/mmau_pro/run19/serve_kimi.py) monkey-patches all six
before starting the stock CLI. `--chat-template` is mandatory — the checkpoint ships none,
so every chat request would 400.

**2. `--stop "[EOS]"`.** Kimi ends its answers with the literal *text* `[EOS]`, not a real
end-of-sequence token, so vLLM never stops on its own. Measured without the stop string:

| | truncated | unparsed | s/item |
|---|---|---|---|
| no-CoT, `max_tokens=64` | **3/3** | 1/3 | 1.90 |
| CoT, `max_tokens=700` | **3/3** | 0/3 | 13.68 |

Every generation ran to the cap and then looped — `'A[EOS][EOS][EOS]A[EOS]AAAAAA…'` — which
both destroys answer parsing and costs ~5× the time. With `--stop "[EOS]"`: 0 truncated,
0 unparsed, **0.27 s/item** and **3.00 s/item** respectively. The stop string is set per
model via `KIMI_AUDIO_STOP` in `config.sh`; the other two models need none.

Note the download skips `audio_detokenizer/` (17.7 GiB) and `vocoder/` (0.9 GiB) — those are
for speech *synthesis*, and vLLM's text path never loads them. 39.7 GiB → 22 GiB.

## Running

```bash
sbatch mmar_sc/sbatch_run.sh          # 4 GPUs, 8 h, gpuH200x8 or gpuA40x4
DRY_RUN=1 bash mmar_sc/run_all.sh     # print the 8-cell plan and exit
```

Resume is free: re-running the identical command is the entire recovery procedure after a
timeout, preemption or crash. `sample_runner` marks an item done only when **all N** of its
samples are present and error-free, so a half-written item is redone in full rather than
topped up — an item's N samples always come from one uninterrupted sampling call.

## Why new code was needed

Nothing already in the repo could do the self-consistency arm:

* `its_hub`'s `OpenAICompatibleLanguageModel` never emits OpenAI's `n` field, so N samples
  would be N separate requests, re-running the audio encoder every time.
* `greedy/greedy_runner.py` is no-CoT only and truncates `content` to **200 characters**,
  which would destroy every reasoning trace.
* `cot_compare.py` keeps full text but does exactly one greedy generation per item.
* No majority-vote aggregator over independent samples existed anywhere.

So `benchmarking/mmar/sample_runner.py` (generate) and `benchmarking/mmar/sc_report.py`
(score) are new; prompts and answer parsing are the repo's own
(`benchmarking.mmau_pro.prompt` / `.scoring`), which keeps these numbers directly
comparable to the existing MMAU-Pro and MMSU baselines.

## Every generation is saved

`sample_runner` writes **one JSONL row per sample** with the full untruncated `response`
text, to `$OUT_ROOT/samples/<model>__<arm>.jsonl` — 512,000 generations, ~1 GB. Row schema:

```
unique_id category length_type arm method temperature n_requested model
gold_letter n_choices sample_idx chunk response pred_letter correct
finish_reason prompt_tokens chunk_completion_tokens latency_s error
```

`sc_report.py` reads only these files, so re-scoring, re-voting at other budgets, or any
downstream use of the traces never needs another GPU-hour.

## Memory model — why the tuning is derived, not hardcoded

KV cost per token differs by **4×** between the two models:

| model | layers × kv_heads × head_dim | KV / token | measured s/item @ n=128 |
|---|---|---|---|
| Qwen2-Audio-7B | 32 × 32 × 128 (full MHA) | **512 KiB** | 4.98 |
| Phi-4-MM | 32 × 8 × 128 (GQA) | **128 KiB** | 5.18 |
| Kimi-Audio-7B | 28 × 4 × 128 (GQA) | **56 KiB** | 10.82 |

All three verified from the actual safetensors shapes, not from `config.json` — Qwen2-Audio's
`text_config` omits the head counts entirely and inherits `Qwen2Config` defaults, so reading
the config would have given the wrong answer for the one model where it matters most.

At `max_tokens=700` one `n=128` Qwen2-Audio request would need **43.8 GiB** of KV. Measured
on an A40: vLLM reports `Available KV cache memory: 21.1 GiB / 43,200 tokens` — so only ~61
full-length sequences fit, and N must be split into `--n-chunk` sized requests. An H200
(141 GB) holds the whole `n=128` at once. `kv_plan()` in `lib.sh` derives `--max-num-seqs`,
`--n-chunk` and `--max-inflight` from the card actually present, so the same script is
correct on either.

Prefix caching is on by default in vLLM V1, so the chunks of one item reuse the audio
prefill and chunking costs essentially nothing.

## Scoring contract

* **Ungradeable items excluded.** MMAR ships 4 whose gold text matches no single choice
  (996 gradeable), matching `verify_data.py`'s pin.
* **Unparsed samples cast no vote.** An item where every sampled vote is unparseable counts
  as *wrong*, not dropped from the denominator.
* **Ties broken uniformly at random** from among the tied letters via the seeded RNG — not
  by option order, which would bias toward `A`.
* **Budget curve**: for each K < N, draw R=50 subsets of size K *without replacement*,
  majority-vote each, report mean ± sd. At K = N there is one possible subset, so it is one
  exact number with no error bar (flagged `(exact)`, R=1).
* A `le30s` cut (the 983 clips Qwen2-Audio's 30 s window can fully hear) is reported
  alongside, derived from `length_type` — no separate run needed.

## Preflight

`run_all.sh` asserts the record count before any GPU work. `load_mmar_mcq` has
`require_audio_exists=True` and **silently drops** items whose wav is missing, which would
shrink the eval set with no error at all — the assertion is the only thing standing between
a missing file and a quietly wrong accuracy.

Data is byte-verified against `greedy/manifests/mmar.sha256` (1,001 files) at download.
