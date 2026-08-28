# Shared config for the random-weight EPF ablation on MMAR (Qwen2.5-Omni 3B/7B).
# Sourced by run_node.sh. Everything lives inside an awaheed folder.

REPO_ROOT=/work/hdd/bcey/awaheed/its-for-audio-reasoning/entropic-particle-filter
EPF_PY=/u/awaheed/envs/epf/bin/python
EPF_VLLM=/u/awaheed/envs/epf/bin/vllm

# The Omni checkpoints do NOT fit in /u's remaining quota, so they live in a second
# cache on /work. Audio stays on /u. Set unconditionally: the login profile exports a
# different HF_HOME and Slurm propagates it, which previously killed every engine with
# LocalEntryNotFoundError.
export HF_HOME=/work/hdd/bcey/awaheed/hf_cache   # Omni cache; model_cfg may override
export HF_HUB_OFFLINE=1

# Benchmark selection: BENCH=mmar (default) | mmau. Same arms, same code path, so the
# two random-reward nulls are directly comparable.
: "${BENCH:=mmar}"
case "$BENCH" in
  mmar) DATA_ROOT=/u/awaheed/epf_data/mmar; B_MODULE=benchmarking.mmar.diversity_probe
        OUT_ROOT=${OUT_ROOT_OVERRIDE:-/u/awaheed/mmar_epf_out}; B_EXPECT_N=1000 ;;
  mmau) # MMAU test-mini: 1,000 items, ALL 1,000 gradeable (music/sound/speech 334/333/333).
        # Loader imports mmau_pro.scoring.match_answer_index — same scorer as MMAR/D1K.
        DATA_ROOT=/u/awaheed/epf_data/mmau; B_MODULE=benchmarking.mmau.diversity_probe
        OUT_ROOT=${OUT_ROOT_OVERRIDE:-/u/awaheed/mmau_epf_out}; B_EXPECT_N=1000 ;;
  mmau_pro)
        # Audio lives on /work (11 G), parquet on /u. Clips are FAR longer than MMAR/MMAU
        # (median 52 s, max 586 s): at maxlen 8192 ~15% of items would 400 on context
        # length, so this bench needs 32768. Single-choice items are dropped before
        # selection (MMAU-Pro ships 106 that are trivially correct).
        DATA_ROOT=/u/awaheed/epf_data/mmau_pro_testmini
        AUDIO_ROOT=/work/hdd/bcey/awaheed/mmau-pro/audio_testmini
        B_MODULE=benchmarking.mmau_pro.diversity_probe
        OUT_ROOT=/u/awaheed/mmau_pro_epf_out; B_EXPECT_N=957
        : "${MAXLEN:=32768}"; : "${MIN_CHOICES:=2}"; : "${MAX_INFLIGHT:=32}" ;;
  mmau_pro_d1k)
        # User-curated D1K subset (macabdul9/MMAU-Pro-D1K): 1,000 single-audio items,
        # clips ≤30 s (median 18 s) so MMAR-scale context suffices — no 32768 needed.
        # Audio + parquet live as a symlink tree under data/ on /u (hf_cache blobs).
        # 94 'open' items ship one choice (kept: matches the SC-D1K convention, 991 gradeable).
        DATA_ROOT=/u/awaheed/epf_data/mmau_pro_d1k
        B_MODULE=benchmarking.mmau_pro.diversity_probe
        OUT_ROOT=${OUT_ROOT_OVERRIDE:-/u/awaheed/mmau_pro_d1k_epf_out}; B_EXPECT_N=1000 ;;
  *)    echo "unknown BENCH='$BENCH' (want mmar|mmau|mmau_pro|mmau_pro_d1k)" >&2; return 1 2>/dev/null || exit 1 ;;
esac
# --allowed-local-media-path must cover wherever the audio actually is
: "${AUDIO_ROOT:=}"
# real path post-migration (symlink at /u/awaheed/epf_data): vLLM's media whitelist
# compares canonicalized request paths against this literal root — must be resolved.
MEDIA_ROOT="${MMAR_EPF_MEDIA_ROOT:-/work/hdd/bcey/awaheed/its-for-audio-reasoning/epf_data}"
[ -n "$AUDIO_ROOT" ] && MEDIA_ROOT=/work/hdd/bcey/awaheed
: "${MIN_CHOICES:=1}"

: "${BASE_PORT:=8100}"
: "${GPU_MEM_UTIL:=0.85}"       # NOT 0.9 — the Omni audio encoder OOM'd an engine there
: "${MAXLEN:=8192}"             # MMAR's longest prompt is ~1.5k tok; 8192 leaves far more KV
: "${MAX_NUM_SEQS:=128}"        # b128 fires 128 concurrent particle requests per item
: "${SERVE_TIMEOUT:=1800}"

# --- the ablation ------------------------------------------------------------------
: "${SUBSET:=full}"
: "${SELECT:=catlen}"           # proportional over (category, length_type): keeps the duration mix
: "${LIMIT:=400}"
: "${PROMPTS:=4}"               # P4 plan-and-solve
: "${SIGNALS:=random_iid}"      # fresh i.i.d. U(0,1) per particle per step; annealing structurally off
: "${BUDGETS:=1,8,16,32,64,128}"
: "${TEMP:=0.8}"
: "${MAX_STEPS:=6}"
: "${MAX_TOKENS_PER_STEP:=300}"
: "${MAX_INFLIGHT:=64}"
: "${SEED:=1234}"

model_cfg () {  # model_cfg <3b|7b|q2a|phi4mm>
  M_SERVE=vllm; M_REQNAME=""; M_HF_HOME=""
  case "$1" in
    3b) M_ID=Qwen/Qwen2.5-Omni-3B; M_REV=f75b40e3da2003cdd6e1829b1f420ca70797c34e
        M_NAME=qwen-omni-3b ;;
    q2a)
        # Same pinned revision + vLLM flags as the SC-era serve_q2a.sh. Plain vLLM
        # path (no LoRA). 30 s encoder window: >30 s clips are only partially heard.
        M_ID=Qwen/Qwen2-Audio-7B-Instruct; M_REV=0a095220c30b7b31434169c3086508ef3ea5bf0a
        M_NAME=qwen2-audio
        M_HF_HOME=/u/awaheed/epf_data/hf_cache ;;
    7b) M_ID=Qwen/Qwen2.5-Omni-7B; M_REV=ae9e1690543ffd5c0221dc27f79834d0294cba00
        M_NAME=qwen-omni ;;
    phi4mm)
        # Base weights alone CANNOT do audio — every request must target the speech-LoRA
        # adapter by name, so it is served as a named LoRA module and M_REQNAME != M_NAME.
        # A base-named request is accepted and silently answers WITHOUT the adapter.
        M_ID=microsoft/Phi-4-multimodal-instruct; M_REV=93f923e1a7727d1c4f446756212d9d3e8fcc5d81
        M_NAME=phi4mm; M_REQNAME=speech; M_SERVE=vllm_lora
        M_HF_HOME=/u/awaheed/epf_data/hf_cache      # phi4mm is in the /u cache, not /work
        SPEECH_LORA_RANK=320
        SPEECH_LORA_DIR="$M_HF_HOME/hub/models--microsoft--Phi-4-multimodal-instruct/snapshots/$M_REV/speech-lora" ;;
    *)  echo "unknown model '$1' (want 3b|7b|q2a|phi4mm)" >&2; return 1 ;;
  esac
}

# defaults filled in after model_cfg
model_post () {
  [ -n "${M_REQNAME:-}" ] || M_REQNAME="$M_NAME"
  [ -n "${M_HF_HOME:-}" ] && export HF_HOME="$M_HF_HOME"
}
