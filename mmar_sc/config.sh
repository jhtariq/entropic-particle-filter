#!/bin/bash
# MMAR greedy + self-consistency study — configuration.
#
# Eight cells: 2 models x 4 arms.
#
#   arm                 prompt        temp  n    max_tokens
#   greedy_nocot        direct        0.0   1     64
#   greedy_p4           P4 CoT        0.0   1    700
#   sc128_nocot_t0.8    direct        0.8   128   64
#   sc128_p4_t0.8       P4 CoT        0.8   128  700
#
# Every knob is environment-overridable, and a gitignored config.local.sh next to this
# file is sourced FIRST so its values win over the := defaults below.
#
# The serving knobs (--max-num-seqs, --n-chunk, --max-inflight) are DERIVED at runtime
# from the GPU's actual VRAM and the model's KV footprint, because those two numbers
# differ by 4x between our models and by 3x between A40 and H200. See kv_plan() in
# lib.sh — hardcoding them would either OOM on A40 or waste most of an H200.

MMAR_SC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$MMAR_SC_DIR/.." && pwd)"

[ -f "$MMAR_SC_DIR/config.local.sh" ] && source "$MMAR_SC_DIR/config.local.sh"

# --- environment --------------------------------------------------------------------
: "${EPF_PY:=/u/awaheed/envs/epf/bin/python}"       # absolute — `conda activate` loses the PATH race
# REAL path (post-migration): /u/awaheed/epf_data is now a symlink here. vLLM's
# --allowed-local-media-path check canonicalizes request paths but not the allowed
# root, so the root MUST be given as the resolved path or every audio request 400s.
: "${EPF_DATA_ROOT:=/work/hdd/bcey/awaheed/its-for-audio-reasoning/epf_data}"
# NOT ":=" — the login profile exports HF_HOME=/work/hdd/bcey/awaheed/.cache and Slurm
# propagates the submitting environment into the job, so a ":=" default would silently
# never apply and every vLLM replica would hunt for the checkpoints in the wrong tree
# (LocalEntryNotFoundError under HF_HUB_OFFLINE, or a 50 GB re-download without it).
# Set it unconditionally; override deliberately via MMAR_SC_HF_HOME or config.local.sh.
HF_HOME="${MMAR_SC_HF_HOME:-$EPF_DATA_ROOT/hf_cache}"; export HF_HOME
: "${DATA_MMAR:=$EPF_DATA_ROOT/mmar}"                # MMAR-meta.json + audio/ (3.5 GB, 1,000 wavs)
: "${MEDIA_ROOT:=$EPF_DATA_ROOT}"                    # vLLM only serves audio under this root
# OUT_ROOT is derived from $BENCH further down (a ":=" default here would win and pin
# every benchmark to the MMAR output tree, colliding on the single-writer lock)
: "${NUM_GPUS:=$(nvidia-smi -L 2>/dev/null | wc -l)}"
: "${BASE_PORT:=8100}"
: "${GPU_MEM_UTIL:=0.85}"                            # NOT 0.90 — the audio encoder spikes outside vLLM's budget
: "${SERVE_TIMEOUT:=2400}"                           # seconds to wait for a server to become healthy
: "${SUBSET:=full}"                                  # full = 1,000 items (996 gradeable)

# --- experiment knobs ---------------------------------------------------------------
: "${SC_N:=128}"                                     # samples per item in the SC arms
: "${SC_TEMP:=0.8}"
: "${COT_METHOD:=4}"                                 # P4 plan-and-solve
: "${COT_MAX_TOKENS:=700}"
: "${NOCOT_MAX_TOKENS:=64}"
: "${SC_SEED:=1234}"                                 # base RNG seed; chunk c uses SC_SEED+c
# Per-benchmark model list (bench_cfg sets B_MODELS). MMAR runs all three; MMAU omits
# Kimi for now — its sc_p4 arm costs ~80 min alone at the mandatory inflight=1.
: "${MMAR_MODELS:=qwen2_audio phi4mm kimi_audio}"
: "${MMAU_MODELS:=qwen2_audio phi4mm}"
: "${MMAR_ARMS:=greedy_nocot greedy_p4 sc_nocot sc_p4}"
: "${DRY_RUN:=0}"

# --- model pins (identical to greedy/config.sh; verified against 9 config occurrences) ---
QWEN2_AUDIO_ID="Qwen/Qwen2-Audio-7B-Instruct"
QWEN2_AUDIO_REV="0a095220c30b7b31434169c3086508ef3ea5bf0a"
QWEN2_AUDIO_NAME="qwen2-audio"
QWEN2_AUDIO_REQNAME="qwen2-audio"
QWEN2_AUDIO_MAXLEN=8192                    # full context; its encoder hears only the first 30 s
QWEN2_AUDIO_SERVE=vllm
QWEN2_AUDIO_WEIGHTS_GIB=17                 # bf16 on-device weights, for the KV budget
QWEN2_AUDIO_KV_KIB=512                     # 32 layers x 32 kv_heads x 128 head_dim, full MHA
QWEN2_AUDIO_STOP=""
QWEN2_AUDIO_MAX_INFLIGHT=0        # 0 = use the kv_plan-derived value

PHI4MM_ID="microsoft/Phi-4-multimodal-instruct"
PHI4MM_REV="93f923e1a7727d1c4f446756212d9d3e8fcc5d81"
PHI4MM_NAME="phi4mm"
PHI4MM_REQNAME="speech"                    # EVERY request targets the speech-LoRA adapter name
PHI4MM_MAXLEN=32768
PHI4MM_SERVE=vllm_lora
PHI4MM_WEIGHTS_GIB=12
PHI4MM_KV_KIB=128                          # 32 layers x 8 kv_heads x 128 head_dim, GQA
PHI4MM_STOP=""
PHI4MM_MAX_INFLIGHT=0        # 0 = use the kv_plan-derived value
: "${SPEECH_LORA_RANK:=320}"
: "${SPEECH_LORA_DIR:=$HF_HOME/hub/models--${PHI4MM_ID//\//--}/snapshots/$PHI4MM_REV/speech-lora}"

# Kimi-Audio: NEVER plain `vllm serve`. vLLM 0.22.1 registers MoonshotKimiaForCausalLM
# but its chat path is broken six ways (empty prompt rendering, engine death on
# concurrent audio, "[EOS]" re-encode failures, audio-vocab decode crashes). The run19
# wrapper monkey-patches all six BEFORE starting the stock CLI — see its docstring.
KIMI_AUDIO_ID="moonshotai/Kimi-Audio-7B-Instruct"
KIMI_AUDIO_REV="9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b"
KIMI_AUDIO_NAME="kimi-audio"
KIMI_AUDIO_REQNAME="kimi-audio"
KIMI_AUDIO_MAXLEN=8192                     # = config.json max_position_embeddings — do NOT raise
KIMI_AUDIO_MAXAUDIOS=1                     # vLLM's kimi path takes at most ONE audio per prompt
KIMI_AUDIO_SERVE=kimi
KIMI_AUDIO_WEIGHTS_GIB=19                  # LLM + whisper-large-v3 encoder + projector
KIMI_AUDIO_KV_KIB=56                       # 28 layers x 4 kv_heads x 128 head_dim (GQA); verified from weights
# MANDATORY: Kimi ends answers with the literal TEXT "[EOS]", not a real EOS token.
# Without this stop string every generation runs to --max-tokens and then loops
# ("A[EOS][EOS]A[EOS]AAAA..."), which both breaks parsing and costs ~5x the time.
KIMI_AUDIO_STOP="[EOS]"
# HARD CAP of ONE audio item in flight per endpoint. Kimi's vLLM path corrupts output
# when >=2 audio items land in one engine step (residual of run19 bug #4: that patch
# fixed the CRASH, not the correctness). Measured on MMAR greedy no-CoT, 1000 items:
#   concurrency 32 -> 190-232 truncated/garbled;  concurrency 1 -> 1.
# The failing set is non-deterministic at temp 0 (Jaccard 0.402 across identical runs),
# so it is cross-item contamination, not a property of any clip. DO NOT RAISE.
KIMI_AUDIO_MAX_INFLIGHT=1
# --- Gemma 4 (audio-capable, MatFormer E2B/E4B) --------------------------------------
# Needs transformers >= 5 for the gemma4 processor; the pinned env stays at 4.57.3, so
# these two serve from an overlay venv (M_VLLM_BIN) that shadows ONLY transformers.
# Checkpoints live in the /work cache (M_HF_HOME) — /u quota cannot take them.
GEMMA4_E2B_ID="google/gemma-4-E2B-it"
GEMMA4_E2B_REV="3e22461f65e89153144f8adb70e3b8c2cc9845a7"
GEMMA4_E2B_NAME="gemma-4-e2b-it"
GEMMA4_E4B_ID="google/gemma-4-E4B-it"
GEMMA4_E4B_REV="ee0ef6023621cff504d758262d4e04895a5af4a2"
GEMMA4_E4B_NAME="gemma-4-e4b-it"
GEMMA4_MAXLEN=8192
GEMMA4_VLLM_BIN=/u/awaheed/envs/gemma4-overlay/bin/vllm
GEMMA4_HF_HOME=/work/hdd/bcey/awaheed/hf_cache

# --- Qwen2.5-Omni (thinker-only text out; same pins as the EPF sweeps) ---------------
# Checkpoints live in the /work cache (re-downloaded there after the /u purge).
QWEN_OMNI_3B_ID="Qwen/Qwen2.5-Omni-3B"
QWEN_OMNI_3B_REV="f75b40e3da2003cdd6e1829b1f420ca70797c34e"
QWEN_OMNI_3B_NAME="qwen-omni-3b"
QWEN_OMNI_3B_WEIGHTS_GIB=12                # 11.15 GiB checkpoint (vLLM load log)
QWEN_OMNI_3B_KV_KIB=36                     # 36 layers x 2 kv_heads x 128 head_dim, GQA
QWEN_OMNI_7B_ID="Qwen/Qwen2.5-Omni-7B"
QWEN_OMNI_7B_REV="ae9e1690543ffd5c0221dc27f79834d0294cba00"
QWEN_OMNI_7B_NAME="qwen-omni"
QWEN_OMNI_7B_WEIGHTS_GIB=22
QWEN_OMNI_7B_KV_KIB=56                     # 28 layers x 4 kv_heads x 128 head_dim, GQA
QWEN_OMNI_MAXLEN=8192
QWEN_OMNI_HF_HOME=/work/hdd/bcey/awaheed/hf_cache

KIMI_WRAPPER="$REPO_ROOT/benchmarking/mmau_pro/run19/serve_kimi.py"
KIMI_CHAT_TEMPLATE="$REPO_ROOT/benchmarking/mmau_pro/run19/template_kimi_audio_epf.jinja"

# --- Mellow (167M, HTSAT->SmolLM2-135M, no vLLM) --------------------------------------
# Served by the run17 FastAPI shim (benchmarking/mmau_pro/serve_mellow.py). Native
# prompt methods 10/11 only — the model was trained on lowercase inline "a) x b) y"
# prompts with a hard 129-token window; the lettered protocol prompts truncate and the
# extended window breaks the model (run17 README). Arms are therefore per-model (M_ARMS).
MELLOW_ID="soham97/mellow"
MELLOW_REV="83672db0dae28764e283210d5bb732621e903d8a"
MELLOW_NAME="mellow"
: "${MELLOW_VARIANT:=v0}"
: "${MELLOW_MAX_TEXT_TOKENS:=0}"          # 0 = faithful 129-token window; KEEP (run17 §17)
: "${SINGLE_AUDIO_FILL:=silence}"
: "${WAVEFORM_CACHE:=1100}"               # all 1,000 MMAR fitted wavs (~1.3 MB each) stay hot
: "${MELLOW_CODE_DIR:=$EPF_DATA_ROOT/mellow_code}"
MELLOW_CODE_COMMIT="349f9b2be84bec713ac71e54fcbcac9bf4d116e5"
SMOLLM2_ID="HuggingFaceTB/SmolLM2-135M"
SMOLLM2_REV="93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
: "${MELLOW_P10_MAX_TOKENS:=64}"          # native MCQ answers are one short burst
: "${MELLOW_P11_MAX_TOKENS:=300}"         # describe-then-answer; matches the EPF per-step cap

# --- benchmark selection ------------------------------------------------------------
# The same 4 arms run against either benchmark; only the loader, data root and expected
# record counts change. OUT_ROOT is per-bench so the single-writer lock and the sample
# JSONLs never collide between a running MMAR job and an MMAU one.
: "${BENCH:=mmar}"
: "${DATA_MMAU:=$EPF_DATA_ROOT/mmau}"        # MMAU-meta.json + audio/ (1.7 GB, 1,000 wavs)

# bench_cfg <bench> — export B_* for the selected benchmark
bench_cfg() {
  case "$1" in
    mmar)
      B_NAME=mmar; B_DATA="$DATA_MMAR"; B_MODULE=benchmarking.mmar.sample_runner
      B_LOADER="from benchmarking.mmar.loader import load_mmar_mcq as load"
      B_EXPECT_N=1000; B_EXPECT_GRADEABLE=996      # 4 golds match no single choice
      B_MODELS="$MMAR_MODELS" ;;
    mmau)
      B_NAME=mmau; B_DATA="$DATA_MMAU"; B_MODULE=benchmarking.mmau.sample_runner
      B_LOADER="from benchmarking.mmau.loader import load_mmau_mcq as load"
      B_EXPECT_N=1000; B_EXPECT_GRADEABLE=1000     # test-mini is fully gradeable
      B_MODELS="$MMAU_MODELS" ;;
    mmau_pro)
      # D1K subset (macabdul9/MMAU-Pro-D1K): 1,000 single-audio items, own audio under
      # one data/ root (symlink tree). Run with SUBSET=d1k. 9 golds match no choice;
      # 94 'open' items ship a single choice (trivially correct when parsed).
      B_NAME=mmau_pro; B_DATA="$EPF_DATA_ROOT/mmau_pro_d1k"
      B_MODULE=benchmarking.mmau_pro.sample_runner
      B_LOADER="from benchmarking.mmau_pro.loader import load_mmau_mcq as load"
      B_EXPECT_N=1000; B_EXPECT_GRADEABLE=991
      B_MODELS="${MMAU_PRO_MODELS:-mellow}" ;;
    *) echo "unknown bench: $1 (want mmar|mmau|mmau_pro)" >&2; return 1 ;;
  esac
}
bench_cfg "$BENCH" || return 1 2>/dev/null || exit 1
: "${OUT_ROOT:=/u/awaheed/${BENCH}_sc_out}"

# model_cfg <key> — export M_* for the selected model
model_cfg() {
  M_HF_HOME=""; M_VLLM_BIN=""; M_ARMS=""   # per-model overrides; empty = harness defaults
  case "$1" in
    mellow)
      M_KEY=mellow; M_ID=$MELLOW_ID; M_REV=$MELLOW_REV
      M_NAME=$MELLOW_NAME; M_REQNAME=$MELLOW_NAME
      M_MAXLEN=0; M_SERVE=mellow
      # kv_plan is vLLM-specific and bypassed for the shim (run_bench2); the dummies
      # only keep `set -u` satisfied. The shim decodes one request at a time, so
      # inflight 2 keeps the pipe full without stacking n=128 KV batches (~4.6 GB each).
      M_WEIGHTS_GIB=2; M_KV_KIB=1; M_STOP=""; M_MAX_INFLIGHT=2
      M_ARMS="${MELLOW_ARMS:-greedy_p10 greedy_p11 sc_p10 sc_p11}" ;;
    qwen2_audio)
      M_KEY=qwen2_audio; M_ID=$QWEN2_AUDIO_ID; M_REV=$QWEN2_AUDIO_REV
      M_NAME=$QWEN2_AUDIO_NAME; M_REQNAME=$QWEN2_AUDIO_REQNAME
      M_MAXLEN=$QWEN2_AUDIO_MAXLEN; M_SERVE=$QWEN2_AUDIO_SERVE
      M_WEIGHTS_GIB=$QWEN2_AUDIO_WEIGHTS_GIB; M_KV_KIB=$QWEN2_AUDIO_KV_KIB; M_STOP=$QWEN2_AUDIO_STOP; M_MAX_INFLIGHT=$QWEN2_AUDIO_MAX_INFLIGHT ;;
    phi4mm)
      M_KEY=phi4mm; M_ID=$PHI4MM_ID; M_REV=$PHI4MM_REV
      M_NAME=$PHI4MM_NAME; M_REQNAME=$PHI4MM_REQNAME
      M_MAXLEN=$PHI4MM_MAXLEN; M_SERVE=$PHI4MM_SERVE
      M_WEIGHTS_GIB=$PHI4MM_WEIGHTS_GIB; M_KV_KIB=$PHI4MM_KV_KIB; M_STOP=$PHI4MM_STOP; M_MAX_INFLIGHT=$PHI4MM_MAX_INFLIGHT ;;
    gemma4_e2b)
      M_KEY=gemma4_e2b; M_ID=$GEMMA4_E2B_ID; M_REV=$GEMMA4_E2B_REV
      M_NAME=$GEMMA4_E2B_NAME; M_REQNAME=$GEMMA4_E2B_NAME
      M_MAXLEN=$GEMMA4_MAXLEN; M_SERVE=vllm
      M_WEIGHTS_GIB=11; M_KV_KIB=35; M_STOP=""; M_MAX_INFLIGHT=0
      M_HF_HOME=$GEMMA4_HF_HOME; M_VLLM_BIN=$GEMMA4_VLLM_BIN ;;
    gemma4_e4b)
      M_KEY=gemma4_e4b; M_ID=$GEMMA4_E4B_ID; M_REV=$GEMMA4_E4B_REV
      M_NAME=$GEMMA4_E4B_NAME; M_REQNAME=$GEMMA4_E4B_NAME
      M_MAXLEN=$GEMMA4_MAXLEN; M_SERVE=vllm
      M_WEIGHTS_GIB=16; M_KV_KIB=84; M_STOP=""; M_MAX_INFLIGHT=0
      M_HF_HOME=$GEMMA4_HF_HOME; M_VLLM_BIN=$GEMMA4_VLLM_BIN ;;
    qwen_omni_3b)
      M_KEY=qwen_omni_3b; M_ID=$QWEN_OMNI_3B_ID; M_REV=$QWEN_OMNI_3B_REV
      M_NAME=$QWEN_OMNI_3B_NAME; M_REQNAME=$QWEN_OMNI_3B_NAME
      M_MAXLEN=$QWEN_OMNI_MAXLEN; M_SERVE=vllm
      M_WEIGHTS_GIB=$QWEN_OMNI_3B_WEIGHTS_GIB; M_KV_KIB=$QWEN_OMNI_3B_KV_KIB; M_STOP=""; M_MAX_INFLIGHT=0
      M_HF_HOME=$QWEN_OMNI_HF_HOME ;;
    qwen_omni_7b)
      M_KEY=qwen_omni_7b; M_ID=$QWEN_OMNI_7B_ID; M_REV=$QWEN_OMNI_7B_REV
      M_NAME=$QWEN_OMNI_7B_NAME; M_REQNAME=$QWEN_OMNI_7B_NAME
      M_MAXLEN=$QWEN_OMNI_MAXLEN; M_SERVE=vllm
      M_WEIGHTS_GIB=$QWEN_OMNI_7B_WEIGHTS_GIB; M_KV_KIB=$QWEN_OMNI_7B_KV_KIB; M_STOP=""; M_MAX_INFLIGHT=0
      M_HF_HOME=$QWEN_OMNI_HF_HOME ;;
    kimi_audio)
      M_KEY=kimi_audio; M_ID=$KIMI_AUDIO_ID; M_REV=$KIMI_AUDIO_REV
      M_NAME=$KIMI_AUDIO_NAME; M_REQNAME=$KIMI_AUDIO_REQNAME
      M_MAXLEN=$KIMI_AUDIO_MAXLEN; M_SERVE=$KIMI_AUDIO_SERVE
      M_WEIGHTS_GIB=$KIMI_AUDIO_WEIGHTS_GIB; M_KV_KIB=$KIMI_AUDIO_KV_KIB; M_STOP=$KIMI_AUDIO_STOP; M_MAX_INFLIGHT=$KIMI_AUDIO_MAX_INFLIGHT ;;
    *) echo "unknown model key: $1" >&2; return 1 ;;
  esac
}

# arm_cfg <arm> — export A_* (prompt method, n, temperature, max_tokens)
arm_cfg() {
  case "$1" in
    greedy_nocot) A_METHOD=0;            A_N=1;      A_TEMP=0.0;      A_MAXTOK=$NOCOT_MAX_TOKENS ;;
    greedy_p4)    A_METHOD=$COT_METHOD;  A_N=1;      A_TEMP=0.0;      A_MAXTOK=$COT_MAX_TOKENS ;;
    sc_nocot)     A_METHOD=0;            A_N=$SC_N;  A_TEMP=$SC_TEMP; A_MAXTOK=$NOCOT_MAX_TOKENS ;;
    sc_p4)        A_METHOD=$COT_METHOD;  A_N=$SC_N;  A_TEMP=$SC_TEMP; A_MAXTOK=$COT_MAX_TOKENS ;;
    # Mellow-native arms (methods 10/11); see the Mellow pin block for why not 0/4
    greedy_p10)   A_METHOD=10;           A_N=1;      A_TEMP=0.0;      A_MAXTOK=$MELLOW_P10_MAX_TOKENS ;;
    greedy_p11)   A_METHOD=11;           A_N=1;      A_TEMP=0.0;      A_MAXTOK=$MELLOW_P11_MAX_TOKENS ;;
    sc_p10)       A_METHOD=10;           A_N=$SC_N;  A_TEMP=$SC_TEMP; A_MAXTOK=$MELLOW_P10_MAX_TOKENS ;;
    sc_p11)       A_METHOD=11;           A_N=$SC_N;  A_TEMP=$SC_TEMP; A_MAXTOK=$MELLOW_P11_MAX_TOKENS ;;
    *) echo "unknown arm: $1" >&2; return 1 ;;
  esac
  A_ARM="$1"
}
