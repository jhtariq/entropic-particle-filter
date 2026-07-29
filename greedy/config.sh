# Greedy-baseline configuration: 6 models × 3 benchmarks, ONE temp-0 chat
# completion per item (no CoT, no particle filtering). Every knob is overridable
# from the environment, e.g.
#   NUM_GPUS=8 bash greedy/run_all.sh
# or via a config.local.sh (gitignored) next to this file — it is sourced first
# and wins. Source this file, then lib.sh, from every greedy script.

GREEDY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$GREEDY_DIR/.." && pwd)"

# local overrides (plain VAR=value lines) — sourced FIRST so the := defaults
# below respect them, including everything derived from EPF_DATA_ROOT
[ -f "$GREEDY_DIR/config.local.sh" ] && source "$GREEDY_DIR/config.local.sh"

# --- machine / environment ----------------------------------------------------
: "${EPF_ENV_NAME:=epf}"
: "${CONDA_BASE:=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")}"
: "${EPF_PY:=$CONDA_BASE/envs/$EPF_ENV_NAME/bin/python}"  # ALWAYS absolute — `conda activate` may lose the PATH race (SETUP_GUIDE §2)
: "${EPF_DATA_ROOT:=$HOME/epf_data}"       # datasets land here (~115 GB total, see README)
: "${HF_HOME:=$EPF_DATA_ROOT/hf_cache}"; export HF_HOME   # model checkpoints (~85 GB)
: "${NUM_GPUS:=$(nvidia-smi -L 2>/dev/null | wc -l)}"
: "${BASE_PORT:=8100}"                     # one server per GPU on BASE_PORT+i
: "${GPU_MEM_UTIL:=0.85}"                  # NOT 0.9 — audio-encoder attention spikes OOM'd an engine at 0.9 (SETUP_GUIDE §10)
# vLLM only serves audio files under this root — must be a parent of ALL DATA_*
# roots below (default layout puts everything under EPF_DATA_ROOT, so it is)
: "${MEDIA_ROOT:=$EPF_DATA_ROOT}"

# --- greedy experiment knobs ----------------------------------------------------
: "${MAX_TOKENS:=64}"                      # short — the prompt asks for ONLY the letter
: "${MAX_INFLIGHT:=16}"                    # per-endpoint concurrency; greedy ≈ b1, which needed
                                           # a low cap on vLLM audio models (SETUP_GUIDE §7)
: "${SERVE_TIMEOUT:=1800}"                 # seconds to wait for a server to become healthy
: "${OUT_ROOT:=$GREEDY_DIR/results}"
: "${DRY_RUN:=0}"                          # 1 = print the 18-cell plan instead of executing
: "${SMOKE:=1}"                            # run_all.sh runs smoke.sh before the full cells

# The 18 cells, in master order: vanilla vLLM serving first, exotic serving last.
GREEDY_MODELS="qwen_omni_7b qwen_omni_3b qwen2_audio phi4mm kimi_audio mellow"
GREEDY_BENCHES="mmau mmar mmsu"

# --- model table -----------------------------------------------------------------
# Per model: _ID/_REV (exact HF pin), _NAME (served name), _REQNAME (the model=
# value REQUESTS use — differs from _NAME only for phi4mm), _MAXLEN, _MAXAUDIOS
# (runner-side clamp: audios beyond this are dropped before the request), _SERVE
# (vllm | vllm_lora | kimi | mellow — lib.sh serve dispatch), _HF_HOME (per-model
# cache override, for boxes where checkpoints live in different caches).

QWEN_OMNI_7B_ID="Qwen/Qwen2.5-Omni-7B"
QWEN_OMNI_7B_REV="ae9e1690543ffd5c0221dc27f79834d0294cba00"
QWEN_OMNI_7B_NAME="qwen-omni"
QWEN_OMNI_7B_MAXLEN=32768
QWEN_OMNI_7B_MAXAUDIOS=3
QWEN_OMNI_7B_SERVE=vllm
: "${QWEN_OMNI_7B_HF_HOME:=$HF_HOME}"

QWEN_OMNI_3B_ID="Qwen/Qwen2.5-Omni-3B"
QWEN_OMNI_3B_REV="f75b40e3da2003cdd6e1829b1f420ca70797c34e"
QWEN_OMNI_3B_NAME="qwen-omni-3b"
QWEN_OMNI_3B_MAXLEN=32768
QWEN_OMNI_3B_MAXAUDIOS=3
QWEN_OMNI_3B_SERVE=vllm
: "${QWEN_OMNI_3B_HF_HOME:=$HF_HOME}"

QWEN2_AUDIO_ID="Qwen/Qwen2-Audio-7B-Instruct"
QWEN2_AUDIO_REV="0a095220c30b7b31434169c3086508ef3ea5bf0a"
QWEN2_AUDIO_NAME="qwen2-audio"
QWEN2_AUDIO_MAXLEN=8192                    # full context; its encoder hears only the first 30 s of each clip
QWEN2_AUDIO_MAXAUDIOS=3
QWEN2_AUDIO_SERVE=vllm
: "${QWEN2_AUDIO_HF_HOME:=$HF_HOME}"

PHI4MM_ID="microsoft/Phi-4-multimodal-instruct"
PHI4MM_REV="93f923e1a7727d1c4f446756212d9d3e8fcc5d81"
PHI4MM_NAME="phi4mm"
PHI4MM_REQNAME="speech"                    # EVERY request targets the speech-LoRA adapter name;
                                           # a request naming the base is VALID but silently
                                           # skips the adapter (run18 README — the #1 gotcha)
PHI4MM_MAXLEN=32768
PHI4MM_MAXAUDIOS=3
PHI4MM_SERVE=vllm_lora
SPEECH_LORA_RANK=320                       # r=320 in the checkpoint's speech_lora config
: "${PHI4MM_HF_HOME:=$HF_HOME}"
: "${SPEECH_LORA_DIR:=$PHI4MM_HF_HOME/hub/models--${PHI4MM_ID//\//--}/snapshots/$PHI4MM_REV/speech-lora}"

KIMI_AUDIO_ID="moonshotai/Kimi-Audio-7B-Instruct"
KIMI_AUDIO_REV="9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b"
KIMI_AUDIO_NAME="kimi-audio"
KIMI_AUDIO_MAXLEN=8192                     # = config.json max_position_embeddings — do NOT raise
KIMI_AUDIO_MAXAUDIOS=1                     # vLLM's kimi path takes at most ONE audio per prompt;
                                           # its Whisper front-end hears only the first 30 s
KIMI_AUDIO_SERVE=kimi
: "${KIMI_AUDIO_HF_HOME:=$HF_HOME}"
# vLLM 0.22.1's kimi chat path is broken without BOTH of these (run19 README):
KIMI_WRAPPER="$REPO_ROOT/benchmarking/mmau_pro/run19/serve_kimi.py"
KIMI_CHAT_TEMPLATE="$REPO_ROOT/benchmarking/mmau_pro/run19/template_kimi_audio_epf.jinja"

MELLOW_ID="soham97/mellow"
MELLOW_REV="83672db0dae28764e283210d5bb732621e903d8a"       # HF ckpt revision (v0.ckpt/v0_s.ckpt/v0.yaml)
MELLOW_NAME="mellow"
MELLOW_MAXAUDIOS=2                         # the shim has exactly 2 audio slots; a 3rd clip
                                           # would crash slot assignment, so the runner clamps
MELLOW_SERVE=mellow
: "${MELLOW_HF_HOME:=$HF_HOME}"
: "${MELLOW_VARIANT:=v0}"
# Faithful mode (0) truncates prompts at Mellow's 129-token training window — the
# lettered MCQ prompt would be cut mid-question. Greedy uses the shim's extended
# window so the model at least SEES the full question + options (README caveat).
: "${MELLOW_MAX_TEXT_TOKENS:=512}"
: "${SINGLE_AUDIO_FILL:=silence}"          # slot-2 filler for single-audio items (silence | duplicate)
: "${WAVEFORM_CACHE:=6000}"                # shim per-GPU waveform LRU (~1.3 MB/entry; lower on small-RAM boxes)
# Mellow model CODE (no PyPI package): pinned clone of github.com/soham97/mellow
MELLOW_CODE_REPO="https://github.com/soham97/mellow"
MELLOW_CODE_COMMIT="349f9b2be84bec713ac71e54fcbcac9bf4d116e5"
: "${MELLOW_CODE_DIR:=$EPF_DATA_ROOT/mellow_code}"
# SmolLM2-135M base (architecture + tokenizer; weights are overwritten by the ckpt)
SMOLLM2_ID="HuggingFaceTB/SmolLM2-135M"
SMOLLM2_REV="93efa2f097d58c2a74874c7e644dbc9b0cee75a2"

# --- benchmark table ---------------------------------------------------------------
# FULL sets everywhere (locked decision): every model runs every item; capability
# limits are handled by the _MAXAUDIOS clamp + in-encoder truncation, not subsetting.
BENCH_MMAU_SUBSET="test"                   # the FULL MMAU-Pro test split
BENCH_MMAU_EXPECTED=5090                   # 5,090 MCQ (24 of them ungradeable — still answered)
BENCH_MMAR_SUBSET="full"
BENCH_MMAR_EXPECTED=1000                   # 1,000 items (996 gradeable)
BENCH_MMSU_SUBSET="full"
BENCH_MMSU_EXPECTED=5000                   # 5,000 items (all gradeable)

: "${DATA_TESTMINI:=$EPF_DATA_ROOT/mmau_pro_testmini}"     # MMAU-Pro parquets
: "${DATA_AUDIO:=$EPF_DATA_ROOT/mmau_pro_audio}"           # MMAU-Pro full-test audio (53 GB)
: "${DATA_MMAR:=$EPF_DATA_ROOT/mmar}"                      # MMAR-meta.json + audio/ (3.5 GB)
: "${DATA_MMSU:=$EPF_DATA_ROOT/mmsu}"                      # MMSU-meta.json + audio/ (1.7 GB)

# --- dataset pins ------------------------------------------------------------------
# MMAU-Pro: same sources as run16-19.
PARQUET_DATASET="macabdul9/MMAU_Pro_Testmini"              # ships testmini AND full-test parquets
PARQUET_REV="81eb01fb86bfc183dcb88b376ab1ca149a9d9c4b"
AUDIO_DATASET="gamma-lab-umd/MMAU-Pro"                     # data.zip: 44 GiB -> 5,787 files / 53 GB
AUDIO_ZIP_SHA256="8fab3e820b27bf7f239ae74a45a1085f1364b6971a7b5ac53f220d091b9b111c"
# MMAR: official release; audio ships as ONE tarball whose members are already
# prefixed audio/<name>.wav. Verified byte-identical to the reference copy
# (meta + evaluation.py + all 1,000 wavs) at this pinned revision.
MMAR_DATASET="BoJack/MMAR"
MMAR_REV="3bd051123480e80d273ae9e8e9f1653f49010ac7"
MMAR_AUDIO_TGZ_SHA256="a84cea951de2e4ef9fbf23a003a7f6a4e319df7d62bf3a36c4f09efc56612c12"
# MMSU: official release. The reference layout (MMSU-meta.json + audio/<id>.wav)
# is MATERIALIZED from the repo's data/*.parquet by greedy/materialize_mmsu.py —
# the repo's loose audio/ folder is a differently-encoded copy and is NOT used.
# All 5,000 parquet audio payloads verified byte-identical to the reference copy.
MMSU_DATASET="ddwang2000/MMSU"
MMSU_REV="548e2283105825bf908a7db5c09c00dbcf42bd4c"

# --- smoke stress items (picked from the reference full-test parquet) --------------
# see smoke.sh: these target the three capability-limit assumptions
SMOKE_Q2A_LONG_IDS="c93e3644-5227-4710-b27b-5c46750afbff,6ed2017e-f95a-4a8e-987f-06b210a870ac"
                                           # 1-audio MMAU items >30 s (truncate-not-400 check)
SMOKE_3AUDIO_ID="af898b05-838b-460b-8108-99471eea9c76"   # 3-audio item (Mellow clamp-to-2 check)
SMOKE_2AUDIO_ID="db44c341-d453-4c33-9739-65eab1c43418"   # 2-audio item (Kimi clamp-to-1 check)
