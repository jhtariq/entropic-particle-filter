# Shared helpers for the greedy scripts. Source config.sh BEFORE this file.
# Serving/verification patterns are lifted from the run16-19 handoff packages:
# run19 (kimi wrapper serve + snapshot shard walk), run18 (speech-LoRA serve +
# adapter-exposed health gate), run17 (mellow shim serve + 3-way identity check).

model_cfg() {  # model_cfg <model_key> -> sets M_ID M_REV M_NAME M_REQNAME M_MAXLEN M_MAXAUDIOS M_SERVE M_HF_HOME
  local key="$1" p
  p="$(echo "$key" | tr '[:lower:]' '[:upper:]')"
  M_ID="$(eval echo "\${${p}_ID:-}")"
  M_REV="$(eval echo "\${${p}_REV:-}")"
  M_NAME="$(eval echo "\${${p}_NAME:-}")"
  M_REQNAME="$(eval echo "\${${p}_REQNAME:-}")"
  M_MAXLEN="$(eval echo "\${${p}_MAXLEN:-}")"
  M_MAXAUDIOS="$(eval echo "\${${p}_MAXAUDIOS:-}")"
  M_SERVE="$(eval echo "\${${p}_SERVE:-}")"
  M_HF_HOME="$(eval echo "\${${p}_HF_HOME:-$HF_HOME}")"
  : "${M_REQNAME:=$M_NAME}"                # only phi4mm requests use a different name (the adapter)
  [ -n "$M_ID" ] || { echo "FATAL: unknown model '$key'" >&2; return 1; }
}

bench_cfg() {  # bench_cfg <mmau|mmar|mmsu> -> sets B_SUBSET B_EXPECTED B_DATA_ROOT B_AUDIO_ROOT
  case "$1" in
    mmau) B_SUBSET="$BENCH_MMAU_SUBSET"; B_EXPECTED="$BENCH_MMAU_EXPECTED"
          B_DATA_ROOT="$DATA_TESTMINI";  B_AUDIO_ROOT="$DATA_AUDIO" ;;
    mmar) B_SUBSET="$BENCH_MMAR_SUBSET"; B_EXPECTED="$BENCH_MMAR_EXPECTED"
          B_DATA_ROOT="$DATA_MMAR";      B_AUDIO_ROOT="" ;;
    mmsu) B_SUBSET="$BENCH_MMSU_SUBSET"; B_EXPECTED="$BENCH_MMSU_EXPECTED"
          B_DATA_ROOT="$DATA_MMSU";      B_AUDIO_ROOT="" ;;
    star_bench) B_SUBSET="$BENCH_STAR_BENCH_SUBSET"; B_EXPECTED="$BENCH_STAR_BENCH_EXPECTED"
          B_DATA_ROOT="$DATA_STAR_BENCH"; B_AUDIO_ROOT="" ;;
    *)    echo "FATAL: unknown bench '$1'" >&2; return 1 ;;
  esac
}

hf_cli() {  # hf (huggingface_hub >= 0.34) with huggingface-cli fallback.
  # HF_HUB_ENABLE_HF_TRANSFER=1 in the caller's profile crashes downloads when the
  # package is missing from the env — enable it only if actually importable.
  local bin_dir xfer=1
  bin_dir="$(dirname "$EPF_PY")"
  "$EPF_PY" -c "import hf_transfer" 2>/dev/null || xfer=0
  if [ -x "$bin_dir/hf" ]; then
    HF_HUB_ENABLE_HF_TRANSFER=$xfer "$bin_dir/hf" "$@"
  else
    HF_HUB_ENABLE_HF_TRANSFER=$xfer "$bin_dir/huggingface-cli" "$@"
  fi
}

check_snapshot() {  # check_snapshot <org/model> <rev> <hf_home> — pinned snapshot, complete shards
  local id="$1" rev="$2" hf_home="$3"
  local dir="$hf_home/hub/models--${id//\//--}/snapshots/$rev"
  [ -d "$dir" ] || { echo "FAIL: snapshot $dir missing — run fetch_models.sh" >&2; return 1; }
  [ -e "$dir/config.json" ] || { echo "FAIL: $dir lacks config.json" >&2; return 1; }
  "$EPF_PY" - "$dir" <<'PYEOF'
import json, os, sys
d = sys.argv[1]
idx = os.path.join(d, "model.safetensors.index.json")
if os.path.exists(idx):
    files = sorted(set(json.load(open(idx))["weight_map"].values()))
    missing = [f for f in files if not os.path.exists(os.path.join(d, f))]
    assert not missing, f"missing weight shards: {missing}"
print(f"snapshot OK (pinned revision present & complete): {d}")
PYEOF
}

check_model() {  # check_model <model_key> — full identity check for one model
  model_cfg "$1"
  local dir="$M_HF_HOME/hub/models--${M_ID//\//--}/snapshots/$M_REV" f
  case "$1" in
    kimi_audio)
      check_snapshot "$M_ID" "$M_REV" "$M_HF_HOME" || return 1
      # kimi extras: tiktoken vocab + tokenizer config feed KimiAudioTokenizer, and
      # vLLM loads the Whisper encoder from whisper-large-v3/. audio_detokenizer/ +
      # vocoder/ are TTS-only and intentionally NOT downloaded (-20 GB).
      for f in tiktoken.model tokenizer_config.json generation_config.json \
               configuration_moonshot_kimia.py \
               whisper-large-v3/model.safetensors whisper-large-v3/config.json \
               whisper-large-v3/preprocessor_config.json; do
        [ -e "$dir/$f" ] || { echo "FAIL: $dir lacks $f" >&2; return 1; }
      done
      echo "kimi assets OK: tokenizer + whisper-large-v3 encoder" ;;
    phi4mm)
      check_snapshot "$M_ID" "$M_REV" "$M_HF_HOME" || return 1
      # the speech adapter IS the audio path; vision-lora is unused for audio-only
      for f in adapter_config.json adapter_model.safetensors; do
        [ -e "$dir/speech-lora/$f" ] || { echo "FAIL: $dir/speech-lora lacks $f" >&2; return 1; }
      done
      echo "speech-lora OK: $dir/speech-lora" ;;
    mellow)
      # raw ckpt files (no safetensors index) + SmolLM2 base + pinned code clone
      [ -d "$dir" ] || { echo "FAIL: snapshot $dir missing — run fetch_models.sh" >&2; return 1; }
      for f in "${MELLOW_VARIANT}.ckpt" v0.yaml; do
        [ -e "$dir/$f" ] || { echo "FAIL: $dir lacks $f" >&2; return 1; }
      done
      echo "mellow snapshot OK: $dir"
      local sdir="$M_HF_HOME/hub/models--${SMOLLM2_ID//\//--}/snapshots/$SMOLLM2_REV"
      [ -e "$sdir/config.json" ] || { echo "FAIL: SmolLM2 snapshot $sdir incomplete" >&2; return 1; }
      echo "smollm2 snapshot OK: $sdir"
      local head
      head="$(git -C "$MELLOW_CODE_DIR" rev-parse HEAD 2>/dev/null)" \
        || { echo "FAIL: $MELLOW_CODE_DIR is not a git clone — run fetch_models.sh" >&2; return 1; }
      [ "$head" = "$MELLOW_CODE_COMMIT" ] \
        || { echo "FAIL: mellow code at $head, want $MELLOW_CODE_COMMIT" >&2; return 1; }
      echo "mellow code OK: $MELLOW_CODE_DIR @ ${head:0:12}" ;;
    *)
      check_snapshot "$M_ID" "$M_REV" "$M_HF_HOME" ;;
  esac
}

ensure_model() {  # ensure_model <model_key> — download only if the pinned snapshot is incomplete
  model_cfg "$1"
  if check_model "$1" > /dev/null 2>&1; then
    echo "model $M_ID@${M_REV:0:12} already present — skipping download"
    return 0
  fi
  case "$1" in
    kimi_audio)
      # ONE --exclude flag with both patterns: a second --exclude OVERRIDES the first
      # (nargs semantics), which silently re-admits the 18 GB TTS detokenizer
      HF_HOME="$M_HF_HOME" hf_cli download "$M_ID" --revision "$M_REV" \
        --exclude "audio_detokenizer/*" "vocoder/*" > /dev/null ;;
    mellow)
      HF_HOME="$M_HF_HOME" hf_cli download "$M_ID" v0.ckpt v0_s.ckpt v0.yaml config.json \
        --revision "$M_REV" > /dev/null
      HF_HOME="$M_HF_HOME" hf_cli download "$SMOLLM2_ID" --revision "$SMOLLM2_REV" > /dev/null
      if [ ! -d "$MELLOW_CODE_DIR/.git" ]; then
        git clone "$MELLOW_CODE_REPO" "$MELLOW_CODE_DIR"
      fi
      git -C "$MELLOW_CODE_DIR" checkout -q "$MELLOW_CODE_COMMIT" ;;
    *)
      HF_HOME="$M_HF_HOME" hf_cli download "$M_ID" --revision "$M_REV" > /dev/null ;;
  esac
  check_model "$1"
}

endpoints_csv() {
  local eps="" i
  for i in $(seq 0 $((NUM_GPUS - 1))); do eps+="${eps:+,}http://localhost:$((BASE_PORT + i))/v1"; done
  echo "$eps"
}

ports_must_be_free() {  # refuse to double-serve: nothing may answer on our ports
  local i port
  for i in $(seq 0 $((NUM_GPUS - 1))); do
    port=$((BASE_PORT + i))
    if curl -s --max-time 2 "http://localhost:$port/v1/models" | grep -q '"id"'; then
      echo "FATAL: something is already serving on :$port — stop it (or change BASE_PORT) first" >&2
      return 1
    fi
  done
}

serve_model() {  # serve_model <model_key> — one server per GPU, dispatch on M_SERVE
  model_cfg "$1"
  local gpu
  mkdir -p "$OUT_ROOT/servers"
  for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    "serve_one_${M_SERVE}" "$gpu"
  done
}

serve_one_vllm() {  # standard vLLM serving (qwen_omni_7b / qwen_omni_3b / qwen2_audio)
  local gpu="$1" port=$((BASE_PORT + gpu))
  local log="$OUT_ROOT/servers/${M_NAME}_gpu${gpu}.log"
  local vllm_bin; vllm_bin="$(dirname "$EPF_PY")/vllm"
  CUDA_VISIBLE_DEVICES=$gpu HF_HOME="$M_HF_HOME" VLLM_USE_FLASHINFER_SAMPLER=0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup setsid "$vllm_bin" serve "$M_ID" --revision "$M_REV" \
    --served-model-name "$M_NAME" --port "$port" --trust-remote-code --dtype bfloat16 \
    --max-model-len "$M_MAXLEN" --enforce-eager --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --allowed-local-media-path "$MEDIA_ROOT" \
    --limit-mm-per-prompt '{"audio":3}' > "$log" 2>&1 &
  echo $! > "$OUT_ROOT/servers/${M_NAME}_gpu${gpu}.pid"
  echo "serving $M_NAME on :$port (GPU $gpu) — log: $log"
}

serve_one_vllm_lora() {  # phi4mm: base + speech-LoRA adapter; requests target "$M_REQNAME"
  local gpu="$1" port=$((BASE_PORT + gpu))
  local log="$OUT_ROOT/servers/${M_NAME}_gpu${gpu}.log"
  local vllm_bin; vllm_bin="$(dirname "$EPF_PY")/vllm"
  CUDA_VISIBLE_DEVICES=$gpu HF_HOME="$M_HF_HOME" VLLM_USE_FLASHINFER_SAMPLER=0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup setsid "$vllm_bin" serve "$M_ID" --revision "$M_REV" \
    --served-model-name "$M_NAME" --port "$port" --trust-remote-code --dtype bfloat16 \
    --max-model-len "$M_MAXLEN" --enforce-eager --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --allowed-local-media-path "$MEDIA_ROOT" \
    --limit-mm-per-prompt '{"audio":3}' \
    --enable-lora --max-lora-rank "$SPEECH_LORA_RANK" --max-loras 1 \
    --lora-modules "${M_REQNAME}=${SPEECH_LORA_DIR}" > "$log" 2>&1 &
  echo $! > "$OUT_ROOT/servers/${M_NAME}_gpu${gpu}.pid"
  echo "serving $M_NAME (+$M_REQNAME LoRA) on :$port (GPU $gpu) — log: $log"
}

serve_one_kimi() {  # kimi_audio: NEVER plain `vllm serve` — the run19 wrapper patches the
  # broken kimi tokenizer and --chat-template is mandatory. The wrapper writes its
  # OWN pid to --pid-file (setsid may re-fork, making the launcher's $! stale).
  local gpu="$1" port=$((BASE_PORT + gpu))
  local log="$OUT_ROOT/servers/${M_NAME}_gpu${gpu}.log"
  local pidf="$OUT_ROOT/servers/${M_NAME}_gpu${gpu}.pid"
  rm -f "$pidf"
  CUDA_VISIBLE_DEVICES=$gpu HF_HOME="$M_HF_HOME" VLLM_USE_FLASHINFER_SAMPLER=0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup setsid "$EPF_PY" "$KIMI_WRAPPER" "$M_ID" --revision "$M_REV" \
    --pid-file "$pidf" \
    --served-model-name "$M_NAME" --port "$port" --trust-remote-code --dtype bfloat16 \
    --max-model-len "$M_MAXLEN" --enforce-eager --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --allowed-local-media-path "$MEDIA_ROOT" \
    --limit-mm-per-prompt '{"audio":1}' \
    --chat-template "$KIMI_CHAT_TEMPLATE" > "$log" 2>&1 &
  echo "serving $M_NAME on :$port (GPU $gpu) — log: $log"
}

serve_one_mellow() {  # mellow: FastAPI+transformers shim (not vLLM-servable); shim records its own pid
  local gpu="$1" port=$((BASE_PORT + gpu))
  local log="$OUT_ROOT/servers/${M_NAME}_gpu${gpu}.log"
  local pidf="$OUT_ROOT/servers/${M_NAME}_gpu${gpu}.pid"
  rm -f "$pidf"
  CUDA_VISIBLE_DEVICES=$gpu HF_HOME="$M_HF_HOME" \
  nohup setsid "$EPF_PY" -m benchmarking.mmau_pro.serve_mellow \
    --port "$port" --model-name "$M_NAME" --variant "$MELLOW_VARIANT" \
    --revision "$M_REV" --smollm2-revision "$SMOLLM2_REV" \
    --code-dir "$MELLOW_CODE_DIR" \
    --max-text-tokens "$MELLOW_MAX_TEXT_TOKENS" --single-audio-fill "$SINGLE_AUDIO_FILL" \
    --allowed-media-root "$MEDIA_ROOT" --waveform-cache "$WAVEFORM_CACHE" \
    --pid-file "$pidf" > "$log" 2>&1 &
  echo "serving $M_NAME on :$port (GPU $gpu) — log: $log"
}

wait_healthy() {  # wait_healthy <port> <req_name> [timeout_s] — healthy AND the request name exposed.
  # For phi4mm req_name is the LoRA adapter: a base-only server FAILS here instead
  # of silently answering without the adapter (run18 lesson).
  local port="$1" req="$2" timeout="${3:-$SERVE_TIMEOUT}" waited=0 body
  while [ "$waited" -lt "$timeout" ]; do
    body="$(curl -s --max-time 3 "http://localhost:$port/v1/models" || true)"
    if echo "$body" | grep -q '"id"'; then
      if echo "$body" | grep -q "\"$req\""; then
        echo "endpoint :$port healthy after ${waited}s ('$req' exposed)"; return 0
      fi
      echo "FATAL: :$port is up but '$req' is missing from /v1/models — got: $(echo "$body" | head -c 300)" >&2
      return 1
    fi
    sleep 5; waited=$((waited + 5))
  done
  echo "FATAL: endpoint :$port not healthy after ${timeout}s — check $OUT_ROOT/servers/*.log" >&2
  return 1
}

sanity_prompt() {  # sanity_prompt <port> <req_name> — text round-trip; catches empty-prompt renders
  local port="$1" req="$2" body ptok
  body="$(curl -s --max-time 120 "http://localhost:$port/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$req\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word OK.\"}],\"max_tokens\":8}")"
  ptok="$(echo "$body" | "$EPF_PY" -c 'import json,sys; print(json.load(sys.stdin).get("usage",{}).get("prompt_tokens",0))' 2>/dev/null || echo 0)"
  if [ "${ptok:-0}" -gt 5 ]; then
    echo "prompt sanity :$port OK (prompt_tokens=$ptok)"; return 0
  fi
  echo "FATAL: prompt sanity FAILED on :$port (prompt_tokens=${ptok:-?})." >&2
  echo "  Response was: $(echo "$body" | head -c 400)" >&2
  return 1
}

sanity_audio() {  # sanity_audio <port> <req_name> <wav_path> — one real audio round-trip.
  # Catches --allowed-local-media-path / audio-decode misconfiguration loudly,
  # BEFORE thousands of items fail with the same error.
  local port="$1" req="$2" wav="$3" body
  body="$(curl -s --max-time 300 "http://localhost:$port/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$req\",\"messages\":[{\"role\":\"user\",\"content\":[
          {\"type\":\"audio_url\",\"audio_url\":{\"url\":\"file://$wav\"}},
          {\"type\":\"text\",\"text\":\"Briefly, what do you hear?\"}]}],\"max_tokens\":32}")"
  if echo "$body" | grep -q '"choices"'; then
    echo "audio sanity :$port OK"; return 0
  fi
  echo "FATAL: audio sanity FAILED on :$port — the server cannot read/decode $wav" >&2
  echo "  Response was: $(echo "$body" | head -c 400)" >&2
  return 1
}

kill_servers() {  # kill_servers — by recorded PID, never pkill -f (SETUP_GUIDE §10).
  # TERM first (uvicorn/vLLM shut down gracefully but not instantly), then wait
  # for actual death and KILL any survivor — a half-dead server would fail the
  # next cell's ports_must_be_free check.
  local f pid pids="" waited
  for f in "$OUT_ROOT"/servers/*_gpu*.pid; do
    [ -e "$f" ] || continue
    pid="$(cat "$f")"
    kill "$pid" 2>/dev/null || true
    pids+=" $pid"
    rm -f "$f"
  done
  waited=0
  for pid in $pids; do
    while kill -0 "$pid" 2>/dev/null && [ "$waited" -lt 30 ]; do sleep 2; waited=$((waited + 2)); done
    kill -9 "$pid" 2>/dev/null || true
  done
  sleep 10  # let GPU memory drain before the next model loads
}

first_item_wav() {  # first_item_wav <bench_key> — absolute path of the cell's first audio clip
  bench_cfg "$1"
  (cd "$REPO_ROOT" && "$EPF_PY" - "$1" "$B_DATA_ROOT" "$B_SUBSET" "${B_AUDIO_ROOT:-}" <<'PYEOF'
import sys
from greedy.greedy_runner import LOADERS
bench, root, subset, audio_root = sys.argv[1:5]
recs = LOADERS[bench](root, subset=subset, limit=1, audio_root=audio_root or None)
print(recs[0].audio_paths[0])
PYEOF
  )
}

run_greedy() {  # run_greedy <bench> <csv> <log> [extra args...] — foreground, uses model_cfg/bench_cfg vars
  local bench="$1" csv="$2" logf="$3"; shift 3
  local extra=()
  [ -n "$B_AUDIO_ROOT" ] && extra+=(--audio-root "$B_AUDIO_ROOT")
  (cd "$REPO_ROOT" && "$EPF_PY" -m greedy.greedy_runner \
    --bench "$bench" --endpoints "$(endpoints_csv)" --model-name "$M_REQNAME" \
    --data-root "$B_DATA_ROOT" --subset "$B_SUBSET" \
    --max-audios "$M_MAXAUDIOS" --max-tokens "$MAX_TOKENS" --max-inflight "$MAX_INFLIGHT" \
    --csv "$csv" --log "$logf" "${extra[@]}" "$@")
}

run_greedy_bg() {  # like run_greedy but backgrounded; sets RUNNER_PID to python's REAL pid.
  # The subshell execs python, and the & is on the subshell itself — NOT on a
  # function call, which would add an outer fork and make $! a wrapper pid the
  # teardown trap kills while python survives to spray errors at dead endpoints.
  local bench="$1" csv="$2" logf="$3"; shift 3
  local extra=()
  [ -n "$B_AUDIO_ROOT" ] && extra+=(--audio-root "$B_AUDIO_ROOT")
  (cd "$REPO_ROOT" && exec "$EPF_PY" -m greedy.greedy_runner \
    --bench "$bench" --endpoints "$(endpoints_csv)" --model-name "$M_REQNAME" \
    --data-root "$B_DATA_ROOT" --subset "$B_SUBSET" \
    --max-audios "$M_MAXAUDIOS" --max-tokens "$MAX_TOKENS" --max-inflight "$MAX_INFLIGHT" \
    --csv "$csv" --log "$logf" "${extra[@]}" "$@") &
  RUNNER_PID=$!
}

run_cell() {  # run_cell <model_key> <bench_key> — THE servant body: serve -> gate -> greedy -> teardown
  local model="$1" bench="$2"
  model_cfg "$model"
  bench_cfg "$bench"
  local cell_dir="$OUT_ROOT/$model/$bench"
  mkdir -p "$cell_dir" "$OUT_ROOT/servers"

  if [ "$DRY_RUN" = "1" ]; then
    echo "DRY: cell $model x $bench -> model=$M_ID@${M_REV:0:12} serve=$M_SERVE request=$M_REQNAME" \
         "subset=$B_SUBSET expected=$B_EXPECTED max_audios=$M_MAXAUDIOS out=$cell_dir/greedy.csv"
    return 0
  fi

  ports_must_be_free
  # the trap kills the RUNNER first (else it would orphan and spray
  # connection-error rows at dead endpoints), then the servers. KILL, not TERM:
  # the runner's asyncio loop was observed surviving TERM long enough to spray;
  # the CSV is flushed per row, so nothing is lost
  RUNNER_PID=0
  trap '[ "${RUNNER_PID:-0}" -gt 0 ] && kill -9 "$RUNNER_PID" 2>/dev/null; kill_servers' EXIT

  serve_model "$model"
  local i port wav
  for i in $(seq 0 $((NUM_GPUS - 1))); do wait_healthy $((BASE_PORT + i)) "$M_REQNAME"; done
  wav="$(first_item_wav "$bench")"
  for i in $(seq 0 $((NUM_GPUS - 1))); do
    port=$((BASE_PORT + i))
    sanity_prompt "$port" "$M_REQNAME"
    sanity_audio "$port" "$M_REQNAME" "$wav"
  done

  # bench_cfg again: first_item_wav ran it in this shell for the wav path — keep vars coherent
  bench_cfg "$bench"
  run_greedy_bg "$bench" "$cell_dir/greedy.csv" "$cell_dir/greedy.log" --expected "$B_EXPECTED"
  if ! wait "$RUNNER_PID"; then
    RUNNER_PID=0
    echo "cell $model x $bench: errors — one automatic resume retry"
    run_greedy_bg "$bench" "$cell_dir/greedy.csv" "$cell_dir/greedy.log" --expected "$B_EXPECTED"
    wait "$RUNNER_PID" \
      || { RUNNER_PID=0; echo "FATAL: cell $model x $bench still failing after retry — see $cell_dir/greedy.log" >&2; return 1; }
  fi
  RUNNER_PID=0

  kill_servers
  trap - EXIT
  echo "cell $model x $bench COMPLETE -> $cell_dir/greedy.csv"
}
