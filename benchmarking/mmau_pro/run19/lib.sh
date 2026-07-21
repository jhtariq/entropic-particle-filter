# Shared helpers for run19 scripts. Source config.sh BEFORE this file.
# Adapted from run18/lib.sh (standard vLLM serving, no LoRA) with the Kimi-Audio
# specifics: serve_one launches serve_kimi.py (the tokenizer-patch wrapper) with
# an explicit --chat-template, requests cap at ONE audio per prompt, and
# check_snapshot verifies the kimi tokenizer + Whisper-encoder assets while the
# TTS-only subfolders (audio_detokenizer/, vocoder/) are excluded by design.

run_cfg() {  # run_cfg run19 -> sets MODEL_ID REV NAME MAXLEN SUBSET STEM PROMPTS RUN_DIR
  local n="${1#run}" p
  p="RUN${n}"
  MODEL_ID="$(eval echo "\$${p}_MODEL_ID")"
  REV="$(eval echo "\$${p}_REV")"
  NAME="$(eval echo "\$${p}_NAME")"
  MAXLEN="$(eval echo "\$${p}_MAXLEN")"
  SUBSET="$(eval echo "\$${p}_SUBSET")"
  STEM="$(eval echo "\$${p}_STEM")"
  PROMPTS="$(eval echo "\$PROMPTS_RUN${n}")"
  RUN_DIR="$OUT_ROOT/$1"
  [ -n "$MODEL_ID" ] || { echo "FATAL: unknown run '$1'" >&2; return 1; }
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

ensure_model() {  # ensure_model <org/model> <rev> — download only if the pinned snapshot is incomplete
  local id="$1" rev="$2"
  if check_snapshot "$id" "$rev" > /dev/null 2>&1; then
    echo "model $id@${rev:0:12} already present — skipping download"
    return 0
  fi
  # ONE --exclude flag with both patterns: a second --exclude OVERRIDES the first
  # (nargs semantics), which silently re-admits the 18 GB TTS detokenizer
  hf_cli download "$id" --revision "$rev" \
    --exclude "audio_detokenizer/*" "vocoder/*" > /dev/null
  check_snapshot "$id" "$rev"
}

endpoints_csv() {
  local eps="" i
  for i in $(seq 0 $((NUM_GPUS - 1))); do eps+="${eps:+,}http://localhost:$((BASE_PORT + i))/v1"; done
  echo "$eps"
}

serve_one() {  # serve_one <gpu_idx> — one vLLM replica per GPU, via the serve_kimi.py wrapper
  local gpu="$1"
  local port=$((BASE_PORT + gpu))
  local log="$OUT_ROOT/servers/${RUN19_NAME}_gpu${gpu}.log"
  local pidf="$OUT_ROOT/servers/${RUN19_NAME}_gpu${gpu}.pid"
  mkdir -p "$OUT_ROOT/servers"
  rm -f "$pidf"
  # NEVER plain `vllm serve` for this model: the wrapper patches the kimi
  # tokenizer's broken apply_chat_template and --chat-template is mandatory
  # (no template is auto-wired for this arch). See serve_kimi.py docstring.
  # The wrapper writes its OWN pid to --pid-file: `setsid` may fork (it does
  # under interactive job control), which makes the launcher's $! stale.
  CUDA_VISIBLE_DEVICES=$gpu VLLM_USE_FLASHINFER_SAMPLER=0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup setsid "$EPF_PY" "$RUN19_DIR/serve_kimi.py" "$RUN19_MODEL_ID" --revision "$RUN19_REV" \
    --pid-file "$pidf" \
    --served-model-name "$RUN19_NAME" --port "$port" --trust-remote-code --dtype bfloat16 \
    --max-model-len "$RUN19_MAXLEN" --enforce-eager --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --allowed-local-media-path "$EPF_DATA_ROOT" \
    --limit-mm-per-prompt '{"audio":1}' \
    --chat-template "$KIMI_CHAT_TEMPLATE" > "$log" 2>&1 &
  echo "serving $RUN19_NAME on :$port (GPU $gpu) — log: $log"
}

wait_healthy() {  # wait_healthy <port> [timeout_s]
  local port="$1" timeout="${2:-1800}" waited=0 body
  while [ "$waited" -lt "$timeout" ]; do
    body="$(curl -s --max-time 3 "http://localhost:$port/v1/models" || true)"
    if echo "$body" | grep -q '"id"'; then
      echo "endpoint :$port healthy after ${waited}s"; return 0
    fi
    sleep 5; waited=$((waited + 5))
  done
  echo "FATAL: endpoint :$port not healthy after ${timeout}s — check $OUT_ROOT/servers/*.log" >&2
  return 1
}

sanity_prompt() {  # sanity_prompt <port> — catch the empty-prompt failure mode before the gates.
  # If the wrapper patch were bypassed, prompts render EMPTY (prompt_tokens ~1-2)
  # or requests 400 on a missing chat template; this healthy render is 10 tokens
  # (3 kimi markers + 7 text tokens), so the bar sits at >5.
  local port="$1" body ptok
  body="$(curl -s --max-time 120 "http://localhost:$port/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$RUN19_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word OK.\"}],\"max_tokens\":8}")"
  ptok="$(echo "$body" | "$EPF_PY" -c 'import json,sys; print(json.load(sys.stdin).get("usage",{}).get("prompt_tokens",0))' 2>/dev/null || echo 0)"
  if [ "${ptok:-0}" -gt 5 ]; then
    echo "prompt sanity :$port OK (prompt_tokens=$ptok)"; return 0
  fi
  echo "FATAL: prompt sanity FAILED on :$port (prompt_tokens=${ptok:-?}) — the chat" >&2
  echo "  template/wrapper is not active. Response was: $(echo "$body" | head -c 400)" >&2
  return 1
}

kill_servers() {  # kill_servers — by recorded PID, never pkill -f (SETUP_GUIDE §10)
  local f pid
  for f in "$OUT_ROOT"/servers/"${RUN19_NAME}"_gpu*.pid; do
    [ -e "$f" ] || continue
    pid="$(cat "$f")"
    kill "$pid" 2>/dev/null || true
    rm -f "$f"
  done
  sleep 10  # let GPU memory drain before the next model loads
}

gate_endpoint() {  # gate_endpoint <port> — both phase0 gates must PASS
  # --single-audio: vLLM's kimi path rejects multi-audio prompts, and the first
  # le30s item happens to have two clips — the flag picks the first 1-audio item
  local port="$1" out
  out="$("$EPF_PY" -m benchmarking.mmau_pro.phase0_gate \
        --endpoint "http://localhost:$port/v1" --model-name "$RUN19_NAME" \
        --data-root "$DATA_TESTMINI" --single-audio 2>&1)" || true
  echo "$out" | tail -3
  if echo "$out" | grep -q '"gate1_logprobs": true, "gate2_continue": true'; then return 0; fi
  echo "FATAL: phase0 gate FAILED on :$port — PF/EPF weights would silently degrade" >&2
  return 1
}

check_snapshot() {  # check_snapshot <org/model> <rev> — pinned snapshot, complete shards + kimi assets
  local id="$1" rev="$2"
  local dir="$HF_HOME/hub/models--${id//\//--}/snapshots/$rev"
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
  # kimi extras: the tiktoken vocab + tokenizer config feed KimiAudioTokenizer,
  # and vLLM loads the Whisper encoder from the whisper-large-v3/ subfolder.
  # audio_detokenizer/ + vocoder/ are TTS-only, unused by the text-out path, and
  # intentionally NOT downloaded (fetch_models.sh excludes them: -20 GB).
  local f
  for f in tiktoken.model tokenizer_config.json generation_config.json \
           configuration_moonshot_kimia.py \
           whisper-large-v3/model.safetensors whisper-large-v3/config.json \
           whisper-large-v3/preprocessor_config.json; do
    [ -e "$dir/$f" ] || { echo "FAIL: $dir lacks $f" >&2; return 1; }
  done
  echo "kimi assets OK: tokenizer + whisper-large-v3 encoder"
}

unpack_seed() {  # unpack_seed <gz> <target_jsonl> — never clobbers an existing (possibly extended) file
  local gz="$1" target="$2"
  if [ -e "$target" ]; then
    echo "seed: $target exists ($(wc -l < "$target") rows) — resuming from it"
    return 0
  fi
  [ -e "$gz" ] || { echo "FATAL: seed $gz missing" >&2; return 1; }
  mkdir -p "$(dirname "$target")"
  gunzip -c "$gz" > "$target"
  echo "seed: unpacked $(wc -l < "$target") rows -> $target"
}
