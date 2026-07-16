# Shared helpers for run18 scripts. Source config.sh BEFORE this file.
# Adapted from run16/lib.sh (standard vLLM serving) with the Phi-4-MM speech-LoRA
# specifics: serve_one attaches the adapter per SERVE_MODE, wait_healthy asserts the
# adapter is actually exposed, and probe_model_name() is the single source of truth
# for the model= value — a request naming the base against a lora-mode server is
# VALID but silently skips the adapter, so no harness call may hard-code the name.

run_cfg() {  # run_cfg run18 -> sets MODEL_ID REV NAME MAXLEN SUBSET STEM PROMPTS RUN_DIR
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
  hf_cli download "$id" --revision "$rev" > /dev/null
  check_snapshot "$id" "$rev"
}

endpoints_csv() {
  local eps="" i
  for i in $(seq 0 $((NUM_GPUS - 1))); do eps+="${eps:+,}http://localhost:$((BASE_PORT + i))/v1"; done
  echo "$eps"
}

probe_model_name() {  # THE model= value for every gate/screen/probe request
  if [ "$SERVE_MODE" = "lora" ]; then echo "$SPEECH_LORA_NAME"; else echo "$RUN18_NAME"; fi
}

serve_one() {  # serve_one <gpu_idx> — one vLLM replica per GPU, mode per SERVE_MODE
  local gpu="$1"
  local port=$((BASE_PORT + gpu))
  local log="$OUT_ROOT/servers/${RUN18_NAME}_gpu${gpu}.log"
  local vllm_bin; vllm_bin="$(dirname "$EPF_PY")/vllm"
  local model_args=("$RUN18_MODEL_ID" --revision "$RUN18_REV") lora_args=()
  case "$SERVE_MODE" in
    lora)   lora_args=(--enable-lora --max-lora-rank "$SPEECH_LORA_RANK" --max-loras 1
                       --lora-modules "${SPEECH_LORA_NAME}=${SPEECH_LORA_DIR}") ;;
    merged) model_args=("$MERGED_DIR") ;;  # local merged-weights dir (no --revision)
    base)   ;;                             # base weights only — quality caveat, see README
    *)      echo "FATAL: SERVE_MODE='$SERVE_MODE' (want lora|merged|base)" >&2; return 1 ;;
  esac
  mkdir -p "$OUT_ROOT/servers"
  CUDA_VISIBLE_DEVICES=$gpu VLLM_USE_FLASHINFER_SAMPLER=0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup setsid "$vllm_bin" serve "${model_args[@]}" \
    --served-model-name "$RUN18_NAME" --port "$port" --trust-remote-code --dtype bfloat16 \
    --max-model-len "$RUN18_MAXLEN" --enforce-eager --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --allowed-local-media-path "$EPF_DATA_ROOT" \
    --limit-mm-per-prompt '{"audio":3}' "${lora_args[@]}" > "$log" 2>&1 &
  echo $! > "$OUT_ROOT/servers/${RUN18_NAME}_gpu${gpu}.pid"
  echo "serving $RUN18_NAME on :$port (GPU $gpu, mode=$SERVE_MODE) — log: $log"
}

wait_healthy() {  # wait_healthy <port> [timeout_s] — healthy AND (lora mode) adapter exposed
  local port="$1" timeout="${2:-1800}" waited=0 body
  while [ "$waited" -lt "$timeout" ]; do
    body="$(curl -s --max-time 3 "http://localhost:$port/v1/models" || true)"
    if echo "$body" | grep -q '"id"'; then
      if [ "$SERVE_MODE" = "lora" ] && ! echo "$body" | grep -q "\"$SPEECH_LORA_NAME\""; then
        echo "FATAL: :$port is serving the BASE model only — '$SPEECH_LORA_NAME' missing from /v1/models" >&2
        return 1
      fi
      echo "endpoint :$port healthy after ${waited}s (mode=$SERVE_MODE)"; return 0
    fi
    sleep 5; waited=$((waited + 5))
  done
  echo "FATAL: endpoint :$port not healthy after ${timeout}s — check $OUT_ROOT/servers/*.log" >&2
  return 1
}

kill_servers() {  # kill_servers — by recorded PID, never pkill -f (SETUP_GUIDE §10)
  local f pid
  for f in "$OUT_ROOT"/servers/"${RUN18_NAME}"_gpu*.pid; do
    [ -e "$f" ] || continue
    pid="$(cat "$f")"
    kill "$pid" 2>/dev/null || true
    rm -f "$f"
  done
  sleep 10  # let GPU memory drain before the next model loads
}

gate_endpoint() {  # gate_endpoint <port> — both phase0 gates must PASS (through the LoRA path)
  local port="$1" out
  out="$("$EPF_PY" -m benchmarking.mmau_pro.phase0_gate \
        --endpoint "http://localhost:$port/v1" --model-name "$(probe_model_name)" \
        --data-root "$DATA_TESTMINI" 2>&1)" || true
  echo "$out" | tail -3
  if echo "$out" | grep -q '"gate1_logprobs": true, "gate2_continue": true'; then return 0; fi
  echo "FATAL: phase0 gate FAILED on :$port — PF/EPF weights would silently degrade" >&2
  return 1
}

check_snapshot() {  # check_snapshot <org/model> <rev> — pinned snapshot, complete shards (+ speech-lora)
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
  # the speech adapter is the run's audio path (lora mode) or the merge input (merged
  # mode); vision-lora is intentionally NOT required — absent on the reference box,
  # unused for an audio-only benchmark
  if [ "$SERVE_MODE" != "base" ]; then
    local f
    for f in adapter_config.json adapter_model.safetensors; do
      [ -e "$dir/speech-lora/$f" ] || { echo "FAIL: $dir/speech-lora lacks $f" >&2; return 1; }
    done
    echo "speech-lora OK: $dir/speech-lora"
  fi
}

check_merged() {  # merged-mode identity: the offline-merge output is present
  [ "$SERVE_MODE" = "merged" ] || return 0
  [ -e "$MERGED_DIR/config.json" ] \
    || { echo "FAIL: $MERGED_DIR incomplete — run fetch_models.sh (merge step)" >&2; return 1; }
  echo "merged dir OK: $MERGED_DIR"
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
