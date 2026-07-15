# Shared helpers for run17 scripts. Source config.sh BEFORE this file.
# Adapted from run16/lib.sh: vLLM serving is replaced by the serve_mellow.py shim
# (Mellow is not vLLM-servable), checkpoint identity covers the raw .ckpt files +
# the pinned GitHub code clone, and there is no seed handling (fresh model).

run_cfg() {  # run_cfg run17 -> sets MODEL_ID REV NAME SUBSET STEM PROMPTS RUN_DIR
  local n="${1#run}" p
  p="RUN${n}"
  MODEL_ID="$(eval echo "\$${p}_MODEL_ID")"
  REV="$(eval echo "\$${p}_REV")"
  NAME="$(eval echo "\$${p}_NAME")"
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

ensure_mellow() {  # download ckpts + SmolLM2 base + code clone, all pinned; idempotent
  if ! check_mellow_snapshot > /dev/null 2>&1; then
    hf_cli download "$RUN17_MODEL_ID" v0.ckpt v0_s.ckpt v0.yaml config.json \
      --revision "$RUN17_REV" > /dev/null
  fi
  if ! check_smollm2_snapshot > /dev/null 2>&1; then
    hf_cli download "$SMOLLM2_ID" --revision "$SMOLLM2_REV" > /dev/null
  fi
  if [ ! -d "$MELLOW_CODE_DIR/.git" ]; then
    git clone "$MELLOW_CODE_REPO" "$MELLOW_CODE_DIR"
  fi
  git -C "$MELLOW_CODE_DIR" checkout -q "$MELLOW_CODE_COMMIT"
  check_mellow_snapshot && check_smollm2_snapshot && check_mellow_code
}

endpoints_csv() {
  local eps="" i
  for i in $(seq 0 $((NUM_GPUS - 1))); do eps+="${eps:+,}http://localhost:$((BASE_PORT + i))/v1"; done
  echo "$eps"
}

serve_one() {  # serve_one <gpu_idx> — one serve_mellow.py shim per GPU
  local gpu="$1"
  local port=$((BASE_PORT + gpu))
  local log="$OUT_ROOT/servers/${RUN17_NAME}_gpu${gpu}.log"
  local pidf="$OUT_ROOT/servers/${RUN17_NAME}_gpu${gpu}.pid"
  mkdir -p "$OUT_ROOT/servers"
  # --pid-file: the shim records its own PID (setsid re-forks, so $! would be stale)
  CUDA_VISIBLE_DEVICES=$gpu nohup setsid "$EPF_PY" -m benchmarking.mmau_pro.serve_mellow \
    --port "$port" --model-name "$RUN17_NAME" --variant "$MELLOW_VARIANT" \
    --revision "$RUN17_REV" --smollm2-revision "$SMOLLM2_REV" \
    --code-dir "$MELLOW_CODE_DIR" \
    --max-text-tokens "$MELLOW_MAX_TEXT_TOKENS" --single-audio-fill "$SINGLE_AUDIO_FILL" \
    --allowed-media-root "$EPF_DATA_ROOT" --waveform-cache "${WAVEFORM_CACHE:-64}" \
    --pid-file "$pidf" > "$log" 2>&1 &
  echo "serving $RUN17_NAME on :$port (GPU $gpu) — log: $log"
}

wait_healthy() {  # wait_healthy <port> [timeout_s]
  local port="$1" timeout="${2:-600}" waited=0
  while [ "$waited" -lt "$timeout" ]; do
    if curl -s --max-time 3 "http://localhost:$port/v1/models" | grep -q '"id"'; then
      echo "endpoint :$port healthy after ${waited}s"; return 0
    fi
    sleep 5; waited=$((waited + 5))
  done
  echo "FATAL: endpoint :$port not healthy after ${timeout}s — check $OUT_ROOT/servers/*.log" >&2
  return 1
}

kill_servers() {  # kill_servers — by recorded PID, never pkill -f (SETUP_GUIDE §10)
  local f pid
  for f in "$OUT_ROOT"/servers/"${RUN17_NAME}"_gpu*.pid; do
    [ -e "$f" ] || continue
    pid="$(cat "$f")"
    kill "$pid" 2>/dev/null || true
    rm -f "$f"
  done
  sleep 5  # let GPU memory drain
}

gate_endpoint() {  # gate_endpoint <port> — both phase0 gates must PASS
  local port="$1" out
  out="$("$EPF_PY" -m benchmarking.mmau_pro.phase0_gate \
        --endpoint "http://localhost:$port/v1" --model-name "$RUN17_NAME" \
        --data-root "$DATA_TESTMINI" 2>&1)" || true
  echo "$out" | tail -3
  if echo "$out" | grep -q '"gate1_logprobs": true, "gate2_continue": true'; then return 0; fi
  echo "FATAL: phase0 gate FAILED on :$port — PF/EPF weights would silently degrade" >&2
  return 1
}

check_mellow_snapshot() {  # pinned HF snapshot with the raw ckpt files (no safetensors index)
  local dir="$HF_HOME/hub/models--${RUN17_MODEL_ID//\//--}/snapshots/$RUN17_REV"
  [ -d "$dir" ] || { echo "FAIL: snapshot $dir missing — run fetch_models.sh" >&2; return 1; }
  local f
  for f in "${MELLOW_VARIANT}.ckpt" v0.yaml; do
    [ -e "$dir/$f" ] || { echo "FAIL: $dir lacks $f" >&2; return 1; }
  done
  echo "mellow snapshot OK: $dir"
}

check_smollm2_snapshot() {
  local dir="$HF_HOME/hub/models--${SMOLLM2_ID//\//--}/snapshots/$SMOLLM2_REV"
  [ -e "$dir/config.json" ] || { echo "FAIL: SmolLM2 snapshot $dir incomplete — run fetch_models.sh" >&2; return 1; }
  echo "smollm2 snapshot OK: $dir"
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

check_mellow_code() {  # code identity: pinned commit checked out
  local head
  head="$(git -C "$MELLOW_CODE_DIR" rev-parse HEAD 2>/dev/null)" \
    || { echo "FAIL: $MELLOW_CODE_DIR is not a git clone — run fetch_models.sh" >&2; return 1; }
  [ "$head" = "$MELLOW_CODE_COMMIT" ] \
    || { echo "FAIL: mellow code at $head, want $MELLOW_CODE_COMMIT" >&2; return 1; }
  echo "mellow code OK: $MELLOW_CODE_DIR @ ${head:0:12}"
}
