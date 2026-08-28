#!/bin/bash
# Serving / scheduling helpers for the MMAR greedy + self-consistency run.
# Source config.sh BEFORE this file.

# ---------------------------------------------------------------------------------
# kv_plan <kv_kib_per_token> <weights_gib> <max_tokens> <n_wanted>
#
# Derives --max-num-seqs (server) and --n-chunk / --max-inflight (client) from the GPU
# actually present. This is not a nicety: at max_tokens=700 one n=128 Qwen2-Audio request
# needs 43.8 GiB of KV, which fits an H200 (141 GB) but NOT an A40 (48 GB). Hardcoding
# either number breaks on the other card.
#
# Exports: KVP_MAXSEQS  KVP_NCHUNK  KVP_INFLIGHT  KVP_NOTE
kv_plan() {
  local kv_kib="$1" weights_gib="$2" max_tokens="$3" n_wanted="$4"
  # Size against the EXPECTED generation length, not --max-tokens. Measured on MMAR:
  # P4 CoT responses average ~38 words / ~52 tokens against a 700 cap, so budgeting at 700
  # left the A40 at 6.6% KV utilisation and 445 tok/s. Budgeting at 256 raised it to
  # 1506 tok/s (3x) with zero preemption. Over-subscription is SAFE — vLLM admits a
  # request only when KV blocks are free and preempts rather than OOMs — so the cost of
  # guessing high is a little recompute, while guessing low idles the GPU.
  local eff="${KV_PLAN_EFFECTIVE_TOKENS:-128}"
  [ "$eff" -gt "$max_tokens" ] && eff="$max_tokens"
  max_tokens="$eff"
  # nvidia-smi prints its failure text to STDOUT on a GPU-less node, so a plain :=default
  # would leave $vram_mib holding that message and silently collapse the pool to the floor.
  local vram_mib
  vram_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
  case "$vram_mib" in
    ''|*[!0-9]*) vram_mib=46068 ;;             # A40 fallback (dry runs, login nodes)
  esac
  # vLLM's budget, minus weights, minus a fixed reserve for the audio encoder's
  # activation spike (which lives OUTSIDE the gpu-memory-utilization budget).
  local budget_mib kv_mib
  budget_mib=$(awk -v m="$vram_mib" -v u="$GPU_MEM_UTIL" 'BEGIN{printf "%d", m*u}')
  kv_mib=$(( budget_mib - weights_gib * 1024 - 3072 ))
  [ "$kv_mib" -lt 2048 ] && kv_mib=2048
  # how many full-length sequences that KV pool holds
  local seqs
  seqs=$(awk -v kv="$kv_mib" -v k="$kv_kib" -v t="$max_tokens" \
             'BEGIN{printf "%d", (kv*1024)/(k*t)}')
  [ "$seqs" -lt 4 ] && seqs=4
  # never schedule more sequences than we can ever have in flight
  local cap=$(( n_wanted * 4 ))
  [ "$seqs" -gt "$cap" ] && seqs=$cap

  # chunk N so a single request never exceeds the server's concurrency; prefer no chunking
  local nchunk="$n_wanted"
  if [ "$n_wanted" -gt "$seqs" ]; then
    # round down to a power of two for tidy sample_idx blocks
    local p=1
    while [ $(( p * 2 )) -le "$seqs" ]; do p=$(( p * 2 )); done
    nchunk="$p"
  fi
  # Keep the server's queue full: ask for ~2x max_num_seqs worth of work in flight so a
  # finishing request is immediately replaced. Excess simply queues (vLLM never OOMs on
  # admission), and this is what took the measured A40 cell from 445 to 1506 tok/s.
  local inflight=$(( 2 * seqs / nchunk ))
  [ "$inflight" -lt 2 ] && inflight=2
  [ "$inflight" -gt 16 ] && inflight=16

  KVP_MAXSEQS="$seqs"
  KVP_NCHUNK="$nchunk"
  KVP_INFLIGHT="$inflight"
  KVP_NOTE="vram=${vram_mib}MiB kv_pool=${kv_mib}MiB @ ${kv_kib}KiB/tok x ${max_tokens}tok"
}

endpoints_csv() {
  local i out=""
  for i in $(seq 0 $((NUM_GPUS - 1))); do
    out="${out}${out:+,}http://localhost:$((BASE_PORT + i))/v1"
  done
  echo "$out"
}

# serve_one <gpu> <max_num_seqs> — dispatch on M_SERVE; records the PID for kill_servers
serve_one() {
  # split declarations: in a single `local a="$1" b=$((a))` bash expands the arithmetic
  # BEFORE assigning a — it only worked because callers leaked a global named `gpu`
  local gpu="$1"; local maxseqs="$2"; local port=$((BASE_PORT + gpu))
  local log="$OUT_ROOT/servers/${M_NAME}_gpu${gpu}.log"
  local vllm_bin="${M_VLLM_BIN:-$(dirname "$EPF_PY")/vllm}"
  mkdir -p "$OUT_ROOT/servers"
  # Kimi-Audio takes a different launcher entirely: the run19 wrapper monkey-patches six
  # vLLM 0.22.1 bugs before starting the stock CLI, --chat-template is MANDATORY (the
  # checkpoint ships none, so every chat request would 400), and it writes its OWN pid
  # because setsid may re-fork and leave the launcher's $! stale.
  if [ "$M_SERVE" = "kimi" ]; then
    local pidf="$OUT_ROOT/servers/${M_NAME}_gpu${gpu}.pid"
    rm -f "$pidf"
    CUDA_VISIBLE_DEVICES=$gpu HF_HOME="$HF_HOME" VLLM_USE_FLASHINFER_SAMPLER=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 \
    VLLM_CACHE_ROOT="${TMPDIR:-/tmp}/vllm_$gpu" TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton_$gpu" \
    TORCHINDUCTOR_CACHE_DIR="${TMPDIR:-/tmp}/inductor_$gpu" \
    nohup setsid "$EPF_PY" "$KIMI_WRAPPER" "$M_ID" --revision "$M_REV" \
      --pid-file "$pidf" \
      --served-model-name "$M_NAME" --port "$port" --trust-remote-code --dtype bfloat16 \
      --max-model-len "$M_MAXLEN" --enforce-eager --gpu-memory-utilization "$GPU_MEM_UTIL" \
      --allowed-local-media-path "$MEDIA_ROOT" \
      --limit-mm-per-prompt '{"audio":1}' \
      --max-num-seqs "$maxseqs" --max-num-batched-tokens 8192 \
      --chat-template "$KIMI_CHAT_TEMPLATE" > "$log" 2>&1 &
    echo "  serving $M_NAME on :$port (GPU $gpu, run19 wrapper, max_num_seqs=$maxseqs) -> $log"
    return 0
  fi
  # Mellow: not servable by vLLM — the run17 FastAPI shim replaces `vllm serve`.
  # It writes its own PID (setsid-safe) and ignores maxseqs (one batched request at a
  # time per process; n=128 SC requests decode as one batch).
  if [ "$M_SERVE" = "mellow" ]; then
    local pidf="$OUT_ROOT/servers/${M_NAME}_gpu${gpu}.pid"
    rm -f "$pidf"
    CUDA_VISIBLE_DEVICES=$gpu HF_HOME="${M_HF_HOME:-$HF_HOME}" HF_HUB_OFFLINE=1 \
    nohup setsid "$EPF_PY" -m benchmarking.mmau_pro.serve_mellow \
      --port "$port" --model-name "$M_NAME" --variant "$MELLOW_VARIANT" \
      --revision "$M_REV" --smollm2-revision "$SMOLLM2_REV" \
      --code-dir "$MELLOW_CODE_DIR" \
      --max-text-tokens "$MELLOW_MAX_TEXT_TOKENS" --single-audio-fill "$SINGLE_AUDIO_FILL" \
      --allowed-media-root "$MEDIA_ROOT" --waveform-cache "$WAVEFORM_CACHE" \
      --pid-file "$pidf" > "$log" 2>&1 &
    echo "  serving $M_NAME (run17 shim) on :$port (GPU $gpu) -> $log"
    return 0
  fi
  local extra=()
  if [ "$M_SERVE" = "vllm_lora" ]; then
    # Phi-4-MM: the base weights alone cannot do audio — every request must target the
    # speech-LoRA adapter by name ($M_REQNAME), so it is served as a named LoRA module.
    extra=(--enable-lora --max-lora-rank "$SPEECH_LORA_RANK" --max-loras 1
           --lora-modules "${M_REQNAME}=${SPEECH_LORA_DIR}")
  fi
  CUDA_VISIBLE_DEVICES=$gpu HF_HOME="${M_HF_HOME:-$HF_HOME}" VLLM_USE_FLASHINFER_SAMPLER=0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 \
  VLLM_CACHE_ROOT="${TMPDIR:-/tmp}/vllm_$gpu" TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton_$gpu" \
  TORCHINDUCTOR_CACHE_DIR="${TMPDIR:-/tmp}/inductor_$gpu" \
  nohup setsid "$vllm_bin" serve "$M_ID" --revision "$M_REV" \
    --served-model-name "$M_NAME" --port "$port" --trust-remote-code --dtype bfloat16 \
    --max-model-len "$M_MAXLEN" --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --allowed-local-media-path "$MEDIA_ROOT" \
    --limit-mm-per-prompt '{"audio":3}' \
    --max-num-seqs "$maxseqs" --max-num-batched-tokens 8192 \
    "${extra[@]}" > "$log" 2>&1 &
  echo $! > "$OUT_ROOT/servers/${M_NAME}_gpu${gpu}.pid"
  echo "  serving $M_NAME on :$port (GPU $gpu, max_num_seqs=$maxseqs) -> $log"
}

wait_healthy() {  # wait_healthy <port> [timeout]
  local port="$1" timeout="${2:-$SERVE_TIMEOUT}" t=0
  while [ "$t" -lt "$timeout" ]; do
    if curl -sf "http://localhost:$port/v1/models" > /dev/null 2>&1; then
      echo "  :$port healthy after ${t}s"; return 0
    fi
    sleep 5; t=$((t + 5))
  done
  echo "  FATAL: :$port not healthy after ${timeout}s" >&2; return 1
}

# Kill by RECORDED PID, never `pkill -f "vllm serve"` — that pattern also matches the
# launching shell and would kill the driver.
kill_servers() {
  local f pid
  for f in "$OUT_ROOT"/servers/*.pid; do
    [ -e "$f" ] || continue
    pid=$(cat "$f" 2>/dev/null) || continue
    if [ -n "$pid" ]; then
      kill -- -"$pid" 2>/dev/null || kill "$pid" 2>/dev/null
    fi
    rm -f "$f"
  done
  sleep 10   # let the GPU memory actually drain before the next model loads
}
