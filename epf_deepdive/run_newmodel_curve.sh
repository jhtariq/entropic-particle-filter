#!/bin/bash
# Probe-EPF budget-curve driver: any model x {mmar, mmau_pro_d1k}.
# Usage: bash run_newmodel_curve.sh <phi|kimi|3b|7b|q2a> <port> [gpu]
# Env:  BENCH=mmar|mmau_pro_d1k (default mmar)
#       FORCE_PM=<4|5>   skip the P4-vs-P5 pcheck (and the APC canary) and use this prompt
#       KIMI_APC=1       kimi only: serve with prefix caching (canary-gated unless FORCE_PM)
#       APC_TOL, KIMI_INFLIGHT   tuning knobs
# Stages: serve -> health -> smoke gate -> [pcheck] -> [APC canary] -> full curve
# budgets 128..1 with --store-texts. Re-runnable: probe_epf resumes by (uid,arm,budget).
set -uo pipefail
D=/work/hdd/bcey/awaheed/its-for-audio-reasoning/epf_deepdive
R=/work/hdd/bcey/awaheed/its-for-audio-reasoning/entropic-particle-filter
PY=/u/awaheed/envs/epf/bin/python
VLLM=/u/awaheed/envs/epf/bin/vllm
MODEL="${1:?phi|kimi|3b|7b|q2a}"; PORT="${2:?port}"; GPU="${3:-}"
[ -n "$GPU" ] && export CUDA_VISIBLE_DEVICES="$GPU"
BENCH="${BENCH:-mmar}"
case "$BENCH" in
  mmar)         PFX=""; CURVEPFX="mmar" ;;
  mmau_pro_d1k) PFX="d1k_"; CURVEPFX="d1k" ;;
  mmau)         PFX="mmau_"; CURVEPFX="mmau" ;;
  *) echo "FATAL bad BENCH $BENCH"; exit 1 ;;
esac

module load cuda-compat/13.0 2>/dev/null || true
export HF_HUB_OFFLINE=1
export TMPDIR="/tmp/$USER/serve_${MODEL}_$$"; mkdir -p "$TMPDIR"
export VLLM_USE_FLASHINFER_SAMPLER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_CACHE_ROOT="$TMPDIR/vllm" TRITON_CACHE_DIR="$TMPDIR/triton" \
       TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor"
# MUST be the resolved real path: vLLM realpaths the file before the subpath
# check, so the /u/awaheed/epf_data symlink form fails ("must be a subpath")
MEDIA=/work/hdd/bcey/awaheed/its-for-audio-reasoning/epf_data
TAG="${MODEL}${CURVEPFX}_$$"

case "$MODEL" in
  3b|7b|q2a)
    # Omni/Q2A: the validated deepdive server (prefix caching, pinned revisions)
    case "$MODEL" in
      3b)  MN=qwen-omni-3b ;;
      7b)  MN=qwen-omni ;;
      q2a) MN=qwen2-audio ;;
    esac
    JMODEL="$MODEL"; INFLIGHT=384
    export SERVE_TAG="$TAG"
    bash "$D/serve_model.sh" "$MODEL" "$PORT" ;;
  phi)
    # Phi-4-multimodal + speech LoRA. GOTCHA: every request MUST target the
    # adapter name "speech" — the base name is valid but silently skips the LoRA.
    export HF_HOME=/u/awaheed/epf_data/hf_cache
    M_ID=microsoft/Phi-4-multimodal-instruct
    M_REV=93f923e1a7727d1c4f446756212d9d3e8fcc5d81
    MN=speech; JMODEL=phi4mm; INFLIGHT=384
    LORA="$HF_HOME/hub/models--microsoft--Phi-4-multimodal-instruct/snapshots/$M_REV/speech-lora"
    [ -d "$LORA" ] || { echo "FATAL no speech-lora dir at $LORA"; exit 1; }
    nohup setsid "$VLLM" serve "$M_ID" --revision "$M_REV" \
      --served-model-name phi4mm --port "$PORT" --trust-remote-code --dtype bfloat16 \
      --max-model-len 32768 --enforce-eager --gpu-memory-utilization 0.85 \
      --allowed-local-media-path "$MEDIA" --limit-mm-per-prompt '{"audio":3}' \
      --enable-lora --max-lora-rank 320 --max-loras 1 --lora-modules "speech=$LORA" \
      --enable-prefix-caching > "$D/serve_${TAG}.log" 2>&1 &
    echo $! > "$D/serve_${TAG}.pid" ;;
  kimi)
    # Kimi-Audio: NEVER plain `vllm serve` — run19 wrapper (tokenizer/EOS/logprobs/
    # concurrent-audio patches) + mandatory chat template.
    # KIMI_APC=1 adds prefix caching (NOT in the run19-validated config: without
    # it every particle step + probe re-encodes the audio — ~40h per curve).
    # Validated on MMAR via the APC canary (drift +0.035 ~= 1 SE, flips balanced).
    export HF_HOME=/u/awaheed/epf_data/hf_cache
    M_ID=moonshotai/Kimi-Audio-7B-Instruct
    M_REV=9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b
    MN=kimi-audio; JMODEL=kimi; INFLIGHT="${KIMI_INFLIGHT:-256}"
    nohup setsid "$PY" "$R/benchmarking/mmau_pro/run19/serve_kimi.py" "$M_ID" \
      --revision "$M_REV" --pid-file "$D/serve_${TAG}.pid" \
      --served-model-name kimi-audio --port "$PORT" --trust-remote-code --dtype bfloat16 \
      --max-model-len 8192 --enforce-eager --gpu-memory-utilization 0.85 \
      --allowed-local-media-path "$MEDIA" --limit-mm-per-prompt '{"audio":1}' \
      --chat-template "$R/benchmarking/mmau_pro/run19/template_kimi_audio_epf.jinja" \
      ${KIMI_APC:+--enable-prefix-caching} \
      > "$D/serve_${TAG}.log" 2>&1 & ;;
  *) echo "FATAL bad MODEL $MODEL"; exit 1 ;;
esac

kill_server() { kill "$(cat "$D/serve_${TAG}.pid" 2>/dev/null)" 2>/dev/null; }
trap kill_server EXIT

t=0; while [ $t -lt 2400 ]; do
  curl -sf "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q "\"$MN\"" && break
  sleep 15; t=$((t+15))
done
[ $t -lt 2400 ] || { echo "FATAL server not healthy (no \"$MN\" on :$PORT)"; tail -30 "$D/serve_${TAG}.log"; exit 1; }
echo "=== $(date '+%F %T') server healthy after ${t}s (:$PORT, model $MN, bench $BENCH) ==="

echo "=== $(date '+%F %T') SMOKE (2 items, b2, store-texts) ==="
rm -f "$D/probe_out/smoke_${PFX}${JMODEL}.jsonl"
"$PY" "$D/probe_epf.py" --endpoint "http://localhost:$PORT/v1" --model-name "$MN" \
  --bench "$BENCH" --limit 2 --budgets 2 --prompt-method 4 --max-steps 6 \
  --store-texts --arms probe_adaptive_t3 --max-inflight 8 \
  --jsonl "$D/probe_out/smoke_${PFX}${JMODEL}.jsonl"
"$PY" - "$D/probe_out/smoke_${PFX}${JMODEL}.jsonl" <<'EOF'
import json, sys
rows = [json.loads(L) for L in open(sys.argv[1])]
ok = [r for r in rows if not r.get("error")]
wt = sum(1 for r in ok if r.get("texts") and any(t.strip() for t in r["texts"]))
tr = sum(1 for r in ok if (r.get("trace") or {}).get("iterations"))
pd = sum(1 for r in ok if any(d for d in (r.get("probe_dists") or [])))
print(f"smoke: {len(ok)}/2 clean, {wt} with texts, {tr} with trace, {pd} with probe letter-mass")
sys.exit(0 if (len(ok) >= 2 and wt >= 2 and tr >= 2 and pd >= 2) else 1)
EOF
[ $? -eq 0 ] || { echo "FATAL smoke gate"; tail -40 "$D/serve_${TAG}.log"; exit 1; }
echo "=== $(date '+%F %T') SMOKE PASS ==="

if [ -n "${FORCE_PM:-}" ]; then
  PM="$FORCE_PM"
else
  CHOICE="$D/probe_out/prompt_choice_${PFX}${JMODEL}.txt"
  if [ ! -s "$CHOICE" ]; then
    echo "=== $(date '+%F %T') P4-vs-P5 greedy check (200 items, b1, temp 0) ==="
    for pm in 4 5; do
      "$PY" "$D/probe_epf.py" --endpoint "http://localhost:$PORT/v1" --model-name "$MN" \
        --bench "$BENCH" --limit 200 --budgets 1 --temp 0.0 --prompt-method "$pm" \
        --max-steps 6 --arms probe_adaptive_t3 --max-inflight 64 \
        --jsonl "$D/probe_out/pcheck_${PFX}${JMODEL}_p${pm}.jsonl"
    done
    "$PY" - "$D/probe_out/pcheck_${PFX}${JMODEL}_p4.jsonl" "$D/probe_out/pcheck_${PFX}${JMODEL}_p5.jsonl" "$CHOICE" <<'EOF'
import json, sys
def acc(path):
    by = {}
    for L in open(path):
        r = json.loads(L)
        if not r.get("error"):
            by[r["unique_id"]] = r
    rs = list(by.values())
    if len(rs) < 180:
        print(f"FATAL pcheck {path}: only {len(rs)} clean rows"); sys.exit(1)
    return sum(1 for r in rs if r["pred_letters"] and r["pred_letters"][0] == r["gold"]) / len(rs), len(rs)
a4, n4 = acc(sys.argv[1]); a5, n5 = acc(sys.argv[2])
w = 5 if a5 > a4 else 4   # tie -> P4 (matches the models' existing baselines)
print(f"pcheck: P4={a4:.4f} (n={n4})  P5={a5:.4f} (n={n5})  -> winner P{w}")
open(sys.argv[3], "w").write(str(w))
EOF
    [ $? -eq 0 ] || { echo "FATAL pcheck"; exit 1; }
  fi
  PM=$(cat "$CHOICE")
fi
echo "=== $(date '+%F %T') prompt method: P$PM ==="

if [ "$MODEL" = kimi ] && [ -n "${KIMI_APC:-}" ] && [ -z "${FORCE_PM:-}" ]; then
  # APC canary: prefix caching is a deviation from the run19-validated shim
  # config. Re-run the greedy pcheck against THIS (APC) server and require the
  # accuracy to reproduce the non-APC result within APC_TOL — a silent
  # KV/mm-hash bug would crater accuracy toward chance before curve compute.
  echo "=== $(date '+%F %T') APC canary (200 items, greedy P$PM) ==="
  "$PY" "$D/probe_epf.py" --endpoint "http://localhost:$PORT/v1" --model-name "$MN" \
    --bench "$BENCH" --limit 200 --budgets 1 --temp 0.0 --prompt-method "$PM" \
    --max-steps 6 --arms probe_adaptive_t3 --max-inflight 64 \
    --jsonl "$D/probe_out/pcheck_${PFX}${JMODEL}_p${PM}_apc.jsonl"
  "$PY" - "$D/probe_out/pcheck_${PFX}${JMODEL}_p${PM}.jsonl" "$D/probe_out/pcheck_${PFX}${JMODEL}_p${PM}_apc.jsonl" <<'EOF'
import json, os, sys
def acc(path):
    by = {}
    for L in open(path):
        r = json.loads(L)
        if not r.get("error"):
            by[r["unique_id"]] = r
    rs = list(by.values())
    return sum(1 for r in rs if r["pred_letters"] and r["pred_letters"][0] == r["gold"]) / max(1, len(rs)), len(rs)
# tolerance: binomial SE at n=200 is ~0.035, and temp-0 trajectory divergence
# under ANY numeric perturbation is chaotic — the gate is for CATASTROPHIC
# drift (wrong-audio KV craters accuracy toward chance), not bitwise parity.
tol = float(os.environ.get("APC_TOL", "0.06"))
a0, n0 = acc(sys.argv[1]); a1, n1 = acc(sys.argv[2])
print(f"APC canary: no-APC={a0:.4f} (n={n0})  APC={a1:.4f} (n={n1})  drift={abs(a1-a0):.4f} tol={tol}")
sys.exit(0 if (n1 >= 180 and abs(a1 - a0) <= tol) else 1)
EOF
  [ $? -eq 0 ] || { echo "FATAL APC canary drift — rerun without KIMI_APC"; exit 1; }
  echo "=== $(date '+%F %T') APC canary PASS ==="
fi

echo "=== $(date '+%F %T') CURVE $CURVEPFX/$JMODEL budgets 128..1 (P$PM, inflight $INFLIGHT) ==="
"$PY" "$D/probe_epf.py" --endpoint "http://localhost:$PORT/v1" --model-name "$MN" \
  --bench "$BENCH" --limit 1000 --budgets 128,64,32,16,8,4,2,1 --prompt-method "$PM" \
  --max-steps 6 --store-texts --arms probe_adaptive_t3 --max-inflight "$INFLIGHT" \
  --jsonl "$D/probe_out/${CURVEPFX}_${JMODEL}_curve.jsonl"
rc=$?
echo "=== $(date '+%F %T') CURVE $CURVEPFX/$JMODEL DONE rc=$rc ==="
exit $rc
