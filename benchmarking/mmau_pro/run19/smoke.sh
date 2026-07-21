#!/bin/bash
# Sanity smoke test. Phase A is offline and exact on any machine (no GPU): unit
# tests, loader counts, snapshot identity, AND the kimi render asserts — the
# wrapper patch + corrected chat template are proven to render non-empty prompts
# and clean continuations before anything is served. Phase B serves one replica
# (GPU 0), runs the prompt-sanity check + both phase0 gates, and a 4-item
# budget-8 probe cell end-to-end (~10-20 min, mostly the model load).
# SMOKE_PHASE=A runs only the offline half (no GPU needed); default runs both.
set -euo pipefail
: "${SMOKE_PHASE:=AB}"
RUN19_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUN19_DIR/config.sh"
source "$RUN19_DIR/lib.sh"
cd "$REPO_ROOT"
mkdir -p "$OUT_ROOT/smoke" "$OUT_ROOT/servers"

echo "=== SMOKE PHASE A (offline: tests, data, scoring, render, artifact identity) ==="
"$EPF_PY" -m pytest tests/ -q --ignore=tests/e2e     # expect: 122+ passed
"$EPF_PY" - "$DATA_TESTMINI" "$DATA_AUDIO" <<'PYEOF'
import sys
from benchmarking.mmau_pro.loader import load_mmau_mcq
from benchmarking.mmau_pro.scoring import extract_letter, match_answer_index, predicted_index
tm, au = sys.argv[1], sys.argv[2]
t = load_mmau_mcq(tm, subset="test_le30s_1a", audio_root=au)
assert (len(t), sum(1 for r in t if r.answer_index is None)) == (1947, 13)
assert all(len(r.audio_paths) == 1 for r in t)
a = load_mmau_mcq(tm, subset="test_1a", audio_root=au)
assert (len(a), sum(1 for r in a if r.answer_index is None)) == (4660, 22)
assert len(load_mmau_mcq(tm, subset="full")) == 957
assert extract_letter('The answer is clear.\n\nAnswer: B', 4) == 1
assert predicted_index('b) dog barking', ['cat', 'dog barking', 'dog']) == 1
assert match_answer_index('The Dog barking!', ['cat', 'dog barking', 'dog']) == 1
print("PHASE A data + scoring: OK")
PYEOF
check_snapshot "$RUN19_MODEL_ID" "$RUN19_REV"
# kimi render asserts — guards the serving bug trio (empty prompts, broken
# continue_final_message, eos-in-prompt encode error) against any future
# vLLM/transformers bump, entirely offline
"$EPF_PY" - "$RUN19_DIR" "$HF_HOME/hub/models--${RUN19_MODEL_ID//\//--}/snapshots/$RUN19_REV" <<'PYEOF'
import sys
run19_dir, snap = sys.argv[1], sys.argv[2]
sys.path.insert(0, run19_dir)
import serve_kimi  # noqa: F401 — applies the apply_chat_template patch
from vllm.tokenizers.kimi_audio import KimiAudioTokenizer
tok = KimiAudioTokenizer.from_pretrained(snap)
tmpl = open(f"{run19_dir}/template_kimi_audio_epf.jinja").read()
msgs = [{"role": "system", "content": "You answer audio MCQs."},
        {"role": "user", "content": "What do you hear? A. rain B. wind"}]
gen = tok.apply_chat_template(msgs, chat_template=tmpl, tokenize=False,
                              add_generation_prompt=True)
assert gen.rstrip().endswith("<|im_kimia_assistant_msg_start|>"), gen[-120:]
assert "What do you hear?" in gen and "You answer audio MCQs." in gen
cont = tok.apply_chat_template(
    msgs + [{"role": "assistant", "content": "## Step 1: I first notice"}],
    chat_template=tmpl, tokenize=False,
    add_generation_prompt=False, continue_final_message=True)
assert cont.rstrip().endswith("## Step 1: I first notice"), cont[-120:]
assert "<|im_kimia_text_eos|>" not in cont
ids = tok.encode(gen, add_special_tokens=False)  # must not raise on the markers
assert len(ids) > 10
# model-generated "[EOS]" text fed back as a prefill must encode as plain text
ids2 = tok.encode("The answer is A. [EOS]", add_special_tokens=False)
assert len(ids2) > 5
# audio-vocab ids in top-logprobs must decode without raising (bug #6)
assert tok.decode([163112]) == "" and tok.decode([ids[0], 163112]) == tok.decode([ids[0]])
print(f"PHASE A kimi render: OK (gen={len(ids)} tokens; continuation clean; "
      "[EOS]-as-text ok; audio-id decode ok)")
PYEOF
echo "=== PHASE A PASS ==="
[ "$SMOKE_PHASE" = "A" ] && { echo "(SMOKE_PHASE=A — skipping the served Phase B)"; exit 0; }

echo "=== SMOKE PHASE B (serve, prompt sanity + gates, tiny b8 cell) ==="
serve_one 0
wait_healthy "$BASE_PORT" 1800
sanity_prompt "$BASE_PORT" || { kill_servers; exit 1; }
gate_endpoint "$BASE_PORT" || { kill_servers; exit 1; }
SM="$OUT_ROOT/smoke/run19_smoke"
rm -f "$SM.jsonl"
extra=(); [ -n "$STOP_REGEX" ] && extra+=(--stop-regex "$STOP_REGEX")
"$EPF_PY" -m benchmarking.mmau_pro.diversity_probe \
  --endpoints "http://localhost:$BASE_PORT/v1" --model-name "$RUN19_NAME" \
  --data-root "$DATA_TESTMINI" --subset "$RUN19_SUBSET" --audio-root "$DATA_AUDIO" \
  --prompts "${PROMPTS_RUN19%%,*}" --signals mean_logprob --budgets 8 \
  --select all --limit 4 --temp 0.8 --ess-threshold 0.6 --early-phase 0.7 \
  --max-inflight "$MAX_INFLIGHT" "${extra[@]}" \
  --jsonl "$SM.jsonl" --csv "$SM.csv" --log "$SM.log"
"$EPF_PY" "$RUN19_DIR/summarize_errors.py" "$SM.jsonl"
kill_servers
echo "=== SMOKE PASS — ready for: nohup bash $RUN19_DIR/run_all.sh > run19.log 2>&1 & ==="
