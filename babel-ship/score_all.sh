#!/usr/bin/env bash
# Score every finished d1k sweep (report-only). Add --fill-csv via
# EXTRA_ARGS="--fill-csv" once BABEL_CSV_DIR points at a consolidated_results
# dir with a mmau-pro-d1k.csv in the campaign schema.
set -euo pipefail

cd "$(dirname "$0")"
EXTRA_ARGS="${EXTRA_ARGS:-}"

declare -A csv_model=(
  [qwen-omni]=qwen2.5-omni-7b
  [qwen-omni-3b]=qwen2.5-omni-3b
  [qwen2-audio]=qwen2-audio
  [phi4mm]=phi4mm
  [kimi-audio]=kimi-audio
)

for model in kimi-audio qwen-omni qwen-omni-3b phi4mm qwen2-audio; do
  jsonl="out/d1k_${model}_prmjudge.jsonl"
  if [[ ! -f "$jsonl" ]]; then
    echo "== $model: no sweep output yet ($jsonl)"
    continue
  fi
  echo "== $model"
  # shellcheck disable=SC2086
  python score_and_fill.py "$jsonl" \
    --benchmark mmau-pro-d1k \
    --model "${csv_model[$model]}" \
    --prompt "P4 CoT" \
    --arm judge_epf_prm \
    --signals judge_marg_mean,judge_marginal,argmax_judge \
    --method epf-judge-prm \
    --expect 991 \
    --n-items 1000 \
    $EXTRA_ARGS
done
