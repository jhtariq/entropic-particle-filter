#!/usr/bin/env bash
# Score finished starbench sweeps (report-only; EXTRA_ARGS="--fill-csv" plus
# BABEL_CSV_DIR to file into a consolidated starbench.csv).
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

for model in qwen-omni-3b qwen-omni qwen2-audio phi4mm kimi-audio; do
  jsonl="out/starbench_${model}_prmjudge.jsonl"
  if [[ ! -f "$jsonl" ]]; then
    echo "== $model: no sweep output yet ($jsonl)"
    continue
  fi
  echo "== $model"
  # shellcheck disable=SC2086
  python score_and_fill.py "$jsonl" \
    --benchmark starbench \
    --model "${csv_model[$model]}" \
    --prompt "P12 Step10" \
    --arm judge_prm_ess \
    --signals judge_marg_mean,judge_marginal,argmax_judge \
    --method epf-judge-prm-ess \
    --expect 626 \
    --n-items 626 \
    $EXTRA_ARGS
done
