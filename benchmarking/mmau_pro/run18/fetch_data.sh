#!/bin/bash
# Download and lay out the MMAU-Pro data exactly as the loader expects:
#   $EPF_DATA_ROOT/mmau_pro_testmini/data/*.parquet  (+ 1,099 testmini audio files)
#   $EPF_DATA_ROOT/mmau_pro_audio/data/              (5,787 files, 53 GB)
# Run 18 uses the stock full-test parquet (subset "test") — no derived subsets.
# Idempotent — safe to re-run after an interrupted download.
set -euo pipefail
RUN18_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUN18_DIR/config.sh"
source "$RUN18_DIR/lib.sh"
cd "$REPO_ROOT"
mkdir -p "$DATA_TESTMINI" "$DATA_AUDIO"

echo "== 1/4 parquets ($PARQUET_DATASET @ ${PARQUET_REV:0:12})"
hf_cli download "$PARQUET_DATASET" --repo-type dataset --revision "$PARQUET_REV" \
  --include "data/*.parquet" --local-dir "$DATA_TESTMINI" > /dev/null
for f in testmini test; do
  [ -e "$DATA_TESTMINI/data/${f}-00000-of-00001.parquet" ] \
    || { echo "FATAL: ${f} parquet missing after download" >&2; exit 1; }
done

echo "== 2/4 full audio set ($AUDIO_DATASET data.zip, 44 GiB — skipped if already extracted)"
NFILES=$(find "$DATA_AUDIO/data" -type f 2>/dev/null | wc -l)
if [ "$NFILES" -ne 5787 ]; then
  hf_cli download "$AUDIO_DATASET" data.zip --repo-type dataset --local-dir "$DATA_AUDIO" > /dev/null
  echo "$AUDIO_ZIP_SHA256  $DATA_AUDIO/data.zip" | sha256sum -c -
  unzip -n -q "$DATA_AUDIO/data.zip" -d "$DATA_AUDIO"
  NFILES=$(find "$DATA_AUDIO/data" -type f | wc -l)
  [ "$NFILES" -eq 5787 ] || { echo "FATAL: expected 5,787 audio files, found $NFILES" >&2; exit 1; }
  [ "${KEEP_ZIP:-0}" = "1" ] || rm -f "$DATA_AUDIO/data.zip"
else
  echo "   already present ($NFILES files) — skipping"
fi

echo "== 3/4 copy the 1,099 testmini clips beside the testmini parquets"
"$EPF_PY" "$RUN18_DIR/prep_testmini_audio.py" \
  --testmini-root "$DATA_TESTMINI" --audio-root "$DATA_AUDIO"

echo "== 4/4 loader verification (exact expected counts)"
"$EPF_PY" - "$DATA_TESTMINI" "$DATA_AUDIO" <<'PYEOF'
import sys
from benchmarking.mmau_pro.loader import load_mmau_mcq
tm, au = sys.argv[1], sys.argv[2]
t = load_mmau_mcq(tm, subset="test", audio_root=au)
n_ungr = sum(1 for r in t if r.answer_index is None)
assert (len(t), n_ungr) == (5090, 24), f"test subset: got {len(t)}/{n_ungr}, want 5090/24"
assert len(load_mmau_mcq(tm, subset="full")) == 957
print("DATA OK: test=5090 MCQ (24 ungradeable), full=957")
PYEOF
