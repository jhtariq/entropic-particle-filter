#!/bin/bash
# Download and lay out the MMAU-Pro data exactly as the loader expects:
#   $EPF_DATA_ROOT/mmau_pro_testmini/data/*.parquet  (+ 1,099 testmini audio files)
#   $EPF_DATA_ROOT/mmau_pro_audio/data/              (5,787 files, 53 GB)
# Run 19 adds the SINGLE-AUDIO derived parquets (vLLM's kimi path takes at most
# one audio per prompt): test_le30s_1a (the grid subset, 1,947 MCQ) and test_1a
# (capacity audit, 4,660 MCQ), committed in run19/data/; the parent test_le30s
# parquet ships in run16/data/ (same pinned sources).
# Idempotent — safe to re-run after an interrupted download.
set -euo pipefail
RUN19_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUN19_DIR/config.sh"
source "$RUN19_DIR/lib.sh"
cd "$REPO_ROOT"
mkdir -p "$DATA_TESTMINI" "$DATA_AUDIO"

echo "== 1/5 parquets ($PARQUET_DATASET @ ${PARQUET_REV:0:12})"
hf_cli download "$PARQUET_DATASET" --repo-type dataset --revision "$PARQUET_REV" \
  --include "data/*.parquet" --local-dir "$DATA_TESTMINI" > /dev/null
for f in testmini test; do
  [ -e "$DATA_TESTMINI/data/${f}-00000-of-00001.parquet" ] \
    || { echo "FATAL: ${f} parquet missing after download" >&2; exit 1; }
done

echo "== 2/5 derived subset parquets (committed in the repo — run16 + run19)"
cp -n "$RUN19_DIR/../run16/data/test_le30s-00000-of-00001.parquet" "$DATA_TESTMINI/data/" || true
cp -n "$RUN19_DIR/data/test_le30s_1a-00000-of-00001.parquet" "$DATA_TESTMINI/data/" || true
cp -n "$RUN19_DIR/data/test_1a-00000-of-00001.parquet" "$DATA_TESTMINI/data/" || true
for f in test_le30s test_le30s_1a test_1a; do
  [ -e "$DATA_TESTMINI/data/${f}-00000-of-00001.parquet" ] \
    || { echo "FATAL: ${f} parquet missing" >&2; exit 1; }
done

echo "== 3/5 full audio set ($AUDIO_DATASET data.zip, 44 GiB — skipped if already extracted)"
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

echo "== 4/5 copy the 1,099 testmini clips beside the testmini parquets"
"$EPF_PY" "$RUN19_DIR/prep_testmini_audio.py" \
  --testmini-root "$DATA_TESTMINI" --audio-root "$DATA_AUDIO"

echo "== 5/5 loader verification (exact expected counts)"
"$EPF_PY" - "$DATA_TESTMINI" "$DATA_AUDIO" <<'PYEOF'
import sys
from benchmarking.mmau_pro.loader import load_mmau_mcq
tm, au = sys.argv[1], sys.argv[2]

def counts(subset):
    recs = load_mmau_mcq(tm, subset=subset, audio_root=au)
    return len(recs), sum(1 for r in recs if r.answer_index is None)

assert counts("test") == (5090, 24), f"test: {counts('test')}, want (5090, 24)"
assert counts("test_le30s_1a") == (1947, 13), f"test_le30s_1a: {counts('test_le30s_1a')}, want (1947, 13)"
assert counts("test_1a") == (4660, 22), f"test_1a: {counts('test_1a')}, want (4660, 22)"
assert len(load_mmau_mcq(tm, subset="full")) == 957
print("DATA OK: test=5090 (24 ungr), test_le30s_1a=1947 (13 ungr), test_1a=4660 (22 ungr), full=957")
PYEOF
