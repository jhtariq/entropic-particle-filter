#!/bin/bash
# Download and lay out all THREE benchmark datasets exactly as the loaders expect,
# byte-verified against the reference box:
#   $DATA_TESTMINI/data/*.parquet        MMAU-Pro parquets (pinned HF revision)
#   $DATA_AUDIO/data/                    MMAU-Pro full-test audio (5,787 files, 53 GB; zip sha256)
#   $DATA_MMAR/{MMAR-meta.json,audio/}   MMAR (pinned HF revision + committed sha256 manifest)
#   $DATA_MMSU/{MMSU-meta.json,audio/}   MMSU (materialized from pinned parquets + manifest)
# Idempotent — safe to re-run after an interrupted download. KEEP_ZIP=1 keeps the
# transient archives (44 GiB MMAU zip, 3 GB MMAR tarball, 1.5 GB MMSU parquets).
set -euo pipefail
GREEDY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$GREEDY_DIR/config.sh"
source "$GREEDY_DIR/lib.sh"
cd "$REPO_ROOT"
mkdir -p "$DATA_TESTMINI" "$DATA_AUDIO" "$DATA_MMAR" "$DATA_MMSU"

echo "== 1/5 MMAU-Pro parquets ($PARQUET_DATASET @ ${PARQUET_REV:0:12})"
if [ ! -e "$DATA_TESTMINI/data/test-00000-of-00001.parquet" ]; then
  hf_cli download "$PARQUET_DATASET" --repo-type dataset --revision "$PARQUET_REV" \
    --include "data/*.parquet" --local-dir "$DATA_TESTMINI" > /dev/null
fi
[ -e "$DATA_TESTMINI/data/test-00000-of-00001.parquet" ] \
  || { echo "FATAL: test parquet missing after download" >&2; exit 1; }

echo "== 2/5 MMAU-Pro full audio set ($AUDIO_DATASET data.zip, 44 GiB — skipped if already extracted)"
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

echo "== 3/5 MMAR ($MMAR_DATASET @ ${MMAR_REV:0:12})"
if (cd "$DATA_MMAR" && sha256sum --quiet -c "$GREEDY_DIR/manifests/mmar.sha256" > /dev/null 2>&1); then
  echo "   already present and byte-verified — skipping"
else
  hf_cli download "$MMAR_DATASET" --repo-type dataset --revision "$MMAR_REV" \
    --local-dir "$DATA_MMAR" > /dev/null
  echo "$MMAR_AUDIO_TGZ_SHA256  $DATA_MMAR/mmar-audio.tar.gz" | sha256sum -c -
  tar -xzf "$DATA_MMAR/mmar-audio.tar.gz" -C "$DATA_MMAR"   # members are already audio/<name>.wav
  [ "${KEEP_ZIP:-0}" = "1" ] || rm -f "$DATA_MMAR/mmar-audio.tar.gz"
  # byte-identity vs the reference box: meta + all 1,000 wavs
  (cd "$DATA_MMAR" && sha256sum --quiet -c "$GREEDY_DIR/manifests/mmar.sha256") \
    || { echo "FATAL: MMAR download does not match the committed manifest — DO NOT run; report this" >&2; exit 1; }
  echo "   byte-verified against manifests/mmar.sha256 (1,001 files)"
fi

echo "== 4/5 MMSU ($MMSU_DATASET @ ${MMSU_REV:0:12} — materialized from parquets)"
if (cd "$DATA_MMSU" && sha256sum --quiet -c "$GREEDY_DIR/manifests/mmsu.sha256" > /dev/null 2>&1); then
  echo "   already present and byte-verified — skipping"
else
  MMSU_PARQUET_DIR="$EPF_DATA_ROOT/mmsu_parquet"
  hf_cli download "$MMSU_DATASET" --repo-type dataset --revision "$MMSU_REV" \
    --include "data/*.parquet" --local-dir "$MMSU_PARQUET_DIR" > /dev/null
  "$EPF_PY" -m greedy.materialize_mmsu \
    --parquet-dir "$MMSU_PARQUET_DIR/data" --out-root "$DATA_MMSU"
  [ "${KEEP_ZIP:-0}" = "1" ] || rm -rf "$MMSU_PARQUET_DIR"
  # byte-identity vs the reference box: meta + all 5,000 audio files
  (cd "$DATA_MMSU" && sha256sum --quiet -c "$GREEDY_DIR/manifests/mmsu.sha256") \
    || { echo "FATAL: MMSU materialization does not match the committed manifest — DO NOT run; report this" >&2; exit 1; }
  echo "   byte-verified against manifests/mmsu.sha256 (5,001 files)"
fi

echo "== 5/5 loader verification (exact expected counts)"
"$EPF_PY" - "$DATA_TESTMINI" "$DATA_AUDIO" "$DATA_MMAR" "$DATA_MMSU" <<'PYEOF'
import sys
from benchmarking.mmau_pro.loader import load_mmau_mcq
from benchmarking.mmar.loader import load_mmar_mcq
from benchmarking.mmsu.loader import load_mmsu_mcq
tm, au, mmar, mmsu = sys.argv[1:5]

recs = load_mmau_mcq(tm, subset="test", audio_root=au)
ungr = sum(1 for r in recs if r.answer_index is None)
assert (len(recs), ungr) == (5090, 24), f"mmau test: {(len(recs), ungr)}, want (5090, 24)"

recs = load_mmar_mcq(mmar, subset="full")
ungr = sum(1 for r in recs if r.answer_index is None)
assert (len(recs), ungr) == (1000, 4), f"mmar full: {(len(recs), ungr)}, want (1000, 4)"

recs = load_mmsu_mcq(mmsu, subset="full")
ungr = sum(1 for r in recs if r.answer_index is None)
assert (len(recs), ungr) == (5000, 0), f"mmsu full: {(len(recs), ungr)}, want (5000, 0)"

print("DATA OK: mmau test=5090 (24 ungr), mmar full=1000 (4 ungr), mmsu full=5000 (0 ungr)")
PYEOF
