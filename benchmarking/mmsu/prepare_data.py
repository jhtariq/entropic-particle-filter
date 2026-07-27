"""Assemble the MMSU data root this harness expects from the upstream HF release.

MMSU ships on the Hub (`ddwang2000/MMSU`) as parquet shards with an embedded audio
column PLUS a top-level `audio/` tree — and no metadata file. `benchmarking/mmsu/
loader.py` wants the house layout instead:

    <out-root>/
      MMSU-meta.json      # JSON array, exactly 5,000 items (the loader contract)
      audio/              # 5,000 .wav, byte-identical to the Hub's

so a fresh box has to build it. That is what this does.

**Audio comes from the parquet `audio` column, written out byte-for-byte.** The Hub
ships the same clips twice — embedded in the parquets AND as a top-level `audio/`
tree — and the two copies DISAGREE. Only the parquet copy reproduces the reference
box's census (channels {1: 4550, 2: 450}, max 35.91 s, 4 clips >30 s); the `audio/`
tree gives {1: 4180, 2: 820} and a 50.73 s max. See `extract_audio`. This is not a
re-encode: HF's Audio feature stores the ORIGINAL encoded file bytes, so nothing is
decoded or resampled. `--audio-source hub` fetches the other copy for provenance
work, but it will NOT pass `--check`.

Column mapping (upstream -> loader contract):

    id                          -> id
    question                    -> question
    choice_a..choice_d          -> choices {"A":…,"B":…,"C":…,"D":…}; missing -> null
                                   (453/5,000 items are 2-choice padded with nulls;
                                   the loader drops the padding, so it must be JSON
                                   null and never "")
    answer_gt                   -> answer — the full choice TEXT, not a letter
                                   (upstream's own mmsu_evaluation.py grades with
                                   `choices['A'] == answer_gt`)
    <id>.wav                    -> audio_path "./audio/<id>.wav"
    task_name, category, sub-category, sub-sub-category,
    linguistics_sub_discipline  -> verbatim

`--check` re-asserts every reference-box fact that `tests/test_mmsu.py` pins. Those
tests are `@requires_mmsu`-gated on the reference box's hardcoded DEFAULT_DATA_ROOT,
so they SKIP on any other machine — this flag is how a fresh box proves the rebuilt
root is item-identical before spending GPU-hours. Exit 1 on any mismatch.

    python -m benchmarking.mmsu.prepare_data \
        --parquet-dir /path/to/MMSU/data --out-root /path/to/data/mmsu
"""

import argparse
import glob
import json
import os
from collections import Counter

# upstream parquet columns we read (the `audio` column is deliberately NOT among them)
META_COLS = [
    "id", "task_name", "question",
    "choice_a", "choice_b", "choice_c", "choice_d",
    "answer_gt", "category", "sub-category", "sub-sub-category",
    "linguistics_sub_discipline",
]
CHOICE_COLS = {"A": "choice_a", "B": "choice_b", "C": "choice_c", "D": "choice_d"}

HF_REPO = "ddwang2000/MMSU"

# --- reference-box pins (mirror tests/test_mmsu.py + benchmarking/mmsu/README.md) ---
EXPECT_RECORDS = 5000
EXPECT_CHOICE_MIX = {4: 4547, 2: 453}
EXPECT_TASKS = 47
EXPECT_TASK_COUNTS = {"gender_prediction": 120, "total_speaker_counting": 120}
EXPECT_CATEGORY = {"Perception": 2580, "Reasoning": 2420}
EXPECT_SUBCATEGORY = {"Linguistics": 3655, "Paralinguistics": 1345}
EXPECT_CHANNELS = {1: 4550, 2: 450}
EXPECT_GT30S = 4                        # clips longer than 30 s
EXPECT_DUR_MIN = (0.25, 0.40)           # ~0.3 s
EXPECT_DUR_P50 = (4.2, 4.8)             # ~4.5 s
EXPECT_DUR_MAX = (35.5, 36.2)           # ~35.9 s


def _clean(v):
    """Upstream nulls/blanks -> None; everything else -> str.

    The literal string "None" is a real choice value (19 items) and a real
    linguistics_sub_discipline, so only true nulls and whitespace are dropped.
    """
    if v is None:
        return None
    s = str(v)
    if s.strip() == "" or s == "nan":
        return None
    return s


def audio_manifest(out_root: str, revision: str | None) -> dict[str, dict]:
    """{'<id>.wav': {'path': <upstream path>, 'size': int}}, cached on disk.

    Fetched once from the tree API (paginated) so the download loop has a
    ground-truth target it can verify against — see download_audio.

    The local key is ALWAYS `<id>.wav` but the upstream path is NOT: the Hub's
    audio/ tree is 4,095 .wav + 899 .mp3 + 3 .flac + 3 .ogg. The loader contract
    is `./audio/<id>.wav` for all 5,000, so every file is stored under `<id>.wav`
    with its bytes untouched, whatever the upstream container — which is why the
    README notes that some `.wav` files carry MP3/ID3 bytes. soundfile/libsndfile
    and vLLM sniff the content, not the extension, so this is safe; renaming to
    the true extension instead would break `audio_path` for 905 items.
    """
    import json as _json
    import re
    import urllib.request

    cache = os.path.join(out_root, ".audio_manifest.json")
    if os.path.exists(cache):
        with open(cache) as f:
            return _json.load(f)

    rev = revision or "main"
    url = f"https://huggingface.co/api/datasets/{HF_REPO}/tree/{rev}/audio?limit=1000"
    token = os.environ.get("HF_TOKEN") or _stored_token()
    headers = {"User-Agent": "mmsu-prepare"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    entries: dict[str, dict] = {}
    while url:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as r:
            page = _json.load(r)
            link = r.headers.get("Link", "")
        for e in page:
            if e.get("type") != "file":
                continue
            path = e["path"]
            local = os.path.basename(os.path.splitext(path)[0]) + ".wav"
            entries[local] = {"path": path,
                              "size": (e.get("lfs") or {}).get("size") or e.get("size") or 0}
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = m.group(1) if m else None

    os.makedirs(out_root, exist_ok=True)
    with open(cache, "w") as f:
        _json.dump(entries, f)
    exts = Counter(os.path.splitext(v["path"])[1] for v in entries.values())
    print(f"manifest: {len(entries)} audio files, "
          f"{sum(v['size'] for v in entries.values()) / 1e9:.3f} GB, upstream {dict(exts)}")
    return entries


def _stored_token() -> str | None:
    for p in (os.path.join(os.environ.get("HF_HOME", ""), "token"),
              os.path.expanduser("~/.cache/huggingface/token")):
        if p and os.path.exists(p):
            return open(p).read().strip() or None
    return None


def download_audio(out_root: str, revision: str | None, workers: int = 8,
                   rounds: int = 40) -> None:
    """Pull audio/*.wav from the Hub verbatim, into <out-root>/audio/.

    Two hard-won details, both worth keeping:

    1. **Fetch from `resolve/`, not the `hf_hub_download` API path.** The Hub meters
       `/api/*` at 1,000 requests per 5-minute window; 5,000 one-per-file metadata
       calls blow through that and the whole download stalls in 429s. The file
       endpoint `…/resolve/<rev>/audio/<name>` is metered separately and far more
       generously (5,000 per 5 minutes), and it is all we actually need — the
       manifest already tells us the expected size of every file.
    2. **Completeness is the only exit condition.** `snapshot_download` responds to
       a 429 on its repo-info call by logging "Returning existing local_dir … as
       remote repo cannot be accessed" and returning NORMALLY, i.e. a partial
       download that looks like success. So this loops until every file is present
       at its manifest size, and raises otherwise.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor

    import requests

    entries = audio_manifest(out_root, revision)
    dest = os.path.join(out_root, "audio")
    os.makedirs(dest, exist_ok=True)
    rev = revision or "main"
    base = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{rev}/"
    token = os.environ.get("HF_TOKEN") or _stored_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    def missing() -> list[str]:
        out = []
        for name, meta in entries.items():
            p = os.path.join(dest, name)
            size = meta["size"]
            if not os.path.exists(p) or (size and os.path.getsize(p) != size):
                out.append(name)
        return out

    todo = missing()
    print(f"audio/: {len(entries) - len(todo)}/{len(entries)} present, fetching {len(todo)}")
    session = requests.Session()

    def grab(name):
        """Returns None on success, or a short reason string."""
        meta = entries[name]
        try:
            r = session.get(base + meta["path"], headers=headers, timeout=120)
            if r.status_code == 429:
                return "429"
            r.raise_for_status()
            want = meta["size"]
            if want and len(r.content) != want:
                return f"size {len(r.content)}!={want}"
            tmp = os.path.join(dest, name + ".part")
            with open(tmp, "wb") as f:
                f.write(r.content)
            os.replace(tmp, os.path.join(dest, name))  # always <id>.wav, bytes verbatim
            return None
        except Exception as e:
            return type(e).__name__

    for rnd in range(1, rounds + 1):
        if not todo:
            break
        with ThreadPoolExecutor(max_workers=workers) as ex:
            reasons = [x for x in ex.map(grab, todo) if x]
        todo = missing()
        if not todo:
            break
        wait = 60 if any(r == "429" for r in reasons) else min(120, 10 * rnd)
        top = Counter(reasons).most_common(2)
        print(f"  round {rnd}: {len(entries) - len(todo)}/{len(entries)} present, {len(todo)} left "
              f"({dict(top)}) — waiting {wait}s for the rate-limit window")
        time.sleep(wait)

    if todo:
        raise SystemExit(f"FAILED: {len(todo)} audio files still missing after {rounds} rounds "
                         f"(e.g. {todo[:3]}) — re-run to resume")
    print(f"audio/: {len(entries)}/{len(entries)} files present at manifest size")


def extract_audio(parquet_dir: str, out_root: str) -> None:
    """Write <out-root>/audio/<id>.wav from the parquet `audio` column, verbatim.

    THIS, not the Hub's audio/ tree, is the reference-box source — the two copies
    disagree, and only this one reproduces the pinned census. Measured over all
    5,000 clips:

                        channels          min   p50    max    >30s
        parquet     {1: 4550, 2: 450}    0.30  4.54  35.91      4   <- matches the pins
        Hub audio/  {1: 4180, 2: 820}    0.30  4.57  50.73      6

    (Sizes mislead here: the Hub tree is 1.685 GB and the parquet 1.78 GB, so the
    README's "1.7 GB" reads like the Hub copy. The census is what settles it.)

    Not a re-encode: HF's Audio feature stores the ORIGINAL encoded file bytes, so
    these are written out byte-for-byte — spot-checked byte-identical to the Hub
    copies on 711 of 721 overlapping ids. The 10 that differ are genuinely
    different recordings upstream. Containers vary (wav/mp3/flac/ogg) but every
    file is stored as `<id>.wav` per the loader contract; soundfile and vLLM sniff
    content, not extension.
    """
    import pyarrow.parquet as pq

    dest = os.path.join(out_root, "audio")
    os.makedirs(dest, exist_ok=True)
    written = skipped = 0
    for shard in sorted(glob.glob(os.path.join(parquet_dir, "train-*.parquet"))):
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=200, columns=["id", "audio"]):
            for rid, aud in zip(batch.column("id").to_pylist(), batch.column("audio").to_pylist()):
                path = os.path.join(dest, f"{rid}.wav")
                data = aud["bytes"]
                if os.path.exists(path) and os.path.getsize(path) == len(data):
                    skipped += 1
                    continue
                tmp = path + ".part"
                with open(tmp, "wb") as f:
                    f.write(data)
                os.replace(tmp, path)
                written += 1
    print(f"audio/: {written} written, {skipped} already correct "
          f"({len(glob.glob(os.path.join(dest, '*.wav')))} total)")


def build_meta(parquet_dir: str, out_root: str) -> str:
    """Read the shards in order and write MMSU-meta.json. Returns the path."""
    import pyarrow.parquet as pq

    shards = sorted(glob.glob(os.path.join(parquet_dir, "train-*.parquet")))
    if not shards:
        raise SystemExit(f"no train-*.parquet under {parquet_dir}")
    rows = []
    for shard in shards:
        table = pq.read_table(shard, columns=META_COLS)  # audio column never touched
        for r in table.to_pylist():
            rid = str(r["id"])
            rows.append({
                "id": rid,
                "task_name": _clean(r["task_name"]),
                "audio_path": f"./audio/{rid}.wav",
                "question": _clean(r["question"]),
                "choices": {k: _clean(r[c]) for k, c in CHOICE_COLS.items()},
                "answer": _clean(r["answer_gt"]),
                "category": _clean(r["category"]),
                "sub-category": _clean(r["sub-category"]),
                "sub-sub-category": _clean(r["sub-sub-category"]),
                "linguistics_sub_discipline": _clean(r["linguistics_sub_discipline"]),
            })
        print(f"  {os.path.basename(shard)}: {table.num_rows} rows")

    os.makedirs(out_root, exist_ok=True)
    path = os.path.join(out_root, "MMSU-meta.json")
    with open(path, "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print(f"wrote {path} ({len(rows)} records)")
    return path


def check(out_root: str) -> bool:
    """Re-assert the reference-box pins against the rebuilt root. True = identical."""
    from benchmarking.mmsu.loader import load_metadata, load_mmsu_mcq

    fails = []

    def want(label, got, expect):
        ok = got == expect
        print(f"  {'ok  ' if ok else 'FAIL'} {label}: {got}" + ("" if ok else f"  (expect {expect})"))
        if not ok:
            fails.append(f"{label}: got {got}, expect {expect}")

    def want_range(label, got, lo_hi):
        lo, hi = lo_hi
        ok = got is not None and lo <= got <= hi
        print(f"  {'ok  ' if ok else 'FAIL'} {label}: {got}" + ("" if ok else f"  (expect {lo}-{hi})"))
        if not ok:
            fails.append(f"{label}: got {got}, expect {lo}-{hi}")

    recs = load_mmsu_mcq(data_root=out_root)
    print("identity check vs the reference box:")
    want("records", len(recs), EXPECT_RECORDS)
    want("ungradeable", len([r for r in recs if r.answer_index is None]), 0)
    want("gold verbatim-matches its own choice",
         all(r.answer_index is not None and r.choices[r.answer_index] == r.answer for r in recs), True)
    want("single-audio", all(len(r.audio_paths) == 1 for r in recs), True)
    want("audio present",
         all(os.path.isabs(p) and os.path.exists(p) for r in recs for p in r.audio_paths), True)
    want("choice-count mix", dict(Counter(len(r.choices) for r in recs)), EXPECT_CHOICE_MIX)
    want("no blank/None choice",
         not any(c is None or str(c).strip() == "" for r in recs for c in r.choices), True)

    tasks = Counter(r.category for r in recs)
    want("task_names", len(tasks), EXPECT_TASKS)
    for name, n in EXPECT_TASK_COUNTS.items():
        want(f"task {name}", tasks.get(name), n)

    md = load_metadata(data_root=out_root)
    want("metadata rows", len(md), EXPECT_RECORDS)
    want("coarse category", dict(Counter(v["category"] for v in md.values())), EXPECT_CATEGORY)
    want("sub-category", dict(Counter(v["sub_category"] for v in md.values())), EXPECT_SUBCATEGORY)
    want('"None" is a real linguistics_sub_discipline',
         "None" in {v["linguistics_sub_discipline"] for v in md.values()}, True)

    # channel + duration census — verify_data reports durations but not channels
    try:
        import soundfile as sf
    except ImportError:
        fails.append("soundfile unavailable — cannot census channels/durations")
        print("  FAIL soundfile unavailable (channels/durations unchecked)")
    else:
        chans, durs = Counter(), []
        for r in recs:
            info = sf.info(r.audio_paths[0])
            chans[info.channels] += 1
            durs.append(info.duration)
        durs.sort()
        want("channel mix", dict(chans), EXPECT_CHANNELS)
        want_range("duration min", round(durs[0], 2), EXPECT_DUR_MIN)
        want_range("duration p50", round(durs[min(len(durs) - 1, len(durs) // 2)], 2), EXPECT_DUR_P50)
        want_range("duration max", round(durs[-1], 2), EXPECT_DUR_MAX)
        want("clips > 30 s", sum(1 for d in durs if d > 30.0), EXPECT_GT30S)

    print(f"\nIDENTITY {'PASS' if not fails else 'FAIL'}")
    for f in fails:
        print(f"  - {f}")
    return not fails


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet-dir", help="dir holding the upstream train-*.parquet shards")
    ap.add_argument("--out-root", required=True, help="the DATA_ROOT to build")
    ap.add_argument("--revision", default=None, help="pin the Hub revision for audio/")
    ap.add_argument("--audio-source", choices=["parquet", "hub"], default="parquet",
                    help="parquet = the reference-box source (see extract_audio); "
                         "hub = download the Hub's audio/ tree, which does NOT match the pins")
    ap.add_argument("--audio-workers", type=int, default=8)
    ap.add_argument("--skip-audio", action="store_true")
    ap.add_argument("--skip-meta", action="store_true", help="leave MMSU-meta.json alone")
    args = ap.parse_args()

    if not args.skip_audio:
        if args.audio_source == "parquet":
            if not args.parquet_dir:
                raise SystemExit("--parquet-dir is required for --audio-source parquet")
            extract_audio(args.parquet_dir, args.out_root)
        else:
            download_audio(args.out_root, args.revision, workers=args.audio_workers)
    if not args.skip_meta:
        if not args.parquet_dir:
            raise SystemExit("--parquet-dir is required unless --skip-meta")
        build_meta(args.parquet_dir, args.out_root)
    if not os.path.exists(os.path.join(args.out_root, "MMSU-meta.json")):
        print("MMSU-meta.json not built yet — skipping the identity check")
        raise SystemExit(0)
    raise SystemExit(0 if check(args.out_root) else 1)


if __name__ == "__main__":
    main()
