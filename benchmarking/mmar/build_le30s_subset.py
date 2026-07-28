"""Build the MMAR le30s subset: items whose clip fits Qwen2-Audio's hard 30 s
encoder window (longer clips are TRUNCATED by that model, not chunk-processed).

Mirrors the mmau_pro test_le30s recipe: filter by soundfile-measured duration
(deterministic, authoritative — NOT the timestamp field, 45/1000 of which are
malformed) and write a pinned sidecar next to the source metadata:

    python -m benchmarking.mmar.build_le30s_subset   # -> <data-root>/MMAR-meta-le30s.json
"""

import json
import os

import click
import soundfile as sf

from benchmarking.mmar.loader import DEFAULT_DATA_ROOT

MAX_SECONDS = 30.0


@click.command()
@click.option("--data-root", default=DEFAULT_DATA_ROOT)
@click.option("--out-name", default="MMAR-meta-le30s.json")
def main(data_root, out_name):
    with open(os.path.join(data_root, "MMAR-meta.json")) as f:
        rows = json.load(f)
    kept, dropped = [], []
    for r in rows:
        path = os.path.abspath(os.path.join(data_root, str(r["audio_path"])))
        dur = sf.info(path).duration
        (kept if dur <= MAX_SECONDS else dropped).append((r, dur))
    out_path = os.path.join(data_root, out_name)
    with open(out_path, "w") as f:
        json.dump([r for r, _ in kept], f, ensure_ascii=False, indent=1)
    print(f"kept {len(kept)} items (duration <= {MAX_SECONDS}s) -> {out_path}")
    print(f"dropped {len(dropped)}:")
    for r, dur in sorted(dropped, key=lambda x: -x[1]):
        print(f"  {dur:6.1f}s  {r['id']}  [{r['modality']}]")


if __name__ == "__main__":
    main()
