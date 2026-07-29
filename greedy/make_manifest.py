"""One-shot tool: build sha256 manifests of the LOCAL MMAR / MMSU dataset copies.

Run once on the reference box; the outputs (greedy/manifests/{mmar,mmsu}.sha256)
are committed so a collaborator's fresh download can be verified byte-identical
with `sha256sum -c` (fetch_data.sh does this). Lines use paths relative to the
dataset root, in sorted order, covering the meta JSON and every audio file.

    python -m greedy.make_manifest --mmar-root .../mmar --mmsu-root .../data/mmsu \
        --out-dir greedy/manifests
"""

import hashlib
import os

import click


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(root: str, meta_name: str, out_path: str) -> int:
    entries = [meta_name]
    audio_dir = os.path.join(root, "audio")
    entries += sorted(
        os.path.join("audio", f) for f in os.listdir(audio_dir)
        if os.path.isfile(os.path.join(audio_dir, f))
    )
    with open(out_path, "w") as out:
        for rel in entries:
            digest = sha256_file(os.path.join(root, rel))
            out.write(f"{digest}  {rel}\n")
    return len(entries)


@click.command()
@click.option("--mmar-root", required=True)
@click.option("--mmsu-root", required=True)
@click.option("--out-dir", default="greedy/manifests")
def main(mmar_root, mmsu_root, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    n = build_manifest(mmar_root, "MMAR-meta.json", os.path.join(out_dir, "mmar.sha256"))
    print(f"mmar.sha256: {n} entries (1 meta + {n - 1} audio)")
    n = build_manifest(mmsu_root, "MMSU-meta.json", os.path.join(out_dir, "mmsu.sha256"))
    print(f"mmsu.sha256: {n} entries (1 meta + {n - 1} audio)")


if __name__ == "__main__":
    main()
