"""Download the exact pinned checkpoints and verify they resolve to the same
commits used in the original campaign.

Usage:  python sanity_check_models.py [judge qwen-omni ...]
        (no args = all six entries in models_manifest.json)

Downloads go to HF_HOME (set it before running if you want them somewhere
specific). Exits non-zero if any revision mismatches - do NOT run sweeps in
that state, the numbers would not be comparable.
"""
import json
import os
import sys

from huggingface_hub import snapshot_download

manifest = json.load(open(os.path.join(os.path.dirname(__file__), "models_manifest.json")))
targets = sys.argv[1:] or list(manifest)

failed = []
for name in targets:
    m = manifest[name]
    print(f"[{name}] {m['repo']} @ {m['revision'][:12]} ...", flush=True)
    path = snapshot_download(m["repo"], revision=m["revision"])
    resolved = os.path.basename(os.path.realpath(path))
    if resolved != m["revision"]:
        print(f"  MISMATCH: snapshot resolved to {resolved}")
        failed.append(name)
        continue
    if name == "phi4mm":
        lora = os.path.join(path, "speech-lora")
        if not os.path.isdir(lora):
            print(f"  MISSING speech-lora dir at {lora}")
            failed.append(name)
            continue
    print(f"  OK -> {path}")

if failed:
    print(f"\nFAIL: {failed}")
    sys.exit(1)
print("\nALL_MODELS_OK")
