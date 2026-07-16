"""Patch vLLM's gemma4_mm._process_audio_input to survive mixed-length audio batches.

WHY (Run 14, RESULTS.md §20): vLLM caches Gemma-4 audio features unpadded per item; a
batch of different-length audios cannot be stacked and reaches the audio tower as a
LIST — upstream code calls .squeeze(1) on it and the engine dies
(`AttributeError: 'list' object has no attribute 'squeeze'`). Single requests work, so
gates pass and the crash only appears under concurrency. Present in upstream vLLM main
as of 2026-07-10 (verified against raw GitHub). The fix re-pads the list to the batch
max mel length and rebuilds the validity mask.

USAGE — run with the *serving* env's python (patches that env's vllm in place;
idempotent; exact-string match, aborts loudly if the installed code differs;
backs up the original to gemma4_mm.py.orig):

    <serving-env>/bin/python scripts/patch_vllm_gemma4.py

Verified on vllm 0.22.1 + transformers 5.13.0.
"""
import os
import shutil
import sys

import vllm  # noqa: F401  (locates the env's vllm)

PATH = os.path.join(os.path.dirname(vllm.__file__),
                    "model_executor", "models", "gemma4_mm.py")

OLD = """        input_features = audio_input["input_features_padded"].squeeze(1)
        input_features_mask = audio_input["input_features_mask"].squeeze(1)
"""

NEW = """        feats = audio_input["input_features_padded"]
        masks_in = audio_input["input_features_mask"]
        if isinstance(feats, (list, tuple)):
            # LOCAL PATCH (its_hub Run 14): a batch of different-length audios cannot
            # be stacked by the mm-kwargs batcher and arrives as a list of per-item
            # tensors; upstream assumes a stacked tensor and crashes. Re-pad to the
            # batch max length and rebuild the validity mask.
            feats = [f.reshape(-1, f.shape[-1]) for f in feats]
            masks_in = [m.reshape(-1) for m in masks_in]
            max_s = max(f.shape[0] for f in feats)
            f0 = feats[0]
            input_features = f0.new_zeros((len(feats), max_s, f0.shape[-1]))
            input_features_mask = torch.zeros(
                (len(feats), max_s), dtype=torch.bool, device=f0.device
            )
            for i, (f, m) in enumerate(zip(feats, masks_in)):
                input_features[i, : f.shape[0]] = f
                input_features_mask[i, : m.shape[0]] = m.to(torch.bool)
        else:
            input_features = feats.squeeze(1)
            input_features_mask = masks_in.squeeze(1)
"""

src = open(PATH).read()
if NEW.strip() in src:
    print(f"already patched — nothing to do ({PATH})")
    sys.exit(0)
n = src.count(OLD)
if n != 1:
    print(f"ERROR: expected exactly 1 match of the target block in {PATH}, found {n}; "
          "aborting (different vllm version? patch by hand or pin vllm==0.22.1)")
    sys.exit(1)
shutil.copyfile(PATH, PATH + ".orig")
open(PATH, "w").write(src.replace(OLD, NEW))
print(f"patched {PATH}\nbackup at {PATH}.orig")
