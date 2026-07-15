"""Run 17 prompt-token audit: how badly does Mellow's fixed text window truncate?

Mellow tokenizes the plain-text prompt to a FIXED ``text_tokenization_len`` (129
SmolLM2 tokens, '!'-padded, pads attended). Any longer prompt is silently truncated —
for MCQ that can eat the options entirely. This audit measures the token-length
distribution of every prompt method (the shim flattens system+user text with "\n\n",
exactly like ``serve_mellow.flatten_messages``) over a subset, reporting p50/p90/max
and %>129 / %>N per method. This is the fixed-window analog of SETUP_GUIDE §11 step 6.

Usage:
    python benchmarking/mmau_pro/run17/audit_prompt_tokens.py \
        --data-root /home/exx/inference-time-scaling/mmau_pro_testmini \
        --audio-root /home/exx/inference-time-scaling/mmau_pro_audio \
        --subset test --methods 1,2,3,4,5,6,7,8,9,10,11
"""

import argparse
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _REPO_ROOT)

from benchmarking.mmau_pro.loader import load_mmau_mcq  # noqa: E402
from benchmarking.mmau_pro.prompt import METHODS, build  # noqa: E402
from benchmarking.mmau_pro.serve_mellow import (  # noqa: E402
    SMOLLM2_REVISION,
    flatten_messages,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/home/exx/inference-time-scaling/mmau_pro_testmini")
    ap.add_argument("--audio-root", default="/home/exx/inference-time-scaling/mmau_pro_audio")
    ap.add_argument("--subset", default="test")
    ap.add_argument("--methods", default=",".join(str(m) for m in METHODS))
    ap.add_argument("--faithful-len", type=int, default=129)
    ap.add_argument("--extended-len", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import numpy as np
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        snapshot_download("HuggingFaceTB/SmolLM2-135M", revision=SMOLLM2_REVISION)
    )

    recs = load_mmau_mcq(args.data_root, subset=args.subset, limit=args.limit,
                         audio_root=args.audio_root)
    recs = [r for r in recs if len(r.audio_paths) <= 2]  # run17 policy: 3-audio excluded
    print(f"subset={args.subset}: {len(recs)} items (<=2 audios)\n")
    print(f"{'method':38s} {'p50':>5s} {'p90':>5s} {'max':>5s} "
          f"{'>'+str(args.faithful_len):>6s} {'>'+str(args.extended_len):>6s}")
    for m in [int(x) for x in args.methods.split(",")]:
        lens = []
        for r in recs:
            msgs, _ = build(m, r, audio_mode="local-path")
            text, _ = flatten_messages([msg.to_dict() for msg in msgs])
            lens.append(len(tok(text, add_special_tokens=True)["input_ids"]))
        a = np.array(lens)
        print(f"P{m:<2d} {METHODS[m]:34s} {int(np.percentile(a,50)):5d} "
              f"{int(np.percentile(a,90)):5d} {a.max():5d} "
              f"{(a > args.faithful_len).mean():6.1%} {(a > args.extended_len).mean():6.1%}")


if __name__ == "__main__":
    main()
