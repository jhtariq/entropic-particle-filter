"""Run 17 shim validation (GPU, no server): reference parity + audio-path checks.

Validates `serve_mellow.MellowEngine` against the reference `MellowWrapper` from the
pinned Mellow clone:

  1. PARITY   — greedy output identity vs `MellowWrapper.generate` on single-audio
                items whose clip is <= 10 s (the deterministic path: no crop involved,
                both audio slots fed the same clip so fill policy is out of the picture).
  2. CROP     — deterministic first-10s crop on a > 10 s clip (two decodes identical,
                equal to the first segment_samples of the full decode).
  3. MP3      — decode path for mp3 clips (testmini has 11, the full test set 50).
  4. FILL A/B — greedy answers + accuracy on single-audio MCQ items with the empty
                second slot filled by silence vs a duplicate of the first clip
                (informs the --single-audio-fill default).

Usage:
    python benchmarking/mmau_pro/run17/validate_shim.py \
        --code-dir /home/exx/inference-time-scaling/mellow_code \
        --data-root /home/exx/inference-time-scaling/mmau_pro_testmini
"""

import argparse
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _REPO_ROOT)

from benchmarking.mmau_pro.loader import load_mmau_mcq  # noqa: E402
from benchmarking.mmau_pro.prompt import format_choices  # noqa: E402
from benchmarking.mmau_pro.scoring import predicted_index  # noqa: E402
from benchmarking.mmau_pro.serve_mellow import (  # noqa: E402
    AudioRef,
    MellowEngine,
    ParsedRequest,
)


def _duration_s(path: str) -> float:
    import soundfile as sf

    info = sf.info(path)
    return info.frames / info.samplerate


def _flattened_len_32k(path: str) -> int:
    """Approximate flattened sample count after the reference preprocessing: channels
    are concatenated end-to-end (reshape(-1)) and resampled to 32 kHz. Clips whose
    flattened length exceeds the 10 s window hit the reference's RANDOM crop branch
    (nondeterministic — e.g. any stereo clip > 5 s), so parity is only defined below it."""
    import soundfile as sf

    info = sf.info(path)
    return int(info.frames * info.channels * 32000 / info.samplerate)


def _mcq_text(rec) -> str:
    return (
        f"{rec.question}\n\nOptions:\n{format_choices(rec.choices)}\n\n"
        "Answer with the letter of the correct option."
    )


def _greedy_req(text: str, paths: list[str], max_tokens: int) -> ParsedRequest:
    return ParsedRequest(
        prompt_text=text,
        continuation="",
        audio_refs=[AudioRef(kind="file", path=p) for p in paths],
        stop=[],
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=None,
        logprobs=True,
        top_logprobs=20,
        include_stop_str=False,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-dir", required=True)
    ap.add_argument("--data-root", default="/home/exx/inference-time-scaling/mmau_pro_testmini")
    ap.add_argument("--audio-root", default="/home/exx/inference-time-scaling/mmau_pro_audio")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--variant", default="v0")
    ap.add_argument("--n-parity", type=int, default=3)
    ap.add_argument("--n-fill", type=int, default=15)
    ap.add_argument("--max-new", type=int, default=60)
    ap.add_argument("--no-kv-cache", action="store_true",
                    help="run the shim in full-reforward mode (bit-path-identical to the reference)")
    args = ap.parse_args()

    print(f"== loading shim engine (faithful mode, kv_cache={not args.no_kv_cache}) ==")
    engine = MellowEngine(code_dir=args.code_dir, variant=args.variant, device=args.device,
                          use_kv_cache=not args.no_kv_cache)
    engine.load()

    print("== loading reference MellowWrapper ==")
    sys.path.insert(0, args.code_dir)

    # torchaudio >= 2.9 delegates decode to torchcodec, which is not installed on this
    # stack; route torchaudio.load through soundfile (same normalized float32 layout)
    # so the reference wrapper's load_audio_into_tensor works.
    import soundfile as _sf
    import torch as _torch
    import torchaudio as _ta

    def _sf_load(path, *a, **kw):
        data, sr = _sf.read(path, dtype="float32", always_2d=True)
        return _torch.from_numpy(data.T), sr

    _ta.load = _sf_load

    from mellow import MellowWrapper

    dev = args.device.split(":")[-1]
    reference = MellowWrapper(config="v0", model=args.variant,
                              device=int(dev) if dev.isdigit() else 0, use_cuda=True)

    # transformers >= 4.57 removed the deprecated pad_to_max_length kwarg the reference
    # wrapper still uses; translate it to the equivalent padding="max_length"
    _orig_encode_plus = reference.tokenizer.encode_plus

    def _encode_plus_compat(*a, **kw):
        if kw.pop("pad_to_max_length", False):
            kw["padding"] = "max_length"
        return _orig_encode_plus(*a, **kw)

    reference.tokenizer.encode_plus = _encode_plus_compat

    records = load_mmau_mcq(args.data_root, subset="full", audio_root=args.audio_root)
    single = [r for r in records if len(r.audio_paths) == 1]

    # ---- 1. greedy parity on clips where the reference path is deterministic ----------
    short = [r for r in single
             if _flattened_len_32k(r.audio_paths[0]) <= engine.segment_samples - 8]
    print(f"\n== 1. PARITY on {args.n_parity} of {len(short)} deterministic-reference items ==")
    failures = 0
    import time as _time
    t0 = _time.time()
    n_tok = 0
    for rec in short[: args.n_parity]:
        text = _mcq_text(rec)
        p = rec.audio_paths[0]
        ref_out = reference.generate(
            examples=[[p, p, text]], max_len=args.max_new, top_p=0.8, temperature=1.0
        )[0]
        shim = engine.generate(_greedy_req(text, [p, p], args.max_new))[0]
        match = shim["content"] == ref_out
        failures += 0 if match else 1
        n_tok += shim["completion_tokens"]
        lp = shim["logprob_entries"]
        print(f"  {rec.unique_id}: match={match} tokens={shim['completion_tokens']} "
              f"prompt_tokens={shim['prompt_tokens']} "
              f"first_lp={lp[0]['logprob']:.4f} top={len(lp[0].get('top_logprobs') or [])}")
        if not match:
            print(f"    ref : {ref_out!r}")
            print(f"    shim: {shim['content']!r}")
    dt = _time.time() - t0
    print(f"  shim+ref wall {dt:.1f}s for {args.n_parity} items ({n_tok} shim tokens)")

    # ---- 1b. batched-n consistency: greedy n=8 rows must all equal the n=1 output ----
    batch_ok = True
    if short:
        rec = short[0]
        text = _mcq_text(rec)
        p = rec.audio_paths[0]
        one = engine.generate(_greedy_req(text, [p, p], args.max_new))[0]["content"]
        req8 = _greedy_req(text, [p, p], args.max_new)
        req8.n = 8
        eight = [r["content"] for r in engine.generate(req8)]
        batch_ok = len(eight) == 8 and all(t == one for t in eight)
        print(f"\n== 1b. BATCH n=8 greedy consistency: {'PASS' if batch_ok else 'FAIL'} ==")
        if not batch_ok:
            print(f"    n=1 : {one!r}")
            for k, t in enumerate(eight):
                if t != one:
                    print(f"    n8[{k}]: {t!r}")

    # ---- 2. crop determinism on a >10 s clip ------------------------------------------
    long_rec = next(r for r in single if _duration_s(r.audio_paths[0]) > 10.0)
    lp_path = long_rec.audio_paths[0]
    print(f"\n== 2. CROP determinism on {os.path.basename(lp_path)} "
          f"({_duration_s(lp_path):.1f}s) ==")
    import numpy as np

    ref = AudioRef(kind="file", path=lp_path)
    full = engine._decode_audio(ref)
    w1 = engine._waveform(ref)
    engine._wav_cache.clear()
    engine._wav_cache_order.clear()
    w2 = engine._waveform(ref)
    crop_ok = (
        np.array_equal(w1, w2)
        and w1.shape[0] == engine.segment_samples
        and np.array_equal(w1, full[: engine.segment_samples])
    )
    print(f"  deterministic first-10s crop: {'PASS' if crop_ok else 'FAIL'}")

    # ---- 3. mp3 decode -----------------------------------------------------------------
    mp3 = next((p for r in records for p in r.audio_paths if p.lower().endswith(".mp3")), None)
    print("\n== 3. MP3 decode ==")
    if mp3 is None:
        print("  no mp3 in this subset -> SKIP")
        mp3_ok = True
    else:
        wav = engine._decode_audio(AudioRef(kind="file", path=mp3))
        mp3_ok = wav.shape[0] > 0
        print(f"  {os.path.basename(mp3)}: {wav.shape[0]} samples @ {engine.sample_rate} Hz "
              f"-> {'PASS' if mp3_ok else 'FAIL'}")

    # ---- 4. single-audio fill A/B ------------------------------------------------------
    print(f"\n== 4. FILL A/B (silence vs duplicate) on {args.n_fill} single-audio items ==")
    rows = []
    for rec in single[: args.n_fill]:
        text = _mcq_text(rec)
        p = rec.audio_paths[0]
        answers = {}
        for fill in ("silence", "duplicate"):
            engine.single_audio_fill = fill
            out = engine.generate(_greedy_req(text, [p], args.max_new))[0]
            answers[fill] = (out["content"], predicted_index(out["content"], rec.choices))
        rows.append((rec, answers))
    engine.single_audio_fill = "silence"

    def _acc(fill):
        good = [r for r, a in rows
                if r.answer_index is not None and a[fill][1] == r.answer_index]
        graded = [r for r, _ in rows if r.answer_index is not None]
        return len(good), len(graded)

    diff = sum(1 for _, a in rows if a["silence"][1] != a["duplicate"][1])
    parse_s = sum(1 for _, a in rows if a["silence"][1] is not None)
    parse_d = sum(1 for _, a in rows if a["duplicate"][1] is not None)
    gs, n = _acc("silence")
    gd, _ = _acc("duplicate")
    print(f"  silence : acc {gs}/{n}, parsed {parse_s}/{len(rows)}")
    print(f"  duplicate: acc {gd}/{n}, parsed {parse_d}/{len(rows)}")
    print(f"  answers differing between fills: {diff}/{len(rows)}")
    for rec, a in rows[:5]:
        print(f"    {rec.unique_id}: silence={a['silence'][0][:60]!r} | "
              f"duplicate={a['duplicate'][0][:60]!r}")

    ok = failures == 0 and crop_ok and mp3_ok and batch_ok
    print(f"\nRESULT: parity_failures={failures} batch={'ok' if batch_ok else 'FAIL'} "
          f"crop={'ok' if crop_ok else 'FAIL'} "
          f"mp3={'ok' if mp3_ok else 'FAIL'} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
