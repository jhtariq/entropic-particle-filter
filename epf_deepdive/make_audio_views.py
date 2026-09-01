"""Build answer-preserving audio views (TTA) for MMAR items.

Views (all preserve pitch, tempo, duration relations and content):
  noise   additive white noise at 27 dB SNR (seed 1234)
  noise2  additive white noise at 27 dB SNR (seed 5678 — seed-stability test)
  snr33   additive white noise at 33 dB SNR (gentler)
  snr21   additive white noise at 21 dB SNR (stronger)
  eq      band tilt: 2nd-order highpass 120 Hz + 4th-order lowpass 7.5 kHz
  shift   0.3 s of silence prepended (temporal offset)

Per-file RNG is seeded from (view, relpath) so results are deterministic and
order-independent (safe under --workers). Existing outputs are skipped, so the
script is idempotent and resume-safe. On per-file failure the original is
copied through so item counts hold.

  python make_audio_views.py --uids full996_uids.txt --views noise,noise2,eq,shift --workers 16
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

SRC_ROOT = "/u/awaheed/epf_data/mmar"
OUT_ROOT = "/work/hdd/bcey/awaheed/its-for-audio-reasoning/epf_data/deepdive_views"


def _rng(view, rel, base_seed):
    h = hashlib.sha1(f"{view}:{rel}".encode()).digest()
    return np.random.default_rng(base_seed + int.from_bytes(h[:4], "little"))


def add_noise(x, sr, rng, snr_db):
    rms = np.sqrt(np.mean(x**2)) or 1e-6
    n = rng.standard_normal(x.shape).astype(x.dtype)
    n *= (rms / (np.sqrt(np.mean(n**2)) or 1e-6)) * 10 ** (-snr_db / 20)
    return np.clip(x + n, -1.0, 1.0)


def v_eq(x, sr, rng):
    sos = butter(2, 120, "highpass", fs=sr, output="sos")
    y = sosfiltfilt(sos, x, axis=0)
    if sr / 2 > 7500:
        y = sosfiltfilt(butter(4, 7500, "lowpass", fs=sr, output="sos"), y, axis=0)
    return np.clip(y, -1.0, 1.0).astype(x.dtype)


def v_shift(x, sr, rng):
    pad = np.zeros((int(0.3 * sr),) + x.shape[1:], dtype=x.dtype)
    return np.concatenate([pad, x], axis=0)


VIEWS = {
    "noise":  (lambda x, sr, rng: add_noise(x, sr, rng, 27), 1234),
    "noise2": (lambda x, sr, rng: add_noise(x, sr, rng, 27), 5678),
    "snr33":  (lambda x, sr, rng: add_noise(x, sr, rng, 33), 1234),
    "snr21":  (lambda x, sr, rng: add_noise(x, sr, rng, 21), 1234),
    "eq":     (v_eq, 0),
    "shift":  (v_shift, 0),
}


def one(job):
    view, rel = job
    src = os.path.join(SRC_ROOT, rel)
    dst = os.path.join(OUT_ROOT, view, rel)
    if os.path.exists(dst):
        return "skip"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    fn, seed = VIEWS[view]
    try:
        x, sr = sf.read(src, dtype="float32", always_2d=False)
        sf.write(dst, fn(x, sr, _rng(view, rel, seed)), sr, subtype="PCM_16")
        return "done"
    except Exception as e:
        print(f"  FALLBACK copy {view}/{rel}: {type(e).__name__}: {e}", file=sys.stderr)
        shutil.copyfile(src, dst)
        return "fallback"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uids", required=True)
    ap.add_argument("--views", required=True, help="comma-separated view names")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    views = a.views.split(",")
    for v in views:
        if v not in VIEWS:
            raise SystemExit(f"unknown view {v!r}; have {list(VIEWS)}")
    uids = set(open(a.uids).read().split())
    meta = json.load(open(os.path.join(SRC_ROOT, "MMAR-meta.json")))
    rels = [str(r["audio_path"]) for r in meta
            if str(r.get("id")) in uids and r.get("audio_path")]
    jobs = [(v, rel) for v in views for rel in rels]
    print(f"{len(rels)} items x {len(views)} views = {len(jobs)} jobs "
          f"({a.workers} workers)", flush=True)
    from collections import Counter
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        c = Counter(ex.map(one, jobs, chunksize=16))
    print(f"done={c.get('done',0)} fallback={c.get('fallback',0)} skipped={c.get('skip',0)}")


if __name__ == "__main__":
    main()
