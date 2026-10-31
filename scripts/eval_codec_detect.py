#!/usr/bin/env python
"""Does the channel detector actually pick the right calibration bucket?

The thresholds in vericall/audio/codec_detect.py are asserted from physics --
band-limiting removes energy above 3.4 kHz, packet loss zeroes 20 ms frames.
This checks them against labelled audio rather than taking the physics on trust,
and prints the separation so a threshold sitting on a boundary is visible rather
than merely passing today.

The labels come free: extract_features.py cycles the channel profiles across
crops and stores a codec id per row, so a fold extracted with --crops 4 is an
already-labelled channel dataset. Note the label is CODEC ID, not profile name --
g711_mulaw and g711_alaw share id 1 by design, because they share a calibration
bucket.

Per-call numbers matter more than per-window ones: the detector is a call-level
accumulator, and a single window of a narrowband call can legitimately contain
no lost frames.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vericall.audio.codec_detect import (
    NARROWBAND_MAX_HIGH_RATIO, CodecDetector, classify_window,
    const_frame_ratio, high_band_ratio, noise_floor_ratio,
)
from vericall.data.cache import CachedFold

# codec id -> calibration bucket. Both G.711 variants share id 1 and one bucket.
ID_TO_BUCKET = {0: "clean", 1: "g711", 2: "noisy_line"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/vericall-cache")
    ap.add_argument("--dataset", default="asvspoof2019_la")
    ap.add_argument("--fold", default="dev")
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows")
    ap.add_argument("--windows-per-call", type=int, default=6)
    ap.add_argument("--out", default="docs/codec_detect.json")
    args = ap.parse_args()

    fold = CachedFold.load(Path(args.cache) / args.dataset / args.fold)
    n = len(fold) if not args.limit else min(args.limit, len(fold))
    codecs = np.asarray(fold.codec)[:n]
    present = sorted(set(codecs.tolist()))
    print(f"{args.dataset}:{args.fold}  {n:,} rows, codec ids present {present}")
    if len(present) < 2:
        print("\nThis fold was extracted with a single channel profile, so it "
              "cannot test the detector. Re-extract with --crops 4.",
              file=sys.stderr)
        return 2

    # ---------------------------------------------------------------- features
    hr = np.zeros(n); cf = np.zeros(n); nf = np.zeros(n)
    for i in range(n):
        w = torch.from_numpy(np.asarray(fold.wave[i], dtype=np.float32))
        hr[i] = high_band_ratio(w)
        cf[i] = const_frame_ratio(w)
        nf[i] = noise_floor_ratio(w)

    print(f"\n{'bucket':<12}{'n':>7}{'high-band':>22}{'const-frame':>20}"
          f"{'noise-floor':>20}")
    print(f"{'':12}{'':>7}{'p5 / med / p95':>22}{'med / p95':>20}{'p5 / med':>20}")
    print("-" * 81)
    stats = {}
    for cid in present:
        m = codecs == cid
        b = ID_TO_BUCKET.get(int(cid), str(cid))
        h, c, f = hr[m], cf[m], nf[m]
        print(f"{b:<12}{int(m.sum()):>7}"
              f"{np.percentile(h,5):>7.1e}/{np.median(h):>6.1e}/{np.percentile(h,95):>6.1e}"
              f"{np.median(c):>10.3f}/{np.percentile(c,95):>8.3f}"
              f"{np.percentile(f,5):>10.3f}/{np.median(f):>8.3f}")
        stats[b] = {"n": int(m.sum()),
                    "high_band_median": float(np.median(h)),
                    "const_frame_p95": float(np.percentile(c, 95)),
                    "noise_floor_median": float(np.median(f))}

    # The band-limit threshold must sit in a gap, not on a boundary -- and the
    # gap is between G.711 and CLEAN only. The noisy profile's added white noise
    # lifts it above clean, which is why classify_window tests noisy first.
    if 0 in present and 1 in present:
        clean_lo = float(np.percentile(hr[codecs == 0], 5))
        g711_hi = float(np.percentile(hr[codecs == 1], 95))
        t = NARROWBAND_MAX_HIGH_RATIO
        print(f"\nband-limit separation (g711 vs clean): g711 p95 = {g711_hi:.2e}, "
              f"clean p5 = {clean_lo:.2e}")
        print(f"  threshold {t:.2e}"
              + ("  [inside the gap]" if g711_hi < t < clean_lo
                 else "  [ON A BOUNDARY -- retune]"))

    # ------------------------------------------------------------- per window
    pred = np.array([classify_window(
        torch.from_numpy(np.asarray(fold.wave[i], dtype=np.float32))) for i in range(n)])
    true = np.array([ID_TO_BUCKET.get(int(c), "?") for c in codecs])
    acc = float((pred == true).mean())
    print(f"\nper-window accuracy: {acc*100:.2f}%")
    conf: dict = defaultdict(Counter)
    for t, p in zip(true, pred):
        conf[t][p] += 1
    buckets = ["clean", "g711", "noisy_line"]
    hdr = "true \\ pred"
    print("  " + hdr.ljust(14) + "".join(f"{b:>12}" for b in buckets))
    for t in buckets:
        if t in conf:
            print(f"  {t:<14}" + "".join(f"{conf[t][p]:>12}" for p in buckets))

    # --------------------------------------------------------------- per call
    # Group consecutive rows of the same codec into synthetic calls: a real call
    # has one channel throughout, which is the condition the accumulator is for.
    k = args.windows_per_call
    calls_ok = calls_n = 0
    per_bucket = defaultdict(lambda: [0, 0])
    for cid in present:
        # Rows of one codec are 4 apart, not adjacent: extract_features cycles
        # the four channel profiles across a clip's four crops. Grouping
        # adjacent ROWS would build a "call" that changes channel every window.
        idx = np.flatnonzero(codecs == cid)
        for s in range(0, len(idx) - k + 1, k):
            det = CodecDetector()
            for i in idx[s:s + k]:
                det.update(torch.from_numpy(np.asarray(fold.wave[i], dtype=np.float32)))
            b = ID_TO_BUCKET.get(int(cid), "?")
            calls_n += 1
            per_bucket[b][1] += 1
            if det.bucket == b:
                calls_ok += 1
                per_bucket[b][0] += 1
    print(f"\nper-call accuracy ({k} windows/call): "
          f"{calls_ok/max(calls_n,1)*100:.2f}%  ({calls_ok}/{calls_n})")
    for b, (ok, tot) in sorted(per_bucket.items()):
        print(f"  {b:<12}{ok}/{tot} = {ok/max(tot,1)*100:6.2f}%")

    out = {"per_window_accuracy": acc,
           "per_call_accuracy": calls_ok / max(calls_n, 1),
           "windows_per_call": k, "stats": stats,
           "per_bucket_call": {b: {"ok": v[0], "n": v[1]}
                               for b, v in per_bucket.items()}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
