#!/usr/bin/env python
"""Verify the Kaggle In-the-Wild mirror against the published release.

The published corpus (Muller et al. 2022) is 31,779 clips from 58 English-
speaking public figures: 19,963 bona fide and 11,816 spoofed, ~37.9 hours.
A mirror that has been re-encoded, subsetted or re-labelled would invalidate
every cross-domain number computed from it, so this is checked rather than
assumed.
"""
from __future__ import annotations

import collections, io, sys, zipfile
from pathlib import Path

Z = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/vericall-data/in_the_wild.zip")

with zipfile.ZipFile(Z) as zf:
    names = zf.namelist()
    wavs = [n for n in names if n.lower().endswith(".wav")]
    meta = [n for n in names if n.lower().endswith((".csv", ".txt", ".json"))]
    print(f"archive: {Z}  ({Z.stat().st_size/2**30:.2f} GB)")
    print(f"  entries: {len(names)}   wav: {len(wavs)}   metadata: {len(meta)}")
    print(f"  top-level: {sorted({n.split('/')[0] for n in names})[:6]}")

    if meta:
        m = meta[0]
        raw = zf.read(m).decode("utf-8", "replace").splitlines()
        print(f"\n  metadata file: {m}  ({len(raw)} lines)")
        print(f"    header: {raw[0][:110]}")
        for line in raw[1:3]:
            print(f"    row:    {line[:110]}")
        if len(raw) > 1 and "," in raw[0]:
            cols = [c.strip().strip('"') for c in raw[0].split(",")]
            label_col = next(
                (i for i, c in enumerate(cols)
                 if c.lower() in ("label", "class", "type", "target")), None)
            if label_col is not None:
                ctr = collections.Counter(
                    r.split(",")[label_col].strip().strip('"')
                    for r in raw[1:] if r.count(",") >= label_col
                )
                print(f"\n  label distribution ({cols[label_col]}):")
                for k, v in ctr.most_common(6):
                    print(f"    {k:<14} {v:>7}")

    # audio format of a sample
    if wavs:
        import struct
        b = zf.read(wavs[0])
        i = 12
        while i < len(b) - 8:
            cid, sz = b[i:i+4], struct.unpack("<I", b[i+4:i+8])[0]
            if cid == b"fmt ":
                ch, sr, _, _, bits = struct.unpack("<HIIHH", b[i+10:i+24])
                print(f"\n  sample audio {wavs[0].split('/')[-1]}: "
                      f"{sr} Hz, {ch}ch, {bits}-bit, "
                      f"{'PCM lossless' if b[:4]==b'RIFF' else '?'}")
                break
            i += 8 + sz + (sz & 1)

EXPECTED = 31779
print(f"\nEXPECTED (published): {EXPECTED} clips "
      f"= 19,963 bona fide + 11,816 spoof")
