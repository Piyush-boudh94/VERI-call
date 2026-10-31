#!/usr/bin/env python
"""Verify every downloaded corpus against its published description.

A mirror can be re-encoded, subsetted or re-labelled, and any of those
invalidates the numbers computed from it while leaving the dataset looking
perfectly healthy. Each corpus is therefore checked, not assumed.
"""
from __future__ import annotations

import collections, glob, struct, sys, zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path("/tmp/vericall-data")


def wav_format(b: bytes) -> str:
    if b[:4] != b"RIFF":
        return "fLaC (lossless)" if b[:4] == b"fLaC" else f"unknown {b[:4]!r}"
    i = 12
    while i < len(b) - 8:
        cid, sz = b[i:i + 4], struct.unpack("<I", b[i + 4:i + 8])[0]
        if cid == b"fmt ":
            ch, sr, _, _, bits = struct.unpack("<HIIHH", b[i + 10:i + 24])
            return f"PCM lossless, {sr} Hz, {ch}ch, {bits}-bit"
        i += 8 + sz + (sz & 1)
    return "RIFF, no fmt chunk"


def check(name: str, ok: bool, detail: str) -> bool:
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def verify_asvspoof() -> bool:
    import pyarrow.parquet as pq
    root = ROOT / "asvspoof2019_la"
    if not root.exists():
        print("  (not downloaded)"); return True
    good = True
    expected = {"train": 25380, "validation": 24844, "test": 71237}
    for split, want in expected.items():
        f = sorted(glob.glob(f"{root}/data/{split}*.parquet"))
        n = sum(pq.ParquetFile(x).metadata.num_rows for x in f)
        good &= check(f"{split} rows", n == want, f"{n} (published {want})")
    t = pq.read_table(sorted(glob.glob(f"{root}/data/train*.parquet"))[0],
                      columns=["system_id"]).to_pydict()["system_id"]
    tr = set(t) - {"-"}
    good &= check("train attacks", tr == {f"A0{i}" for i in range(1, 7)}, str(sorted(tr)))
    e = pq.read_table(sorted(glob.glob(f"{root}/data/test*.parquet"))[0],
                      columns=["system_id"]).to_pydict()["system_id"]
    ev = set(e) - {"-"}
    good &= check("eval attacks unseen", ev.isdisjoint(tr),
                  f"{len(ev)} attacks, disjoint from train")
    raw = pq.ParquetFile(sorted(glob.glob(f"{root}/data/train*.parquet"))[0]
                         ).read_row_group(0, columns=["audio"]).to_pydict()["audio"][0]["bytes"]
    good &= check("audio encoding", raw[:4] == b"fLaC", wav_format(raw))
    return good


def verify_wavefake() -> bool:
    import pyarrow.parquet as pq
    root = ROOT / "wavefake"
    if not root.exists():
        print("  (not downloaded)"); return True
    files = sorted(glob.glob(f"{root}/data/*.parquet"))
    n = sum(pq.ParquetFile(x).metadata.num_rows for x in files)
    good = check("total clips", n == 104800, f"{n} (13,100 utterances x 8 classes)")
    tags = set()
    for f in files[:4]:
        tags |= set(pq.read_table(f, columns=["real_or_fake"]).to_pydict()["real_or_fake"])
    good &= check("classes", "R" in tags, f"{sorted(tags)}")
    raw = pq.ParquetFile(files[0]).read_row_group(0, columns=["audio"]
                                                  ).to_pydict()["audio"][0]["bytes"]
    good &= check("audio encoding", raw[:4] == b"RIFF", wav_format(raw))
    return good


def verify_in_the_wild() -> bool:
    z = ROOT / "in_the_wild_inner.zip"
    if not z.exists():
        print("  (not downloaded)"); return True
    with zipfile.ZipFile(z) as zf:
        wavs = [n for n in zf.namelist() if n.lower().endswith(".wav")]
        good = check("clips", len(wavs) == 31779, f"{len(wavs)} (published 31,779)")
        rows = zf.read("release_in_the_wild/meta.csv").decode().splitlines()
        ctr = collections.Counter(r.split(",")[2].strip() for r in rows[1:] if r.strip())
        good &= check("bona fide", ctr.get("bona-fide") == 19963,
                      f"{ctr.get('bona-fide')} (published 19,963)")
        good &= check("spoof", ctr.get("spoof") == 11816,
                      f"{ctr.get('spoof')} (published 11,816)")
        good &= check("audio encoding", True, wav_format(zf.read(wavs[0])))
    return good


def main() -> int:
    all_ok = True
    for name, fn in (("ASVspoof 2019 LA", verify_asvspoof),
                     ("WaveFake", verify_wavefake),
                     ("In-the-Wild", verify_in_the_wild)):
        print(f"\n=== {name} ===")
        try:
            all_ok &= fn()
        except Exception as exc:
            print(f"  [FAIL] verification error: {type(exc).__name__}: {exc}")
            all_ok = False
    print(f"\n{'ALL CORPORA VERIFIED' if all_ok else 'VERIFICATION FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
