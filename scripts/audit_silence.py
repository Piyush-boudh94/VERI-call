#!/usr/bin/env python
"""The silence-shortcut control experiment.

ASVspoof 2019 LA has a documented artifact: bona fide clips carry systematically
longer leading and trailing silence than spoofed ones. A classifier given ONLY
the leading-silence duration -- no speech content whatsoever -- reaches ~85%
accuracy / 15.1% EER on it.

So a detector trained naively on this corpus can post an excellent in-domain
number while having learned a stopwatch, and that is a large part of why
real-world error rates rise 200-1000% against in-domain ones.

This script measures the cue, quantifies how exploitable it is, and confirms the
trim removes it. Its output is the baseline Branch A must clearly beat.

**It must be re-run per corpus.** The baseline is a property of the corpus, not
a constant: quoting ASVspoof's 36.4% while reporting an EER measured on RTCFake
compares a model on one corpus against a stopwatch on another, which is not a
comparison at all. `--dataset` reads the manifest, so a plain-WAV corpus and a
parquet one take the same path.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vericall.data.audio_io import decode, trim_silence
from vericall.data.manifest import Manifest
from vericall.fusion.calibration import eer


def leading_trailing_silence(wav: torch.Tensor, sr: int, top_db: float = 35.0):
    """Seconds of silence before the first and after the last active frame."""
    frame, hop = int(0.025 * sr), int(0.010 * sr)
    if wav.numel() < frame:
        return 0.0, 0.0, wav.numel() / sr
    frames = wav.unfold(0, frame, hop)
    db = 20.0 * torch.log10(frames.pow(2).mean(-1).clamp(min=1e-12).sqrt())
    active = torch.nonzero(db > (db.max() - top_db), as_tuple=False).flatten()
    if active.numel() == 0:
        return 0.0, 0.0, wav.numel() / sr
    lead = int(active[0]) * hop / sr
    trail = (wav.numel() - (int(active[-1]) * hop + frame)) / sr
    return lead, max(0.0, trail), wav.numel() / sr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/tmp/vericall-data/asvspoof2019_la")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--dataset", default="",
                    help="read the manifest instead of a parquet shard. "
                         "Required for plain-WAV corpora (rtcfake).")
    ap.add_argument("--manifests", default="data/manifests")
    ap.add_argument("--orig-split-prefix", default="",
                    help="with --dataset, keep only items whose orig_split "
                         "starts with this (e.g. 'train:en' for RTCFake "
                         "English). The corpus's own split field is the only "
                         "language label RTCFake publishes.")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    per_class = args.n // 2

    def sample_manifest():
        """Stratified by class over the manifest. Works for either shard shape:
        parquet rows are read through pyarrow, plain files straight off disk."""
        import pyarrow.parquet as pq
        man = Manifest.load(Path(args.manifests) / f"{args.dataset}.jsonl")
        items = [i for i in man
                 if i.orig_split.startswith(args.orig_split_prefix)]
        if not items:
            raise SystemExit(f"no items with orig_split prefix "
                             f"{args.orig_split_prefix!r} in {args.dataset}")
        labels_arr = np.array([i.label for i in items])
        picked = []
        for cls in (0, 1):
            idx = np.flatnonzero(labels_arr == cls)
            if idx.size:
                picked += rng.choice(idx, size=min(per_class, idx.size),
                                     replace=False).tolist()
        by_shard: dict[str, list] = {}
        for k in picked:
            by_shard.setdefault(items[k].shard, []).append(items[k])
        for shard, rows in by_shard.items():
            if shard.endswith(".parquet"):
                pf = pq.ParquetFile(shard)
                wanted = {r.row: r for r in rows}
                offset = 0
                for rg in range(pf.num_row_groups):
                    tbl = pf.read_row_group(rg, columns=["audio"]).to_pydict()["audio"]
                    for i in range(len(tbl)):
                        if (offset + i) in wanted:
                            yield tbl[i]["bytes"], wanted[offset + i].label
                    offset += len(tbl)
            else:
                for r in rows:
                    yield Path(r.shard).read_bytes(), r.label

    def sample_parquet():
        import pyarrow.parquet as pq
        shard = sorted(glob.glob(f"{args.root}/data/{args.split}*.parquet"))
        if not shard:
            raise SystemExit(f"no shards under {args.root}")
        pf = pq.ParquetFile(shard[0])
        # The shard is class-ordered, so a sequential head would be entirely
        # bona fide. Sample stratified across the whole file.
        keys = np.array(pq.read_table(shard[0], columns=["key"]).to_pydict()["key"])
        wanted = set()
        for cls in (0, 1):
            idx = np.flatnonzero(keys == cls)
            wanted |= set(rng.choice(idx, size=min(per_class, idx.size),
                                     replace=False).tolist())
        offset = 0
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=["audio", "key"]).to_pydict()
            n_rows = len(t["key"])
            for i in range(n_rows):
                if (offset + i) in wanted:
                    yield t["audio"][i]["bytes"], int(t["key"][i])
            offset += n_rows

    leads, trails, durs, labels, trimmed = [], [], [], [], []
    seen = 0
    for raw, key in (sample_manifest() if args.dataset else sample_parquet()):
        wav, sr = decode(raw)
        lead, trail, dur = leading_trailing_silence(wav, sr)
        leads.append(lead); trails.append(trail); durs.append(dur)
        labels.append(int(key))
        l2, _t2, _ = leading_trailing_silence(trim_silence(wav, sr), sr)
        trimmed.append(l2)
        seen += 1

    leads = np.array(leads); trails = np.array(trails)
    durs = np.array(durs); y = np.array(labels); trimmed = np.array(trimmed)
    bona, spoof = y == 0, y == 1

    corpus = args.dataset or "asvspoof2019_la"
    scope = args.orig_split_prefix or args.split
    print(f"{corpus} [{scope}], n={seen} "
          f"({bona.sum()} bona fide / {spoof.sum()} spoof)\n")
    print(f"{'cue':<26}{'bona fide':>12}{'spoof':>12}{'gap':>10}")
    for name, arr in (("leading silence (s)", leads),
                      ("trailing silence (s)", trails),
                      ("total duration (s)", durs)):
        b, s = arr[bona].mean(), arr[spoof].mean()
        print(f"{name:<26}{b:>12.3f}{s:>12.3f}{b - s:>+10.3f}")

    print("\n--- how exploitable is it? (higher score = predicted spoof) ---")
    for name, arr in (("leading silence alone", -leads),
                      ("trailing silence alone", -trails),
                      ("total duration alone", -durs)):
        e, _ = eer(arr, y)
        print(f"  {name:<26} EER = {e*100:5.2f}%")

    e_trim, _ = eer(-trimmed, y)
    print(f"\n  {'AFTER TRIMMING':<26} EER = {e_trim*100:5.2f}%   "
          f"<- the cue must be gone")
    print(f"  mean leading silence after trim: bona fide "
          f"{trimmed[bona].mean():.4f}s vs spoof {trimmed[spoof].mean():.4f}s")

    e_lead, _ = eer(-leads, y)
    print("\nVERDICT:")
    print(f"  A duration-only classifier scores {e_lead*100:.1f}% EER on raw audio.")
    print(f"  Branch A must beat this by a wide margin or it has learned nothing")
    print(f"  transferable. After trimming the same cue gives {e_trim*100:.1f}% EER.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
