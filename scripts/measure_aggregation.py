#!/usr/bin/env python
"""Measure whether the n-of-m binomial error model holds on real VeriCall scores.

The 3-of-5 aggregation scheme borrowed from cryptographic call authentication
computes its combined false-positive rate from the binomial distribution over
Bernoulli trials. This script tests that independence assumption against real
per-chunk score streams from bona fide ASVspoof calls.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vericall.audio.chunker import ChunkerConfig, DualRateChunker
from vericall.branches.a_ssl import BranchA, BranchAConfig, BranchAHead
from vericall.data.audio_io import load_audio
from vericall.data.manifest import Manifest
from vericall.fusion.calibration import load_calibration, logit
from vericall.risk.aggregation import evaluate_aggregation


def collect_streams(model, head, cal, manifest, plan, fold, device,
                    n_calls, clips_per_call, label):
    import pyarrow.parquet as pq
    groups = defaultdict(list)
    for it in manifest:
        if fold in plan.get(it.utt_id, []) and it.label == label:
            groups[(it.speaker_id, it.attack_id)].append(it)

    runs = []
    for key, clips in sorted(groups.items(), key=lambda kv: str(kv[0])):
        for i in range(0, len(clips) - clips_per_call + 1, clips_per_call):
            runs.append(clips[i:i + clips_per_call])
            if len(runs) >= n_calls:
                break
        if len(runs) >= n_calls:
            break

    need = defaultdict(set)
    for run in runs:
        for it in run:
            need[it.shard].add(it.row)
    cache = {}
    for shard, rows in need.items():
        pf = pq.ParquetFile(shard)
        off = 0
        for rg in range(pf.num_row_groups):
            n = pf.metadata.row_group(rg).num_rows
            hits = [r for r in range(off, off + n) if r in rows]
            if hits:
                tbl = pf.read_row_group(rg, columns=["audio"]).to_pydict()["audio"]
                for r in hits:
                    cache[(shard, r)] = load_audio(tbl[r - off]["bytes"])
            off += n

    streams = []
    hop = int(ChunkerConfig().tick_s * 16000)
    with torch.inference_mode():
        for run in runs:
            parts = [cache.get((it.shard, it.row)) for it in run]
            parts = [p for p in parts if p is not None and p.numel() > 0]
            if len(parts) < 4:
                continue
            wav = torch.cat(parts)
            ch = DualRateChunker("c", ChunkerConfig())
            wins = []
            for i in range(0, wav.numel(), hop):
                wins += [w for w in ch.push(wav[i:i + hop])
                         if w.lane == "fast" and w.quality > 0.15]
            if len(wins) < 6:
                continue
            batch = torch.stack([w.samples for w in wins]).to(device)
            p = torch.softmax(head(model.frontend(batch)), dim=-1)[:, 1]
            streams.append(cal.apply(logit(p.float().cpu().numpy())))
    return streams


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/branch_A_seed0.pt")
    ap.add_argument("--manifest", default="data/manifests/asvspoof2019_la.jsonl")
    ap.add_argument("--splits", default="data/splits/asvspoof2019_la.json")
    ap.add_argument("--calib", default="calib/A@clean.json")
    ap.add_argument("--summary", default="calib/summary.json")
    ap.add_argument("--calls", type=int, default=120)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(args.checkpoint, map_location=device, weights_only=False)
    head = BranchAHead(768, BranchAConfig()).to(device)
    head.load_state_dict(blob["state_dict"]); head.eval()
    model = BranchA(BranchAConfig()).to(device).eval()
    cal = load_calibration(args.calib)
    eta = json.loads(Path(args.summary).read_text())["eta"]["A"]

    manifest = Manifest.load(args.manifest)
    plan = json.loads(Path(args.splits).read_text())
    print(f"collecting bona fide chunk streams (eta = {eta:.3f})...")
    streams = collect_streams(model, head, cal, manifest, plan, "dev",
                              device, args.calls, 8, label=0)
    lens = [len(s) for s in streams]
    print(f"  {len(streams)} calls, {sum(lens)} chunks, "
          f"{np.mean(lens):.0f} chunks/call\n")

    print("Testing the binomial independence assumption on real scores:")
    print("-" * 104)
    rows = []
    for thr in (0.0, 1.0, 2.0, 3.0):
        for n, m in ((3, 5), (2, 5), (4, 5), (5, 5)):
            try:
                rep = evaluate_aggregation(streams, thr, n, m, eta)
            except ValueError:
                continue
            if rep.chunk_fp_rate == 0.0:
                continue
            rows.append((thr, rep))
            print(f"  thr={thr:>4.1f} nats  {rep.line()}")
    print("-" * 104)
    if rows:
        # Only rows where the event actually occurred are informative. Where
        # empirical == 0 the vote simply never fired in this sample -- that is a
        # sample-size limit, not evidence the binomial model is correct, and
        # averaging those zeros in would understate the discrepancy badly.
        informative = [(t, r) for t, r in rows
                       if r.empirical_call_fp > 0 and r.predicted_call_fp_binomial > 0]
        skipped = len(rows) - len(informative)
        if informative:
            infl = [r.inflation for _, r in informative]
            print(f"\n  binomial optimism over {len(informative)} informative "
                  f"operating points: median {np.median(infl):.1f}x, "
                  f"range {min(infl):.1f}-{max(infl):.1f}x")
            three = [r.inflation for _, r in informative if (r.n, r.m) == (3, 5)]
            if three:
                print(f"  for the proposed 3-of-5 specifically: "
                      f"{min(three):.1f}-{max(three):.1f}x optimistic")
        if skipped:
            print(f"  ({skipped} operating points excluded: the vote never fired "
                  f"in {sum(len(s) for s in streams)} chunks -- no data, not agreement)")
        print(f"  5 correlated chunks are worth {5*eta:.2f} independent "
              f"observations at the measured eta")
    Path("docs").mkdir(exist_ok=True)
    Path("docs/aggregation.json").write_text(json.dumps(
        [{"threshold": t, **r.__dict__} for t, r in rows], indent=2))
    print("\nwrote docs/aggregation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
