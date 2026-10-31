#!/usr/bin/env python
"""Measure the effective-sample-size deflation for EVERY branch, not just A.

`FusionConfig.eta` ships {"A": 0.6, "A2": 0.6, "B": 0.9}. Only A's has ever been
measured, and the measurement came back **0.080** -- roughly 7x away from its
default, in the direction that makes confidence bounds too tight. Carrying
unmeasured defaults for the other two after that is not a conservative choice;
it is an untested assumption with a known track record of being wrong.

The mechanism eta has to capture is call-level consistency, not window overlap.
Overlap couples adjacent windows only: a 1 s window at a 0.5 s hop shares no
samples at all with its neighbour twice removed. But one speaker, one channel
and one generator for a whole call couples windows at every lag, and that term
is the larger of the two. So the sequences here are built the same way
fit_calibration.py builds them -- consecutive clips from the SAME speaker with
the SAME label and generator, concatenated into a ~15 s stream -- and each
branch is scored on the lane it actually runs on:

    A, A2   fast lane, 1.0 s window / 0.5 s hop
    B       slow lane, 3.0 s window / 1.0 s hop

Scoring B on the fast lane would measure a configuration that never runs, and
its longer window and slower hop are exactly the things that change how
correlated its outputs are.

Caveat carried over unchanged: the calls are concatenated clips, so splice
points add noise a real call would not have.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vericall.audio.chunker import ChunkerConfig, DualRateChunker
from vericall.branches.a2_phase import BranchA2, BranchA2Config
from vericall.branches.a_ssl import BranchA, BranchAConfig
from vericall.branches.b_prosody import BranchB, BranchBConfig
from vericall.data.audio_io import load_audio
from vericall.data.manifest import Manifest
from vericall.fusion.calibration import estimate_eta, logit

LANE = {"A": "fast", "A2": "fast", "B": "slow"}


def build_calls(manifest: Manifest, plan: dict, n_calls: int,
                clips_per_call: int, fold: str):
    """Homogeneous synthetic calls: one speaker, one label, one generator.

    Mixing a speaker's bona fide and spoof clips into one stream injects label
    jumps that cancel the very correlation being measured and yield a falsely
    optimistic eta near 1.0.
    """
    items = [it for it in manifest if fold in plan.get(it.utt_id, [])]
    by_group: dict[tuple, list] = defaultdict(list)
    for it in items:
        by_group[(it.speaker_id, it.label, it.attack_id)].append(it)

    wanted: dict[str, list] = {}
    for key, clips in sorted(by_group.items(), key=lambda kv: str(kv[0])):
        spk, lab, atk = key
        for i in range(0, len(clips) - clips_per_call + 1, clips_per_call):
            wanted[f"{spk}_{lab}_{atk}_{i}"] = clips[i:i + clips_per_call]
            if len(wanted) >= n_calls:
                return wanted
    return wanted


def read_audio(wanted: dict) -> dict:
    """Sequential row-group reads for parquet, direct reads for plain WAV."""
    import pyarrow.parquet as pq

    need = {(it.shard, it.row) for clips in wanted.values() for it in clips}
    by_shard: dict[str, set] = defaultdict(set)
    for shard, row in need:
        by_shard[shard].add(row)

    audio: dict[tuple[str, int], torch.Tensor] = {}
    for shard, rows in by_shard.items():
        if not shard.endswith(".parquet"):
            audio[(shard, 0)] = load_audio(Path(shard).read_bytes())
            continue
        pf = pq.ParquetFile(shard)
        offset = 0
        for rg in range(pf.num_row_groups):
            n = pf.metadata.row_group(rg).num_rows
            hits = [r for r in range(offset, offset + n) if r in rows]
            if hits:
                tbl = pf.read_row_group(rg, columns=["audio"]).to_pydict()["audio"]
                for r in hits:
                    audio[(shard, r)] = load_audio(tbl[r - offset]["bytes"])
            offset += n
    return audio


@torch.no_grad()
def branch_sequences(branch: str, models, wanted: dict, audio: dict, device,
                     max_lag: int, quality_floor: float = 0.15):
    a, a2, b = models
    lane = LANE[branch]
    seqs, lengths = [], []
    for spk, clips in wanted.items():
        parts = [audio.get((it.shard, it.row)) for it in clips]
        parts = [p for p in parts if p is not None and p.numel() > 0]
        if len(parts) < 4:
            continue
        stream = torch.cat(parts)
        ch = DualRateChunker(f"spk_{spk}", ChunkerConfig())
        wins = [w for w in ch.push(stream)
                if w.lane == lane and w.quality > quality_floor]
        if len(wins) < max_lag + 2:
            continue
        x = torch.stack([w.samples for w in wins]).to(device)
        if branch == "A":
            p = torch.softmax(a.head(a.frontend(x)), dim=-1)[:, 1]
            s = logit(p.float().cpu().numpy())
        elif branch == "A2":
            s = a2.score(x).float().cpu().numpy()
        else:
            codec = torch.zeros(len(wins), dtype=torch.long, device=device)
            s = b.score(x, codec).float().cpu().numpy()
        seqs.append(list(np.asarray(s, dtype=float)))
        lengths.append(len(wins))
    return seqs, lengths


def lag_profile(seqs, max_lag: int) -> dict:
    prof = {}
    for k in range(1, max_lag + 1):
        rs = []
        for seq in seqs:
            x = np.asarray(seq, dtype=float)
            if x.size < k + 2:
                continue
            x = x - x.mean()
            d = float(np.dot(x, x))
            if d > 0:
                rs.append(float(np.dot(x[:-k], x[k:])) / d)
        prof[f"lag{k}"] = float(np.mean(rs)) if rs else 0.0
    return prof


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="asvspoof2019_la")
    ap.add_argument("--fold", default="calib")
    ap.add_argument("--calls", type=int, default=400)
    ap.add_argument("--clips-per-call", type=int, default=8)
    ap.add_argument("--max-lag", type=int, default=8)
    ap.add_argument("--branches", nargs="+", default=["A", "A2", "B"])
    ap.add_argument("--a-ckpt", default="checkpoints/branch_A_seed0.pt")
    ap.add_argument("--manifests", default="data/manifests")
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--out", default="docs/eta_branches.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    man = Manifest.load(Path(args.manifests) / f"{args.dataset}.jsonl")
    plan = json.loads((Path(args.splits) / f"{args.dataset}.json").read_text())

    a = BranchA(BranchAConfig()).to(device).eval()
    a.head.load_state_dict(torch.load(args.a_ckpt, map_location=device,
                                      weights_only=False)["state_dict"])
    a2 = BranchA2(BranchA2Config()).to(device)
    a2.load_state_dict(torch.load("checkpoints/branch_A2_seed0.pt",
                                  map_location=device,
                                  weights_only=False)["state_dict"])
    a2.eval()
    b = BranchB(BranchBConfig()).to(device)
    b.load_state_dict(torch.load("checkpoints/branch_B.pt", map_location=device,
                                 weights_only=False)["state_dict"])
    b.eval()
    models = (a, a2, b)

    wanted = build_calls(man, plan, args.calls, args.clips_per_call, args.fold)
    print(f"{args.dataset}:{args.fold} -> {len(wanted)} synthetic calls "
          f"({args.clips_per_call} clips each, one speaker/label/generator per call)")
    audio = read_audio(wanted)
    print(f"  decoded {len(audio):,} clips\n")

    defaults = {"A": 0.6, "A2": 0.6, "B": 0.9}
    results = {}
    print(f"{'branch':<8}{'lane':<7}{'calls':>7}{'win/call':>10}"
          f"{'eta':>9}{'inflation':>11}{'default':>9}")
    print("-" * 61)
    for br in args.branches:
        seqs, lengths = branch_sequences(br, models, wanted, audio, device,
                                         args.max_lag)
        if not seqs:
            print(f"{br:<8}{LANE[br]:<7}  no usable sequences")
            continue
        eta = estimate_eta(seqs, max_lag=args.max_lag)
        prof = lag_profile(seqs, args.max_lag)
        prof["mean_windows_per_call"] = float(np.mean(lengths))
        print(f"{br:<8}{LANE[br]:<7}{len(seqs):>7}"
              f"{np.mean(lengths):>10.1f}{eta:>9.3f}{1/max(eta,1e-9):>10.1f}x"
              f"{defaults[br]:>9.2f}")
        results[br] = {"eta": float(eta), "lane": LANE[br],
                       "n_sequences": len(seqs), "default": defaults[br],
                       "profile": prof}

    print("\nlag profile (correlation that barely decays is call-level "
          "consistency, not window overlap):")
    for br, r in results.items():
        lags = " ".join(f"{r['profile'][f'lag{k}']:+.3f}"
                        for k in range(1, min(args.max_lag, 8) + 1))
        print(f"  {br:<4}{lags}")

    print("\nverdict:")
    for br, r in results.items():
        ratio = r["default"] / max(r["eta"], 1e-9)
        if ratio > 1.5:
            print(f"  {br}: default {r['default']:.2f} is {ratio:.1f}x too "
                  f"optimistic -- bounds computed with it are "
                  f"{np.sqrt(ratio):.1f}x too tight")
        elif ratio < 0.67:
            print(f"  {br}: default {r['default']:.2f} is "
                  f"{1/ratio:.1f}x too pessimistic -- safe, but wasteful")
        else:
            print(f"  {br}: default {r['default']:.2f} is about right "
                  f"(measured {r['eta']:.3f})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
