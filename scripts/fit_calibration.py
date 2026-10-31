#!/usr/bin/env python
"""Fit the calibration artifacts that connect the trained heads to the fusion engine.

The fusion engine never sees a raw model score. Branch A emits a posterior,
Branch B an unbounded one-class anomaly score, Branch C a cosine similarity --
three incompatible scales. Everything crossing a branch boundary must already be
a log-likelihood ratio in nats, or the fusion weights silently absorb an
order-of-magnitude scale mismatch and every threshold downstream becomes
meaningless.

Two artifacts are produced:

1. **Per-channel-bucket Platt calibration.** Fit on the `calib` fold, which is
   disjoint from both train and dev. Prior-weighted, so the output is prior-free
   and the operational prior is injected once at decision time. Fit separately
   per channel condition (clean / G.711 / noisy line) rather than adapting
   per-call: on a call spoofed from t=0, per-call normalisation subtracts off
   exactly the shift being detected.

2. **eta, the effective-sample-size deflation.** Measured from *streamed
   overlapping windows* over whole utterances -- not from the cache, whose crops
   are independent by construction and would therefore report eta = 1 and
   measure nothing. This is the number that stops 50%-overlapping windows from
   being counted as independent evidence.
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
from vericall.data.cache import CachedFold
from vericall.data.manifest import Manifest
from vericall.fusion.calibration import (
    PlattCalibration, cllr, eer, estimate_eta, fit_platt, logit, min_cllr,
    save_calibration,
)

CHANNEL_NAMES = {0: "clean", 1: "g711", 2: "noisy_line"}


def load_head(ckpt: Path, device) -> BranchAHead:
    blob = torch.load(ckpt, map_location=device)
    head = BranchAHead(768, BranchAConfig()).to(device)
    head.load_state_dict(blob["state_dict"])
    head.eval()
    return head


@torch.inference_mode()
def score_fold(head, fold: CachedFold, device, batch: int = 256) -> np.ndarray:
    out = np.zeros(len(fold), dtype=np.float64)
    for x, _, _, sel in fold.batches(batch, shuffle=False, use="feat", device=device):
        out[sel] = torch.softmax(head(x), dim=-1)[:, 1].float().cpu().numpy()
    return out


def fit_channel_calibrations(scores, labels, codecs, out_dir: Path) -> dict:
    report = {}
    raw = logit(scores)
    for cid in sorted(set(int(c) for c in codecs)):
        name = CHANNEL_NAMES.get(cid, f"codec{cid}")
        m = codecs == cid
        if m.sum() < 200 or len(set(labels[m])) < 2:
            print(f"  {name:<12} skipped (n={int(m.sum())}, needs both classes)")
            continue
        cal = fit_platt(raw[m], labels[m], branch="A", channel_bucket=name,
                        calib_id=f"A@{name}")
        e, _ = eer(scores[m], labels[m])
        save_calibration(out_dir / f"A@{name}.json", cal)
        report[name] = {
            "n": int(m.sum()), "eer": e, "a": cal.a, "b": cal.b,
            "cllr": cal.cllr, "min_cllr": cal.min_cllr,
            "calibration_loss": cal.calibration_loss,
        }
        print(f"  {name:<12} n={int(m.sum()):>6}  EER {e*100:5.2f}%  "
              f"a={cal.a:6.3f} b={cal.b:+6.3f}  "
              f"Cllr {cal.cllr:.3f}  minCllr {cal.min_cllr:.3f}  "
              f"loss {cal.calibration_loss:.3f}")
    return report


@torch.inference_mode()
def measure_eta(model: BranchA, head, manifest: Manifest, plan: dict,
                device, n_calls: int, clips_per_call: int = 8,
                max_lag: int = 8) -> tuple[float, int, dict]:
    """Measure the effective-sample-size deflation from overlapping windows.

    Two design points, both forced by the data:

    *Why synthetic calls.* ASVspoof clips are short -- median 1.87 s, max 4.07 s
    after trimming -- so one clip yields 2-3 windows, far too few to estimate an
    autocorrelation. We therefore concatenate consecutive clips from the SAME
    speaker into a ~15 s stream. That is also closer to what we are actually
    modelling: a call is continuous speech from one speaker, not a 2 s clip. The
    caveat, stated rather than hidden, is that splice points introduce
    discontinuities a real call would not have; they add a little noise to the
    estimate but do not bias the overlap correlation being measured.

    *What eta actually has to capture.* Two correlations, not one. Window
    overlap couples adjacent windows (lag-1 only: with a 1 s window and 0.5 s
    hop, window i shares nothing with i+2). But call-level consistency -- one
    speaker, one channel, one generator for the whole call -- couples windows at
    EVERY lag, and it is the larger of the two. Both must be in eta, or the
    confidence bound over an accumulated call score is too tight. Hence
    homogeneous groups and a lag window long enough to see the call-level term;
    estimate_eta truncates at the first non-positive lag anyway.

    Calibration data only -- the cross-domain eval sets are never touched.
    """
    import pyarrow.parquet as pq

    items = [it for it in manifest if "calib" in plan.get(it.utt_id, [])]

    # Group by (speaker, label, attack) -- a real call is ONE speaker with ONE
    # provenance throughout. Concatenating a speaker's bona fide and spoof clips
    # into a single stream injects label jumps that cancel the very correlation
    # being measured and yield a falsely optimistic eta ~ 1.0.
    by_group: dict[tuple, list] = defaultdict(list)
    for it in items:
        by_group[(it.speaker_id, it.label, it.attack_id)].append(it)

    # LA dev has only ~20 speakers, so one call per speaker gives a handful of
    # sequences. Chunk each group into as many calls as its clips allow.
    wanted: dict[str, list] = {}
    for (spk, lab, atk), clips in sorted(by_group.items(), key=lambda kv: str(kv[0])):
        for i in range(0, len(clips) - clips_per_call + 1, clips_per_call):
            wanted[f"{spk}_{lab}_{atk}_{i}"] = clips[i:i + clips_per_call]
            if len(wanted) >= n_calls:
                break
        if len(wanted) >= n_calls:
            break

    need = {(it.shard, it.row): (spk, i)
            for spk, clips in wanted.items() for i, it in enumerate(clips)}
    audio: dict[tuple[str, int], torch.Tensor] = {}
    by_shard: dict[str, list[int]] = defaultdict(list)
    for shard, row in need:
        by_shard[shard].append(row)

    for shard, rows in by_shard.items():
        # Corpora ship either parquet shards (ASVspoof, WaveFake, IndicSynth)
        # or one audio file per item (RTCFake, In-the-Wild, hindi_paired).
        # Dispatch on the suffix, the same rule extract_features.py uses --
        # without this, fitting calibration for any file-addressed corpus dies
        # in pyarrow with "Parquet magic bytes not found", which is a confusing
        # way to say "this corpus is not parquet".
        if not str(shard).endswith(".parquet"):
            if Path(shard).exists():
                audio[(shard, 0)] = load_audio(Path(shard).read_bytes())
            continue
        rowset = set(rows)
        pf = pq.ParquetFile(shard)
        offset = 0
        for rg in range(pf.num_row_groups):
            n = pf.metadata.row_group(rg).num_rows
            hits = [r for r in range(offset, offset + n) if r in rowset]
            if hits:
                tbl = pf.read_row_group(rg, columns=["audio"]).to_pydict()["audio"]
                for r in hits:
                    audio[(shard, r)] = load_audio(tbl[r - offset]["bytes"])
            offset += n

    sequences: list[list[float]] = []
    lengths: list[int] = []
    for spk, clips in wanted.items():
        parts = [audio.get((it.shard, it.row)) for it in clips]
        parts = [p for p in parts if p is not None and p.numel() > 0]
        if len(parts) < 4:
            continue
        stream = torch.cat(parts)
        ch = DualRateChunker(f"spk_{spk}", ChunkerConfig())
        wins = [w for w in ch.push(stream)
                if w.lane == "fast" and w.quality > 0.15]
        if len(wins) < max_lag + 2:
            continue
        batch = torch.stack([w.samples for w in wins]).to(device)
        p = torch.softmax(head(model.frontend(batch)), dim=-1)[:, 1]
        sequences.append(list(logit(p.float().cpu().numpy())))
        lengths.append(len(wins))

    if not sequences:
        return 1.0, 0, {}

    eta = estimate_eta(sequences, max_lag=max_lag)
    # report the lag profile too: lag-1 should dominate and lag-2+ be ~0
    prof = {}
    for k in range(1, max_lag + 1):
        rs = []
        for seq in sequences:
            x = np.asarray(seq, dtype=float)
            if x.size < k + 2:
                continue
            x = x - x.mean()
            d = float(np.dot(x, x))
            if d > 0:
                rs.append(float(np.dot(x[:-k], x[k:])) / d)
        prof[f"lag{k}"] = float(np.mean(rs)) if rs else 0.0
    prof["mean_windows_per_call"] = float(np.mean(lengths))
    return eta, len(sequences), prof


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/branch_A_seed0.pt")
    ap.add_argument("--cache", default="/tmp/vericall-cache")
    ap.add_argument("--dataset", default="asvspoof2019_la")
    ap.add_argument("--manifest", default="data/manifests/asvspoof2019_la.jsonl")
    ap.add_argument("--splits", default="data/splits/asvspoof2019_la.json")
    ap.add_argument("--out", default="calib")
    ap.add_argument("--eta-calls", type=int, default=120)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    head = load_head(Path(args.checkpoint), device)
    fold = CachedFold.load(Path(args.cache) / args.dataset / "calib")
    print(f"calib fold: {len(fold):,} rows "
          f"({int((fold.label==0).sum()):,} bona / {int((fold.label==1).sum()):,} spoof)")

    scores = score_fold(head, fold, device)
    labels = np.asarray(fold.label)
    codecs = np.asarray(fold.codec)

    pooled_eer, _ = eer(scores, labels)
    raw = logit(scores)
    print(f"\nuncalibrated: EER {pooled_eer*100:5.2f}%  "
          f"Cllr {cllr(raw, labels):.3f}  minCllr {min_cllr(raw, labels):.3f}")

    print("\nper-channel-bucket calibration (fit on calib, disjoint from train/dev):")
    report = fit_channel_calibrations(scores, labels, codecs, out_dir)

    pooled = fit_platt(raw, labels, branch="A", channel_bucket="pooled",
                       calib_id="A@pooled")
    save_calibration(out_dir / "A@pooled.json", pooled)
    print(f"  {'pooled':<12} n={len(fold):>6}  "
          f"a={pooled.a:6.3f} b={pooled.b:+6.3f}  "
          f"Cllr {pooled.cllr:.3f}  loss {pooled.calibration_loss:.3f}")

    print(f"\nmeasuring eta from streamed overlapping windows "
          f"({args.eta_calls} utterances)...")
    model = BranchA(BranchAConfig()).to(device).eval()
    manifest = Manifest.load(args.manifest)
    plan = json.loads(Path(args.splits).read_text())
    eta, n_seq, prof = measure_eta(model, head, manifest, plan, device,
                                   args.eta_calls)
    print(f"  eta = {eta:.3f} from {n_seq} synthetic calls "
          f"({prof.get('mean_windows_per_call', 0):.0f} windows each)")
    print("  autocorrelation profile: " + "  ".join(
        f"{k}={v:+.3f}" for k, v in prof.items() if k.startswith("lag")))
    print(f"  -> {1/eta:.2f}x variance inflation; treating {1/eta:.2f} overlapping "
          f"windows as one independent observation")

    artifact = {
        "branch": "A",
        "checkpoint": str(args.checkpoint),
        "calib_rows": len(fold),
        "uncalibrated_eer": pooled_eer,
        "channels": report,
        "pooled": {"a": pooled.a, "b": pooled.b, "cllr": pooled.cllr,
                   "min_cllr": pooled.min_cllr,
                   "calibration_loss": pooled.calibration_loss},
        "eta": {"A": eta, "n_sequences": n_seq, "profile": prof},
    }
    (out_dir / "summary.json").write_text(json.dumps(artifact, indent=2))
    print(f"\nwrote {out_dir}/ ({len(list(out_dir.glob('*.json')))} artifacts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
