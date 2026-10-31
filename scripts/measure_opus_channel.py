#!/usr/bin/env python
"""Is the fitted calibration still valid through the codec WebRTC uses?

web/call.html streams live two-party audio into the detector, and that audio
has been through Opus. Every calibration bucket in calib/ was fitted on clean
audio, G.711 or a simulated noisy line -- never Opus. The page says so on
screen; this measures how much it matters.

The question is deliberately NOT "does EER get worse". EER is
threshold-free, so a branch can hold its EER while its LLRs become
meaningless -- and this system does not threshold on a score, it accumulates
calibrated log-likelihood ratios over a call. What breaks a calibrated
accumulator is the mapping from score to LLR drifting, and the quantity that
measures exactly that is **calibration loss**: Cllr - min_Cllr. min_Cllr is
what a perfectly re-fitted mapping would achieve on this data, so the gap is
the part of the cost that is the mapping's fault rather than the detector's.

So three numbers per branch, all on held-out dev utterances:

  EER              does the channel destroy discrimination?
  min_Cllr         how much information survives, at best?
  Cllr             what the SERVED mapping actually delivers.

And one more thing worth knowing before trusting a live demo: what does the
runtime channel detector call Opus audio? It chooses between clean, g711 and
noisy_line, and Opus is none of them, so it will pick one. Which one decides
which wrong calibration gets applied.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vericall.data.opus import opus_roundtrip_batch

from vericall.audio.codec_detect import CodecDetector
from vericall.branches.a2_phase import BranchA2, BranchA2Config
from vericall.branches.a_ssl import BranchA, BranchAConfig
from vericall.data.audio_io import load_audio
from vericall.fusion.calibration import (
    bootstrap_eer_ci, cllr, eer, load_calibration, logit, min_cllr,
)

# 1.0 s, because that is what BOTH the fast lane and the calibration fit use:
# chunker.fast_window_s = 1.0 and extract_features.py --window defaults to 1.0.
# An earlier version of this script used a 4 s crop and zero-padded anything
# shorter, which was wrong twice over. 95% of dev clips are under 4 s (median
# 2.04 s), so almost every row became half speech and half digital silence --
# and zero padding drives const_frame_ratio to 0.25, which trips the channel
# detector's packet-loss test (c > 0.0) and made CLEAN audio classify as
# noisy_line 53% of the time. That was the harness, not the detector.
CROP_S = 1.0
SR = 16_000
DETECT_MIN_S = 4.0      # CodecDetector.min_windows = 4 one-second windows


def load_rows(manifest: Path, splits: Path, fold: str, n: int, seed: int):
    keep = {u for u, v in json.loads(splits.read_text()).items()
            if v and v[0] == fold}
    rows = []
    with manifest.open() as fh:
        for line in fh:
            r = json.loads(line)
            if r["utt_id"] in keep:
                rows.append(r)
    rng = random.Random(seed)
    bona = [r for r in rows if r["label"] == 0]
    spoof = [r for r in rows if r["label"] == 1]
    # Balanced, because Cllr's prior-weighted cost is defined against a
    # balanced prior and an unbalanced draw would silently reweight it.
    k = min(n // 2, len(bona), len(spoof))
    return rng.sample(bona, k) + rng.sample(spoof, k)


def read_audio(rows, quiet: bool):
    import pyarrow.parquet as pq

    by_shard: dict[str, list] = {}
    for i, r in enumerate(rows):
        by_shard.setdefault(r["shard"], []).append((i, r["row"]))

    want = int(CROP_S * SR)
    crops: list[torch.Tensor | None] = [None] * len(rows)
    fulls: list[torch.Tensor | None] = [None] * len(rows)
    for shard, items in by_shard.items():
        if not Path(shard).exists():
            raise SystemExit(f"missing shard {shard}\n"
                             "Run: python scripts/fetch_datasets.py "
                             "--datasets asvspoof2019_la")
        col = pq.read_table(shard, columns=["audio"]).column("audio")
        if not quiet:
            print(f"  {Path(shard).name}: {len(items)} rows")
        for out_i, row_i in items:
            cell = col[row_i].as_py()
            raw = cell["bytes"] if isinstance(cell, dict) else cell
            w = load_audio(raw)
            if w.numel() < want:
                continue            # dropped, never padded
            # Centre crop: the same region for clean and Opus, so the only
            # difference between the two arms is the codec.
            start = (w.numel() - want) // 2
            crops[out_i] = w[start:start + want]
            fulls[out_i] = w
    keep = [i for i, w in enumerate(crops) if w is not None]
    return keep, torch.stack([crops[i] for i in keep]), [fulls[i] for i in keep]


@torch.inference_mode()
def score(models, x: torch.Tensor, device, batch: int = 32):
    a, a2 = models
    sa, sa2 = [], []
    for i in range(0, len(x), batch):
        chunk = x[i:i + batch].to(device)
        sa.append(a.score(chunk).float().cpu())
        sa2.append(a2.score(chunk).float().cpu())
    return torch.cat(sa).numpy(), torch.cat(sa2).numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifests/asvspoof2019_la.jsonl")
    ap.add_argument("--splits", default="data/splits/asvspoof2019_la.json")
    ap.add_argument("--fold", default="dev",
                    help="dev is held out from calibration fitting, which is "
                         "what makes the Cllr comparison honest")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--bitrates", type=int, nargs="*", default=[24_000])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="docs/opus_channel.json")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}")

    rows = load_rows(Path(args.manifest), Path(args.splits), args.fold,
                     args.n, args.seed)
    labels = np.array([r["label"] for r in rows])
    groups = np.array([r["speaker_id"] for r in rows])

    print("reading audio")
    keep, clean, fulls = read_audio(rows, args.quiet)
    dropped = len(rows) - len(keep)
    rows = [rows[i] for i in keep]
    labels, groups = labels[keep], groups[keep]
    print(f"kept {len(rows)} utterances >= {CROP_S:g}s "
          f"({int((labels == 0).sum())} bona / {int((labels == 1).sum())} spoof), "
          f"{dropped} dropped as too short -- never padded")

    a = BranchA(BranchAConfig()).to(device).eval()
    a.head.load_state_dict(torch.load("checkpoints/branch_A_seed0.pt",
                                      map_location=device,
                                      weights_only=True)["state_dict"])
    a2 = BranchA2(BranchA2Config()).to(device)
    a2.load_state_dict(torch.load("checkpoints/branch_A2_seed0.pt",
                                  map_location=device,
                                  weights_only=True)["state_dict"])
    a2.eval()
    models = (a, a2)

    cal = {"A": load_calibration("calib/A@clean.json"),
           "A2": load_calibration("calib/A2@clean.json")}

    arms: dict[str, torch.Tensor] = {"clean": clean}
    for br in args.bitrates:
        print(f"encoding through libopus at {br // 1000} kbps "
              f"({len(clean)} utterances, one stream each)")
        # One stream per utterance: libopus carries prediction state across
        # frames, so a shared stream would let each utterance's artifacts
        # depend on the previous one.
        arms[f"opus@{br // 1000}k"] = opus_roundtrip_batch(
            clean, bitrate=br).clamp(-1.0, 1.0)

    # What does the runtime detector call each arm? This decides which wrong
    # calibration a live call actually receives, so it is worth knowing.
    #
    # Measured on FULL clips, not on the 1 s crops: CodecDetector needs 4
    # one-second windows before it will commit, and below that it reports its
    # default rather than a measurement. Clips shorter than 4 s are excluded
    # from this arm only -- padding them to reach the window count is exactly
    # the mistake that produced the previous run's bogus 53% noisy_line.
    long_idx = [i for i, w in enumerate(fulls) if w.numel() >= int(DETECT_MIN_S * SR)]
    print(f"\nchannel detection measured on {len(long_idx)} clips of "
          f">= {DETECT_MIN_S:g}s (the rest cannot reach min_windows=4 honestly)")
    buckets = {}
    for name in arms:
        votes = Counter()
        for i in long_idx:
            wav = fulls[i] if name == "clean" else opus_roundtrip_batch(
                fulls[i].unsqueeze(0), bitrate=int(name.split("@")[1][:-1]) * 1000
            )[0].clamp(-1.0, 1.0)
            det = CodecDetector()
            for st in range(0, wav.numel() - SR + 1, SR):
                det.update(wav[st:st + SR])
            votes[det.bucket if det.confident else "not_confident"] += 1
        buckets[name] = dict(votes)

    results = {}
    for name, wav in arms.items():
        s_a, s_a2 = score(models, wav, device)
        row = {}
        for branch, raw in (("A", s_a), ("A2", s_a2)):
            e, lo, hi = bootstrap_eer_ci(raw, labels, groups, n_boot=300)
            # logit FIRST. The Platt mapping was fitted on logit(posterior),
            # not on the posterior (scripts/fit_calibration.py: raw =
            # logit(scores)). Feeding it a probability produced a Cllr of
            # 0.93 against a recorded 0.083 in calib/A@clean.json -- an
            # 11x discrepancy that was this script's bug, not a finding.
            llr = cal[branch].apply(logit(raw))
            row[branch] = {
                "eer": float(e), "ci95": [float(lo), float(hi)],
                "cllr_served_mapping": float(cllr(llr, labels)),
                "min_cllr": float(min_cllr(logit(raw), labels)),
                "mean_llr_bona": float(llr[labels == 0].mean()),
                "mean_llr_spoof": float(llr[labels == 1].mean()),
            }
            row[branch]["calibration_loss"] = (
                row[branch]["cllr_served_mapping"] - row[branch]["min_cllr"])
        results[name] = row

    # ------------------------------------------------------------- report
    print(f"\n{'arm':<14}{'branch':<8}{'EER':>18}{'min_Cllr':>11}"
          f"{'Cllr':>9}{'cal loss':>11}")
    print("-" * 74)
    for name in arms:
        for branch in ("A", "A2"):
            r = results[name][branch]
            print(f"{name:<14}{branch:<8}"
                  f"{r['eer']*100:7.2f}% [{r['ci95'][0]*100:.1f}-{r['ci95'][1]*100:.1f}]"
                  f"{r['min_cllr']:11.4f}{r['cllr_served_mapping']:9.4f}"
                  f"{r['calibration_loss']:11.4f}")

    print("\nwhat the runtime channel detector calls each arm:")
    for name, votes in buckets.items():
        total = sum(votes.values())
        parts = ", ".join(f"{k} {v/total*100:.0f}%"
                          for k, v in sorted(votes.items(), key=lambda kv: -kv[1]))
        print(f"  {name:<14} {parts}")

    print("\nthe LLRs the accumulator would bank (mean per class):")
    for name in arms:
        for branch in ("A", "A2"):
            r = results[name][branch]
            print(f"  {name:<14}{branch:<4}bona {r['mean_llr_bona']:+8.3f}   "
                  f"spoof {r['mean_llr_spoof']:+8.3f}")

    payload = {
        "fold": args.fold, "n": len(rows), "crop_s": CROP_S,
        "speakers": len(set(groups)),
        "calibration_used": {k: {"a": v.a, "b": v.b} for k, v in cal.items()},
        "detected_buckets": buckets,
        "results": results,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
