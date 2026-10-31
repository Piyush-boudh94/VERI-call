#!/usr/bin/env python
"""Fit per-channel calibration for Branch A or A2.

Per-channel buckets matter because the slopes genuinely differ: A is 0.510 clean
vs 0.411 G.711, and A2 is worse -- it degrades 10.54x on a noisy line
(docs/RESULTS.md section 10), the most channel-sensitive component in the system,
so a single pooled mapping calibrates it against the wrong distribution exactly
where its behaviour changes most.

Was fit_a2_channels.py, A2-only. Generalised because runtime codec detection
(vericall/audio/codec_detect.py) can now identify a **noisy_line** call at 98.8%
per-call accuracy, and the gateway then had nothing to map it with: calib/ held
clean and g711 buckets only, and /health said so. Detecting a channel you cannot
calibrate for is worse than not detecting it.

The calib fold carries codec ids from augmentation at extraction time, so the
labels are free -- but the fold must have been extracted with all four channel
profiles (--crops 4) or the noisy bucket simply will not be present.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vericall.branches.a2_phase import BranchA2, BranchA2Config
from vericall.branches.a_ssl import BranchAConfig, BranchAHead
from vericall.data.cache import CachedFold
from vericall.fusion.calibration import (
    cllr, eer, fit_platt, logit, min_cllr, save_calibration,
)

CHANNEL_NAMES = {0: "clean", 1: "g711", 2: "noisy_line"}


@torch.inference_mode()
def score(model, fold, device, branch: str, batch=64):
    """P(spoof) per row. A reads cached SSL features, A2 the waveform."""
    out = np.zeros(len(fold))
    src = fold.feat if branch == "A" else fold.wave
    for i in range(0, len(fold), batch):
        sel = np.arange(i, min(i + batch, len(fold)))
        x = torch.from_numpy(np.asarray(src[sel], dtype=np.float32)).to(device)
        out[sel] = (torch.softmax(model(x), -1)[:, 1] if branch == "A"
                    else model.score(x)).float().cpu().numpy()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/vericall-cache")
    ap.add_argument("--dataset", default="asvspoof2019_la")
    ap.add_argument("--out", default="calib")
    ap.add_argument("--branch", choices=["A", "A2"], default="A2")
    ap.add_argument("--ckpt", default="",
                    help="defaults to checkpoints/branch_<branch>_seed0.pt")
    ap.add_argument("--fold", default="calib")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = args.ckpt or f"checkpoints/branch_{args.branch}_seed0.pt"
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    if args.branch == "A":
        model = BranchAHead(768, BranchAConfig()).to(device)
    else:
        model = BranchA2(BranchA2Config()).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()

    fold = CachedFold.load(Path(args.cache) / args.dataset / args.fold)
    labels = np.asarray(fold.label)
    codecs = np.asarray(fold.codec)
    raw = logit(score(model, fold, device, args.branch))
    out_dir = Path(args.out)

    print(f"branch {args.branch} | {ckpt}")
    print(f"{args.fold} fold: {len(fold):,} rows")
    print(f"uncalibrated: EER {eer(raw, labels)[0]*100:5.2f}%  "
          f"Cllr {cllr(raw, labels):.3f}  minCllr {min_cllr(raw, labels):.3f}\n")

    report = {}
    print(f"per-channel calibration for Branch {args.branch}:")
    for cid in sorted(set(int(c) for c in codecs)):
        name = CHANNEL_NAMES.get(cid, f"codec{cid}")
        m = codecs == cid
        if m.sum() < 200 or len(set(labels[m])) < 2:
            print(f"  {name:<12} skipped (n={int(m.sum())})")
            continue
        cal = fit_platt(raw[m], labels[m], branch=args.branch,
                        channel_bucket=name, calib_id=f"{args.branch}@{name}")
        e, _ = eer(raw[m], labels[m])
        save_calibration(out_dir / f"{args.branch}@{name}.json", cal)
        report[name] = {"n": int(m.sum()), "eer": e, "a": cal.a, "b": cal.b,
                        "cllr": cal.cllr, "min_cllr": cal.min_cllr,
                        "calibration_loss": cal.calibration_loss}
        print(f"  {name:<12} n={int(m.sum()):>6}  EER {e*100:5.2f}%  "
              f"a={cal.a:6.3f} b={cal.b:+6.3f}  "
              f"Cllr {cal.cllr:.3f}  loss {cal.calibration_loss:.3f}")

    if len(report) > 1:
        slopes = {k: v["a"] for k, v in report.items()}
        spread = max(slopes.values()) / max(min(slopes.values()), 1e-9)
        print(f"\n  slope spread across channels: {spread:.2f}x  {slopes}")
        print("  A pooled fit would mis-scale every non-clean channel by "
              "roughly this factor." if spread > 1.2 else
              "  Channels are similar; the pooled fit was close to adequate.")

    (out_dir / f"{args.branch}_channels.json").write_text(
        json.dumps(report, indent=2))
    print(f"\nwrote {out_dir}/{args.branch}@*.json and "
          f"{args.branch}_channels.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
