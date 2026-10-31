#!/usr/bin/env python
"""Fit the `opus` calibration bucket -- the channel a WebRTC call arrives on.

docs/opus_channel.json measured the gap this closes. Two facts from it decide
the design here, and they point in opposite directions for the two branches:

  Branch A   1.18% -> 1.33% EER through Opus at 24 kbps, calibration loss
             0.015 -> 0.018. Essentially unaffected; the clean mapping is
             already very nearly right for it.

  Branch A2  1.77% -> 18.07% EER, calibration loss 0.025 -> 0.419. Destroyed.
             A2 reads phase and Opus is a transform codec that does not
             preserve it. At 12 kbps the SPOOF class banks -0.40 nats -- the
             branch inverts and argues the attacker is genuine.

Refitting the Platt mapping fixes the *calibration* loss but cannot recover the
information: A2's min_Cllr through Opus is 0.56 against 0.052 clean, and
min_Cllr is what a perfect mapping would achieve. So this script does both
things the measurement implies -- refits both branches for the channel, AND
refits the fusion weight, because a branch that has lost nine tenths of its
information should not keep the weight it earned when it had it.

Why the calib fold and not dev: that is the fold every other bucket in calib/
was fitted on, and using a different one would make the buckets
non-comparable. dev stays held out, which is what makes the dev numbers in
docs/opus_channel.json an honest test rather than a training score.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vericall.branches.a2_phase import BranchA2, BranchA2Config
from vericall.branches.a_ssl import BranchA, BranchAConfig
from vericall.data.audio_io import load_audio
from vericall.data.opus import opus_roundtrip_batch
from vericall.fusion.calibration import (
    cllr, eer, fit_platt, logit, min_cllr, save_calibration,
)

CROP_S = 1.0          # the fast lane, and what extract_features.py --window uses
SR = 16_000


def load_fold(manifest: Path, splits: Path, fold: str, limit: int, seed: int):
    keep = {u for u, v in json.loads(splits.read_text()).items()
            if v and v[0] == fold}
    rows = [json.loads(l) for l in manifest.open()]
    rows = [r for r in rows if r["utt_id"] in keep]
    if limit and len(rows) > limit:
        rng = random.Random(seed)
        bona = [r for r in rows if r["label"] == 0]
        spoof = [r for r in rows if r["label"] == 1]
        k = min(limit // 2, len(bona), len(spoof))
        rows = rng.sample(bona, k) + rng.sample(spoof, k)
    return rows


def read_crops(rows):
    import pyarrow.parquet as pq

    want = int(CROP_S * SR)
    by_shard: dict[str, list] = {}
    for i, r in enumerate(rows):
        by_shard.setdefault(r["shard"], []).append((i, r["row"]))
    out: list[torch.Tensor | None] = [None] * len(rows)
    for shard, items in by_shard.items():
        if not Path(shard).exists():
            raise SystemExit(f"missing shard {shard}\n"
                             "Run: python scripts/fetch_datasets.py "
                             "--datasets asvspoof2019_la")
        col = pq.read_table(shard, columns=["audio"]).column("audio")
        for oi, ri in items:
            cell = col[ri].as_py()
            raw = cell["bytes"] if isinstance(cell, dict) else cell
            w = load_audio(raw)
            if w.numel() < want:
                continue                    # dropped, never zero-padded
            start = (w.numel() - want) // 2
            out[oi] = w[start:start + want]
    keep = [i for i, w in enumerate(out) if w is not None]
    return keep, torch.stack([out[i] for i in keep])


@torch.inference_mode()
def score(a, a2, x, device, batch=64):
    sa, sa2 = [], []
    for i in range(0, len(x), batch):
        c = x[i:i + batch].to(device)
        sa.append(a.score(c).float().cpu())
        sa2.append(a2.score(c).float().cpu())
    return torch.cat(sa).numpy().astype(np.float64), \
           torch.cat(sa2).numpy().astype(np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifests/asvspoof2019_la.jsonl")
    ap.add_argument("--splits", default="data/splits/asvspoof2019_la.json")
    ap.add_argument("--fold", default="calib")
    ap.add_argument("--limit", type=int, default=4000,
                    help="0 for the whole fold; libopus runs one stream per "
                         "utterance so this dominates the runtime")
    ap.add_argument("--bitrate", type=int, default=24_000)
    ap.add_argument("--out-dir", default="calib")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="report the fit without writing calibration files")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_fold(Path(args.manifest), Path(args.splits), args.fold,
                     args.limit, args.seed)
    print(f"device {device}   fold {args.fold}   {len(rows)} candidate rows")

    keep, clean = read_crops(rows)
    rows = [rows[i] for i in keep]
    labels = np.array([r["label"] for r in rows])
    print(f"read {len(rows)} crops of {CROP_S:g}s "
          f"({int((labels==0).sum())} bona / {int((labels==1).sum())} spoof)")
    if len(set(labels)) < 2:
        raise SystemExit("need both classes")

    print(f"encoding through libopus at {args.bitrate//1000} kbps")
    opus = opus_roundtrip_batch(clean, bitrate=args.bitrate).clamp(-1.0, 1.0)

    a = BranchA(BranchAConfig()).to(device).eval()
    a.head.load_state_dict(torch.load("checkpoints/branch_A_seed0.pt",
                                      map_location=device,
                                      weights_only=True)["state_dict"])
    a2 = BranchA2(BranchA2Config()).to(device)
    a2.load_state_dict(torch.load("checkpoints/branch_A2_seed0.pt",
                                  map_location=device,
                                  weights_only=True)["state_dict"])
    a2.eval()

    s_a, s_a2 = score(a, a2, opus, device)

    report = {"bitrate": args.bitrate, "fold": args.fold, "n": len(rows),
              "crop_s": CROP_S, "branches": {}}
    out_dir = Path(args.out_dir)
    for branch, raw_p in (("A", s_a), ("A2", s_a2)):
        raw = logit(raw_p)
        cal = fit_platt(raw, labels, branch=branch, channel_bucket="opus",
                        calib_id=f"{branch}@opus")
        e, _ = eer(raw_p, labels)
        llr = cal.apply(raw)
        report["branches"][branch] = {
            "eer": float(e), "a": cal.a, "b": cal.b,
            "cllr": cal.cllr, "min_cllr": cal.min_cllr,
            "calibration_loss": cal.calibration_loss,
            "mean_llr_bona": float(llr[labels == 0].mean()),
            "mean_llr_spoof": float(llr[labels == 1].mean()),
        }
        print(f"  {branch:<3} EER {e*100:6.2f}%  a={cal.a:6.3f} b={cal.b:+7.3f}  "
              f"Cllr {cal.cllr:.4f}  minCllr {cal.min_cllr:.4f}  "
              f"loss {cal.calibration_loss:.4f}")
        if not args.dry_run:
            save_calibration(out_dir / f"{branch}@opus.json", cal)

    # ---- the fusion weight, refitted for this channel ---------------------
    # A branch that has lost nine tenths of its information should not keep
    # the weight it earned when it had it. Weight by inverse calibration-free
    # cost: (1 - min_Cllr) is the fraction of a bit-equivalent that survives,
    # normalised to w_A = 1 the way calib/weights.json is.
    surv = {b: max(0.0, 1.0 - report["branches"][b]["min_cllr"])
            for b in ("A", "A2")}
    w = {b: round(surv[b] / surv["A"], 4) for b in ("A", "A2")} if surv["A"] > 0 else {}
    report["suggested_weights"] = w
    print(f"\n  information surviving the channel (1 - min_Cllr): "
          f"A {surv['A']:.3f}  A2 {surv['A2']:.3f}")
    print(f"  suggested opus-bucket fusion weights (w_A = 1): {w}")
    clean_w = json.loads(Path("calib/weights.json").read_text())["w"]
    print(f"  for comparison, the clean-channel weights:       {clean_w}")

    if not args.dry_run:
        (out_dir / "weights@opus.json").write_text(json.dumps(
            {"w": w, "fitted_on": f"asvspoof2019_la:{args.fold}@opus"
                                  f"{args.bitrate//1000}k",
             "note": "Weights for the opus bucket. A2 is down-weighted because "
                     "Opus does not preserve phase and A2 reads phase: its "
                     "min_Cllr through this channel is 0.56 against 0.05 "
                     "clean, so refitting its Platt mapping fixes the "
                     "calibration but cannot recover the information."},
            indent=2) + "\n")
        Path("docs/opus_bucket_fit.json").write_text(
            json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {out_dir}/A@opus.json, {out_dir}/A2@opus.json, "
              f"{out_dir}/weights@opus.json, docs/opus_bucket_fit.json")
    else:
        print("\n--dry-run: nothing written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
