#!/usr/bin/env python
"""What does runtime channel detection actually buy?

The feature is only worth its complexity if selecting the right calibration
bucket measurably improves the LLRs. Three conditions on held-out audio:

    always-clean    what the gateway did before -- one bucket for every call
    detected        vericall/audio/codec_detect.py chooses per call
    oracle          the true channel label, i.e. the ceiling

**Cllr, not EER, is the metric that answers this.** A Platt mapping is monotonic,
so swapping buckets cannot change the ranking of one branch's scores and cannot
change its EER at all -- min-Cllr is identical in all three conditions by
construction. What changes is whether the numbers are LLRs, which is exactly what
every threshold and the whole fusion layer assume. So: Cllr, and the calibration
loss Cllr - min-Cllr that isolates the mapping's contribution.

EER does move for the FUSED score, because fusion sums two differently
mis-scaled branches -- and that is the operational consequence.

**Each condition gets its own subset (alpha, scale), fitted on calib under that
same condition and applied to the eval fold.** Sharing one table across all
three is a confound, and a flattering one in the wrong direction: the first run
of this script applied a per-channel-fitted table to always-clean LLRs and
reported that per-channel calibration made the fused Cllr worse. It had measured
the mismatch, not the feature.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vericall.audio.codec_detect import CodecDetector
from vericall.data.cache import CachedFold
from vericall.fusion.calibration import (
    cllr, eer, load_calibration, logit, min_cllr,
)
from vericall.fusion.config import FusionConfig, load_weights

ID_TO_BUCKET = {0: "clean", 1: "g711", 2: "noisy_line"}


@torch.inference_mode()
def score(fold, device, branch, ckpt, batch=128):
    from vericall.branches.a2_phase import BranchA2, BranchA2Config
    from vericall.branches.a_ssl import BranchAConfig, BranchAHead
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    m = (BranchAHead(768, BranchAConfig()) if branch == "A"
         else BranchA2()).to(device)
    m.load_state_dict(blob["state_dict"]); m.eval()
    src = fold.feat if branch == "A" else fold.wave
    out = np.zeros(len(fold))
    for i in range(0, len(fold), batch):
        sel = np.arange(i, min(i + batch, len(fold)))
        x = torch.from_numpy(np.asarray(src[sel], dtype=np.float32)).to(device)
        out[sel] = torch.softmax(m(x), -1)[:, 1].float().cpu().numpy()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/vericall-cache")
    ap.add_argument("--dataset", default="asvspoof2019_la")
    ap.add_argument("--fold", default="dev")
    ap.add_argument("--calib-dir", default="calib")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--windows-per-call", type=int, default=6)
    ap.add_argument("--fit-fold", default="calib",
                    help="fold the per-condition subset calibration is fitted "
                         "on. Must be disjoint from --fold.")
    ap.add_argument("--out", default="docs/bucket_benefit.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold = CachedFold.load(Path(args.cache) / args.dataset / args.fold)
    n = len(fold) if not args.limit else min(args.limit, len(fold))
    labels = np.asarray(fold.label)[:n]
    codecs = np.asarray(fold.codec)[:n]
    buckets = sorted({ID_TO_BUCKET[int(c)] for c in codecs})
    print(f"{args.dataset}:{args.fold}  {n:,} rows, channels {buckets}")

    cdir = Path(args.calib_dir)
    cal = {}
    for b in ("clean", "g711", "noisy_line"):
        for br in ("A", "A2"):
            p = cdir / f"{br}@{b}.json"
            if p.exists():
                cal[(br, b)] = load_calibration(p)
    have = sorted({b for (_, b) in cal})
    print(f"calibration buckets available: {have}")
    if len(have) < len(buckets):
        print(f"  missing {[b for b in buckets if b not in have]} -- "
              f"rows on those channels cannot be scored correctly under any "
              f"condition", file=sys.stderr)

    raw = {br: logit(score(fold, device, br,
                           f"checkpoints/branch_{br}_seed0.pt")[:n])
           for br in ("A", "A2")}

    # --- detection, per synthetic call ------------------------------------
    # Rows of one channel are 4 apart (the four crops of a clip cycle the four
    # profiles), so a "call" is k rows of the SAME channel, taken in stride.
    k = args.windows_per_call
    detected = np.array([None] * n, dtype=object)
    for cid in sorted(set(int(c) for c in codecs)):
        idx = np.flatnonzero(codecs == cid)
        for s in range(0, len(idx), k):
            grp = idx[s:s + k]
            det = CodecDetector()
            for i in grp:
                det.update(torch.from_numpy(
                    np.asarray(fold.wave[i], dtype=np.float32)))
            for i in grp:
                detected[i] = det.bucket
    truth = np.array([ID_TO_BUCKET[int(c)] for c in codecs], dtype=object)
    print(f"detection agrees with truth on "
          f"{float((detected == truth).mean())*100:.2f}% of rows "
          f"({k} windows/call)\n")

    def apply(br, chosen):
        out = np.zeros(n)
        for b in set(chosen):
            m = np.array([c == b for c in chosen])
            c = cal.get((br, b)) or cal.get((br, "clean"))
            out[m] = c.apply(raw[br][m])
        return out

    cfg = FusionConfig()
    w = load_weights(cdir / "weights.json", cfg.w)
    print(f"weights: " + "  ".join(f"{k}={v}" for k, v in sorted(w.items()))
          + (f"   [fitted, {cdir}/weights.json]"
             if (cdir / "weights.json").exists() else "   [FusionConfig default]"))

    # Per-condition subset calibration, fitted on the CALIB fold under the same
    # condition and applied here. See the module docstring.
    fit_fold = CachedFold.load(Path(args.cache) / args.dataset / args.fit_fold)
    fy = np.asarray(fit_fold.label)
    fit_codecs = np.asarray(fit_fold.codec)
    fit_raw = {br: logit(score(fit_fold, device, br,
                               f"checkpoints/branch_{br}_seed0.pt"))
               for br in ("A", "A2")}
    fit_truth = np.array([ID_TO_BUCKET[int(c)] for c in fit_codecs], dtype=object)

    def apply_on(raw_map, chosen, br):
        out = np.zeros(len(chosen))
        for b in set(chosen):
            m = np.array([c == b for c in chosen])
            c = cal.get((br, b)) or cal.get((br, "clean"))
            out[m] = c.apply(raw_map[br][m])
        return out

    def fit_subset_for(chosen_fit):
        from sklearn.linear_model import LogisticRegression
        comb = (w["A"] * apply_on(fit_raw, chosen_fit, "A")
                + w["A2"] * apply_on(fit_raw, chosen_fit, "A2"))
        lr = LogisticRegression(class_weight="balanced", C=1e6,
                                solver="lbfgs", max_iter=1000)
        lr.fit(comb.reshape(-1, 1), fy)
        return float(lr.intercept_[0]), float(lr.coef_[0][0])

    conds = {
        "always-clean": (np.array(["clean"] * n, dtype=object),
                         np.array(["clean"] * len(fy), dtype=object)),
        "detected": (detected, fit_truth),   # calib fit uses truth: the
        "oracle": (truth, fit_truth),        # detector is 98.8% accurate there
    }

    rows = {}
    print(f"{'condition':<15}{'branch':<8}{'Cllr':>9}{'minCllr':>10}"
          f"{'calib loss':>13}")
    print("-" * 55)
    for name, (chosen, chosen_fit) in conds.items():
        alpha, scale = fit_subset_for(chosen_fit)
        rows[name] = {"subset": {"alpha": alpha, "scale": scale}}
        for br in ("A", "A2"):
            v = apply(br, chosen)
            c, mc = cllr(v, labels), min_cllr(v, labels)
            rows[name][br] = {"cllr": c, "min_cllr": mc, "loss": c - mc}
            print(f"{name:<15}{br:<8}{c:>9.3f}{mc:>10.3f}{c-mc:>13.3f}")
        fused = alpha + scale * (w["A"] * apply("A", chosen)
                                 + w["A2"] * apply("A2", chosen))
        fc, fmc = cllr(fused, labels), min_cllr(fused, labels)
        fe, _ = eer(fused, labels)
        rows[name]["fused"] = {"cllr": fc, "min_cllr": fmc, "loss": fc - fmc,
                               "eer": fe}
        print(f"{name:<15}{'A+A2':<8}{fc:>9.3f}{fmc:>10.3f}{fc-fmc:>13.3f}"
              f"   EER {fe*100:5.2f}%   (alpha={alpha:+.3f} scale={scale:.3f})")
        print()

    base = rows["always-clean"]["fused"]
    det = rows["detected"]["fused"]
    orc = rows["oracle"]["fused"]
    print("fused, relative to serving every call as clean:")
    print(f"  detected  Cllr {base['cllr']:.3f} -> {det['cllr']:.3f} "
          f"({(1-det['cllr']/base['cllr'])*100:+.1f}%)   "
          f"EER {base['eer']*100:.2f}% -> {det['eer']*100:.2f}%")
    print(f"  oracle    Cllr {base['cllr']:.3f} -> {orc['cllr']:.3f} "
          f"({(1-orc['cllr']/base['cllr'])*100:+.1f}%)   "
          f"EER {base['eer']*100:.2f}% -> {orc['eer']*100:.2f}%")
    gap = det["cllr"] - orc["cllr"]
    avail = base["cllr"] - orc["cllr"]
    if abs(avail) < 1e-6:
        print(f"  detection matches the oracle to {abs(gap):.4f} Cllr, but the "
              f"oracle itself is no better than always-clean here -- the "
              f"detector is not the limiting factor.")
    else:
        print(f"  detection leaves {gap:+.4f} Cllr against the oracle "
              f"({abs(gap)/abs(avail)*100:.1f}% of the available gain "
              f"unrealised)")

    # Per-branch is where the effect actually lives, and it is large. Print it
    # explicitly so a reader does not stop at the fused line and conclude the
    # feature does nothing.
    print("\nper branch, calibration loss (the part a mapping controls):")
    for br in ("A", "A2"):
        b0 = rows["always-clean"][br]["loss"]
        b1 = rows["detected"][br]["loss"]
        print(f"  {br:<4}{b0:.3f} -> {b1:.3f}  ({(1-b1/max(b0,1e-9))*100:+.0f}%)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
