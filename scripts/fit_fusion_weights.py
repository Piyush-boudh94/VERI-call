#!/usr/bin/env python
"""Fit the fusion weights, instead of carrying the ones somebody chose.

`FusionConfig.w` is {"A": 1.0, "A2": 0.45, "B": 0.0}. A and A2's weights were set
by hand and have never been fitted -- which was harmless while the branch LLRs
were uniformly mis-calibrated, and stopped being harmless the moment per-channel
calibration fixed them.

Measured (docs/RESULTS.md section 18): turning on runtime channel selection cuts
Branch A2's calibration loss by 93% and Branch A's by 18%, improves the fused
*ranking* (min-Cllr 0.057 -> 0.053), and makes the fused *calibration* worse
(Cllr 0.069 -> 0.082). Two errors were cancelling: hand-set weights that
over-weighted A2 against LLRs that were over-confident. Fix one and the other
stops being hidden.

So fit them. A logistic regression of the label on the per-branch LLRs gives
exactly the right object -- its coefficients ARE the weights and its intercept
IS the subset alpha, on the same LLR scale the fusion engine works in. Fitted on
calib, reported on dev, and the defaults are NOT changed by this script: it
prints what they should be so that changing them stays a deliberate act.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_subsets import score_A, score_A2                    # noqa: E402

from vericall.data.cache import CachedFold                   # noqa: E402
from vericall.fusion.calibration import (                    # noqa: E402
    cllr, eer, load_calibration, logit, min_cllr,
)
from vericall.fusion.config import FusionConfig              # noqa: E402

ID_TO_BUCKET = {0: "clean", 1: "g711", 2: "noisy_line"}


def per_channel_llr(fold, raw, branch, cdir):
    codecs = np.asarray(fold.codec)
    out = np.zeros_like(raw, dtype=float)
    for cid in sorted(set(int(c) for c in codecs)):
        b = ID_TO_BUCKET.get(cid, "pooled")
        path = cdir / f"{branch}@{b}.json"
        if not path.exists():
            path = cdir / f"{branch}@pooled.json"
        m = codecs == cid
        out[m] = load_calibration(path).apply(raw[m])
    return out


def report(name, s, y):
    c, mc = cllr(s, y), min_cllr(s, y)
    e, _ = eer(s, y)
    print(f"  {name:<26}EER {e*100:5.2f}%   Cllr {c:.3f}   "
          f"minCllr {mc:.3f}   loss {c-mc:.3f}")
    return {"eer": e, "cllr": c, "min_cllr": mc, "loss": c - mc}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/vericall-cache")
    ap.add_argument("--dataset", default="asvspoof2019_la")
    ap.add_argument("--fit-fold", default="calib")
    ap.add_argument("--eval-fold", default="dev")
    # The checkpoints were hardcoded to branch_A_seed0 / branch_A2_seed0, which
    # meant this script could only ever fit weights for the DEFAULT models --
    # and it did so silently against whatever calibration directory it was
    # pointed at. Fitting the Hindi profile that way reported "Branch A alone
    # 42.29%" (the ASVspoof number; the Hindi head scores 7.54% on the same
    # fold) and produced a degenerate w_A = -0.05 with w_A2 normalised to 8.09.
    # Weights are only valid for the heads they were fitted against, so the
    # heads have to be nameable.
    ap.add_argument("--ckpt-a", default="checkpoints/branch_A_seed0.pt")
    ap.add_argument("--ckpt-a2", default="checkpoints/branch_A2_seed0.pt")
    ap.add_argument("--calib-dir", default="calib")
    ap.add_argument("--out", default="docs/fusion_weights.json")
    ap.add_argument("--write", action="store_true",
                    help="also write <calib-dir>/weights.json, which the "
                         "gateway loads. Weights live in the calibration "
                         "directory rather than in FusionConfig because they "
                         "are only valid for the branch calibrations they were "
                         "fitted against -- a profile whose A2 mapping is "
                         "different needs a different A2 weight, and a global "
                         "default would silently apply one profile's weights "
                         "to another's LLRs.")
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cdir = Path(args.calib_dir)

    def llrs(fold_name):
        f = CachedFold.load(Path(args.cache) / args.dataset / fold_name)
        a = per_channel_llr(f, logit(score_A(f, device, args.ckpt_a)), "A", cdir)
        a2 = per_channel_llr(f, logit(score_A2(f, device, args.ckpt_a2)),
                             "A2", cdir)
        return f, np.asarray(f.label), a, a2

    ffit, yfit, afit, a2fit = llrs(args.fit_fold)
    fev, yev, aev, a2ev = llrs(args.eval_fold)
    print(f"fit on {args.fit_fold} ({len(ffit):,} rows), "
          f"evaluated on {args.eval_fold} ({len(fev):,} rows)\n")

    lr = LogisticRegression(class_weight="balanced", C=1e6, solver="lbfgs",
                            max_iter=1000)
    lr.fit(np.column_stack([afit, a2fit]), yfit)
    wa, wa2 = (float(v) for v in lr.coef_[0])
    alpha = float(lr.intercept_[0])
    # Report weights normalised so A = 1.0, which is the convention
    # FusionConfig uses -- the overall scale is absorbed by the subset alpha and
    # scale, so only the RATIO between branches is a free parameter.
    print(f"fitted:   w_A = {wa:.4f}   w_A2 = {wa2:.4f}   alpha = {alpha:+.4f}")
    print(f"normalised to w_A = 1: w_A2 = {wa2/wa:.4f}   "
          f"(current default {FusionConfig().weight_of('A2')})\n")

    cfg = FusionConfig()
    out = {"fitted": {"w_A": wa, "w_A2": wa2, "alpha": alpha},
           "w_A2_normalised": wa2 / wa,
           "current_default_w_A2": cfg.weight_of("A2"), "eval": {}}

    print(f"on {args.eval_fold}:")
    out["eval"]["hand_set"] = report(
        "hand-set 1.0 / 0.45",
        cfg.weight_of("A") * aev + cfg.weight_of("A2") * a2ev, yev)
    out["eval"]["fitted"] = report(
        f"fitted {wa:.2f} / {wa2:.2f}", alpha + wa * aev + wa2 * a2ev, yev)
    out["eval"]["A_alone"] = report("Branch A alone", aev, yev)
    out["eval"]["A2_alone"] = report("Branch A2 alone", a2ev, yev)

    h, f = out["eval"]["hand_set"], out["eval"]["fitted"]
    print(f"\nfitted vs hand-set:  Cllr {h['cllr']:.3f} -> {f['cllr']:.3f} "
          f"({(1-f['cllr']/max(h['cllr'],1e-9))*100:+.1f}%)   "
          f"calibration loss {h['loss']:.3f} -> {f['loss']:.3f}")
    best_single = min(out["eval"]["A_alone"]["eer"], out["eval"]["A2_alone"]["eer"])
    print(f"fused EER {f['eer']*100:.2f}% vs best single branch "
          f"{best_single*100:.2f}%  -> "
          f"{'helps' if f['eer'] < best_single else 'HURTS'}")
    if args.write:
        wpath = cdir / "weights.json"
        wpath.write_text(json.dumps(
            {"w": {"A": 1.0, "A2": round(wa2 / wa, 4)},
             "fitted_on": f"{args.dataset}:{args.fit_fold}",
             "note": ("Normalised to w_A = 1; the overall scale is absorbed by "
                      "the subset (alpha, scale). Valid only for the branch "
                      "calibrations in this directory.")}, indent=2) + "\n")
        print(f"\nwrote {wpath} -- now re-run "
              f"scripts/fit_subsets.py --per-channel so the subset table "
              f"matches these weights")
    else:
        print("\nNothing written to the calibration directory. Pass --write "
              "to adopt these weights, then re-run "
              "scripts/fit_subsets.py --per-channel.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
