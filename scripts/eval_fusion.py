#!/usr/bin/env python
"""Does refitting the fusion layer on a corpus stop it hurting on that corpus?

docs/RESULTS.md section 16 finding 4: with weights and a subset table fitted on
ASVspoof, fusion scored worse than its own best single branch in three of four
RTCFake conditions -- because the weights encode a branch hierarchy that
retraining changed. The diagnosis implies a fix. This measures whether the fix
works, on **held-out eval folds**, not on the calib fold the table was fitted on.

Fusion here is the single-window arithmetic the engine accumulates over time:

    S    = sum_k w_k * llr_k          weighted sum of calibrated branch LLRs
    S_K  = alpha_K + scale_K * S      per-subset recalibration for the live set K

Time discounting and eta play no part in a per-window EER, so this isolates the
combination rule from the accumulator.
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
from fit_subsets import score_A, score_A2               # noqa: E402

from vericall.data.cache import CachedFold               # noqa: E402
from vericall.fusion.calibration import (                # noqa: E402
    bootstrap_eer_ci, eer, load_calibration, logit,
)
from vericall.fusion.config import FusionConfig, load_subset_calib  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+",
                    default=["rtcfake:eval_rtc_en", "rtcfake:eval_rtc_zh"])
    ap.add_argument("--cache", default="/tmp/vericall-cache")
    ap.add_argument("--a-ckpt", default="checkpoints/branch_A_rtc_seed0.pt")
    ap.add_argument("--a2-ckpt", default="checkpoints/branch_A2_rtc_seed0.pt")
    ap.add_argument("--calib-dir", default="calib_rtc")
    ap.add_argument("--subsets", default="calib_rtc/subsets.json")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--out", default="docs/fusion_refit.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = FusionConfig()
    cdir = Path(args.calib_dir)
    cal_A = load_calibration(cdir / "A@pooled.json")
    cal_A2 = load_calibration(cdir / "A2@pooled.json")
    table = load_subset_calib(args.subsets)
    key = frozenset({"A", "A2"})
    if key not in table:
        print(f"subset table has no entry for {sorted(key)}", file=sys.stderr)
        return 2
    alpha, scale = table[key]
    print(f"weights  A={cfg.weight_of('A')}  A2={cfg.weight_of('A2')}")
    print(f"subset A+A2  alpha={alpha:+.3f}  scale={scale:.3f}\n")

    results: dict = {}
    print(f"{'fold':<26}{'A':>10}{'A2':>10}{'A+A2 fused':>14}   verdict")
    print("-" * 78)
    for spec in args.folds:
        ds, name = spec.split(":")
        path = Path(args.cache) / ds / name
        if not path.exists():
            print(f"{spec:<26}  (not extracted)")
            continue
        fold = CachedFold.load(path)
        y = np.asarray(fold.label)
        llr_a = cal_A.apply(logit(score_A(fold, device, args.a_ckpt)))
        llr_a2 = cal_A2.apply(logit(score_A2(fold, device, args.a2_ckpt)))
        fused = alpha + scale * (cfg.weight_of("A") * llr_a
                                 + cfg.weight_of("A2") * llr_a2)

        e_a = eer(llr_a, y)[0]
        e_a2 = eer(llr_a2, y)[0]
        e_f = eer(fused, y)[0]
        lo, hi = bootstrap_eer_ci(fused, y, fold.groups, n_boot=args.boot)[1:]
        best = min(e_a, e_a2)
        # The bar is the BEST SINGLE BRANCH, not the weakest one. Fusion that
        # beats its worst branch and loses to its best is a regression dressed
        # as a combination -- that is exactly what section 16 measured.
        verdict = "helps" if e_f < best else f"HURTS ({e_f/best:.2f}x worse)"
        print(f"{spec:<26}{e_a*100:>9.2f}%{e_a2*100:>9.2f}%{e_f*100:>13.2f}%"
              f"   {verdict}")
        results[spec] = {"A": e_a, "A2": e_a2, "fused": e_f,
                         "fused_ci95": [lo, hi], "best_single": best,
                         "helps": bool(e_f < best), "n": len(fold)}

    print()
    for spec, r in results.items():
        print(f"  {spec:<26}fused {r['fused']*100:5.2f}% "
              f"[{r['fused_ci95'][0]*100:.2f}-{r['fused_ci95'][1]*100:.2f}]"
              f"   best single {r['best_single']*100:5.2f}%")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
