#!/usr/bin/env python
"""Who does this system wrongly accuse, and is it the same rate for everyone?

Every accuracy number in this project is an EER or a recall -- aggregate figures
that say how often the system is right. None of them says **who it is wrong
about**, and for a fraud detector that is the number with a person on the other
end of it. A false positive here is not a metric: it is a real customer whose
payment is held, whose account is frozen, and who has to prove they are
themselves.

So: fix an operating threshold on one population, then measure the realised
false-positive rate on another. If the rates differ, the system distributes the
harm of being wrongly accused unevenly, and the group it falls on did nothing to
deserve it.

The axis measured here is **language**, because it is the one this project can
actually test and the one that matters for the deployment it is aimed at: the
models were trained on English, and RTCFake supplies Mandarin bona fide speech
recorded through the same platforms. That makes it a controlled comparison --
same corpus, same channels, same recording pipeline, different language.

**This is a lower bound on the problem, not a survey of it.** The population
that matters for an AICTE problem statement is Indian-language and
Indian-accented speakers, and this project has no bona fide corpus for them
(docs/RESULTS.md section 16: IndicSynth is spoof-only). Mandarin is the
available proxy for "a language the model never trained on", nothing more.
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

from vericall.data.cache import CachedFold
from vericall.fusion.calibration import load_calibration, logit
from vericall.fusion.config import FusionConfig, load_weights


@torch.inference_mode()
def branch_llr(fold, device, branch, ckpt, cal, batch=128):
    from vericall.branches.a2_phase import BranchA2
    from vericall.branches.a_ssl import BranchAConfig, BranchAHead
    blob = torch.load(ckpt, map_location=device, weights_only=True)
    m = (BranchAHead(768, BranchAConfig()) if branch == "A"
         else BranchA2()).to(device)
    m.load_state_dict(blob["state_dict"]); m.eval()
    src = fold.feat if branch == "A" else fold.wave
    raw = np.zeros(len(fold))
    for i in range(0, len(fold), batch):
        sel = np.arange(i, min(i + batch, len(fold)))
        x = torch.from_numpy(np.asarray(src[sel], dtype=np.float32)).to(device)
        raw[sel] = torch.softmax(m(x), -1)[:, 1].float().cpu().numpy()
    return cal.apply(logit(raw))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/vericall-cache")
    ap.add_argument("--reference", default="rtcfake:eval_rtc_en",
                    help="population the threshold is set on")
    ap.add_argument("--compare", nargs="+", default=["rtcfake:eval_rtc_zh"],
                    help="populations the threshold is then applied to")
    ap.add_argument("--calib-dir", default="calib_rtc")
    ap.add_argument("--a-ckpt", default="checkpoints/branch_A_rtc_seed0.pt")
    ap.add_argument("--a2-ckpt", default="checkpoints/branch_A2_rtc_seed0.pt")
    ap.add_argument("--fpr", type=float, nargs="+", default=[0.01, 0.05])
    ap.add_argument("--out", default="docs/fairness.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cdir = Path(args.calib_dir)
    cal_a = load_calibration(cdir / "A@pooled.json")
    cal_a2 = load_calibration(cdir / "A2@pooled.json")
    w = load_weights(cdir / "weights.json", FusionConfig().w)

    def fused_scores(spec):
        ds, name = spec.split(":")
        path = Path(args.cache) / ds / name
        if not path.exists():
            return None, None
        fold = CachedFold.load(path)
        a = branch_llr(fold, device, "A", args.a_ckpt, cal_a)
        a2 = branch_llr(fold, device, "A2", args.a2_ckpt, cal_a2)
        return w["A"] * a + w["A2"] * a2, np.asarray(fold.label)

    ref_s, ref_y = fused_scores(args.reference)
    if ref_s is None:
        print(f"{args.reference} not extracted", file=sys.stderr)
        return 2
    ref_bona = ref_s[ref_y == 0]
    print(f"reference population: {args.reference}  "
          f"{len(ref_bona):,} bona fide utterances\n")

    results = {"reference": args.reference, "populations": {}}
    # Both error types, always. Reporting only the false-positive rate is a
    # half-truth that reads as reassuring: a group the model scores LOWER is
    # flagged less often AND protected less often, and the second is a harm too
    # -- it is just the one that nobody complains about, because the victim of a
    # missed attack does not know the system was watching.
    print(f"{'target FPR':<12}{'threshold':>11}   population"
          f"{'':>10}{'bona fide n':>13}{'realised FPR':>14}{'ratio':>8}"
          f"{'miss rate':>13}{'ratio':>8}")
    print("-" * 103)
    for target in args.fpr:
        thr = float(np.quantile(ref_bona, 1.0 - target))
        ref_fpr = float((ref_bona >= thr).mean())
        ref_spoof = ref_s[ref_y == 1]
        ref_miss = (float((ref_spoof < thr).mean()) if len(ref_spoof)
                    else float("nan"))
        print(f"{target*100:>5.1f}%{'':<6}{thr:>11.3f}   "
              f"{args.reference:<20}{len(ref_bona):>13,}"
              f"{ref_fpr*100:>13.2f}%{1.0:>8.2f}{ref_miss*100:>12.1f}%"
              f"{1.0:>8.2f}")
        row = {"threshold": thr,
               args.reference: {"n": len(ref_bona), "fpr": ref_fpr,
                                "n_spoof": int(len(ref_spoof)),
                                "miss_rate": ref_miss}}
        for spec in args.compare:
            s, y = fused_scores(spec)
            if s is None:
                print(f"{'':<12}{'':>11}   {spec:<20}  (not extracted)")
                continue
            bona, spoof = s[y == 0], s[y == 1]
            fpr = float((bona >= thr).mean())
            miss = float((spoof < thr).mean()) if len(spoof) else float("nan")
            ratio = fpr / max(ref_fpr, 1e-9)
            miss_ratio = miss / max(ref_miss, 1e-9)
            print(f"{'':<12}{'':>11}   {spec:<20}{len(bona):>13,}"
                  f"{fpr*100:>13.2f}%{ratio:>8.2f}{miss*100:>13.1f}%"
                  f"{miss_ratio:>8.2f}")
            row[spec] = {"n": len(bona), "fpr": fpr,
                         "ratio_to_reference": ratio,
                         "n_spoof": int(len(spoof)), "miss_rate": miss,
                         "miss_ratio_to_reference": miss_ratio}
        results["populations"][f"{target:.2f}"] = row
        print()

    # The four-fifths rule from US employment law (EEOC) is not a legal standard
    # for fraud detection, but it is the most widely used rule of thumb for
    # "is this disparity large enough to act on", and a 5x gap needs no rule to
    # be obviously unacceptable.
    worst_fpr = worst_miss = 0.0
    for row in results["populations"].values():
        for v in row.values():
            if isinstance(v, dict) and "ratio_to_reference" in v:
                worst_fpr = max(worst_fpr, v["ratio_to_reference"])
                worst_miss = max(worst_miss, v.get("miss_ratio_to_reference", 0.0))
    results["worst_fpr_ratio"] = worst_fpr
    results["worst_miss_ratio"] = worst_miss
    print(f"worst false-positive disparity: {worst_fpr:.2f}x")
    print(f"worst miss-rate disparity:      {worst_miss:.2f}x")
    if worst_fpr >= 1.25:
        print("\n  -> OVER-ACCUSED. A customer in the compared population is "
              "materially more\n     likely to be wrongly flagged at the same "
              "operating point. That is a\n     deployment blocker, not a "
              "tuning note.")
    if worst_miss >= 1.25:
        print("\n  -> UNDER-PROTECTED. The compared population is flagged less "
              "often, and\n     that is not good news: at the same threshold "
              "more real attacks against\n     them get through. A group the "
              "model scores lower is both accused less\n     AND defended "
              "less, and only the first shows up as a complaint.")
    if worst_fpr < 1.25 and worst_miss < 1.25:
        print("\n  -> No material disparity in either direction on this axis "
              "at these\n     operating points.")
    print("\nAxes NOT measured: accent within a language, age, gender, "
          "speech disorders,\nrecording device. Absence of a measurement is "
          "not evidence of fairness.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
