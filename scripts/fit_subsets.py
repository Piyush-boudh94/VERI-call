#!/usr/bin/env python
"""Fit the per-subset calibration table -- the piece that makes a missing branch
a dropped term rather than a different model.

The fusion engine combines branch LLRs as a weighted sum. But which branches are
live varies constantly: Branch B has not fired in the first 3 s of any call,
Branch C is absent for every unenrolled caller, and under load the degradation
ladder sheds branches deliberately. A sum over {A} and a sum over {A, A2, B} are
on different scales, so a single threshold cannot serve both.

The alternative -- one model per availability subset, or imputing the missing
branch -- is worse. Imputing an input to a *detector* means inventing evidence.

So: two scalars (alpha_K, scale_K) per subset, fit by logistic regression on
CACHED dev-set branch outputs. No model is retrained, no GPU is touched, the
whole table is a couple of kB of JSON and refits in seconds. Without it every
subset silently falls back to (0.0, 1.0), which is only correct for whichever
subset the weights happened to be tuned on.
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
from vericall.branches.a_ssl import BranchA, BranchAConfig, BranchAHead
from vericall.branches.b_prosody import BranchB, BranchBConfig
from vericall.data.cache import CachedFold
from vericall.fusion.calibration import (
    eer, fit_platt, fit_subset_calibration, load_calibration, logit,
    one_class_to_llr,
)
from vericall.fusion.config import FusionConfig, load_weights

SUBSETS = [("A",), ("A", "A2"), ("A", "A2", "B"), ("A", "B"), ("A2",), ("A2", "B")]


@torch.inference_mode()
def score_A(fold, device, ckpt):
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    head = BranchAHead(768, BranchAConfig()).to(device)
    head.load_state_dict(blob["state_dict"]); head.eval()
    out = np.zeros(len(fold))
    for x, _, _, sel in fold.batches(256, shuffle=False, use="feat", device=device):
        out[sel] = torch.softmax(head(x), -1)[:, 1].float().cpu().numpy()
    return out


@torch.inference_mode()
def score_A2(fold, device, ckpt):
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    m = BranchA2(BranchA2Config()).to(device)
    m.load_state_dict(blob["state_dict"]); m.eval()
    out = np.zeros(len(fold))
    for x, _, _, sel in fold.batches(128, shuffle=False, use="wave", device=device):
        out[sel] = torch.softmax(m(x), -1)[:, 1].float().cpu().numpy()
    return out


@torch.inference_mode()
def score_B(fold, device, ckpt):
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    m = BranchB(BranchBConfig()).to(device)
    m.load_state_dict(blob["state_dict"]); m.eval()
    out = np.zeros(len(fold))
    for i in range(0, len(fold), 128):
        sel = np.arange(i, min(i + 128, len(fold)))
        x = torch.from_numpy(np.asarray(fold.wave[sel], dtype=np.float32)).to(device)
        c = torch.from_numpy(fold.codec[sel]).to(device).long()
        out[sel] = m.score(x, c).float().cpu().numpy()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/vericall-cache")
    ap.add_argument("--dataset", default="asvspoof2019_la")
    ap.add_argument("--fold", default="calib")
    ap.add_argument("--out", default="calib/subsets.json")
    ap.add_argument("--a-ckpt", default="checkpoints/branch_A_seed0.pt")
    ap.add_argument("--a2-ckpt", default="checkpoints/branch_A2_seed0.pt")
    ap.add_argument("--calib-dir", default="calib",
                    help="where per-branch Platt files are read and written. "
                         "Point a refit at its own directory: a table fitted "
                         "for one corpus's heads must not overwrite another's.")
    ap.add_argument("--per-channel", action="store_true",
                    help="apply each row's per-channel Platt bucket (by codec "
                         "id) instead of one pooled mapping. Use this when the "
                         "gateway has runtime codec detection on: the subset "
                         "table must be fitted against the LLRs serving will "
                         "actually produce, or the weights compensate for a "
                         "mis-calibration that no longer happens.")
    ap.add_argument("--fit-a", action="store_true",
                    help="fit Branch A's Platt mapping on this fold instead of "
                         "loading it. Required when --a-ckpt is a head the "
                         "existing A@pooled.json was not fitted for -- applying "
                         "one head's calibration to another's scores produces "
                         "confident nonsense, not a degraded number.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold = CachedFold.load(Path(args.cache) / args.dataset / args.fold)
    labels = np.asarray(fold.label)
    print(f"{args.fold} fold: {len(fold):,} rows "
          f"({int((labels==0).sum()):,} bona / {int((labels==1).sum()):,} spoof)")

    llrs: dict[str, np.ndarray] = {}

    from vericall.fusion.calibration import save_calibration
    cdir = Path(args.calib_dir)
    cdir.mkdir(parents=True, exist_ok=True)

    ID_TO_BUCKET = {0: "clean", 1: "g711", 2: "noisy_line"}

    def per_channel(raw: np.ndarray, branch: str) -> np.ndarray:
        """Map each row through the bucket its codec id names."""
        codecs = np.asarray(fold.codec)
        out = np.zeros_like(raw, dtype=float)
        used = []
        for cid in sorted(set(int(c) for c in codecs)):
            b = ID_TO_BUCKET.get(cid, "pooled")
            path = cdir / f"{branch}@{b}.json"
            if not path.exists():
                path = cdir / f"{branch}@pooled.json"
            m = codecs == cid
            out[m] = load_calibration(path).apply(raw[m])
            used.append(f"{b}:{int(m.sum())}")
        print(f"  {branch}   per-channel buckets applied ({', '.join(used)})")
        return out

    ck = Path(args.a_ckpt)
    raw_a = logit(score_A(fold, device, ck))
    if args.per_channel:
        llrs["A"] = per_channel(raw_a, "A")
        print(f"  A   EER {eer(llrs['A'], labels)[0]*100:5.2f}%   [{ck.name}]")
    elif args.fit_a:
        cal_A = fit_platt(raw_a, labels, branch="A", channel_bucket="pooled",
                          calib_id="A@pooled")
        save_calibration(cdir / "A@pooled.json", cal_A)
        print(f"  A   fitted on this fold "
              f"(calibration loss {cal_A.calibration_loss:.3f})")
    else:
        cal_A = load_calibration(cdir / "A@pooled.json")
    if not args.per_channel:
        llrs["A"] = cal_A.apply(raw_a)
        print(f"  A   EER {eer(llrs['A'], labels)[0]*100:5.2f}%   [{ck.name}]")

    ck2 = Path(args.a2_ckpt)
    if ck2.exists() and args.per_channel:
        llrs["A2"] = per_channel(logit(score_A2(fold, device, ck2)), "A2")
        print(f"  A2  EER {eer(llrs['A2'], labels)[0]*100:5.2f}%   [{ck2.name}]")
    elif ck2.exists():
        raw2 = score_A2(fold, device, ck2)
        cal2 = fit_platt(logit(raw2), labels, branch="A2", channel_bucket="pooled",
                         calib_id="A2@pooled")
        save_calibration(cdir / "A2@pooled.json", cal2)
        llrs["A2"] = cal2.apply(logit(raw2))
        print(f"  A2  EER {eer(llrs['A2'], labels)[0]*100:5.2f}%  "
              f"(calibration loss {cal2.calibration_loss:.3f})   [{ck2.name}]")
    else:
        print("  A2  checkpoint missing -- skipping")

    ckb = Path("checkpoints/branch_B.pt")
    cohort_path = cdir / "B_bonafide_cohort.npy"
    if ckb.exists() and cohort_path.exists():
        rawb = score_B(fold, device, ckb)
        cohort = np.load(cohort_path)
        cfg = FusionConfig()
        llrs["B"] = one_class_to_llr(rawb, cohort, kappa=cfg.kappa_B,
                                     floor=cfg.floor_B)
        print(f"  B   EER {eer(llrs['B'], labels)[0]*100:5.2f}%  "
              f"(clipped to [{cfg.floor_B}, {cfg.kappa_B}] nats)")
    else:
        print("  B   checkpoint or cohort missing -- skipping")

    available = [s for s in SUBSETS if all(b in llrs for b in s)]
    if not available:
        print("\nno subset has all its branches; nothing to fit")
        return 1

    # The subset table is fitted FOR a set of weights: (alpha, scale) absorbs
    # whatever scale the weighted sum lands on. Fitting it against the default
    # weights while the gateway serves fitted ones from this same directory
    # would recalibrate a sum nobody computes.
    w = load_weights(cdir / "weights.json", FusionConfig().w)
    if (cdir / "weights.json").exists():
        print(f"\nusing fitted weights from {cdir}/weights.json: "
              + "  ".join(f"{k}={v}" for k, v in sorted(w.items())))
    table = fit_subset_calibration(llrs, labels, w, available)

    print(f"\nper-subset calibration ({len(table)} subsets):")
    print(f"  {'subset':<16}{'alpha':>9}{'scale':>9}{'EER':>9}")
    out: dict[str, list[float]] = {}
    for K, (alpha, scale) in table.items():
        key = "+".join(sorted(K))
        # `w`, not `cfg.w`. Two defects in the original line:
        #
        #   1. `cfg` is only bound inside the Branch B block above, so this
        #      raised NameError on any profile where B's checkpoint or cohort
        #      is absent -- which is every fresh calibration directory. The
        #      --per-channel path recommended in this script's own help text
        #      had therefore never run to completion.
        #   2. cfg.w is the DEFAULT weight set, while the table three lines up
        #      was fitted against the LOADED weights. The comment above that
        #      fit says fitting against defaults "would recalibrate a sum
        #      nobody computes" -- and then the EER column printed beside it
        #      was computed exactly that way, so the number reported next to
        #      each subset did not describe the mapping being written.
        combined = sum(w.get(b, 0.0) * llrs[b] for b in K)
        e, _ = eer(combined, labels)
        print(f"  {key:<16}{alpha:>9.3f}{scale:>9.3f}{e*100:>8.2f}%")
        out[key] = [alpha, scale]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")

    # Provenance: which heads this table was fitted FOR. A Platt mapping applied
    # to a head it was not fitted for produces confident nonsense rather than a
    # degraded number, and nothing downstream can tell from the numbers alone.
    # The serving gateway compares this against the checkpoints it loaded and
    # says so in /health.
    prov = {"fitted_for": {"A": Path(args.a_ckpt).name,
                           "A2": Path(args.a2_ckpt).name},
            "dataset": args.dataset, "fold": args.fold,
            "rows": int(len(fold))}
    (cdir / "provenance.json").write_text(json.dumps(prov, indent=2))
    print(f"wrote {cdir / 'provenance.json'}  (fitted for "
          f"{prov['fitted_for']['A']} + {prov['fitted_for']['A2']} "
          f"on {args.dataset}:{args.fold})")
    print("Load into FusionConfig.subset_calib as {frozenset(k.split('+')): tuple(v)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
