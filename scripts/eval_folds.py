#!/usr/bin/env python
"""Score one or more Branch A/A2 checkpoints against one or more cached folds.

Why this exists as its own script: every cross-domain number in docs/RESULTS.md
so far was produced at the end of the training run that made the checkpoint, so
comparing "the ASVspoof model" against "the RTCFake model" meant comparing two
numbers produced by two different runs on two different fold definitions. That
is not a comparison, it is two facts.

Here every checkpoint meets every fold through one code path, so a row and a
column are directly subtractable.

    scripts/eval_folds.py \
        --ckpt checkpoints/branch_A_seed0.pt checkpoints/branch_A_rtc_seed0.pt \
        --folds asvspoof2019_la:eval_in_domain rtcfake:eval_rtc_en rtcfake:eval_rtc_zh

EER above 50% is reported as INVERTED rather than quietly printed, because an
inverted branch is a different failure from an uninformative one: flipped, it
carries signal, and fusion will amplify it in the wrong direction. That is the
failure docs/RESULTS.md section 13 found and section 14 built a guard for.
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

from vericall.branches.a2_phase import BranchA2
from vericall.branches.a_ssl import BranchAConfig, BranchAHead
from vericall.data.cache import CachedFold
from vericall.fusion.calibration import bootstrap_eer_ci, eer


def load_head(path: str, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    branch = ck.get("branch", "A")
    head = (BranchAHead(768, BranchAConfig()) if branch == "A"
            else BranchA2()).to(device)
    head.load_state_dict(ck["state_dict"])
    head.eval()

    # A head trained with OC-Softmax scores through its one-class centre, not
    # through the 2-way classifier -- which in that run was never trained.
    # Reading head(x) on such a checkpoint does not error, it just scores
    # badly, so the checkpoint carries the OC head and we use it when present.
    oc = None
    if ck.get("ocsoftmax"):
        from vericall.train.ocsoftmax import OCSoftmax
        oc = OCSoftmax(BranchAConfig().hidden * 4).to(device)
        oc.load_state_dict(ck["ocsoftmax"])
        oc.eval()
    return (head, oc), ("feat" if branch == "A" else "wave")


@torch.no_grad()
def score_fold(head_pair, fold: CachedFold, device, use: str, batch: int = 256):
    """P(spoof), matching train.py's evaluate() exactly.

    Same convention deliberately: a matrix whose rows were produced by a
    different scoring rule than the numbers already in docs/RESULTS.md would
    invite subtraction between two things that are not comparable.
    """
    head, oc = head_pair
    scores = np.zeros(len(fold), dtype=np.float64)
    for x, y, c, sel in fold.batches(batch, shuffle=False, use=use,
                                     device=device):
        logits = oc.to_logits(head.pooled(x)) if oc is not None else head(x)
        scores[sel] = torch.softmax(logits, dim=-1)[:, 1].float().cpu().numpy()
    return scores


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--folds", nargs="+", required=True,
                    help="dataset:fold, e.g. rtcfake:eval_rtc_zh")
    ap.add_argument("--cache", default="/tmp/vericall-cache")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--out", default="docs/eval_matrix.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = {}
    for spec in args.folds:
        ds, name = spec.split(":")
        p = Path(args.cache) / ds / name
        if not p.exists():
            print(f"  ({spec} not extracted; skipping)")
            continue
        folds[spec] = CachedFold.load(p)

    if not folds:
        print("no folds available", file=sys.stderr)
        return 2

    results: dict = {}
    width = max(len(s) for s in folds) + 2
    print(f"{'checkpoint':<38}" + "".join(f"{s:>{width}}" for s in folds))
    print("-" * (38 + width * len(folds)))

    for ck in args.ckpt:
        if not Path(ck).exists():
            print(f"{Path(ck).name:<38}  (missing)")
            continue
        head, use = load_head(ck, device)
        row, cells = {}, []
        for spec, fold in folds.items():
            sc = score_fold(head, fold, device, use)
            e, _ = eer(sc, fold.label)
            lo, hi = bootstrap_eer_ci(sc, fold.label, fold.groups,
                                      n_boot=args.boot)[1:]
            row[spec] = {"eer": e, "ci95": [lo, hi], "n": len(fold),
                         "inverted": e > 0.5}
            cells.append(f"{e*100:>{width-1}.2f}%")
        results[ck] = row
        print(f"{Path(ck).name:<38}" + "".join(cells))

    print()
    for ck, row in results.items():
        for spec, r in row.items():
            flag = "  <-- INVERTED (anti-correlated, not merely uninformative)" \
                if r["inverted"] else ""
            print(f"  {Path(ck).name:<34}{spec:<26}"
                  f"{r['eer']*100:6.2f}%  "
                  f"[{r['ci95'][0]*100:.2f}-{r['ci95'][1]*100:.2f}]{flag}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
