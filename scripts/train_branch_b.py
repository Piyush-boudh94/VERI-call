#!/usr/bin/env python
"""Train Branch B: one-class prosody detector (Deep-SVDD).

Branch B sees **bona fide speech only**, by design. It learns what genuine
prosody looks like and scores departure from it, rather than separating two
classes. That matters for a detector meant to survive attack families it has
never seen: a two-class model can only recognise the spoofs it was shown, while
a one-class model flags anything that does not look like natural pitch and
rhythm.

The consequence the fusion layer must absorb is that the output is an unbounded
anomaly score with no probabilistic meaning and no negative distribution behind
it. It is mapped onto the shared LLR scale by
`fusion.calibration.one_class_to_llr` against a bona fide cohort, and hard
clipped there so an uncalibrated branch can never dominate a decision.

Collapse guard: the centre is a fixed buffer, never a parameter. If it were
learnable the objective would be trivially minimised by mapping every input to
one point -- the classic failure of one-class deep models.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vericall.branches.b_prosody import BranchB, BranchBConfig
from vericall.data.cache import CachedFold
from vericall.fusion.calibration import eer


@torch.inference_mode()
def embedding_spread(model: BranchB, fold: CachedFold, device,
                     n: int = 512) -> float:
    """Per-dimension std of the embeddings.

    The collapse detector. Deep-SVDD's failure mode is the encoder learning a
    constant function: loss goes to zero, every input lands on the centre, and
    EER sits at chance. Loss alone cannot distinguish that from success -- it
    looks *better* -- so spread is checked independently every epoch.
    """
    sel = np.linspace(0, len(fold) - 1, min(n, len(fold))).astype(int)
    x = torch.from_numpy(np.asarray(fold.wave[sel], dtype=np.float32)).to(device)
    c = torch.from_numpy(fold.codec[sel]).to(device).long()
    return float(model.embed(x, c).float().std(dim=0).mean())


COLLAPSE_STD = 0.05      # measured: a collapsed run sat at 0.0057


@torch.inference_mode()
def score_fold(model: BranchB, fold: CachedFold, device, batch: int = 128):
    out = np.zeros(len(fold), dtype=np.float64)
    for i in range(0, len(fold), batch):
        sel = np.arange(i, min(i + batch, len(fold)))
        x = torch.from_numpy(np.asarray(fold.wave[sel], dtype=np.float32)).to(device)
        c = torch.from_numpy(fold.codec[sel]).to(device).long()
        out[sel] = model.score(x, c).float().cpu().numpy()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/vericall-cache")
    ap.add_argument("--dataset", default="asvspoof2019_la")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoints")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = Path(args.cache) / args.dataset
    train = CachedFold.load(base / "train")
    dev = CachedFold.load(base / "dev")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    bona = np.flatnonzero(train.label == 0)
    print(f"Branch B (one-class) | device {device}")
    print(f"  train  {len(train):,} rows -> using {len(bona):,} BONA FIDE only")
    print(f"  dev    {len(dev):,} rows (both classes, for EER only)")

    model = BranchB(BranchBConfig()).to(device)

    init = torch.from_numpy(
        np.asarray(train.wave[bona[:512]], dtype=np.float32)
    ).to(device)
    centre = model.init_center(init)
    print(f"  centre initialised from 512 bona fide clips "
          f"(norm {float(centre.norm()):.3f})")

    params = [p for p in model.encoder.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-5)
    rng = np.random.default_rng(args.seed)

    best = {"eer": 1.0, "epoch": -1}
    history = []
    for epoch in range(args.epochs):
        model.train()
        idx = bona.copy()
        rng.shuffle(idx)
        total, n, t0 = 0.0, 0, time.time()
        for i in range(0, len(idx), args.batch):
            sel = np.sort(idx[i:i + args.batch])
            x = torch.from_numpy(np.asarray(train.wave[sel], dtype=np.float32)).to(device)
            c = torch.from_numpy(train.codec[sel]).to(device).long()
            loss = model(x, c).mean()          # pull bona fide toward the centre
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            total += float(loss) * len(sel)
            n += len(sel)

        model.eval()
        spread = embedding_spread(model, dev, device)
        scores = score_fold(model, dev, device)
        e, _ = eer(scores, dev.label)
        history.append({"epoch": epoch, "loss": total / max(n, 1),
                        "dev_eer": e, "spread": spread})
        if spread < COLLAPSE_STD:
            print(f"    epoch {epoch:2d}  COLLAPSE: embedding std {spread:.5f} "
                  f"< {COLLAPSE_STD} -- the encoder is emitting a constant.")
            print("    Deep-SVDD requires a bias-free network; check that every "
                  "Conv1d/Linear has bias=False and BatchNorm affine=False.")
            return 2
        flag = ""
        if e < best["eer"]:
            best = {"eer": e, "epoch": epoch}
            torch.save({"state_dict": model.state_dict(), "branch": "B",
                        "dev_eer": e, "seed": args.seed},
                       out_dir / "branch_B.pt")
            flag = "  *"
        print(f"    epoch {epoch:2d}  loss {total/max(n,1):9.4f}  "
              f"dev EER {e*100:5.2f}%  spread {spread:.4f}  "
              f"({time.time()-t0:.1f}s){flag}")

    print(f"\n  best dev EER {best['eer']*100:.2f}% at epoch {best['epoch']}")
    print("  NOTE: a one-class branch is expected to be weaker in-domain than")
    print("  the supervised branches. Its value is that it does not depend on")
    print("  having seen the attack, and that prosody survives codec")
    print("  compression better than fine spectral detail.")

    model.load_state_dict(torch.load(out_dir / "branch_B.pt", map_location=device,
                                     weights_only=False)["state_dict"])
    model.eval()
    calib = CachedFold.load(base / "calib")
    cs = score_fold(model, calib, device)
    cohort = cs[calib.label == 0]
    Path("calib").mkdir(exist_ok=True)
    np.save("calib/B_bonafide_cohort.npy", cohort)
    ce, _ = eer(cs, calib.label)
    print(f"\n  wrote calib/B_bonafide_cohort.npy ({len(cohort):,} bona fide scores)")
    print(f"  calib-fold EER {ce*100:.2f}%")

    (out_dir / "results_B.json").write_text(json.dumps(
        {"best": best, "history": history, "calib_eer": ce}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
