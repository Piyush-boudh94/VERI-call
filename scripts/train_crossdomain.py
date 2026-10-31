#!/usr/bin/env python
"""Train Branch A to survive a change of corpus.

docs/RESULTS.md section 16 is the reason this exists. Measured there:

  * trained on ASVspoof -> 11.35% in-domain, and CHANCE on RTCFake (55.34% zh,
    which is worse than guessing -- the branch had inverted)
  * trained on RTCFake  -> 22.93% there, and chance on ASVspoof and WaveFake
  * trained on BOTH, naively concatenated -> best in-domain number in the
    project (9.12%) and WaveFake DESTROYED, 30.08% -> 49.89%, with ASVspoof
    still 64% of the training rows

That last result is the interesting one: the mixture did not dilute WaveFake
performance in proportion, it removed it. The mechanism offered was that a naive
mixture lets the model take whichever cue is cheapest across the mixture, and
"which corpus is this" is extremely cheap when one corpus is clean studio audio,
another is single-speaker resynthesis and a third is RTC telephony.

Section 16 named three fixes and tested none. Two are here:

  --balanced   equal draws per corpus per batch, so no corpus can dominate by
               row count alone
  --dann       a domain classifier on the pooled representation behind a
               gradient reversal layer, so corpus identity becomes actively
               unusable rather than merely under-weighted

Evaluated against the same four folds every other configuration was, through
scripts/eval_folds.py, so the rows are comparable to the published matrix.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vericall.branches.a_ssl import BranchAConfig, BranchAHead
from vericall.data.cache import CachedFold, ConcatFold, class_weights
from vericall.fusion.calibration import bootstrap_eer_ci, eer
from vericall.train.dann import DomainHead, dann_lambda
from vericall.train.ocsoftmax import OCSoftmax


def load_split(cache: Path, datasets, name: str):
    parts, names = [], []
    for ds in datasets:
        p = cache / ds / name
        if p.exists():
            parts.append(CachedFold.load(p))
            names.append(ds)
        else:
            print(f"  ({ds}:{name} not extracted; skipping)")
    if not parts:
        raise SystemExit(f"no '{name}' fold for {datasets}")
    return parts[0] if len(parts) == 1 else ConcatFold(parts, names)


@torch.no_grad()
def evaluate(head, fold, device, batch=256, oc=None):
    head.eval()
    if oc is not None:
        oc.eval()
    scores = np.zeros(len(fold))
    for x, y, c, sel in fold.batches(batch, shuffle=False, use="feat",
                                     device=device):
        if oc is not None:
            logits = oc.to_logits(head.pooled(x))
        else:
            logits = head(x)
        scores[sel] = torch.softmax(logits, -1)[:, 1].float().cpu().numpy()
    head.train()
    if oc is not None:
        oc.train()
    return eer(scores, np.asarray(fold.label))[0], scores


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/vericall-cache")
    ap.add_argument("--datasets", nargs="+",
                    default=["asvspoof2019_la", "rtcfake"])
    ap.add_argument("--eval-on", nargs="*", default=[
        "asvspoof2019_la:eval_in_domain", "wavefake:eval_unseen_vocoder",
        "rtcfake:eval_rtc_en", "rtcfake:eval_rtc_zh"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--balanced", action="store_true")
    ap.add_argument("--dann", action="store_true")
    ap.add_argument("--ocsoftmax", action="store_true",
                    help="one-class softmax instead of 2-class cross-entropy. "
                         "Asks the model to enclose bona fide rather than "
                         "separate two closed sets, which is the right shape "
                         "for an open set of future attacks.")
    ap.add_argument("--max-lambda", type=float, default=1.0)
    ap.add_argument("--train-fold", default="train",
                    help="fold name to train on. The layer sweep passes "
                         "train@L7 and friends, which are complete folds "
                         "written by extract_features --layers.")
    ap.add_argument("--dev-fold", default="dev")
    ap.add_argument("--tag", default="_xd")
    ap.add_argument("--out", default="checkpoints")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = Path(args.cache)
    train_fold = load_split(cache, args.datasets, args.train_fold)
    dev_fold = load_split(cache, args.datasets, args.dev_fold)
    evals = {}
    for spec in args.eval_on:
        ds, name = spec.split(":")
        p = cache / ds / name
        if p.exists():
            evals[spec] = CachedFold.load(p)
        else:
            print(f"  (eval {spec} not extracted; skipping)")

    n_domains = len(getattr(train_fold, "names", [args.datasets[0]]))
    print(f"corpora {'+'.join(args.datasets)} | balanced={args.balanced} "
          f"| dann={args.dann} | domains={n_domains}")
    print(f"  train {len(train_fold):,} rows  dev {len(dev_fold):,}")
    if hasattr(train_fold, "names"):
        for n, c in zip(train_fold.names, train_fold.info["rows"]):
            print(f"        {n:<20}{c:,} ({c/len(train_fold)*100:.0f}%)")
    if args.dann and n_domains < 2:
        raise SystemExit("--dann needs at least two corpora to be adversarial "
                         "about; with one domain the reversal is a no-op that "
                         "just adds a useless head.")

    results = []
    for seed in args.seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        head = BranchAHead(768, BranchAConfig()).to(device)
        dom = DomainHead(BranchAConfig().hidden * 4, n_domains).to(device) \
            if args.dann else None
        oc = OCSoftmax(BranchAConfig().hidden * 4).to(device) \
            if args.ocsoftmax else None

        w = class_weights(train_fold.label).to(device)
        lossfn = nn.CrossEntropyLoss(weight=w)
        dom_loss = nn.CrossEntropyLoss()
        params = (list(head.parameters())
                  + (list(dom.parameters()) if dom else [])
                  + (list(oc.parameters()) if oc else []))
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

        best = {"eer": 1.0, "epoch": -1, "state": None}
        step, total_steps = 0, args.epochs * max(1, len(train_fold) // args.batch)
        print(f"\n--- seed {seed} ---")
        for epoch in range(args.epochs):
            t0 = time.time()
            tot = totd = nb = 0.0
            for x, y, c, sel in train_fold.batches(
                    args.batch, seed=seed * 1000 + epoch, use="feat",
                    device=device, balanced=args.balanced):
                opt.zero_grad(set_to_none=True)
                pooled = head.pooled(x)
                if oc is not None:
                    loss, _ = oc(pooled, y)
                else:
                    loss = lossfn(head.classifier(pooled), y)
                dl = torch.zeros((), device=device)
                if dom is not None:
                    lam = dann_lambda(step, total_steps, args.max_lambda)
                    d = torch.from_numpy(train_fold.domain_ids(sel)).to(device)
                    dl = dom_loss(dom(pooled, lam), d)
                    loss = loss + dl
                loss.backward()
                opt.step()
                tot += float(loss); totd += float(dl); nb += 1; step += 1
            sched.step()

            dev_eer, _ = evaluate(head, dev_fold, device, oc=oc)
            extra = "  ".join(
                f"{k.split(':')[-1][:14]} {evaluate(head, f, device, oc=oc)[0]*100:5.2f}%"
                for k, f in evals.items())
            lam_now = dann_lambda(step, total_steps, args.max_lambda) if dom else 0.0
            print(f"    epoch {epoch:2d}  loss {tot/max(nb,1):.4f}"
                  + (f"  dom {totd/max(nb,1):.3f} lam {lam_now:.2f}" if dom else "")
                  + f"  dev {dev_eer*100:5.2f}%  {extra}  ({time.time()-t0:.0f}s)",
                  flush=True)
            if dev_eer < best["eer"]:
                best = {"eer": dev_eer, "epoch": epoch,
                        "state": {k: v.detach().cpu().clone()
                                  for k, v in head.state_dict().items()},
                        "oc": ({k: v.detach().cpu().clone()
                                for k, v in oc.state_dict().items()}
                               if oc else None)}

        head.load_state_dict(best["state"])
        if oc is not None and best.get("oc"):
            oc.load_state_dict(best["oc"])
        row = {"seed": seed, "best_epoch": best["epoch"], "dev_eer": best["eer"]}
        for spec, f in evals.items():
            e, sc = evaluate(head, f, device, oc=oc)
            lo, hi = bootstrap_eer_ci(sc, np.asarray(f.label), f.groups,
                                      n_boot=200)[1:]
            row[spec] = {"eer": e, "ci95": [lo, hi]}
        results.append(row)
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        blob = {"state_dict": head.state_dict(), "branch": "A", "seed": seed,
                "dev_eer": best["eer"], "config": vars(args)}
        if oc is not None:
            # Saved alongside, because a head trained with OC-Softmax scores
            # through the one-class centre. Loading it without this and reading
            # head(x) would use a classifier that was never trained -- and would
            # not error, only score badly.
            blob["ocsoftmax"] = oc.state_dict()
        torch.save(blob, out / f"branch_A{args.tag}_seed{seed}.pt")

    print("\n" + "=" * 78)
    print(f"{'fold':<34}{'EER':>9}{'95% CI':>18}")
    print("-" * 78)
    dev = np.array([r["dev_eer"] for r in results])
    print(f"{'dev (mixed)':<34}{dev.mean()*100:>8.2f}%")
    for spec in evals:
        v = np.array([r[spec]["eer"] for r in results])
        ci = results[0][spec]["ci95"]
        flag = "  <-- INVERTED" if v.mean() > 0.5 else ""
        print(f"{spec:<34}{v.mean()*100:>8.2f}%"
              f"{ci[0]*100:>10.2f}-{ci[1]*100:.2f}%{flag}")
    Path(f"{args.out}/results_A{args.tag}.json").write_text(
        json.dumps(results, indent=2))
    print(f"\nwrote {args.out}/results_A{args.tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
