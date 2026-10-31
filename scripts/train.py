#!/usr/bin/env python
"""Train the branch heads on the extracted cache.

Discipline enforced here, each for a stated reason:

* **Early stopping on dev EER, not dev loss.** With ~1:8.8 class imbalance the
  two diverge, and stopping on loss silently selects a worse operating point.
* **Prior-weighted loss**, so the output stays prior-free.
* **An overfit-a-tiny-subset check before the real run.** If 100 samples cannot
  be driven to ~0 loss there is a capacity or optimisation bug, and it is much
  cheaper to find that on day one than in week three.
* **The duration-only baseline must be beaten.** `scripts/audit_silence.py`
  shows leading-silence duration alone scores 36.4% EER on raw ASVspoof. A model
  that does not clearly beat that has learned a stopwatch.
* **Multiple seeds**, reported as mean +/- spread rather than a single lucky run.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vericall.branches.a2_phase import BranchA2
from vericall.branches.a_ssl import BranchAConfig, BranchAHead
from vericall.data.cache import CachedFold, ConcatFold, class_weights
from vericall.fusion.calibration import bootstrap_eer_ci, eer

DURATION_BASELINE_EER = 0.364      # measured, scripts/audit_silence.py


def evaluate(head: nn.Module, fold: CachedFold, device, use: str,
             batch: int = 256) -> tuple[float, np.ndarray]:
    head.eval()
    scores = np.zeros(len(fold), dtype=np.float64)
    with torch.inference_mode():
        for x, y, c, sel in fold.batches(batch, shuffle=False, use=use,
                                          device=device):
            p = torch.softmax(head(x), dim=-1)[:, 1]
            scores[sel] = p.float().cpu().numpy()
    head.train()
    e, _ = eer(scores, fold.label)
    return e, scores


def overfit_check(head: nn.Module, fold: CachedFold, device, use: str,
                  n: int = 100, steps: int = 150) -> float:
    """Can the model memorise 100 samples? If not, something is broken."""
    import copy

    probe = copy.deepcopy(head).to(device).train()
    opt = torch.optim.AdamW(probe.parameters(), lr=3e-3)
    lossfn = nn.CrossEntropyLoss()
    src = fold.feat if use == "feat" else fold.wave
    idx = np.r_[np.flatnonzero(fold.label == 0)[: n // 2],
                np.flatnonzero(fold.label == 1)[: n // 2]]
    x = torch.from_numpy(np.asarray(src[idx], dtype=np.float32)).to(device)
    y = torch.from_numpy(fold.label[idx]).to(device)
    loss = float("nan")
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        l = lossfn(probe(x), y)
        l.backward()
        opt.step()
        loss = float(l)
    return loss


def train_one(seed: int, train_fold, dev_fold, evals: dict, args, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if args.branch == "A":
        head = BranchAHead(768, BranchAConfig()).to(device)
        use = "feat"
    else:
        head = BranchA2().to(device)
        use = "wave"

    if seed == args.seeds[0]:
        loss0 = overfit_check(head, train_fold, device, use)
        status = "PASS" if loss0 < 0.05 else "FAIL"
        print(f"  overfit-100 sanity check: final loss {loss0:.4f}  [{status}]")
        if status == "FAIL":
            print("  ^ cannot memorise 100 samples: capacity or optimisation bug")

    w = class_weights(train_fold.label).to(device)
    lossfn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = {"eer": 1.0, "epoch": -1, "state": None}
    history = []
    for epoch in range(args.epochs):
        t0 = time.time()
        head.train()
        tot, nb = 0.0, 0
        for x, y, c, _ in train_fold.batches(args.batch, seed=seed * 1000 + epoch,
                                             use=use, device=device):
            opt.zero_grad(set_to_none=True)
            l = lossfn(head(x), y)
            l.backward()
            opt.step()
            tot += float(l); nb += 1
        sched.step()

        dev_eer, _ = evaluate(head, dev_fold, device, use)
        row = {"epoch": epoch, "loss": tot / max(nb, 1), "dev_eer": dev_eer,
               "secs": round(time.time() - t0, 1)}
        # the most informative diagnostic in the project: if in-domain improves
        # while cross-domain worsens, that is overfitting in progress
        for name, f in evals.items():
            row[f"{name}_eer"] = evaluate(head, f, device, use)[0]
        history.append(row)

        if dev_eer < best["eer"]:
            best = {"eer": dev_eer, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone()
                              for k, v in head.state_dict().items()}}
        extra = "  ".join(f"{k.replace('_eer','')} {row[k]*100:5.2f}%"
                          for k in row if k.endswith("_eer") and k != "dev_eer")
        print(f"    epoch {epoch:2d}  loss {row['loss']:.4f}  "
              f"dev {dev_eer*100:5.2f}%  {extra}  ({row['secs']:.1f}s)", flush=True)

    head.load_state_dict(best["state"])
    return head, best, history


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", choices=["A", "A2"], default="A")
    ap.add_argument("--cache", default="/tmp/vericall-cache")
    ap.add_argument("--dataset", nargs="+", default=["asvspoof2019_la"],
                    help="one or more corpora. Several are concatenated into "
                         "one train/dev fold -- the fix for the cross-domain "
                         "collapse in docs/RESULTS.md section 13, where a model "
                         "fitted on one corpus inverted on another.")
    ap.add_argument("--tag", default="",
                    help="suffix for checkpoint filenames, so a mixed-corpus "
                         "run does not overwrite the single-corpus baseline")
    ap.add_argument("--eval-on", nargs="*", default=[])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="checkpoints")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = list(args.dataset)

    def load_split(name: str):
        parts, names = [], []
        for ds in datasets:
            p = Path(args.cache) / ds / name
            if p.exists():
                parts.append(CachedFold.load(p))
                names.append(ds)
            else:
                print(f"  ({ds}:{name} not extracted; skipping)")
        if not parts:
            raise SystemExit(f"no '{name}' fold found for {datasets}")
        return parts[0] if len(parts) == 1 else ConcatFold(parts, names)

    train_fold = load_split("train")
    dev_fold = load_split("dev")
    evals = {}
    for spec in args.eval_on:
        ds, fold = spec.split(":") if ":" in spec else (datasets[0], spec)
        p = Path(args.cache) / ds / fold
        if p.exists():
            evals[f"{ds.split('_')[0]}:{fold}"] = CachedFold.load(p)
        else:
            print(f"  (eval fold {spec} not extracted; skipping)")

    print(f"Branch {args.branch} | device {device} | corpora {'+'.join(datasets)}")
    print(f"  train {len(train_fold):,} rows  "
          f"({int((train_fold.label==0).sum()):,} bona / "
          f"{int((train_fold.label==1).sum()):,} spoof)")
    if isinstance(train_fold, ConcatFold):
        # A pooled row count hides which corpus the model is actually fitting.
        for n, c in zip(train_fold.names, train_fold.info["rows"]):
            print(f"          {n:<20} {c:,} rows ({c/len(train_fold)*100:.0f}%)")
    print(f"  dev   {len(dev_fold):,} rows")
    for k, v in evals.items():
        print(f"  eval  {k}: {len(v):,} rows")

    results, out = [], Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        print(f"\n--- seed {seed} ---")
        head, best, hist = train_one(seed, train_fold, dev_fold, evals, args, device)
        use = "feat" if args.branch == "A" else "wave"
        row = {"seed": seed, "best_epoch": best["epoch"], "dev_eer": best["eer"]}
        for name, f in evals.items():
            e, scores = evaluate(head, f, device, use)
            pt, lo, hi = bootstrap_eer_ci(scores, f.label, f.groups, n_boot=200)
            row[name] = {"eer": e, "ci95": [lo, hi]}
        results.append(row)
        torch.save({"state_dict": head.state_dict(), "branch": args.branch,
                    "seed": seed, "dev_eer": best["eer"], "history": hist},
                   out / f"branch_{args.branch}{args.tag}_seed{seed}.pt")

    print("\n" + "=" * 70)
    print("RESULTS  (mean +/- spread over seeds)")
    print("=" * 70)
    dev = np.array([r["dev_eer"] for r in results])
    print(f"  dev EER              {dev.mean()*100:5.2f}% +/- {dev.std()*100:.2f}")
    for name in evals:
        vals = np.array([r[name]["eer"] for r in results])
        ci = results[0][name]["ci95"]
        print(f"  {name:<20} {vals.mean()*100:5.2f}% +/- {vals.std()*100:.2f}"
              f"   (95% CI on seed 0: {ci[0]*100:.2f}-{ci[1]*100:.2f}%)")

    print(f"\n  duration-only baseline: {DURATION_BASELINE_EER*100:.1f}% EER")
    ref = dev.mean()
    verdict = "PASS" if ref < DURATION_BASELINE_EER * 0.5 else "TOO CLOSE"
    print(f"  this model:             {ref*100:.2f}% EER   [{verdict}]")
    if verdict != "PASS":
        print("  ^ a model that does not clearly beat a stopwatch has learned nothing")

    # The tag belongs in the results filename too. Without it a tagged run
    # silently overwrites the untagged baseline's results -- which is exactly
    # what happened on the first mixed-corpus run, destroying the file holding
    # the published 2.48/11.01/30.29 numbers (recoverable only from git).
    (out / f"results_{args.branch}{args.tag}.json").write_text(
        json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
