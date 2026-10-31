#!/usr/bin/env python
"""Fine-tune the SSL frontend -- the one fix for cross-corpus collapse that
has never been attempted.

Seven attempts are on record and none worked: naive corpus concatenation,
balanced sampling, a domain adversary, OC-Softmax (§20), SSL layer selection
(§22), and the same interventions re-tested on In-the-Wild (§24) and CodecFake
(§26). Every one of them changed what happens AFTER the frontend. Not one
changed the frontend, because there was no code path that could: every trainer
in this repo reads `use="feat"` from a cache of precomputed WavLM activations,
so the encoder is frozen by construction, not by choice.

§22 named this explicitly and could not test it. Its layer sweep found depth
monotonically harmful and failed to replicate the 2026 readout-layer result,
and recorded two candidate explanations it could not separate: that work
**fine-tunes** the encoder where this one freezes, and reads a learned weighted
sum where this one truncates. A frozen top layer has specialised toward
phonetic content with no opportunity to adapt back toward artefacts.

So this is not another reweighting of a fixed representation. It lets the
representation move.

**Partial unfreezing, and why.** `--unfreeze N` releases the top N transformer
layers and leaves the rest frozen. Full fine-tuning of 94.4 M parameters on
25 k clips overfits a corpus almost immediately, which would produce a better
in-domain number and a worse out-of-domain one -- the exact failure already
recorded seven times. The encoder also gets a much smaller learning rate than
the head (`--ssl-lr`), because a pretrained representation that moves as fast
as a randomly-initialised classifier is a representation being destroyed.

**The column that decides it** is WaveFake-unseen, held out of training
entirely. ASVspoof-only frozen scores 30.08% there and every mixing
configuration scored 46.8-49.8%, i.e. chance. Beating 30.08% would be the
first real repair of the project's central weakness. Not beating it is worth
just as much written down.
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

from vericall.branches.a_ssl import BranchA, BranchAConfig
from vericall.data.cache import CachedFold
from vericall.fusion.calibration import bootstrap_eer_ci, eer


def unfreeze_top(model: BranchA, n: int) -> tuple[int, int]:
    """Release the top `n` transformer layers of the frontend.

    torchaudio's WavLM exposes them as model.encoder.transformer.layers. The
    frontend also pins itself to eval() in .train(), which is correct while it
    is frozen and wrong once it is not -- dropout and layernorm must behave as
    training layers again, or the encoder trains in a mode it will never be
    served in.
    """
    fe = model.frontend
    fe.train = lambda mode=True: nn.Module.train(fe, mode)   # undo the eval pin

    # SSLFrontend.forward is decorated @torch.no_grad(). requires_grad_(True)
    # on the parameters is therefore not enough: the decorator discards the
    # graph before the optimiser can ever see a gradient, and training runs
    # bit-identically to the frozen baseline.
    #
    # This was caught only because the unfreeze=2 arm of the sweep produced
    # losses equal to unfreeze=0 to four decimal places. Without that control
    # arm the experiment would have reported "fine-tuning does not help" --
    # a false negative on the one intervention nobody had tried.
    if n <= 0:
        return 0, 0          # control arm: leave the frozen forward in place

    def grad_forward(wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        feats, _ = fe.model.extract_features(wav, num_layers=fe.layer)
        return feats[-1]

    fe.forward = grad_forward


    # The top N layers OF THE ONES ACTUALLY RUN, which is not the top N of the
    # encoder. Branch A truncates at cfg.layer (5 of 12), so
    # extract_features(num_layers=5) never executes layers 6-12 -- and
    # unfreezing those sets requires_grad on parameters that are not in the
    # graph. Gradients are then exactly zero and training runs bit-identically
    # to the frozen baseline, which is precisely what the first version of this
    # function did.
    #
    # Two controls caught it: the unfreeze=2 arm matched unfreeze=0 to four
    # decimal places, and a direct check showed the frontend output with
    # requires_grad False. Neither the "unfroze 14.2M parameters" log line nor
    # the optimiser complained, because both were true and irrelevant.
    used = fe.model.encoder.transformer.layers[:fe.layer]
    n = max(0, min(n, len(used)))
    for p in fe.model.parameters():
        p.requires_grad_(False)
    for layer in used[len(used) - n:]:
        for p in layer.parameters():
            p.requires_grad_(True)
    trainable = sum(p.numel() for p in fe.model.parameters() if p.requires_grad)
    return n, trainable


@torch.inference_mode()
def score_fold(model: BranchA, fold: CachedFold, device, batch: int) -> np.ndarray:
    model.eval()
    out = np.zeros(len(fold))
    for x, _, _, sel in fold.batches(batch, shuffle=False, use="wave",
                                     device=device):
        out[sel] = model.score(x).float().cpu().numpy()
    return out


def trainable_state(model: nn.Module) -> dict:
    """Only the parameters that could have moved.

    `model.state_dict()` here is 363.5 MB, because Branch A carries the whole
    WavLM encoder and the state dict does not care that almost none of it was
    trained. Eight of these were committed before anyone looked: 2.9 GB of
    checkpoints, of which 2.9 GB is a copy of weights `torchaudio.pipelines`
    downloads on its own. The frozen control arm is the absurd case -- 363.5 MB
    to record that nothing in the encoder changed.

    So save the parameters with `requires_grad` and nothing else. Buffers come
    along because BatchNorm statistics do move under training and are not
    recoverable from the pretrained weights.

      unfreeze 0  ->   0.4 MB   (was 363.5)
      unfreeze 4  -> ~114 MB    (the four layers that actually trained)

    The loader pairs with this: build the model, which arrives with pretrained
    encoder weights, then `load_state_dict(..., strict=False)`.
    """
    train = {k for k, p in model.named_parameters() if p.requires_grad}
    keep = {k: v for k, v in model.state_dict().items()
            if k in train or k not in dict(model.named_parameters())}
    return keep


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="/scratch/vericall-cache")
    ap.add_argument("--dataset", default="asvspoof2019_la")
    ap.add_argument("--train-fold", default="train")
    ap.add_argument("--dev-fold", default="dev")
    ap.add_argument("--eval-on", nargs="*", default=[],
                    help="dataset:fold, the held-out columns")
    ap.add_argument("--unfreeze", type=int, default=3,
                    help="top N transformer layers to release; 0 reproduces "
                         "the frozen baseline through this same code path")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-4, help="head")
    ap.add_argument("--ssl-lr", type=float, default=1e-5, help="encoder")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="ft")
    ap.add_argument("--out", default="checkpoints")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("refusing to fine-tune 94M parameters on CPU; this needs the pod")
        return 2

    model = BranchA(BranchAConfig()).to(device)
    n_layers, n_ssl = unfreeze_top(model, args.unfreeze)
    n_head = sum(p.numel() for p in model.head.parameters())
    print(f"device {device} | unfroze top {n_layers} encoder layers "
          f"({n_ssl/1e6:.1f}M) + head ({n_head/1e6:.2f}M)")

    root = Path(args.cache) / args.dataset
    tr = CachedFold.load(root / args.train_fold)
    dv = CachedFold.load(root / args.dev_fold)
    print(f"train {len(tr):,} rows | dev {len(dv):,} rows")

    # Discriminative learning rates: a pretrained encoder that moves as fast as
    # a fresh head is an encoder being overwritten.
    opt = torch.optim.AdamW([
        {"params": [p for p in model.frontend.parameters() if p.requires_grad],
         "lr": args.ssl_lr},
        {"params": model.head.parameters(), "lr": args.lr},
    ], weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")

    evals = []
    for spec in args.eval_on:
        ds, fold = spec.split(":")
        evals.append((spec, CachedFold.load(Path(args.cache) / ds / fold)))

    best = (1e9, -1)
    ckpt = Path(args.out) / f"branch_A{args.tag}_seed{args.seed}.pt"
    for ep in range(args.epochs):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for x, y, _, _ in tr.batches(args.batch, shuffle=True, use="wave",
                                     device=device, balanced=True):
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = lossf(model(x), y.long())
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 5.0)
            scaler.step(opt)
            scaler.update()
            tot += float(loss.detach()); nb += 1

        dv_eer, _ = eer(score_fold(model, dv, device, args.batch),
                        np.asarray(dv.label))
        line = (f"  epoch {ep:2d}  loss {tot/max(1,nb):.4f}  "
                f"dev {dv_eer*100:5.2f}%")
        for spec, f in evals:
            e, _ = eer(score_fold(model, f, device, args.batch),
                       np.asarray(f.label))
            line += f"  {spec.split(':')[1]} {e*100:5.2f}%"
        line += f"  ({time.time()-t0:.0f}s)"
        if dv_eer < best[0]:
            best = (dv_eer, ep)
            torch.save({"state_dict": trainable_state(model), "branch": "A",
                        "seed": args.seed, "dev_eer": dv_eer,
                        "unfroze": n_layers, "partial": True}, ckpt)
            line += " *"
        print(line, flush=True)

    # strict=False because the checkpoint holds only what moved; everything
    # absent from it is the pretrained encoder the fresh model already has.
    model.load_state_dict(torch.load(ckpt, map_location=device,
                                     weights_only=True)["state_dict"],
                          strict=False)
    print(f"\nbest epoch {best[1]}  dev {best[0]*100:.2f}%")
    report = {"unfreeze": n_layers, "dev_eer": best[0], "folds": {}}
    for spec, f in evals:
        s = score_fold(model, f, device, args.batch)
        e, lo, hi = bootstrap_eer_ci(s, np.asarray(f.label),
                                     np.asarray(f.groups), n_boot=300)
        report["folds"][spec] = {"eer": e, "ci95": [lo, hi]}
        print(f"  {spec:<34} {e*100:6.2f}%  [{lo*100:.2f}-{hi*100:.2f}]")
    Path(f"docs/finetune_{args.tag}.json").write_text(
        json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
