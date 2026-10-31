#!/usr/bin/env python
"""Does prosody actually survive codec degradation better than spectral detail?

This is the v3 blueprint's central claim for Branch B -- that it is "the main
defence against the survives-compression requirement" -- and it has never been
measured. `docs/RESULTS.md` section 6 retracted the claim on the grounds that
prosody is weakly informative on clean ASVspoof LA, but explicitly recorded the
compression argument as *untested, not disproven*. This tests it.

RTCFake would have been the ideal corpus (real transmission through seven live
platforms) but its HuggingFace repo is gated behind owner approval. The
substitute here is arguably a cleaner instrument for this specific question: it
is a **paired** design -- identical utterances, identical speakers, only the
channel changes -- so any EER movement is attributable to the codec rather than
confounded with a corpus or platform shift.

What it does NOT capture, stated plainly: real platform transmission also
applies adaptive noise suppression, echo cancellation and bursty packet loss
that a fixed simulator does not reproduce. This measures the mechanism, not the
deployment.

The question that actually decides the blueprint's claim is not "does B degrade
less than A" -- B starts near chance, so a floor effect makes that trivially
true and meaningless. It is: **does A's EER ever cross B's under degradation?**
If A stays below B at every severity we can simulate, B is never the main
defence under any of these channel conditions.
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
from vericall.branches.a_ssl import BranchA, BranchAConfig
from vericall.branches.b_prosody import BranchB, BranchBConfig
from vericall.data.augment import CHANNEL_BY_NAME, apply_channel
from vericall.data.cache import CachedFold
from vericall.fusion.calibration import bootstrap_eer_ci, eer

CHANNELS = ["clean", "g711_mulaw", "noisy_line"]


def load_models(device):
    a = BranchA(BranchAConfig()).to(device).eval()
    a.head.load_state_dict(torch.load("checkpoints/branch_A_seed0.pt",
                                      map_location=device,
                                      weights_only=False)["state_dict"])
    a2 = BranchA2(BranchA2Config()).to(device)
    a2.load_state_dict(torch.load("checkpoints/branch_A2_seed0.pt",
                                  map_location=device,
                                  weights_only=False)["state_dict"])
    a2.eval()
    b = BranchB(BranchBConfig()).to(device)
    b.load_state_dict(torch.load("checkpoints/branch_B.pt", map_location=device,
                                 weights_only=False)["state_dict"])
    b.eval()
    return a, a2, b


@torch.inference_mode()
def score_channel(models, fold, idx, channel_name, device, batch=64):
    a, a2, b = models
    ch = CHANNEL_BY_NAME[channel_name]
    sa = np.zeros(len(idx)); sa2 = np.zeros(len(idx)); sb = np.zeros(len(idx))
    codec = torch.full((1,), ch.codec_id, dtype=torch.long, device=device)
    for i in range(0, len(idx), batch):
        sel = idx[i:i + batch]
        x = torch.from_numpy(np.asarray(fold.wave[sel], dtype=np.float32)).to(device)
        x = apply_channel(x, ch)
        sa[i:i + len(sel)] = a.score(x).float().cpu().numpy()
        sa2[i:i + len(sel)] = a2.score(x).float().cpu().numpy()
        sb[i:i + len(sel)] = b.score(x, codec.expand(len(sel))).float().cpu().numpy()
    return {"A": sa, "A2": sa2, "B": sb}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/vericall-cache")
    ap.add_argument("--dataset", default="asvspoof2019_la")
    ap.add_argument("--fold", default="dev")
    ap.add_argument("--stride", type=int, default=6, help="subsample the fold")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(device)
    fold = CachedFold.load(Path(args.cache) / args.dataset / args.fold)
    idx = np.arange(0, len(fold), args.stride)
    labels = fold.label[idx]
    groups = fold.groups[idx]
    print(f"{args.fold} fold, every {args.stride}th row: {len(idx):,} windows "
          f"({int((labels==0).sum()):,} bona / {int((labels==1).sum()):,} spoof)\n")

    results: dict[str, dict[str, float]] = {}
    for cname in CHANNELS:
        scores = score_channel(models, fold, idx, cname, device)
        row = {}
        for branch, s in scores.items():
            e, lo, hi = bootstrap_eer_ci(s, labels, groups, n_boot=200)
            row[branch] = {"eer": e, "ci": [lo, hi]}
        results[cname] = row

    print(f"{'channel':<14}{'Branch A':>22}{'Branch A2':>22}{'Branch B':>22}")
    print("-" * 80)
    for cname in CHANNELS:
        cells = []
        for br in ("A", "A2", "B"):
            r = results[cname][br]
            cells.append(f"{r['eer']*100:6.2f}% [{r['ci'][0]*100:.1f}-{r['ci'][1]*100:.1f}]")
        print(f"{cname:<14}{cells[0]:>22}{cells[1]:>22}{cells[2]:>22}")

    print(f"\n{'channel':<14}{'A degradation':>16}{'A2 degradation':>16}"
          f"{'B degradation':>16}")
    print("-" * 64)
    base = {br: results["clean"][br]["eer"] for br in ("A", "A2", "B")}
    for cname in CHANNELS[1:]:
        cells = []
        for br in ("A", "A2", "B"):
            e = results[cname][br]["eer"]
            ratio = e / base[br] if base[br] > 0 else float("nan")
            cells.append(f"{ratio:5.2f}x ({e*100:5.2f}%)")
        print(f"{cname:<14}{cells[0]:>16}{cells[1]:>16}{cells[2]:>16}")

    print("\n--- the question that decides the claim ---")
    crossed = False
    for cname in CHANNELS:
        ea, eb = results[cname]["A"]["eer"], results[cname]["B"]["eer"]
        mark = "A WORSE THAN B" if ea > eb else "A still better"
        crossed |= ea > eb
        print(f"  {cname:<14} A {ea*100:6.2f}%  vs  B {eb*100:6.2f}%   -> {mark}")

    print()
    if crossed:
        print("VERDICT: A's EER crosses B's under degradation. The blueprint's")
        print("  compression claim has a regime where it holds -- report which.")
    else:
        print("VERDICT: A stays below B at every simulated severity. Branch B is")
        print("  not the main defence under any channel condition tested here,")
        print("  so the blueprint's claim does not hold on this evidence.")
        print("  (Not the same as disproven for REAL platform transmission --")
        print("   that still needs RTCFake, which is gated.)")

    Path("docs").mkdir(exist_ok=True)
    Path("docs/codec_robustness.json").write_text(json.dumps(results, indent=2))
    print("\nwrote docs/codec_robustness.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
