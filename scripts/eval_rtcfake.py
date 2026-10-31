#!/usr/bin/env python
"""Does prosody survive REAL platform transmission? The paired test.

docs/RESULTS.md section 6 retracted the blueprint's claim that Branch B is "the
main defence against the survives-compression requirement", and section 10
tested it with a channel simulator -- while explicitly recording that a fixed
simulator cannot reproduce what real transmission adds: adaptive noise
suppression, echo cancellation, and bursty rather than uniform packet loss.

RTCFake closes that gap. Its metadata pairs each utterance with itself: the same
speech offline (clean) and online (after passing through a real RTC platform).
Same speaker, same content, same label -- only the channel differs. That is a
stronger instrument than the simulator, and it is the condition Branch B has
never been tested under.

The question that decides whether Branch B stays in the architecture is NOT
"does B degrade less than A" -- B sits near chance, so a floor effect makes that
trivially true. It is: **does A's EER ever cross B's under real transmission?**
If A stays below B even here, Branch B is not the main defence under any
condition anyone has tested, and it should be cut rather than carried.

The en/zh split is a second, free result: these models only ever saw English
(ASVspoof LA), so Chinese is genuine cross-lingual generalisation.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vericall.branches.a2_phase import BranchA2, BranchA2Config
from vericall.branches.a_ssl import BranchA, BranchAConfig
from vericall.branches.b_prosody import BranchB, BranchBConfig
from vericall.fusion.calibration import bootstrap_eer_ci, eer

ROOT = Path("/tmp/vericall-data/rtcfake/data")
SR = 16_000


def _z(x):
    """Standardise before weighting: the branches emit different raw scales and
    the fusion weights assume calibrated LLRs, which we do not have here."""
    x = np.asarray(x, dtype=float)
    sd = x.std()
    return (x - x.mean()) / (sd if sd > 1e-9 else 1.0)


def load_models(device, a_ckpt: str = "checkpoints/branch_A_seed0.pt"):
    a = BranchA(BranchAConfig()).to(device).eval()
    a.head.load_state_dict(torch.load(a_ckpt, map_location=device,
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


def load_pairs(split: str, lang: str | None, limit: int):
    labels: dict[str, int] = {}
    with (ROOT / f"{split}_label.txt").open() as fh:
        for line in fh:
            parts = line.split()
            if len(parts) == 2:
                labels[parts[0]] = 0 if parts[1] == "bonafide" else 1

    pairs = []
    with (ROOT / "meta" / f"{split}_offline_online_pairs.csv").open() as fh:
        for row in csv.DictReader(fh):
            off, on = row["offline_id"], row["online_id"]
            # label file drops the split prefix that the pair file carries
            key = off.split("/", 1)[1] if off.startswith(split + "/") else off
            if key not in labels:
                continue
            if lang and f"/{lang}/" not in off:
                continue
            if not off or not on:
                continue
            po, pn = ROOT / "wav" / off, ROOT / "wav" / on
            # is_file(), not exists(): an empty id resolves to the wav DIRECTORY,
            # which exists, so exists() silently admits broken rows and the
            # failure surfaces much later as "Format not recognised".
            if po.is_file() and pn.is_file():
                pairs.append((str(po), str(pn), labels[key], key))
            if len(pairs) >= limit:
                break
    return pairs


@torch.inference_mode()
def score(models, paths, device, seconds=4.0, batch=32):
    """Fixed-length centre crop, so offline and online are compared on equal
    duration -- transmission changes length slightly and an unequal crop would
    confound duration with channel."""
    import soundfile as sf
    a, a2, b = models
    n = int(seconds * SR)
    out = {"A": np.zeros(len(paths)), "A2": np.zeros(len(paths)),
           "B": np.zeros(len(paths))}
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        waves = []
        for p in chunk:
            w, _ = sf.read(p, dtype="float32", always_2d=False)
            if w.ndim > 1:
                w = w.mean(axis=1)
            if len(w) < n:
                w = np.pad(w, (0, n - len(w)))
            else:
                s = (len(w) - n) // 2
                w = w[s:s + n]
            waves.append(w)
        x = torch.from_numpy(np.stack(waves)).to(device)
        codec = torch.zeros(len(chunk), dtype=torch.long, device=device)
        out["A"][i:i + len(chunk)] = a.score(x).float().cpu().numpy()
        out["A2"][i:i + len(chunk)] = a2.score(x).float().cpu().numpy()
        out["B"][i:i + len(chunk)] = b.score(x, codec).float().cpu().numpy()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=1200)
    ap.add_argument("--langs", nargs="+", default=["en", "zh"])
    ap.add_argument("--branch-a", default="checkpoints/branch_A_seed0.pt",
                    help="Branch A checkpoint. Point this at an RTCFake-trained "
                         "head to re-run THIS protocol against it: same 4 s "
                         "crops, same paired utterances, same script. A number "
                         "measured a different way is not a comparison.")
    ap.add_argument("--out", default="docs/rtcfake_paired.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(device, args.branch_a)
    print(f"Branch A checkpoint: {args.branch_a}")
    results: dict = {}

    for lang in args.langs:
        pairs = load_pairs(args.split, lang, args.limit)
        if not pairs:
            print(f"[{lang}] no pairs found"); continue
        labels = np.array([p[2] for p in pairs])
        groups = np.array([p[3] for p in pairs])
        print(f"\n=== {lang.upper()} — {len(pairs)} paired utterances "
              f"({int((labels==0).sum())} bona / {int((labels==1).sum())} spoof) ===")

        off = score(models, [p[0] for p in pairs], device)
        on = score(models, [p[1] for p in pairs], device)

        print(f"{'branch':<10}{'offline (clean)':>22}{'online (real RTC)':>22}"
              f"{'degradation':>14}")
        print("-" * 70)
        row = {}
        for br in ("A", "A2", "B"):
            eo, lo_o, hi_o = bootstrap_eer_ci(off[br], labels, groups, n_boot=200)
            en_, lo_n, hi_n = bootstrap_eer_ci(on[br], labels, groups, n_boot=200)
            ratio = en_ / eo if eo > 0 else float("nan")
            row[br] = {"offline": eo, "online": en_, "ratio": ratio,
                       "ci_offline": [lo_o, hi_o], "ci_online": [lo_n, hi_n]}
            print(f"{br:<10}{eo*100:>13.2f}% [{lo_o*100:4.1f}-{hi_o*100:4.1f}]"
                  f"{en_*100:>13.2f}% [{lo_n*100:4.1f}-{hi_n*100:4.1f}]"
                  f"{ratio:>13.2f}x")
        # Does the ASVspoof-fitted fusion help or hurt here? The weights
        # (A=1.0, A2=0.45, B=0.30) put the most weight on Branch A. If A has
        # inverted on this corpus, fusion actively drags the score the wrong
        # way and would underperform its own best single branch.
        from vericall.fusion.config import FusionConfig
        w = FusionConfig().w
        for cond, sc in (("offline", off), ("online", on)):
            fused = sum(w[b] * _z(sc[b]) for b in ("A", "A2", "B"))
            ef, _lo, _hi = bootstrap_eer_ci(fused, labels, groups, n_boot=200)
            best_single = min(row[b][cond] for b in ("A", "A2", "B"))
            row[f"fused_{cond}"] = {"eer": ef, "best_single": best_single}
            verdict = ("HURTS -- worse than its own best branch"
                       if ef > best_single + 0.01 else "helps")
            print(f"  fused {cond:<8} {ef*100:6.2f}%   best single branch "
                  f"{best_single*100:6.2f}%   -> {verdict}")

        results[lang] = row

        print(f"\n  the question that decides Branch B:")
        for cond in ("offline", "online"):
            ea, eb = row["A"][cond], row["B"][cond]
            print(f"    {cond:<8} A {ea*100:6.2f}%  vs  B {eb*100:6.2f}%   -> "
                  f"{'A WORSE THAN B' if ea > eb else 'A still better'}")

    # "A crosses B" is NOT sufficient to justify Branch B, and an earlier
    # version of this verdict said it was. A can cross B by COLLAPSING, which is
    # what happened: A sits at 50-79% EER here. B only earns its place if it is
    # usable in absolute terms, so require that explicitly.
    USABLE = 0.40
    b_usable = any(r["B"][c] < USABLE for r in results.values()
                   for c in ("offline", "online"))
    a_collapsed = any(r["A"][c] > 0.45 for r in results.values()
                      for c in ("offline", "online"))
    a_inverted = any(r["A"][c] > 0.60 for r in results.values()
                     for c in ("offline", "online"))

    print("\n" + "=" * 70)
    if a_inverted:
        print("FINDING: Branch A is INVERTED on at least one language -- EER above")
        print("  60% means systematically anti-correlated, not merely uninformative.")
        print("  It has learned something corpus-specific that points the wrong way")
        print("  under domain shift. This is the dominant result, not Branch B.")
    if b_usable:
        print("VERDICT: Branch B is usable somewhere here -- report where.")
    else:
        print(f"VERDICT: Branch B is at chance (>{USABLE*100:.0f}% EER) in every")
        print("  condition including REAL platform transmission, which was its last")
        print("  untested justification. It contributes nothing to fusion")
        print("  (docs/RESULTS.md section 8) and is not the compression defence it")
        print("  was claimed to be. Cut it, or carry it at zero weight and say so.")
    if a_collapsed:
        print("\n  NOTE: A crossing B here is NOT evidence for B -- A collapsed.")
    if False:
        print("VERDICT: A stays below B under REAL platform transmission, which")
        print("  was Branch B's last untested justification. On this evidence the")
        print("  branch does not earn its place and should be cut rather than")
        print("  carried -- it contributes nothing to fusion (docs/RESULTS.md 8)")
        print("  and is not the compression defence it was claimed to be.")

    Path("docs").mkdir(exist_ok=True)
    results["_meta"] = {"branch_a_ckpt": args.branch_a, "split": args.split,
                        "limit": args.limit}
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
