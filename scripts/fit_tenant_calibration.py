#!/usr/bin/env python
"""Refit calibration on one deployment's own traffic.

docs/RESULTS.md section 17 ends: *"A deployment procedure that fits calibration
on a sample of the tenant's own traffic is the obvious design. It is not
implemented."* This is that procedure.

**Why calibration and not retraining.** Sections 13, 17, 20, 22 and 23 are a
record of five separate attempts to make the *representation* transfer across
corpora, languages, generators and transports. All five failed. The one thing
that has consistently worked is refitting the score->LLR mapping for the
conditions actually being served: section 18 measured per-channel calibration
buying real accuracy, and section 23 cut Branch A2's calibration loss 10x on
the WebRTC channel by refitting two numbers. The artifacts are a few kB of
JSON and refit in seconds on cached branch outputs with no GPU, so this is
practical in a way retraining is not.

**What it does not do.** It cannot fix a representation that has no signal on
your audio. If min_Cllr comes back near 1.0 this script will say so and refuse
to write a profile, because a well-calibrated uninformative detector is still
uninformative -- and a confident-looking artifact would hide that.

**Input.** A directory of the tenant's own recordings, labelled:

    tenant_audio/
      bonafide/*.wav      real calls, from the channel you actually serve
      spoof/*.wav         known attacks against you, or your own red-team clones

Both classes are required. Fitting a prior-free LLR needs both, and there is no
honest way to synthesise the missing one -- pairing your bona fide with spoof
from a public corpus lets the mapping separate them by recording conditions
rather than by synthesis, which is the leak that inflates published numbers.

**Output.** A complete profile directory, servable directly:

    VERICALL_CALIB_DIR=calib-acme python run_local.py

Artifacts that are channel-specific are refit; the ones that are not (the
cohorts, the subset table, the fusion weights) are copied so the profile is
self-contained, then the whole directory is sealed.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vericall.branches.a2_phase import BranchA2, BranchA2Config
from vericall.branches.a_ssl import BranchA, BranchAConfig
from vericall.data.audio_io import load_audio
from vericall.fusion.calibration import (
    cllr, eer, fit_platt, load_calibration, logit, min_cllr, save_calibration,
)
from vericall.serving.integrity import build_manifest

CROP_S = 1.0
SR = 16_000

# Below this the fit is noise. 2 parameters per branch, but the Cllr estimate
# that decides whether to trust the profile needs far more than 2 points, and a
# tenant who hands over 40 clips should be told that rather than handed a
# number. Both classes must clear it independently.
MIN_PER_CLASS = 150

# Above this min_Cllr the detector is carrying almost no information on this
# tenant's audio and no mapping will rescue it. 1.0 is the cost of answering
# "I don't know" to every trial.
MAX_USABLE_MIN_CLLR = 0.90

# Artifacts that are not channel-specific and are copied through unchanged.
COPY_THROUGH = (
    "B_bonafide_cohort.npy", "C_cohort.npy", "C@asv.json", "C_summary.json",
    "subsets.json", "weights.json",
)


def crops_from(dirpath: Path, want: int) -> tuple[torch.Tensor, list[str]]:
    files = sorted(p for p in dirpath.rglob("*")
                   if p.suffix.lower() in (".wav", ".flac", ".mp3", ".ogg"))
    out, kept = [], []
    for f in files:
        try:
            w = load_audio(f.read_bytes())
        except Exception as exc:                              # noqa: BLE001
            print(f"    skipped {f.name}: {type(exc).__name__}")
            continue
        if w.numel() < want:
            continue                       # dropped, never zero-padded
        start = (w.numel() - want) // 2
        out.append(w[start:start + want])
        kept.append(f.name)
    if not out:
        return torch.empty(0, want), []
    return torch.stack(out), kept


@torch.inference_mode()
def score(a, a2, x, device, batch=32):
    sa, sa2 = [], []
    for i in range(0, len(x), batch):
        c = x[i:i + batch].to(device)
        sa.append(a.score(c).float().cpu())
        sa2.append(a2.score(c).float().cpu())
    return (torch.cat(sa).numpy().astype(np.float64),
            torch.cat(sa2).numpy().astype(np.float64))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio_dir", help="directory with bonafide/ and spoof/")
    ap.add_argument("--tenant", required=True, help="name, e.g. acme-bank")
    ap.add_argument("--bucket", default="pooled",
                    help="channel bucket to write (clean/g711/noisy_line/"
                         "opus/pooled). Use the one you actually serve; "
                         "pooled is right when you serve a mixture.")
    ap.add_argument("--base", default="calib",
                    help="profile to inherit the non-channel artifacts from")
    ap.add_argument("--out", default=None, help="default: calib-<tenant>")
    ap.add_argument("--force", action="store_true",
                    help="write the profile even if the guards fail. You are "
                         "asserting you understand what the numbers say.")
    args = ap.parse_args()

    root = Path(args.audio_dir)
    out = Path(args.out or f"calib-{args.tenant}")
    base = Path(args.base)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    want = int(CROP_S * SR)

    print(f"tenant {args.tenant}   bucket {args.bucket}   device {device}\n")
    for sub in ("bonafide", "spoof"):
        if not (root / sub).is_dir():
            print(f"FAIL: {root/sub} not found.\n\n"
                  "Both classes are required. Fitting a prior-free LLR needs\n"
                  "both, and pairing your bona fide against a public corpus's\n"
                  "spoof would let the mapping separate them by recording\n"
                  "conditions rather than by synthesis.")
            return 2

    print("reading audio")
    bona, _ = crops_from(root / "bonafide", want)
    spoof, _ = crops_from(root / "spoof", want)
    print(f"  bona fide {len(bona):>5} clips >= {CROP_S:g}s")
    print(f"  spoof     {len(spoof):>5} clips >= {CROP_S:g}s")

    short = [n for n, v in (("bonafide", len(bona)), ("spoof", len(spoof)))
             if v < MIN_PER_CLASS]
    if short and not args.force:
        print(f"\nFAIL: {', '.join(short)} below {MIN_PER_CLASS} usable clips.\n\n"
              "A mapping fitted on this little data is noise wearing a\n"
              "confident face, and it would replace a mapping that was fitted\n"
              "on tens of thousands. Collect more, or pass --force if you\n"
              "genuinely mean to.")
        return 2
    if len(bona) == 0 or len(spoof) == 0:
        print("\nFAIL: a class is empty; nothing can be fitted.")
        return 2

    x = torch.cat([bona, spoof])
    labels = np.concatenate([np.zeros(len(bona), int), np.ones(len(spoof), int)])

    a = BranchA(BranchAConfig()).to(device).eval()
    a.head.load_state_dict(torch.load("checkpoints/branch_A_seed0.pt",
                                      map_location=device,
                                      weights_only=True)["state_dict"])
    a2 = BranchA2(BranchA2Config()).to(device)
    a2.load_state_dict(torch.load("checkpoints/branch_A2_seed0.pt",
                                  map_location=device,
                                  weights_only=True)["state_dict"])
    a2.eval()

    print("\nscoring")
    s_a, s_a2 = score(a, a2, x, device)

    report = {"tenant": args.tenant, "bucket": args.bucket,
              "fitted_at": time.time(), "n_bonafide": int(len(bona)),
              "n_spoof": int(len(spoof)), "crop_s": CROP_S, "branches": {}}

    print(f"\n{'branch':<8}{'EER':>9}{'min_Cllr':>11}"
          f"{'Cllr (base)':>13}{'Cllr (fitted)':>15}{'improvement':>13}")
    print("-" * 69)

    fits, unusable = {}, []
    for branch, raw_p in (("A", s_a), ("A2", s_a2)):
        raw = logit(raw_p)
        e, _ = eer(raw_p, labels)
        mc = min_cllr(raw, labels)

        # What the CURRENTLY SERVED mapping costs on this tenant's audio. This
        # is the number the refit has to beat; without it a tenant cannot tell
        # whether the procedure helped or hurt.
        base_file = base / f"{branch}@{args.bucket}.json"
        if not base_file.exists():
            base_file = base / f"{branch}@pooled.json"
        base_cllr = float("nan")
        if base_file.exists():
            base_cllr = cllr(load_calibration(base_file).apply(raw), labels)

        cal = fit_platt(raw, labels, branch=branch,
                        channel_bucket=args.bucket,
                        calib_id=f"{branch}@{args.bucket}")
        fits[branch] = cal
        delta = base_cllr - cal.cllr
        print(f"{branch:<8}{e*100:8.2f}%{mc:11.4f}{base_cllr:13.4f}"
              f"{cal.cllr:15.4f}{delta:+13.4f}")
        report["branches"][branch] = {
            "eer": float(e), "min_cllr": float(mc),
            "cllr_before": base_cllr, "cllr_after": cal.cllr,
            "a": cal.a, "b": cal.b,
        }
        if mc > MAX_USABLE_MIN_CLLR:
            unusable.append(branch)

    if unusable and not args.force:
        print(f"\nFAIL: min_Cllr above {MAX_USABLE_MIN_CLLR} for "
              f"{', '.join(unusable)}.\n\n"
              "min_Cllr is what a PERFECT mapping would cost, so this says the\n"
              "detector carries almost no information on your audio and no\n"
              "calibration will rescue it. A well-calibrated uninformative\n"
              "detector is still uninformative, and writing the profile would\n"
              "hide that behind a confident-looking artifact.\n\n"
              "This is a model problem, not a calibration one. See\n"
              "docs/RESULTS.md sections 13 and 20 for what has already been\n"
              "tried against it.")
        return 3

    out.mkdir(parents=True, exist_ok=True)
    for branch, cal in fits.items():
        save_calibration(out / f"{branch}@{args.bucket}.json", cal)
    copied = []
    for name in COPY_THROUGH:
        src = base / name
        if src.exists():
            shutil.copy2(src, out / name)
            copied.append(name)

    (out / "provenance.json").write_text(json.dumps({
        "fitted_for": {"A": "branch_A_seed0.pt", "A2": "branch_A2_seed0.pt"},
        "dataset": f"tenant:{args.tenant}",
        "fold": args.bucket,
        "rows": int(len(x)),
        "inherited_from": str(base),
        "inherited_artifacts": copied,
        "note": "Channel mappings refit on this tenant's own traffic; "
                "cohorts, subset table and fusion weights inherited.",
    }, indent=2) + "\n")
    (out / "tenant_fit.json").write_text(json.dumps(report, indent=2) + "\n")

    man = build_manifest(sorted(p for p in out.iterdir() if p.is_file()), out)
    (out / "MANIFEST.sha256").write_text(json.dumps(man, indent=2) + "\n")

    print(f"\nwrote {out}/  ({len(fits)} refit, {len(copied)} inherited, sealed)")
    print(f"\nServe it with:\n    VERICALL_CALIB_DIR={out} python run_local.py")
    print("\nCheck /health afterwards: calibration_buckets must list "
          f"'{args.bucket}' and integrity must be clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
