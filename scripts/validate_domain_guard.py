#!/usr/bin/env python
"""Does the domain guard fire on real data, and stay quiet in-domain?

The guard was written from the RTCFake finding, so testing it only on synthetic
signatures would be circular. This replays REAL branch scores through it:

    ASVspoof dev   -- the corpus the calibration was fitted on. Must stay quiet.
    RTCFake dev zh -- where Branch A inverted (79% EER). Must fire.

A guard that cannot tell those apart is decoration, and the honest outcome of
this script is either "it discriminates" or "it does not".
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
from vericall.fusion.calibration import load_calibration, logit
from vericall.fusion.domain_guard import BranchReference, DomainGuard

RTC = Path("/tmp/vericall-data/rtcfake/data")
CACHE = Path("/tmp/vericall-cache/asvspoof2019_la")
SR = 16_000


def load(device):
    a = BranchA(BranchAConfig()).to(device).eval()
    a.head.load_state_dict(torch.load("checkpoints/branch_A_seed0.pt",
                                      map_location=device,
                                      weights_only=False)["state_dict"])
    a2 = BranchA2(BranchA2Config()).to(device)
    a2.load_state_dict(torch.load("checkpoints/branch_A2_seed0.pt",
                                  map_location=device,
                                  weights_only=False)["state_dict"])
    a2.eval()
    return a, a2, (load_calibration("calib/A@clean.json"),
                   load_calibration("calib/A2@clean.json"))


@torch.inference_mode()
def llrs_from_wavs(models, paths, device, seconds=4.0, batch=32):
    import soundfile as sf
    a, a2, (cal_a, cal_a2) = models
    n = int(seconds * SR)
    out_a, out_a2 = [], []
    for i in range(0, len(paths), batch):
        waves = []
        for p in paths[i:i + batch]:
            w, _ = sf.read(p, dtype="float32", always_2d=False)
            if w.ndim > 1:
                w = w.mean(axis=1)
            w = np.pad(w, (0, max(0, n - len(w))))[:n] if len(w) < n else \
                w[(len(w) - n) // 2:(len(w) - n) // 2 + n]
            waves.append(w)
        x = torch.from_numpy(np.stack(waves)).to(device)
        out_a.append(cal_a.apply(logit(a.score(x).float().cpu().numpy())))
        out_a2.append(cal_a2.apply(logit(a2.score(x).float().cpu().numpy())))
    return np.concatenate(out_a), np.concatenate(out_a2)


@torch.inference_mode()
def llrs_from_cache(models, fold, device, stride=40, batch=32):
    from vericall.data.cache import CachedFold
    a, a2, (cal_a, cal_a2) = models
    f = CachedFold.load(fold)
    idx = np.arange(0, len(f), stride)
    out_a, out_a2 = [], []
    for i in range(0, len(idx), batch):
        sel = idx[i:i + batch]
        x = torch.from_numpy(np.asarray(f.wave[sel], dtype=np.float32)).to(device)
        out_a.append(cal_a.apply(logit(a.score(x).float().cpu().numpy())))
        out_a2.append(cal_a2.apply(logit(a2.score(x).float().cpu().numpy())))
    return np.concatenate(out_a), np.concatenate(out_a2), f.label[idx]


def rtc_bonafide(lang: str, limit: int) -> list[str]:
    """Bona fide only: the guard's inversion signature is 'branch reports
    genuine audio as spoof', so mixing real spoofs in would muddy it."""
    labels = {}
    for line in (RTC / "dev_label.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            labels[parts[0]] = parts[1]
    out = []
    with (RTC / "meta" / "dev_offline_online_pairs.csv").open() as fh:
        for row in csv.DictReader(fh):
            off = row["offline_id"]
            if not off or f"/{lang}/" not in off:
                continue
            key = off.split("/", 1)[1] if off.startswith("dev/") else off
            if labels.get(key) != "bonafide":
                continue
            p = RTC / "wav" / off
            if p.is_file():
                out.append(str(p))
            if len(out) >= limit:
                break
    return out


def run_guard(name: str, a: np.ndarray, a2: np.ndarray, ref) -> dict:
    g = DomainGuard(reference=ref)
    for x, y in zip(a, a2):
        g.observe("A", float(x))
        g.observe("A2", float(y))
    tripped = g.evaluate()
    j = g.to_json()
    print(f"{name:<26} A {j['branch_mean_llr'].get('A', 0):+7.2f}  "
          f"A2 {j['branch_mean_llr'].get('A2', 0):+7.2f}  "
          f"-> {'SUSPECT' if tripped else 'quiet'}")
    for r in g.reasons:
        print(f"    {r}")
    return {"tripped": tripped, **j}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load(device)

    # Reference: the distribution Branch A/A2 actually produce on the corpus
    # their calibration was fitted on. Measured here rather than assumed.
    a_cal, a2_cal, labels = llrs_from_cache(models, CACHE / "calib", device)
    bona = labels == 0
    ref = {
        "A": BranchReference("A", float(a_cal[bona].mean()),
                             float(a_cal[bona].std())),
        "A2": BranchReference("A2", float(a2_cal[bona].mean()),
                              float(a2_cal[bona].std())),
    }
    print("reference from ASVspoof calib fold, bona fide only:")
    for b, r in ref.items():
        print(f"  {b:<3} mean {r.mean:+.2f}  sd {r.sd:.2f}")
    print()

    results = {}
    a_dev, a2_dev, lab_dev = llrs_from_cache(models, CACHE / "dev", device)
    m = lab_dev == 0
    results["asvspoof_dev_bonafide"] = run_guard(
        "ASVspoof dev (in-domain)", a_dev[m], a2_dev[m], ref)

    for lang in ("en", "zh"):
        paths = rtc_bonafide(lang, args.limit)
        if not paths:
            print(f"RTCFake {lang}: no bona fide found"); continue
        a_r, a2_r = llrs_from_wavs(models, paths, device)
        results[f"rtcfake_{lang}_bonafide"] = run_guard(
            f"RTCFake {lang} (out-of-domain)", a_r, a2_r, ref)

    quiet_in = not results["asvspoof_dev_bonafide"]["tripped"]
    fires_out = any(v["tripped"] for k, v in results.items()
                    if k.startswith("rtcfake"))
    print()
    if quiet_in and fires_out:
        print("PASS: quiet in-domain, fires out-of-domain. The guard "
              "discriminates on real data.")
    elif not fires_out:
        print("FAIL: silent on RTCFake, the exact case it was built for. "
              "The guard does not work -- do not ship it as a safeguard.")
    else:
        print("FAIL: fires in-domain too. It would penalise normal calls.")

    Path("docs").mkdir(exist_ok=True)
    Path("docs/domain_guard_validation.json").write_text(
        json.dumps({"reference": {b: {"mean": r.mean, "sd": r.sd}
                                  for b, r in ref.items()},
                    "results": results}, indent=2))
    print("wrote docs/domain_guard_validation.json")
    return 0 if (quiet_in and fires_out) else 1


if __name__ == "__main__":
    raise SystemExit(main())
