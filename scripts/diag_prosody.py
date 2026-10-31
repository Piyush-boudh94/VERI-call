#!/usr/bin/env python
"""Is prosody discriminative on this corpus at all?

Branch B is a one-class model, and a one-class model that fails is ambiguous:
it may be collapsing (an optimisation failure) or the features may simply carry
no signal (a data fact). Those need different responses, so they must be
separated before spending more time on the architecture.

The test: fit a SUPERVISED classifier on the same prosody statistics. Supervised
learning with both labels is strictly easier than one-class learning. If a
supervised model on these features is also near chance, prosody carries no
usable spoof signal on this data and no amount of one-class tuning will help.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vericall.data.cache import CachedFold
from vericall.features.prosody import (
    F0Config, F0Tracker, frame_energy, prosody_statistics,
)
from vericall.fusion.calibration import eer

STAT_NAMES = ["logF0 mean", "logF0 sd", "F0 slope", "F0 jitter", "logE mean",
              "logE sd", "logE delta", "voiced ratio", "longest pause",
              "voiced prob"]


@torch.inference_mode()
def stats_for(fold: CachedFold, tracker, device, stride: int):
    idx = np.arange(0, len(fold), stride)
    out = []
    for i in range(0, len(idx), 128):
        sel = idx[i:i + 128]
        x = torch.from_numpy(np.asarray(fold.wave[sel], dtype=np.float32)).to(device)
        f0, voiced = tracker(x)
        e = frame_energy(x)[:, : f0.shape[1]]
        out.append(prosody_statistics(f0, voiced, e).float().cpu().numpy())
    return np.concatenate(out), fold.label[idx]


def main() -> int:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tracker = F0Tracker(F0Config()).to(device)
    base = Path("/tmp/vericall-cache/asvspoof2019_la")

    Xtr, ytr = stats_for(CachedFold.load(base / "train"), tracker, device, 12)
    Xte, yte = stats_for(CachedFold.load(base / "dev"), tracker, device, 8)
    print(f"train {Xtr.shape}  dev {Xte.shape}")
    print(f"  train bona/spoof: {(ytr==0).sum()}/{(ytr==1).sum()}")

    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

    print("\nSUPERVISED upper bound on these features:")
    lr = LogisticRegression(class_weight="balanced", max_iter=2000).fit(Xtr_s, ytr)
    e_lr, _ = eer(lr.decision_function(Xte_s), yte)
    print(f"  logistic regression   dev EER {e_lr*100:5.2f}%")

    gb = GradientBoostingClassifier(random_state=0, n_estimators=200,
                                    max_depth=3).fit(Xtr_s, ytr)
    e_gb, _ = eer(gb.predict_proba(Xte_s)[:, 1], yte)
    print(f"  gradient boosting     dev EER {e_gb*100:5.2f}%")

    print("\nPer-feature discriminative power (single-feature EER):")
    rows = []
    for j, name in enumerate(STAT_NAMES):
        e, _ = eer(Xte_s[:, j], yte)
        rows.append((min(e, 1 - e), name, e))
    for best, name, e in sorted(rows)[:10]:
        print(f"  {name:16s} EER {e*100:5.2f}%   (informativeness {abs(0.5-e)*100:4.1f} pp)")

    print()
    best = min(e_lr, e_gb)
    # 0.40 EER is not "discriminative" in any useful sense -- it is 10 pp from
    # chance. An earlier version of this script used 0.40 as the pass mark and
    # printed a reassuring verdict at 39.4%, which was simply the wrong bar.
    if best > 0.45:
        print(f"VERDICT: prosody carries NO usable spoof signal here "
              f"(supervised bound {best*100:.1f}% EER).")
    elif best > 0.30:
        print(f"VERDICT: prosody is only WEAKLY informative on this corpus.")
        print(f"  Supervised upper bound {best*100:.1f}% EER -- about "
              f"{(0.5-best)*100:.0f} pp better than chance.")
        print("  A one-class model cannot exceed this bound, so Branch B is")
        print("  ceiling-limited by the data, not only by its optimisation.")
        print("  It cannot be 'the main defence' against anything at this level.")
    else:
        print(f"VERDICT: prosody IS discriminative supervised "
              f"({best*100:.1f}% EER); the one-class formulation is the problem.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
