#!/usr/bin/env python
"""Fit Branch C: AS-Norm cohort and the identity-LLR calibration.

Branch C answers "is this the enrolled person?", NOT "is this synthetic?". The
distinction is the single most important correction to the original blueprint: a
convincing clone of an enrolled speaker *raises* voiceprint similarity, so a
branch that added similarity into the spoof score would report the best attacks
as the safest calls.

It is therefore calibrated on its own terms -- target vs nontarget trials, no
spoof data involved -- and consumed at the policy layer, where high spoof score
AND high identity match is the *worst* case (a convincing clone of a known
customer), not the most reassuring one.

Adaptive score normalisation (AS-Norm) against an imposter cohort is standard
speaker-verification practice: raw cosine similarity has a speaker-dependent
offset, so a fixed threshold across speakers is meaningless without it.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vericall.branches.c_speaker import BranchC, BranchCConfig
from vericall.data.cache import CachedFold
from vericall.fusion.calibration import eer, fit_platt


@torch.inference_mode()
def embed_fold(model: BranchC, fold: CachedFold, device, limit: int, batch: int = 64,
              seed: int = 0):
    """Embed bona fide rows only -- identity calibration must not be polluted by
    synthetic speech, whose speaker identity is by construction ambiguous.

    Stratified across speakers, not a prefix of the fold. The cache stores rows
    in file order, which is grouped by speaker (each speaker's clips run
    consecutively), so `[:limit]` silently draws almost entirely from the first
    few speakers in the file rather than the full roster. Measured effect: with
    limit=1500 on ASVspoof LA train (20 speakers, 10,320 bona fide rows), the
    naive prefix covered only 3 speakers. AS-Norm's imposter cohort needs broad
    speaker coverage to characterise "typical" similarity across a population --
    3 speakers means the normalisation statistics are dominated by within-
    speaker variation from a handful of people rather than true population
    diversity, which is close to the opposite of what AS-Norm is for.
    """
    bona = np.flatnonzero(fold.label == 0)
    by_speaker: dict[str, list[int]] = {}
    for i in bona:
        by_speaker.setdefault(fold.meta[i]["speaker_id"], []).append(i)

    rng = np.random.default_rng(seed)
    per_speaker = max(1, limit // max(1, len(by_speaker)))
    idx = []
    for rows in by_speaker.values():
        rows = np.array(rows)
        rng.shuffle(rows)
        idx.extend(rows[:per_speaker].tolist())
    idx = np.array(sorted(idx))[:limit]           # sorted: mmap likes locality

    embs, speakers = [], []
    for i in range(0, len(idx), batch):
        sel = idx[i:i + batch]
        x = torch.from_numpy(np.asarray(fold.wave[sel], dtype=np.float32)).to(device)
        embs.append(model.embed(x).float().cpu())
        speakers += [fold.meta[j]["speaker_id"] for j in sel]
    return torch.cat(embs), np.array(speakers)


def build_trials(embs: torch.Tensor, speakers: np.ndarray, rng, n_per: int = 4000):
    by_spk = defaultdict(list)
    for i, s in enumerate(speakers):
        by_spk[s].append(i)
    multi = {s: v for s, v in by_spk.items() if len(v) >= 2}
    if len(multi) < 2:
        raise ValueError("need >= 2 speakers with >= 2 clips each")

    target, nontarget = [], []
    keys = list(multi)
    for _ in range(n_per):
        s = keys[rng.integers(len(keys))]
        a, b = rng.choice(multi[s], size=2, replace=False)
        target.append((a, b))
        s1, s2 = rng.choice(len(keys), size=2, replace=False)
        target_pool, other_pool = multi[keys[s1]], multi[keys[s2]]
        nontarget.append((target_pool[rng.integers(len(target_pool))],
                          other_pool[rng.integers(len(other_pool))]))
    return target, nontarget


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/vericall-cache")
    ap.add_argument("--dataset", default="asvspoof2019_la")
    ap.add_argument("--cohort", type=int, default=1500)
    ap.add_argument("--trials", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    model = BranchC(BranchCConfig(), device=device).to(device).eval()

    base = Path(args.cache) / args.dataset
    print("embedding bona fide clips...")
    tr_emb, tr_spk = embed_fold(model, CachedFold.load(base / "train"),
                                device, args.cohort)
    ca_emb, ca_spk = embed_fold(model, CachedFold.load(base / "calib"),
                                device, 4000)
    print(f"  cohort  {tuple(tr_emb.shape)} from {len(set(tr_spk))} speakers (train)")
    print(f"  trials  {tuple(ca_emb.shape)} from {len(set(ca_spk))} speakers (calib)")

    # Cohort from TRAIN, trials from CALIB: fitting the normalisation on the
    # same data it is evaluated on would flatter it.
    model.asnorm = __import__(
        "vericall.branches.c_speaker", fromlist=["ASNorm"]
    ).ASNorm(tr_emb.to(device), top_k=min(model.cfg.cohort_size, len(tr_emb)))

    target, nontarget = build_trials(ca_emb, ca_spk, rng, args.trials)
    e = torch.nn.functional.normalize(ca_emb, dim=-1).to(device)

    def raw_scores(pairs):
        a = e[[p[0] for p in pairs]]
        b = e[[p[1] for p in pairs]]
        return model.asnorm(a, b).float().cpu().numpy()

    st, sn = raw_scores(target), raw_scores(nontarget)
    scores = np.concatenate([sn, st])
    labels = np.concatenate([np.zeros(len(sn), int), np.ones(len(st), int)])

    asv_eer, _ = eer(scores, labels)
    print(f"\n  AS-Norm target/nontarget EER {asv_eer*100:5.2f}%  "
          f"(identity, not spoof)")
    print(f"    target   mean {st.mean():+.3f}  sd {st.std():.3f}")
    print(f"    nontarget mean {sn.mean():+.3f}  sd {sn.std():.3f}")

    cal = fit_platt(scores, labels, branch="C", channel_bucket="asv",
                    calib_id="C@asv")
    print(f"  identity LLR calibration: a={cal.a:.3f} b={cal.b:+.3f}  "
          f"Cllr {cal.cllr:.3f}  loss {cal.calibration_loss:.3f}")

    Path("calib").mkdir(exist_ok=True)
    np.save("calib/C_cohort.npy", tr_emb.numpy())
    from vericall.fusion.calibration import save_calibration
    save_calibration("calib/C@asv.json", cal)
    Path("calib/C_summary.json").write_text(json.dumps({
        "asv_eer": asv_eer, "a": cal.a, "b": cal.b,
        "cohort_size": int(tr_emb.shape[0]),
        "cohort_speakers": int(len(set(tr_spk))),
        "trials": int(len(scores)),
        "note": "identity signal; never enters the spoof score",
    }, indent=2))
    print("\n  wrote calib/C_cohort.npy, calib/C@asv.json, calib/C_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
