#!/usr/bin/env python
"""Does the detector catch voice-cloning attacks in an Indian language?

**This script does not report an EER, and that is deliberate.** IndicSynth
publishes synthetic speech only -- 50,044 rows checked across 16 strided Hindi
shards, two generator families, 101 target speakers, zero bona fide. An EER
needs both classes. The tempting move is to borrow bona fide Hindi from some
other corpus, but then a detector can separate the two by recording conditions
instead of by synthesis, and the resulting number would measure the pairing, not
the model. That failure is invisible in the output and flattering, which is the
worst combination.

What this corpus CAN answer, honestly:

    at a threshold that gives a chosen false-positive rate on bona fide speech
    we DO have, what fraction of Hindi attacks does the system catch?

The threshold comes from a real bona fide fold (RTCFake English by default); the
recall comes from Hindi. Both halves are measured, neither is invented, and the
operating point is stated rather than optimised.

Two things this cannot tell you, stated here so they are not inferred away:

  * **The Hindi false-positive rate is unknown.** A model that called everything
    spoof would score 100% here. That is why every row prints the FPR its
    threshold was set at -- recall without the paired FPR is not a result.
  * The threshold transfers only if the score scale transfers, and section 13
    measured that failing badly across corpora. A low number here may mean the
    threshold moved, not that the attacks were missed; the score-distribution
    summary is printed so the two can be told apart.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vericall.branches.a_ssl import BranchA, BranchAConfig
from vericall.data.audio_io import fixed_length, load_audio
from vericall.data.cache import CachedFold
from vericall.data.manifest import Manifest

SR = 16_000


def load_branch_a(path: str, device):
    a = BranchA(BranchAConfig()).to(device).eval()
    a.head.load_state_dict(torch.load(path, map_location=device,
                                      weights_only=False)["state_dict"])
    return a


@torch.no_grad()
def score_cached(model, fold: CachedFold, device, batch: int = 256):
    """Log-odds of spoof on an already-extracted fold, via the head only.

    Log-odds rather than softmax P(spoof) deliberately. The head saturates: on
    the first run of this script the 1%, 5% and 10% FPR thresholds all came back
    as exactly 1.0000 and the realised FPR missed its target by 3x, because a
    probability clipped at 1.0 has no resolution left to place a quantile in.
    The logit is monotonic with the probability -- same ranking, same EER -- but
    keeps the dynamic range the quantile needs. Same tie problem as the EER
    fix in vericall/fusion/calibration.py, in a different disguise.
    """
    out = np.zeros(len(fold), dtype=np.float64)
    for s in range(0, len(fold), batch):
        sel = np.arange(s, min(s + batch, len(fold)))
        x = torch.from_numpy(np.asarray(fold.feat[sel], dtype=np.float32)).to(device)
        z = model.head(x)
        out[sel] = (z[:, 1] - z[:, 0]).float().cpu().numpy()
    return out


@torch.no_grad()
def score_raw(model, items, device, window: float, batch: int = 64):
    """Log-odds of spoof straight from parquet audio -- no cache needed for an
    eval-only corpus, and one fewer place for an extraction setting to drift."""
    import pyarrow.parquet as pq

    by_shard = defaultdict(list)
    for it in items:
        by_shard[it.shard].append(it)
    scores, order = [], []
    for shard, rows in by_shard.items():
        pf = pq.ParquetFile(shard)
        wanted = {r.row: r for r in rows}
        offset, buf, buf_it = 0, [], []

        def flush():
            if not buf:
                return
            x = torch.from_numpy(np.stack(buf)).to(device).float()
            z = model.head(model.frontend(x))
            scores.extend((z[:, 1] - z[:, 0]).float().cpu().numpy().tolist())
            order.extend(buf_it)
            buf.clear(); buf_it.clear()

        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg, columns=["audio"]).to_pydict()["audio"]
            for i in range(len(tbl)):
                if (offset + i) in wanted:
                    wav = load_audio(tbl[i]["bytes"])
                    buf.append(fixed_length(wav, window, SR,
                                            random_crop=False).numpy())
                    buf_it.append(wanted[offset + i])
                    if len(buf) >= batch:
                        flush()
            offset += len(tbl)
        flush()
    return np.array(scores), order


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+",
                    default=["checkpoints/branch_A_seed0.pt"])
    ap.add_argument("--dataset", default="indicsynth_hi")
    ap.add_argument("--manifests", default="data/manifests")
    ap.add_argument("--cache", default="/tmp/vericall-cache")
    ap.add_argument("--bonafide-fold", default="rtcfake:eval_rtc_en",
                    help="where the threshold comes from. Must contain real "
                         "bona fide audio; the Hindi corpus has none.")
    ap.add_argument("--fpr", type=float, nargs="+", default=[0.01, 0.05, 0.10])
    ap.add_argument("--limit", type=int, default=4000)
    ap.add_argument("--window", type=float, default=1.0)
    ap.add_argument("--out", default="docs/indic_hindi.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    man = Manifest.load(Path(args.manifests) / f"{args.dataset}.jsonl")
    if any(i.label == 0 for i in man):
        print("NOTE: this manifest has bona fide rows -- use an EER, not this "
              "script.", file=sys.stderr)
    rng = np.random.default_rng(0)
    items = list(man)
    # Stratify by generator: the shards are ordered by generative model, so a
    # positional prefix draws from one family (the bug this project already hit
    # twice -- Branch C's cohort, then the first IndicSynth fetch).
    by_gen = defaultdict(list)
    for it in items:
        by_gen[it.attack_id].append(it)
    per = max(1, args.limit // max(len(by_gen), 1))
    picked = []
    for gen, rows in sorted(by_gen.items()):
        idx = rng.choice(len(rows), size=min(per, len(rows)), replace=False)
        picked += [rows[i] for i in idx]
    print(f"{args.dataset}: {len(man):,} rows, {len(by_gen)} generator(s), "
          f"{len({i.speaker_id for i in man})} target speakers")
    print(f"  sampled {len(picked):,} stratified by generator: "
          + ", ".join(f"{g}={sum(1 for p in picked if p.attack_id == g)}"
                      for g in sorted(by_gen)))

    ds, fold_name = args.bonafide_fold.split(":")
    bf_path = Path(args.cache) / ds / fold_name
    if not bf_path.exists():
        print(f"threshold fold {args.bonafide_fold} not extracted", file=sys.stderr)
        return 2
    bf = CachedFold.load(bf_path)
    n_bona = int((bf.label == 0).sum())
    print(f"  threshold fold {args.bonafide_fold}: {n_bona:,} bona fide rows\n")

    results: dict = {}
    for ck in args.ckpt:
        if not Path(ck).exists():
            print(f"{ck}: missing"); continue
        model = load_branch_a(ck, device)
        bona = score_cached(model, bf, device)[bf.label == 0]
        hindi, _ = score_raw(model, picked, device, args.window)

        print(f"=== {Path(ck).name} ===")
        print(f"  log-odds on bona fide {args.bonafide_fold}: "
              f"mean {bona.mean():+7.2f}  median {np.median(bona):+7.2f}")
        print(f"  log-odds on Hindi spoof:{'':<17}"
              f"mean {hindi.mean():+7.2f}  median {np.median(hindi):+7.2f}")
        row = {"bona_mean": float(bona.mean()),
               "hindi_mean": float(hindi.mean()),
               "n_hindi": int(len(hindi)), "n_bona": int(len(bona)),
               "operating_points": {}}
        print(f"\n  {'FPR set on ' + args.bonafide_fold:<34}"
              f"{'threshold':>11}{'Hindi caught':>14}")
        for target in args.fpr:
            thr = float(np.quantile(bona, 1.0 - target))
            recall = float((hindi >= thr).mean())
            actual_fpr = float((bona >= thr).mean())
            print(f"  {target*100:>5.1f}%  (actual {actual_fpr*100:4.1f}%)"
                  f"{'':<12}{thr:>11.3f}{recall*100:>13.1f}%")
            row["operating_points"][f"{target:.2f}"] = {
                "threshold": thr, "actual_fpr": actual_fpr, "recall": recall}

        per_gen = {}
        for gen in sorted(by_gen):
            mask = np.array([p.attack_id == gen for p in picked])
            if mask.any():
                thr = float(np.quantile(bona, 0.95))
                per_gen[gen] = float((hindi[mask] >= thr).mean())
        if len(per_gen) > 1:
            print("\n  per generator, at the 5% FPR threshold:")
            for gen, r in per_gen.items():
                print(f"    {gen:<20}{r*100:>6.1f}% caught")
        row["per_generator_at_5pct_fpr"] = per_gen
        results[ck] = row
        print()

    print("Reminder: the Hindi FALSE-POSITIVE rate is unmeasured -- this corpus "
          "publishes no bona fide.\nA model that flagged everything would "
          "score 100% above. Read recall only beside its FPR.")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
