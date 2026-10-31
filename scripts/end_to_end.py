#!/usr/bin/env python
"""End-to-end validation on real audio: waveform -> risk decision.

This is the first point at which every piece is exercised together on real
speech rather than synthetic events: chunker -> Branch A -> the calibration
fitted on the held-out calib fold -> the fusion engine -> the tiered decision.

Calls are built by concatenating consecutive clips from one speaker with one
provenance, because ASVspoof clips are 1-4 s and a call is not. A mid-call swap
is built by splicing a bona fide run onto a spoofed run from the same speaker --
which is exactly the attack the dual-window design and change-point detector
exist for, and the one a single-shot classifier cannot express at all.

Nothing here is tuned. The calibration constants and eta are read from
calib/, fitted on data disjoint from both training and this evaluation.
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

from vericall.audio.chunker import ChunkerConfig, DualRateChunker
from vericall.branches.a_ssl import BranchA, BranchAConfig, BranchAHead
from vericall.data.audio_io import load_audio
from vericall.data.manifest import Manifest
from vericall.fusion.config import FusionConfig, TierPolicy
from vericall.fusion.engine import FusionState, apply_hysteresis, update
from vericall.fusion.types import BranchEvent
from vericall.fusion.calibration import load_calibration, logit


def build_calls(manifest, plan, fold, clips_per_call, n_per_class):
    """Homogeneous runs: one speaker, one label, one attack."""
    groups = defaultdict(list)
    for it in manifest:
        if fold in plan.get(it.utt_id, []):
            groups[(it.speaker_id, it.label, it.attack_id)].append(it)
    bona, spoof = [], []
    for (spk, lab, atk), clips in sorted(groups.items(), key=lambda kv: str(kv[0])):
        for i in range(0, len(clips) - clips_per_call + 1, clips_per_call):
            run = clips[i:i + clips_per_call]
            (bona if lab == 0 else spoof).append((spk, atk, run))
    return bona[:n_per_class], spoof[:n_per_class]


def read_audio(runs, cache: dict):
    import pyarrow.parquet as pq
    need = defaultdict(set)
    for _, _, run in runs:
        for it in run:
            if (it.shard, it.row) not in cache:
                need[it.shard].add(it.row)
    for shard, rows in need.items():
        # Parquet shards (ASVspoof, WaveFake, IndicSynth) or one audio file per
        # item (RTCFake, In-the-Wild, hindi_paired). Dispatch on the suffix,
        # the same rule extract_features.py uses. Third script to need this;
        # without it a file-addressed corpus fails in pyarrow with "Parquet
        # magic bytes not found", which is an unhelpful way to say "not parquet".
        if not str(shard).endswith(".parquet"):
            from pathlib import Path as _P
            if _P(shard).exists():
                cache[(shard, 0)] = load_audio(_P(shard).read_bytes())
            continue
        pf = pq.ParquetFile(shard)
        offset = 0
        for rg in range(pf.num_row_groups):
            n = pf.metadata.row_group(rg).num_rows
            hits = [r for r in range(offset, offset + n) if r in rows]
            if hits:
                tbl = pf.read_row_group(rg, columns=["audio"]).to_pydict()["audio"]
                for r in hits:
                    cache[(shard, r)] = load_audio(tbl[r - offset]["bytes"])
            offset += n
    return cache


@torch.inference_mode()
def run_call(model, head, cal, cfg, policy, wav, device, label):
    """Stream one call and return its trajectory."""
    st = FusionState("call")
    chunker = DualRateChunker("call", ChunkerConfig())
    traj, first_alert = [], None

    # Feed incrementally, as a live stream would. Pushing a whole call in one
    # go is wrong and silently so: the ring buffer holds only the longest
    # window (3 s), so every emitted window would read the SAME tail of the
    # audio. On a mid-call swap that makes the entire call look spoofed from
    # t=0 and the measured detection latency meaningless.
    hop = int(ChunkerConfig().tick_s * 16000)
    windows = []
    for i in range(0, wav.numel(), hop):
        windows += [w for w in chunker.push(wav[i:i + hop]) if w.lane == "fast"]
    if not windows:
        return None
    batch = torch.stack([w.samples for w in windows]).to(device)
    p = torch.softmax(head(model.frontend(batch)), dim=-1)[:, 1].float().cpu().numpy()
    llrs = cal.apply(logit(p))                     # raw score -> nats

    for w, llr in zip(windows, llrs):
        ev = BranchEvent("A", "call", w.t_start, w.t_end, float(llr),
                         0.04, w.quality, w.hop_s)
        view = update(st, ev, cfg)
        change = apply_hysteresis(st, view, cfg, policy)
        traj.append((w.t_end, float(llr), view.S, view.S_lo, st.tier))
        if change and change.direction == "escalate" and first_alert is None:
            first_alert = w.t_end
    return {"label": label, "traj": traj, "tier": st.tier,
            "latch": st.tier_latch, "first_alert": first_alert,
            "duration": wav.numel() / 16000,
            "changes": [c["t"] for c in st.change_events]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/branch_A_seed0.pt")
    ap.add_argument("--manifest", default="data/manifests/asvspoof2019_la.jsonl")
    ap.add_argument("--splits", default="data/splits/asvspoof2019_la.json")
    ap.add_argument("--calib", default="calib/A@clean.json")
    ap.add_argument("--summary", default="calib/summary.json")
    ap.add_argument("--fold", default="dev")
    ap.add_argument("--calls", type=int, default=40)
    ap.add_argument("--clips-per-call", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(args.checkpoint, map_location=device, weights_only=False)
    head = BranchAHead(768, BranchAConfig()).to(device)
    head.load_state_dict(blob["state_dict"]); head.eval()
    model = BranchA(BranchAConfig()).to(device).eval()

    cal = load_calibration(args.calib)
    eta = json.loads(Path(args.summary).read_text())["eta"]["A"]
    cfg = FusionConfig(eta={"A": eta, "A2": eta, "B": min(1.0, eta * 1.5)})
    policy = TierPolicy()
    print(f"calibration: a={cal.a:.3f} b={cal.b:+.3f} ({cal.channel_bucket})   "
          f"eta={eta:.3f} ({1/eta:.1f}x variance inflation)\n")

    manifest = Manifest.load(args.manifest)
    plan = json.loads(Path(args.splits).read_text())
    bona, spoof = build_calls(manifest, plan, args.fold,
                              args.clips_per_call, args.calls)
    cache: dict = {}
    read_audio(bona + spoof, cache)

    def audio_of(run):
        parts = [cache.get((it.shard, it.row)) for it in run]
        parts = [p for p in parts if p is not None and p.numel() > 0]
        return torch.cat(parts) if parts else None

    results = {"bonafide": [], "spoof": [], "swap": []}
    for spk, atk, run in bona:
        wav = audio_of(run)
        if wav is not None and wav.numel() > 16000 * 3:
            r = run_call(model, head, cal, cfg, policy, wav, device, "bonafide")
            if r: results["bonafide"].append(r)
    for spk, atk, run in spoof:
        wav = audio_of(run)
        if wav is not None and wav.numel() > 16000 * 3:
            r = run_call(model, head, cal, cfg, policy, wav, device, "spoof")
            if r: results["spoof"].append(r)

    # mid-call swap: a genuine run spliced onto a spoofed run
    for (sb, ab, rb), (ss, as_, rs) in zip(bona, spoof):
        wb, ws = audio_of(rb), audio_of(rs)
        if wb is None or ws is None:
            continue
        wav = torch.cat([wb, ws])
        r = run_call(model, head, cal, cfg, policy, wav, device, "swap")
        if r:
            r["onset"] = wb.numel() / 16000
            results["swap"].append(r)

    print(f"{'scenario':<12}{'n':>4}{'alerted':>9}{'median TTA':>12}"
          f"{'median S_lo':>13}{'change pts':>12}")
    print("-" * 62)
    for name in ("bonafide", "spoof", "swap"):
        rs = results[name]
        if not rs:
            continue
        alerted = [r for r in rs if r["first_alert"] is not None]
        if name == "swap":
            early = [r for r in alerted if r["first_alert"] < r["onset"]]
            tta = [r["first_alert"] - r["onset"] for r in alerted
                   if r["first_alert"] >= r["onset"]]
            if early:
                print(f"  (note: {len(early)}/{len(alerted)} swap alerts fired "
                      f"BEFORE the splice -- premature)")
        else:
            tta = [r["first_alert"] for r in alerted]
        med_tta = f"{np.median(tta):.1f}s" if tta else "--"
        med_s = np.median([r["traj"][-1][3] for r in rs])
        cps = np.mean([len(r["changes"]) for r in rs])
        print(f"{name:<12}{len(rs):>4}{len(alerted)/len(rs)*100:>8.0f}%"
              f"{med_tta:>12}{med_s:>13.2f}{cps:>12.2f}")

    fa = len([r for r in results['bonafide'] if r['first_alert'] is not None])
    n_b = max(1, len(results['bonafide']))
    det = len([r for r in results['spoof'] if r['first_alert'] is not None])
    n_s = max(1, len(results['spoof']))
    print(f"\n  per-call FPR (bona fide alerted): {fa}/{n_b} = {fa/n_b*100:.1f}%")
    print(f"  per-call TPR (spoof alerted):     {det}/{n_s} = {det/n_s*100:.1f}%")

    Path("docs").mkdir(exist_ok=True)
    Path("docs/end_to_end.json").write_text(json.dumps(
        {k: [{kk: vv for kk, vv in r.items() if kk != "traj"} for r in v]
         for k, v in results.items()}, indent=2))
    print("\nwrote docs/end_to_end.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
