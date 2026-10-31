#!/usr/bin/env python
"""End-to-end validation with all FOUR branches live on real audio.

Every prior end-to-end run (scripts/end_to_end.py) exercised Branch A alone.
That was not an oversight left for later -- it is the exact configuration that
hid bug E24 (failed escalation gates silently suppressing every alert despite
perfect score separation). This script is the first time A, A2, B and C have
all run together against real streamed audio, with:

  - Branch A / A2 calibrated per the fitted Platt constants (channel-bucket
    "clean", since ASVspoof LA is uncompressed 16 kHz PCM)
  - Branch B mapped to LLR via the bona fide cohort (one_class_to_llr),
    hard-clipped so it can never dominate a decision
  - Branch C computing turn-detection embeddings but NEVER contributing to the
    spoof score -- consumed separately as an identity signal, never fused in
  - The per-subset calibration table (calib/subsets.json), so a partial branch
    set (e.g. the first 3 s of a call, before B has enough context) is scored
    on its own correctly-fitted scale rather than silently defaulting to
    (alpha=0, scale=1)

eta is now measured for all three scoring branches -- A=0.207, A2=0.928,
B=0.565, from 800 synthetic calls, each branch scored on the lane it actually
runs on (scripts/measure_eta_branches.py). The earlier A=0.080 was an artifact
of a chunker bug and is not used anywhere.
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
from vericall.branches.a2_phase import BranchA2, BranchA2Config
from vericall.branches.a_ssl import BranchA, BranchAConfig, BranchAHead
from vericall.branches.b_prosody import BranchB, BranchBConfig
from vericall.branches.c_speaker import BranchC, BranchCConfig
from vericall.data.manifest import Manifest
from vericall.fusion.calibration import load_calibration, logit, one_class_to_llr
from vericall.fusion.config import (
    FusionConfig, TierPolicy, load_subset_calib, load_weights,
)
from vericall.fusion.engine import FusionState, apply_hysteresis, update
from vericall.fusion.types import BranchEvent

# reuse the dataset-level helpers unchanged
sys.path.insert(0, str(Path(__file__).resolve().parent))
from end_to_end import build_calls, read_audio  # noqa: E402


def load_models(device):
    head = BranchAHead(768, BranchAConfig()).to(device)
    blob = torch.load("checkpoints/branch_A_seed0.pt", map_location=device,
                      weights_only=False)
    head.load_state_dict(blob["state_dict"]); head.eval()
    model_a = BranchA(BranchAConfig()).to(device).eval()

    model_a2 = BranchA2(BranchA2Config()).to(device)
    blob2 = torch.load("checkpoints/branch_A2_seed0.pt", map_location=device,
                       weights_only=False)
    model_a2.load_state_dict(blob2["state_dict"]); model_a2.eval()

    model_b = BranchB(BranchBConfig()).to(device)
    blobb = torch.load("checkpoints/branch_B.pt", map_location=device,
                       weights_only=False)
    model_b.load_state_dict(blobb["state_dict"]); model_b.eval()

    model_c = BranchC(BranchCConfig(), device=device).to(device).eval()
    b_cohort = np.load("calib/B_bonafide_cohort.npy")

    cal_a = load_calibration("calib/A@clean.json")
    cal_a2 = load_calibration("calib/A2@pooled.json")

    return {
        "a_head": head, "a_model": model_a, "a2": model_a2, "b": model_b,
        "c": model_c, "b_cohort": b_cohort, "cal_a": cal_a, "cal_a2": cal_a2,
    }


@torch.inference_mode()
def run_call(models, cfg, policy, wav, device, label):
    """Stream a call incrementally -- never push a whole call at once. The
    ring buffer holds only the longest window (3 s), so a single bulk push
    would make every emitted window read the same tail of audio, and a
    mid-call swap would look spoofed from t=0. (This exact bug was caught and
    fixed in the single-branch script; it is not re-introduced here.)"""
    st = FusionState("call")
    chunker = DualRateChunker("call", ChunkerConfig())
    hop = int(ChunkerConfig().tick_s * 16000)

    fast_windows, slow_windows = [], []
    for i in range(0, wav.numel(), hop):
        for w in chunker.push(wav[i:i + hop]):
            (fast_windows if w.lane == "fast" else slow_windows).append(w)

    events: list[BranchEvent] = []
    codec0 = torch.zeros(1, dtype=torch.long, device=device)

    for w in fast_windows:
        x = w.samples.unsqueeze(0).to(device)
        p_a = torch.softmax(models["a_model"].head(models["a_model"].frontend(x)),
                            -1)[:, 1].item()
        llr_a = float(models["cal_a"].apply(logit(np.array([p_a])))[0])
        events.append(BranchEvent("A", "call", w.t_start, w.t_end, llr_a,
                                  0.04, w.quality, w.hop_s))

        p_a2 = torch.softmax(models["a2"](x), -1)[:, 1].item()
        llr_a2 = float(models["cal_a2"].apply(logit(np.array([p_a2])))[0])
        events.append(BranchEvent("A2", "call", w.t_start, w.t_end, llr_a2,
                                  0.04, w.quality, w.hop_s))

    for w in slow_windows:
        x = w.samples.unsqueeze(0).to(device)
        raw_b = models["b"].score(x, codec0).item()
        llr_b = float(one_class_to_llr(np.array([raw_b]), models["b_cohort"],
                                       kappa=cfg.kappa_B, floor=cfg.floor_B)[0])
        events.append(BranchEvent("B", "call", w.t_start, w.t_end, llr_b,
                                  0.04, w.quality, w.hop_s))

        emb = models["c"].embed(x).squeeze(0).cpu().tolist()
        events.append(BranchEvent("C", "call", w.t_start, w.t_end, 0.0,
                                  0.04, w.quality, w.hop_s, emb=emb))

    events.sort(key=lambda e: (e.t_end, e.branch))

    first_alert, subsets_seen = None, set()
    for ev in events:
        view = update(st, ev, cfg)
        subsets_seen.add(view.subset)
        change = apply_hysteresis(st, view, cfg, policy)
        if change and change.direction == "escalate" and first_alert is None:
            first_alert = ev.t_end

    return {"label": label, "tier": st.tier, "latch": st.tier_latch,
            "first_alert": first_alert, "duration": wav.numel() / 16000,
            "changes": [c["t"] for c in st.change_events],
            "final_S_lo": view.S_lo, "subsets_seen": sorted(subsets_seen,
                                                            key=lambda s: (len(s), s))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifests/asvspoof2019_la.jsonl")
    ap.add_argument("--splits", default="data/splits/asvspoof2019_la.json")
    ap.add_argument("--fold", default="dev")
    ap.add_argument("--calls", type=int, default=40)
    ap.add_argument("--clips-per-call", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(device)

    subset_table = load_subset_calib("calib/subsets.json")
    # eta now comes from FusionConfig, where all three branches carry MEASURED
    # values (scripts/measure_eta_branches.py, 800 calls, docs/eta_branches.json).
    #
    # This used to override Branch A's eta from calib/summary.json. That file
    # still holds 0.080, which is wrong: it was measured while the chunker
    # emitted byte-identical windows for a multi-tick push, so the
    # autocorrelation behind it was computed on float non-determinism. Reading
    # it here would silently reinstate the bad number in the one script whose
    # whole job is to check the end-to-end decision, so the override is gone
    # rather than merely updated.
    # Weights from the calibration directory, not FusionConfig's defaults. The
    # subset table absorbs whatever scale the weighted sum lands on, so a table
    # fitted for one set of weights applied to another is a mis-pairing -- the
    # same class of error as serving a head against another head's Platt map.
    cfg = FusionConfig(subset_calib=subset_table,
                       w=load_weights(Path("calib/weights.json"),
                                      FusionConfig().w))
    policy = TierPolicy()
    print("All four branches loaded. Subset calibration table has "
          f"{len(subset_table)} entries.")
    print("weights: " + "  ".join(f"{k}={v}" for k, v in sorted(cfg.w.items())))
    print("eta (all MEASURED, 800 calls -- docs/eta_branches.json): "
          f"A={cfg.eta['A']}  A2={cfg.eta['A2']}  B={cfg.eta['B']}\n")

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
            results["bonafide"].append(
                run_call(models, cfg, policy, wav, device, "bonafide"))
    for spk, atk, run in spoof:
        wav = audio_of(run)
        if wav is not None and wav.numel() > 16000 * 3:
            results["spoof"].append(
                run_call(models, cfg, policy, wav, device, "spoof"))

    for (sb, ab, rb), (ss, as_, rs) in zip(bona, spoof):
        wb, ws = audio_of(rb), audio_of(rs)
        if wb is None or ws is None:
            continue
        wav = torch.cat([wb, ws])
        r = run_call(models, cfg, policy, wav, device, "swap")
        r["onset"] = wb.numel() / 16000
        results["swap"].append(r)

    print(f"{'scenario':<12}{'n':>4}{'alerted':>9}{'median TTA':>12}"
          f"{'median S_lo':>13}{'change pts':>12}  subsets seen")
    print("-" * 100)
    for name in ("bonafide", "spoof", "swap"):
        rs = results[name]
        if not rs:
            continue
        alerted = [r for r in rs if r["first_alert"] is not None]
        if name == "swap":
            tta = [r["first_alert"] - r["onset"] for r in alerted
                   if r["first_alert"] >= r["onset"]]
            early = [r for r in alerted if r["first_alert"] < r["onset"]]
        else:
            tta = [r["first_alert"] for r in alerted]
            early = []
        med_tta = f"{np.median(tta):.1f}s" if tta else "--"
        med_s = np.median([r["final_S_lo"] for r in rs])
        cps = np.mean([len(r["changes"]) for r in rs])
        all_subsets = sorted({s for r in rs for s in r["subsets_seen"]},
                             key=lambda s: (len(s), s))
        print(f"{name:<12}{len(rs):>4}{len(alerted)/len(rs)*100:>8.0f}%"
              f"{med_tta:>12}{med_s:>13.2f}{cps:>12.2f}  "
              f"{[('+'.join(s) or 'none') for s in all_subsets]}")
        if early:
            print(f"  (note: {len(early)}/{len(alerted)} swap alerts fired "
                  f"BEFORE the splice -- would indicate a streaming bug)")

    fa = len([r for r in results['bonafide'] if r['first_alert'] is not None])
    n_b = max(1, len(results['bonafide']))
    det = len([r for r in results['spoof'] if r['first_alert'] is not None])
    n_s = max(1, len(results['spoof']))
    print(f"\n  per-call FPR (bona fide alerted): {fa}/{n_b} = {fa/n_b*100:.1f}%")
    print(f"  per-call TPR (spoof alerted):     {det}/{n_s} = {det/n_s*100:.1f}%")

    Path("docs").mkdir(exist_ok=True)
    Path("docs/end_to_end_full.json").write_text(json.dumps(
        {k: [{kk: vv for kk, vv in r.items()} for r in v]
         for k, v in results.items()}, indent=2))
    print("\nwrote docs/end_to_end_full.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
