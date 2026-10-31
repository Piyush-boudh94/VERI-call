#!/usr/bin/env python
"""Server-side capacity, measured without a network client in the way.

`vericall.bench.load` drives the gateway over real WebSockets, which is the
right end-to-end test and the wrong instrument for finding the server's ceiling:
it runs on the same 16 cores as the server it is measuring. At N=50 that stopped
being a caveat and became the answer -- the batched GPU forward measured 296 ms
under load against 9 ms in isolation, a 33x inflation that is CPU starvation by
the measuring apparatus, not a property of the service.

This drives the same pipeline the gateway does -- chunker, cross-call batcher,
scorer, fusion, tier policy -- in process, with no sockets and no JSON. What it
measures is therefore **server capacity excluding transport**, and the gap
between this and bench.load is the transport cost. Both numbers are worth
having; conflating them is how a service gets sized on the wrong one.

Pacing is real-time: each simulated call pushes one tick of audio per tick of
wall clock. That is what makes the result "N concurrent calls" rather than
"N calls' worth of batch throughput", which a GPU can always do faster.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Match the serving process. Without this the probe measures a configuration
# nobody runs: torch's intra-op pool is sized from the host core count (112 on
# this 16-core pod), so concurrent forwards each try to claim the whole machine.
torch.set_num_threads(1)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vericall.audio.chunker import ChunkerConfig
from vericall.branches.a2_phase import BranchA2, BranchA2Config
from vericall.branches.a_ssl import BranchA, BranchAConfig
from vericall.fusion.calibration import load_calibration
from vericall.fusion.config import FusionConfig, TierPolicy, load_subset_calib, load_weights
from vericall.serving.batcher import BatcherOverloaded, CrossCallBatcher
from vericall.serving.scorer import BranchCalibrations, BranchScorer
from vericall.serving.session import CallSession, SessionConfig

SR = 16_000


def build(device, calib_dir: Path):
    a = BranchA(BranchAConfig()).to(device).eval()
    a.head.load_state_dict(torch.load("checkpoints/branch_A_seed0.pt",
                                      map_location=device,
                                      weights_only=False)["state_dict"])
    a2 = BranchA2(BranchA2Config()).to(device)
    a2.load_state_dict(torch.load("checkpoints/branch_A2_seed0.pt",
                                  map_location=device,
                                  weights_only=False)["state_dict"])
    a2.eval()
    cal = BranchCalibrations()
    for br in ("A", "A2"):
        p = calib_dir / f"{br}@clean.json"
        if p.exists():
            setattr(cal, br, load_calibration(p))
    scorer = BranchScorer(a, a2, None, cal, device)
    fusion = FusionConfig(
        subset_calib=load_subset_calib(calib_dir / "subsets.json"),
        w=load_weights(calib_dir / "weights.json", FusionConfig().w))
    return scorer, cal, SessionConfig(fusion=fusion, policy=TierPolicy())


async def one_call(idx, seconds, session, batcher, scorer, cal, lat, drops):
    tick = ChunkerConfig().tick_s
    step = int(tick * SR)
    rng = np.random.default_rng(idx)
    audio = torch.from_numpy(
        (0.1 * rng.standard_normal(int(seconds * SR))).astype(np.float32))
    t0 = time.perf_counter()
    for i in range(0, audio.numel() - step, step):
        windows = session.push_audio(audio[i:i + step])
        fast = [w for w in windows if w.lane == "fast"]
        if fast:
            t_s = time.perf_counter()
            try:
                rows = await batcher.submit(fast)
                for w, row in zip(fast, rows):
                    scorer.events_from_raw(
                        [w], {b: v.unsqueeze(0) for b, v in row.items()}, cal)
            except BatcherOverloaded:
                drops.append(1)
            lat.append((time.perf_counter() - t_s) * 1000.0)
        target = t0 + (i + step) / SR
        delay = target - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
    return time.perf_counter() - t0


async def run(args) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scorer, cal, scfg = build(device, Path(args.calib_dir))
    batcher = CrossCallBatcher(scorer.forward_fast, max_batch=args.max_batch,
                               collect_ms=args.collect_ms)

    # Warm every shape the batcher will hit, or the first real batch of each
    # size pays CUDA's re-planning cost and lands in the p99.
    from vericall.audio.chunker import Window
    with torch.inference_mode():
        for n in (1, 2, 4, 8, 16, args.max_batch):
            ws = [Window(lane="fast", call_id="w", t_start=0.0, t_end=1.0,
                         hop_s=0.5, samples=torch.zeros(SR), quality=1.0)] * n
            scorer.forward_fast(ws)
    scorer.stats.latencies_ms.clear()

    lat: list[float] = []
    drops: list[int] = []
    sessions = [CallSession(f"probe-{i:04d}", scfg) for i in range(args.calls)]
    t0 = time.perf_counter()
    await asyncio.gather(*[
        one_call(i, args.seconds, sessions[i], batcher, scorer, cal, lat, drops)
        for i in range(args.calls)])
    wall = time.perf_counter() - t0

    lat.sort()
    def pct(q):
        return lat[min(len(lat) - 1, int(q * len(lat)))] if lat else 0.0
    bs = batcher.stats()
    out = {
        "calls": args.calls, "audio_s": args.seconds, "wall_s": round(wall, 2),
        "realtime_factor": round(wall / args.seconds, 2),
        "windows": bs.get("windows", 0),
        "windows_per_s": round(bs.get("windows", 0) / max(wall, 1e-9), 1),
        "mean_batch": bs.get("mean_batch", 0.0),
        "submit_to_result_p50_ms": round(pct(0.50), 1),
        "submit_to_result_p99_ms": round(pct(0.99), 1),
        "gpu_forward_p99_ms": round(scorer.stats.percentiles().get("p99", 0.0), 1),
        "dropped": len(drops), "rejected": bs.get("rejected", 0),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, nargs="+", default=[1, 16, 50, 64])
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--max-batch", type=int, default=32)
    ap.add_argument("--collect-ms", type=float, default=30.0)
    ap.add_argument("--calib-dir", default="calib")
    ap.add_argument("--out", default="docs/capacity_probe.json")
    args = ap.parse_args()

    rows = []
    print(f"collect_ms={args.collect_ms}  max_batch={args.max_batch}\n")
    print(f"{'N':>5}{'wall/audio':>12}{'win/s':>9}{'mean_batch':>12}"
          f"{'p50 ms':>9}{'p99 ms':>9}{'gpu p99':>9}{'drop':>6}  verdict")
    print("-" * 82)
    for n in args.calls:
        r = asyncio.run(run(argparse.Namespace(**{**vars(args), "calls": n})))
        rows.append(r)
        # Real time is the binding constraint: a server that takes 2 s to
        # process 1 s of audio is not serving a live call at any latency.
        ok = r["realtime_factor"] <= 1.15 and r["submit_to_result_p99_ms"] <= 250
        print(f"{n:>5}{r['realtime_factor']:>12.2f}{r['windows_per_s']:>9.1f}"
              f"{r['mean_batch']:>12.2f}{r['submit_to_result_p50_ms']:>9.1f}"
              f"{r['submit_to_result_p99_ms']:>9.1f}{r['gpu_forward_p99_ms']:>9.1f}"
              f"{r['dropped']:>6}  {'pass' if ok else 'FAIL'}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {args.out}")
    print("Excludes WebSocket and JSON cost by construction -- compare against "
          "vericall.bench.load for the transport-inclusive number.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
