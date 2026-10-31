#!/usr/bin/env python
"""Drive the live gateway with REAL audio and print the risk trajectory.

The point of this script is to be un-foolable by a mock. It streams a genuine
ASVspoof clip and a spoofed one through the actual WebSocket path and prints
what comes back. A mock returning random numbers produces two indistinguishable
noisy traces; a working detector produces one that stays negative and one that
climbs. That difference is the test.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vericall.data.manifest import Manifest
from end_to_end import build_calls, read_audio


async def stream_call(url: str, call_id: str, wav: torch.Tensor,
                      chunk_s: float = 0.5, sr: int = 16000) -> list[dict]:
    import websockets

    frames: list[dict] = []
    n = int(chunk_s * sr)
    async with websockets.connect(f"{url}/ws/{call_id}", max_size=None) as ws:
        hello = json.loads(await ws.recv())
        if not hello.get("calibrated"):
            print(f"  !! server reports calibrated=False: {hello.get('notes')}")
        for i in range(0, wav.numel(), n):
            # int16 PCM -- same wire format as the load generator and the
            # dashboard, so all three clients exercise one code path
            block = wav[i:i + n].numpy()
            await ws.send((np.clip(block, -1.0, 1.0) * 32767
                           ).astype(np.int16).tobytes())
            # pace to wall clock so this exercises the streaming path, not a
            # throughput path -- otherwise it measures batch speed, not latency
            await asyncio.sleep(chunk_s * 0.05)
            try:
                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.01)
                    frames.append(json.loads(msg))
            except asyncio.TimeoutError:
                pass
        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                frames.append(json.loads(msg))
        except asyncio.TimeoutError:
            pass
    return frames


def summarise(name: str, frames: list[dict]) -> dict:
    scored = [f for f in frames if "risk" in f]
    if not scored:
        print(f"{name:<12} no scored frames returned")
        return {}
    risks = [f["risk"] for f in scored]
    tiers = [f["tier"] for f in scored]
    alerts = [a for f in scored for a in f.get("alerts", [])
              if a["direction"] == "escalate"]
    branches = sorted({b for f in scored for b in f.get("active_branches", [])})
    print(f"{name:<12} frames={len(scored):>3}  risk {risks[0]:+7.2f} -> "
          f"{risks[-1]:+7.2f}  (min {min(risks):+.2f} max {max(risks):+.2f})  "
          f"tier_final={tiers[-1]}  alerts={len(alerts)}  branches={branches}")
    return {"risks": risks, "tiers": tiers, "alerts": alerts}


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8001")
    ap.add_argument("--manifest", default="data/manifests/asvspoof2019_la.jsonl")
    ap.add_argument("--splits", default="data/splits/asvspoof2019_la.json")
    ap.add_argument("--clips-per-call", type=int, default=8)
    args = ap.parse_args()

    manifest = Manifest.load(args.manifest)
    plan = json.loads(Path(args.splits).read_text())
    bona, spoof = build_calls(manifest, plan, "dev", args.clips_per_call, 2)
    cache: dict = {}
    read_audio(bona + spoof, cache)

    def audio_of(run):
        parts = [cache.get((it.shard, it.row)) for it in run]
        parts = [p for p in parts if p is not None and p.numel() > 0]
        return torch.cat(parts) if parts else None

    wav_bona = audio_of(bona[0][2])
    wav_spoof = audio_of(spoof[0][2])
    print(f"bona fide {wav_bona.numel()/16000:.1f}s | "
          f"spoof {wav_spoof.numel()/16000:.1f}s\n")

    out = {}
    out["bonafide"] = summarise("bona fide",
                                await stream_call(args.url, "smoke-bona", wav_bona))
    out["spoof"] = summarise("spoof",
                             await stream_call(args.url, "smoke-spoof", wav_spoof))

    rb = out.get("bonafide", {}).get("risks") or []
    rs = out.get("spoof", {}).get("risks") or []
    print()
    if rb and rs:
        if max(rb) < min(rs):
            print(f"VERDICT: cleanly separated -- bona fide max {max(rb):+.2f} "
                  f"< spoof min {min(rs):+.2f}. Not a mock.")
        elif rs[-1] > rb[-1]:
            print(f"VERDICT: correct direction but overlapping "
                  f"(bona {rb[-1]:+.2f} vs spoof {rs[-1]:+.2f}).")
        else:
            print("VERDICT: NOT SEPARATED -- spoof did not score above bona "
                  "fide. Investigate before trusting the demo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
