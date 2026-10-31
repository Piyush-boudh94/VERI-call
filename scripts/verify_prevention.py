#!/usr/bin/env python
"""Same cloned call, different transaction context, different intervention.

This is the claim the prevention layer rests on, and it is the one a panel will
want to see rather than be told: detection produces one number, but what should
HAPPEN depends on what the caller is attempting. At ~1e-4 prevalence no
achievable detector FPR makes an unconditioned alert precise; conditioning on
context is the only lever that moves the prior enough to matter.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import wave
from pathlib import Path

import numpy as np


def read_wav(path: Path):
    with wave.open(str(path), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2"), w.getframerate()


async def run(url: str, wav: np.ndarray, sr: int, context: str) -> dict:
    import websockets
    step = int(0.5 * sr)
    last = {}
    async with websockets.connect(
            f"{url}/ws/prev-{context}?context={context}", max_size=None) as ws:
        await ws.recv()
        for i in range(0, len(wav), step):
            await ws.send(wav[i:i + step].tobytes())
            await asyncio.sleep(0.03)
            try:
                while True:
                    m = json.loads(await asyncio.wait_for(ws.recv(), timeout=0.01))
                    if "risk" in m:
                        last = m
            except asyncio.TimeoutError:
                pass
        try:
            while True:
                m = json.loads(await asyncio.wait_for(ws.recv(), timeout=1.5))
                if "risk" in m:
                    last = m
        except asyncio.TimeoutError:
            pass
    return last


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8001")
    ap.add_argument("--clip", default="web/demo/spoofed.wav")
    args = ap.parse_args()

    wav, sr = read_wav(Path(args.clip))
    print(f"one cloned call ({Path(args.clip).name}, {len(wav)/sr:.1f}s), "
          f"seven transaction contexts\n")
    print(f"{'context':<24}{'risk':>9}{'threshold':>11}  action")
    print("-" * 74)

    seen = {}
    for ctx in ("balance_enquiry", "statement_request", "inbound_unclassified",
                "transfer_small", "contact_change", "payee_add", "transfer_large"):
        last = await run(args.url, wav, sr, ctx)
        p = last.get("prevention") or {}
        act = p.get("action", "?")
        seen[ctx] = act
        print(f"{ctx:<24}{last.get('risk', 0):>9.2f}{p.get('threshold_nats', 0):>11.2f}"
              f"  {act}")

    distinct = set(seen.values())
    print(f"\ndistinct actions from one identical audio stream: {len(distinct)}"
          f"  {sorted(distinct)}")
    ok = (seen.get("balance_enquiry") == "allow"
          and seen.get("transfer_large") in ("fraud_desk", "hold"))
    print("\n" + ("PASS: identical detection, context-dependent intervention"
                  if ok else "FAIL: context did not change the action"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
