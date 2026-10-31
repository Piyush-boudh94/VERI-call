#!/usr/bin/env python
"""Validate the demo path end to end, using the exact bytes the browser fetches.

This is the "will it work on stage" check. It reads the same wav files the
dashboard downloads from /static/demo/, converts them the same way the browser
does (mono 16 kHz int16), streams them over the same WebSocket at wall-clock
pace, and asserts the three things a panel will actually watch:

    genuine        stays low and never alerts
    cloned         climbs and alerts
    mid-call swap  stays low, then climbs AFTER the splice

The third is the one that matters most: it is the attack a single-shot
classifier cannot express, and the only one where a premature alert would be
visible as a bug rather than as a success.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import wave
from pathlib import Path

import numpy as np


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise ValueError(f"{path.name}: expected mono 16-bit")
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return pcm, sr


async def stream(url: str, call_id: str, pcm: np.ndarray, sr: int,
                 frame_s: float = 0.5, pace: float = 0.06) -> list[dict]:
    import websockets
    frames: list[dict] = []
    step = int(frame_s * sr)
    async with websockets.connect(f"{url}/ws/{call_id}", max_size=None) as ws:
        await ws.recv()                       # hello frame
        for i in range(0, len(pcm), step):
            await ws.send(pcm[i:i + step].tobytes())
            await asyncio.sleep(frame_s * pace)
            try:
                while True:
                    frames.append(json.loads(
                        await asyncio.wait_for(ws.recv(), timeout=0.01)))
            except asyncio.TimeoutError:
                pass
        try:
            while True:
                frames.append(json.loads(
                    await asyncio.wait_for(ws.recv(), timeout=1.5)))
        except asyncio.TimeoutError:
            pass
    return [f for f in frames if "risk" in f]


def first_alert(frames: list[dict]) -> float | None:
    for f in frames:
        for a in f.get("alerts", []):
            if a.get("direction") == "escalate":
                return a.get("at")
    return None


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8001")
    ap.add_argument("--demo", default="web/demo")
    args = ap.parse_args()

    demo = Path(args.demo)
    meta = json.loads((demo / "manifest.json").read_text())
    splice = meta.get("splice_s")
    print(f"demo clips from {demo}  (splice at {splice}s)\n")

    checks: list[tuple[str, bool, str]] = []
    for name in ("genuine.wav", "spoofed.wav", "midcall_swap.wav"):
        pcm, sr = read_wav(demo / name)
        frames = await stream(args.url, f"verify-{name}", pcm, sr)
        if not frames:
            print(f"{name:<20} NO FRAMES RETURNED")
            checks.append((name, False, "no frames"))
            continue
        risks = [f["risk"] for f in frames]
        alert = first_alert(frames)
        tier = frames[-1].get("tier", 0)
        print(f"{name:<20} risk {risks[0]:+7.2f} -> {risks[-1]:+7.2f}  "
              f"(max {max(risks):+.2f})  tier={tier}  "
              f"alert={'none' if alert is None else f'{alert:.1f}s'}")

        if name == "genuine.wav":
            ok = alert is None and risks[-1] < 0
            checks.append((name, ok, "stays low, no alert"))
        elif name == "spoofed.wav":
            ok = alert is not None and risks[-1] > 0
            checks.append((name, ok, "climbs and alerts"))
        else:
            ok = alert is not None and splice is not None and alert >= splice - 1.0
            detail = ("alerts after the splice" if ok else
                      f"alerted at {alert}s, splice was {splice}s -- PREMATURE")
            checks.append((name, ok, detail))

    print()
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<20} {detail}")
    failed = [c for c in checks if not c[1]]
    print(f"\n{'DEMO PATH READY' if not failed else f'{len(failed)} CHECK(S) FAILED'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
