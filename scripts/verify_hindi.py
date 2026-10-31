#!/usr/bin/env python
"""Verify the Hindi profile against a running gateway.

A separate file rather than another heredoc inside scripts/verify_export.sh:
that script already nests two, and a third collided with them badly enough to
corrupt an edit. Shell heredocs do not nest, and a verifier is easier to read
as a program than as a quoted string.

Called as: verify_hindi.py <port>
Exits nonzero with a reason on the first failure.

What it asserts, and why each one is worth a gate:

  channel buckets   The deployment target is Indian-language TELEPHONY.
                    calib_hindi_ch fits clean/g711/noisy_line because Branch A
                    degrades 9.24% -> 13.93% -> 23.21% across them and the
                    fitted slopes differ 1.25x; serving one pooled mapping
                    mis-scales every non-clean call by roughly that factor.
  pairing           A Platt mapping applied to a head it was not fitted for
                    produces confident nonsense, not a degraded number.
  nothing degraded  The gateway has nineteen ways to lose a capability and
                    still answer status "ok". An absent auth token is expected
                    on a laptop; anything else here is a real problem.
  verdicts          Real Kathbath speech must not alert; clones of Kathbath
                    speakers by BOTH generator families must.
"""
from __future__ import annotations

import asyncio
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
import websockets

# Genuine must stay clear; both clone families must alert. Both are required
# because a detector that catches only voice conversion would have passed a
# single-family gate while scoring 40% -- no detection at all -- on zero-shot
# TTS, which is exactly what an earlier Hindi head did.
EXPECT = {
    "hindi_genuine": False,
    "hindi_clone_vc": True,
    "hindi_clone_tts": True,
}

FRAME_S = 0.32


def die(msg: str) -> None:
    print(f"\033[31mFAIL: {msg}\033[0m")
    raise SystemExit(1)


def check_health(port: str) -> None:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health") as fh:
        h = json.load(fh)

    want = {"clean", "g711", "noisy_line"}
    have = set(h.get("calibration_buckets", []))
    if want - have:
        die(f"hindi profile is missing channel buckets: {sorted(want - have)}")
    if h.get("calibration_pairing") != "ok":
        die("hindi calibration was not fitted for the heads that loaded "
            f"({h.get('calibration_pairing')})")
    if not h.get("calibrated"):
        die("hindi profile started uncalibrated -- every threshold assumes LLRs")
    missing = {"A", "A2"} - set(h.get("branches_loaded", []))
    if missing:
        die(f"branches did not load: {sorted(missing)}")
    degraded = [d for d in h.get("degraded", []) if "NO AUTH TOKEN" not in d]
    if degraded:
        die("hindi profile is degraded: " + "; ".join(degraded))
    print(f"  ok -- buckets {sorted(have)}, pairing ok, "
          f"branches {h['branches_loaded']}, nothing degraded")


async def stream(port: str, name: str, must_alert: bool) -> None:
    path = Path("web/demo") / f"{name}.wav"
    if not path.is_file():
        die(f"demo clip missing from the clone: {path}")
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != 16_000:
        die(f"{name} is {sr} Hz; the wire format is 16 kHz")
    pcm = (np.clip(wav[:, 0], -1.0, 1.0) * 32767).astype(np.int16)
    n = int(16_000 * FRAME_S)
    last = None

    async with websockets.connect(f"ws://127.0.0.1:{port}/ws/hi-{name}",
                                  max_size=2 ** 20) as ws:
        async def rx():
            nonlocal last
            try:
                async for m in ws:
                    d = json.loads(m)
                    if "error" in d:
                        die(f"server error on {name}: {d}")
                    if isinstance(d.get("risk"), (int, float)):
                        last = d
            except websockets.ConnectionClosed:
                pass

        task = asyncio.create_task(rx())
        # Wall-clock pace. The chunker ticks on real time, so streaming faster
        # emits almost no windows and every verdict collapses toward zero --
        # a harness that got this wrong cost real debugging time and briefly
        # looked like a model failure.
        for i in range(0, len(pcm) - n, n):
            await ws.send(pcm[i:i + n].tobytes())
            await asyncio.sleep(FRAME_S)
        await asyncio.sleep(1.5)
        await ws.close()
        await task

    if last is None:
        die(f"no risk updates for {name}")
    alerted = last["tier"] > 0
    if alerted != must_alert:
        die(f"{name}: tier {last['tier']}, risk {last['risk']:+.1f} -- expected "
            f"{'an alert' if must_alert else 'no alert'}")
    if last.get("dropped_chunks"):
        die(f"{name} dropped {last['dropped_chunks']} chunks")
    print(f"  ok -- {name:18s} tier {last['tier']}  risk {last['risk']:+8.1f}  "
          f"{'ALERT' if alerted else 'clear'}  "
          f"{last['confidence']['audio_scored_s']:.1f}s scored")


def main() -> int:
    if len(sys.argv) != 2:
        die("usage: verify_hindi.py <port>")
    port = sys.argv[1]
    check_health(port)
    for name, must in EXPECT.items():
        asyncio.run(stream(port, name, must))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
