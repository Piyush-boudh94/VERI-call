#!/usr/bin/env python
"""Does the browser's live-mic path score the same as the known-good clip path?

The dashboard now captures a live microphone. Browsers hand back whatever the
device runs at -- commonly 48 kHz -- so the page resamples to 16 kHz before
sending. That resampler is the single most likely thing to be silently wrong in
the whole demo: send 48 kHz samples to a 16 kHz model and nothing errors, the
audio simply sounds three times too fast and every score is meaningless.

No browser here, so this reimplements **the exact algorithm in index.html**
(linear interpolation at `micCtx.sampleRate / 16000`) and streams the result to
the live gateway. A clip upsampled to 48 kHz and then put through the page's
resampler must score the same as the original 16 kHz clip. If the resampler is
wrong the two diverge, which is a number rather than an opinion.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16_000
FRAME = 0.5


def js_resample(x: np.ndarray, in_rate: int, out_rate: int = SR) -> np.ndarray:
    """Line-for-line port of the page's resampler. Deliberately not scipy:
    the point is to test what the browser actually runs, not a better method."""
    ratio = in_rate / out_rate
    out_len = int(len(x) / ratio)
    out = np.empty(out_len, dtype=np.float32)
    for i in range(out_len):
        pos = i * ratio
        i0 = int(pos)
        frac = pos - i0
        a = x[i0]
        b = x[min(i0 + 1, len(x) - 1)]
        out[i] = a + (b - a) * frac
    return out


async def stream(url: str, call_id: str, pcm: np.ndarray) -> dict:
    import websockets
    step = int(SR * FRAME)
    last: dict = {}
    async with websockets.connect(f"{url}/{call_id}", open_timeout=20) as ws:
        async def reader():
            nonlocal last
            async for raw in ws:
                m = json.loads(raw)
                if m.get("error"):
                    last = {"error": m["error"]}
                    return
                if "risk" in m:
                    last = m
        rt = asyncio.create_task(reader())
        for i in range(0, len(pcm) - step, step):
            chunk = np.clip(pcm[i:i + step], -1.0, 1.0)
            await ws.send((chunk * 32767).astype("<i2").tobytes())
            await asyncio.sleep(FRAME)          # wall-clock pace, as the page does
        await asyncio.sleep(1.5)
        rt.cancel()
    return last


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8001/ws")
    ap.add_argument("--clip", default="web/demo/genuine.wav")
    ap.add_argument("--device-rate", type=int, default=48_000,
                    help="what a real browser would capture at")
    args = ap.parse_args()

    wav, sr = sf.read(args.clip, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    print(f"{Path(args.clip).name}: {len(wav)/sr:.1f}s at {sr} Hz")

    # Simulate a device by upsampling, then run the page's own resampler back.
    up = np.interp(np.arange(int(len(wav) * args.device_rate / sr))
                   * sr / args.device_rate, np.arange(len(wav)), wav
                   ).astype(np.float32)
    back = js_resample(up, args.device_rate)
    print(f"  simulated {args.device_rate} Hz capture -> "
          f"{len(back)} samples back at {SR} Hz "
          f"({len(back)/SR:.1f}s, drift {abs(len(back)-len(wav))} samples)")

    direct = await stream(args.url, "mic-ref", wav)
    if direct.get("error"):
        print(f"gateway refused: {direct['error']}", file=sys.stderr)
        return 2
    await asyncio.sleep(1.0)
    viamic = await stream(args.url, "mic-resampled", back)

    d, v = direct.get("risk"), viamic.get("risk")
    print(f"\n  original 16 kHz          risk {d:+.2f} nats  tier {direct.get('tier')}")
    print(f"  via page resampler       risk {v:+.2f} nats  tier {viamic.get('tier')}")
    if d is None or v is None:
        print("no risk returned", file=sys.stderr)
        return 2
    # Same sign and same tier is the bar. The resampler is lossy by design, so
    # demanding identical nats would fail for the wrong reason.
    rel = abs(v - d) / max(abs(d), 1.0)
    same_sign = (d < 0) == (v < 0)
    same_tier = direct.get("tier") == viamic.get("tier")
    print(f"  relative difference      {rel*100:.1f}%")
    ok = same_sign and same_tier and rel < 0.25
    print(f"\n  {'PASS' if ok else 'FAIL'} — sign {'same' if same_sign else 'FLIPPED'}, "
          f"tier {'same' if same_tier else 'different'}")
    if not ok:
        print("  The live-mic path does not agree with the clip path. A wrong\n"
              "  resample ratio is the first thing to check.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
