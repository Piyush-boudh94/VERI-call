"""Load generator: N simulated concurrent calls over WebSocket.

Frames are paced to **wall clock**, 20 ms at a time. That is not a detail:
streaming as fast as possible measures batch throughput, not concurrent-call
capacity, and the two differ by more than an order of magnitude. A number
produced without pacing would be meaningless.

Reports the SLOs separately, because conflating them is how "85 ms" ends up
printed next to a system whose real detection latency is seconds:

    chunk_inference_latency  window buffered -> score available
    time_to_first_score      first audio byte -> first risk view
    time_to_alert            spoof onset -> tier crossing
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from dataclasses import dataclass, field

import numpy as np

SR = 16_000
FRAME_MS = 20


@dataclass
class CallResult:
    call_id: str
    first_score_s: float | None = None
    alert_s: float | None = None
    spoof_onset_s: float | None = None
    updates: int = 0
    max_risk: float = -1e9
    final_tier: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def time_to_alert(self) -> float | None:
        if self.alert_s is None:
            return None
        return self.alert_s - (self.spoof_onset_s or 0.0)


def synth_audio(seconds: float, f0: float = 140.0, seed: int = 0) -> np.ndarray:
    """Harmonic speech-like signal with jitter, so VAD sees real activity."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    f = f0 * (1.0 + 0.03 * np.sin(2 * math.pi * 0.7 * t))
    phase = 2 * math.pi * np.cumsum(f) / SR
    x = sum(0.7 ** k * np.sin((k + 1) * phase) for k in range(6))
    env = 0.5 + 0.5 * np.sin(2 * math.pi * 1.3 * t) ** 2      # syllabic rhythm
    x = x * env + 0.01 * rng.standard_normal(t.size)
    return (x / max(abs(x).max(), 1e-6) * 0.9).astype(np.float32)


async def one_call(url: str, call_id: str, seconds: float,
                   spoof_onset: float | None, res: CallResult) -> None:
    import websockets

    res.spoof_onset_s = spoof_onset
    audio = synth_audio(seconds, f0=140.0, seed=hash(call_id) % 1000)
    frame = int(FRAME_MS * SR / 1000)
    t0 = time.perf_counter()

    try:
        async with websockets.connect(url, max_queue=64) as sock:
            async def reader():
                async for raw in sock:
                    msg = json.loads(raw)
                    if msg.get("error"):
                        res.errors.append(msg["error"]); return
                    now = time.perf_counter() - t0
                    if "risk" in msg:
                        res.updates += 1
                        res.max_risk = max(res.max_risk, msg["risk"])
                        res.final_tier = msg.get("tier", 0)
                        if res.first_score_s is None:
                            res.first_score_s = now
                        if msg.get("tier", 0) >= 1 and res.alert_s is None:
                            res.alert_s = now

            rt = asyncio.create_task(reader())
            for i in range(0, len(audio) - frame, frame):
                chunk = audio[i:i + frame]
                pcm = (chunk * 32767).astype(np.int16).tobytes()
                await sock.send(pcm)
                # pace to wall clock: this is what makes the number mean
                # "concurrent calls" rather than "batch throughput"
                target = t0 + (i + frame) / SR
                delay = target - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
            await sock.send(json.dumps({"op": "close"}))
            await asyncio.sleep(1.0)
            rt.cancel()
    except Exception as exc:                                  # noqa: BLE001
        res.errors.append(f"{type(exc).__name__}: {exc}"[:120])


async def run(args) -> int:
    results = [CallResult(f"call-{i:04d}") for i in range(args.calls)]
    onset = args.seconds * 0.5 if args.mid_call_swap else None
    url = f"ws://{args.host}:{args.port}/ws"
    suffix = f"?token={args.token}" if args.token else ""

    print(f"{args.calls} concurrent calls x {args.seconds}s, "
          f"paced at {FRAME_MS} ms/frame -> {url}")
    t0 = time.perf_counter()
    await asyncio.gather(*[
        one_call(f"{url}/{r.call_id}{suffix}", r.call_id, args.seconds, onset, r)
        for r in results
    ])
    wall = time.perf_counter() - t0

    ok = [r for r in results if not r.errors]
    failed = [r for r in results if r.errors]
    first = [r.first_score_s for r in ok if r.first_score_s is not None]
    alerts = [r.time_to_alert for r in ok if r.time_to_alert is not None]

    print(f"\n  wall clock            {wall:.1f}s (audio was {args.seconds}s)")
    print(f"  calls completed       {len(ok)}/{args.calls}")
    if failed:
        print(f"  failures              {len(failed)}  e.g. {failed[0].errors[0]}")
    print(f"  score updates/call    "
          f"{statistics.mean([r.updates for r in ok]):.1f}" if ok else "  none")
    if first:
        print(f"  time_to_first_score   p50 {statistics.median(first):.2f}s  "
              f"max {max(first):.2f}s   (SLO p99 <= 1.6s)")
    if alerts:
        print(f"  time_to_alert         p50 {statistics.median(alerts):.2f}s  "
              f"max {max(alerts):.2f}s   (SLO median <= 3s)")
    else:
        print("  time_to_alert         no alerts fired")

    try:
        import urllib.request
        with urllib.request.urlopen(
                f"http://{args.host}:{args.port}/metrics", timeout=10) as r:
            m = json.loads(r.read().decode())
        p = m.get("chunk_inference_latency_ms", {})
        if p:
            print(f"  chunk_inference       p50 {p['p50']:.1f}ms  "
                  f"p95 {p['p95']:.1f}ms  p99 {p['p99']:.1f}ms  "
                  f"(SLO p99 <= 250ms)")
        print(f"  windows scored        {m.get('windows_scored')} "
              f"in {m.get('batches')} batches")
    except Exception as exc:                                  # noqa: BLE001
        print(f"  (metrics unavailable: {type(exc).__name__})")
    return 0 if len(ok) == args.calls else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8760)
    ap.add_argument("--mid-call-swap", action="store_true")
    ap.add_argument("--token", default="",
                    help="shared secret, when the gateway has "
                         "VERICALL_AUTH_TOKEN set")
    args = ap.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
