"""Measure real-time factor and derive concurrent-call capacity.

Exists to replace a derived estimate with a measurement. Two numbers commonly
quoted for this kind of system do not survive contact with it:

  - "AASIST RTF 0.0155" is the graph-attention head alone on a full GPU. With an
    SSL frontend in front of it the real cost is roughly an order of magnitude
    higher.
  - "1.2M events/sec" comes from transaction-fraud streaming, where an event is
    a ~500-byte JSON record rather than 16,000 audio samples through a
    95M-parameter encoder.

Capacity is reported as concurrent calls and as core-seconds per call-minute,
which extrapolates honestly to other hardware.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Callable, Optional

import torch


@dataclass
class BenchResult:
    name: str
    device: str
    batch: int
    window_s: float
    n_iters: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    windows_per_s: float
    rtf: float                     # audio-seconds processed per wall-second, inverted
    calls_at_2hz: float            # fast lane needs 2 windows/s/call
    vram_mib: Optional[float] = None

    def line(self) -> str:
        v = f" vram {self.vram_mib:.0f}MiB" if self.vram_mib else ""
        return (
            f"{self.name:<26} b={self.batch:<4} "
            f"p50 {self.p50_ms:7.2f}ms  p99 {self.p99_ms:7.2f}ms  "
            f"{self.windows_per_s:9.1f} win/s  RTF {self.rtf:.5f}  "
            f"~{self.calls_at_2hz:6.0f} calls{v}"
        )


def benchmark(
    fn: Callable[[torch.Tensor], object],
    batch: int,
    window_s: float,
    device: torch.device,
    name: str,
    n_iters: int = 30,
    n_warmup: int = 5,
    sample_rate: int = 16_000,
) -> BenchResult:
    x = torch.randn(batch, int(window_s * sample_rate), device=device)
    cuda = device.type == "cuda"

    for _ in range(n_warmup):                  # warm up kernels and autotune
        fn(x)
    if cuda:
        torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(n_iters):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(x)
        if cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    times.sort()
    p50 = statistics.median(times)
    p95 = times[min(len(times) - 1, int(0.95 * len(times)))]
    p99 = times[min(len(times) - 1, int(0.99 * len(times)))]

    wps = batch / (p50 / 1000.0)
    audio_s_per_wall_s = wps * window_s
    vram = None
    if cuda:
        free, total = torch.cuda.mem_get_info()
        vram = (total - free) / 2 ** 20

    return BenchResult(
        name=name, device=str(device), batch=batch, window_s=window_s,
        n_iters=n_iters, p50_ms=p50, p95_ms=p95, p99_ms=p99,
        windows_per_s=wps, rtf=1.0 / audio_s_per_wall_s,
        calls_at_2hz=wps / 2.0, vram_mib=vram,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64])
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    from vericall.branches.a_ssl import BranchA, BranchAConfig
    from vericall.features.phase import PhaseFeatures
    from vericall.features.spectral import LFCC, STFT, STFTConfig

    torch.set_grad_enabled(False)
    branch_a = BranchA(BranchAConfig()).to(device).eval()
    phase = PhaseFeatures().to(device).eval()
    cfg = STFTConfig()
    stft, lfcc = STFT(cfg).to(device), LFCC(cfg).to(device)

    stages = [
        ("features: LFCC", lambda x: lfcc(stft(x))),
        ("features: phase (A2)", lambda x: phase(x)),
        ("Branch A: SSL layer-5", lambda x: branch_a.frontend(x)),
        ("Branch A: full", lambda x: branch_a.score(x)),
    ]

    print(f"device={device}  "
          f"{torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    print("-" * 118)
    results: list[BenchResult] = []
    for name, fn in stages:
        for b in args.batches:
            try:
                r = benchmark(fn, b, 1.0, device, name, n_iters=args.iters)
            except RuntimeError as exc:                 # OOM at large batch
                print(f"{name:<26} b={b:<4} FAILED: {str(exc)[:60]}")
                break
            results.append(r)
            print(r.line())
        print("-" * 118)

    full = [r for r in results if r.name == "Branch A: full"]
    if full:
        best = max(full, key=lambda r: r.calls_at_2hz)
        core_s_per_call_min = 60.0 * 2.0 * (best.p50_ms / 1000.0) / best.batch
        print(f"\nBest Branch A throughput at batch {best.batch}: "
              f"{best.windows_per_s:.0f} win/s")
        print(f"  fast lane needs 2 win/s/call -> "
              f"~{best.calls_at_2hz:.0f} concurrent calls, GPU-side")
        print(f"  cost: {core_s_per_call_min*1000:.1f} GPU-ms per call-minute")
        print("  NOTE: this is the GPU ceiling only. End-to-end capacity is "
              "CPU-bound\n        (ingest, VAD, ring buffers, slow lane) and is "
              "measured by the\n        load test, not by this benchmark.")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump([asdict(r) for r in results], fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
