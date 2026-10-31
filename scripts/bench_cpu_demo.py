#!/usr/bin/env python
"""Can the demo run on a laptop instead of the cluster?

The cluster is the wrong place to run a live presentation: the namespace is
wiped roughly twice a day and pod scratch goes with it. If the model holds
real-time on CPU, the demo becomes a laptop artifact that cannot be killed by
someone else's kubectl.

Real-time for the fast lane means 2 one-second windows per second per call, i.e.
an RTF below 0.5 per stream.
"""
from __future__ import annotations

import sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vericall.branches.a2_phase import BranchA2
from vericall.branches.a_ssl import BranchA, BranchAConfig

torch.set_grad_enabled(False)


def timed(fn, n=10, warm=3):
    for _ in range(warm):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def main() -> int:
    for threads in (1, 2, 4, 8):
        torch.set_num_threads(threads)
        a = BranchA(BranchAConfig()).eval()
        a2 = BranchA2().eval()
        x = torch.randn(1, 16000)

        dt_a = timed(lambda: a.score(x))
        dt_a2 = timed(lambda: a2.score(x))
        total = dt_a + dt_a2
        rtf = total / 1.0
        streams = 0.5 / rtf if rtf > 0 else 0
        print(f"threads={threads:<3} BranchA {dt_a*1000:7.1f} ms  "
              f"A2 {dt_a2*1000:5.1f} ms  total {total*1000:7.1f} ms  "
              f"RTF {rtf:.3f}  -> {streams:4.1f} concurrent calls "
              f"{'OK real-time' if rtf < 0.5 else 'TOO SLOW'}")
    print("\nfast lane needs 2 windows/s/call => RTF < 0.5 for one live call")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
