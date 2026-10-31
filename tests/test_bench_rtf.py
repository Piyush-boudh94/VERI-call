"""Real-time-factor measurement.

The last module without a test. It is the thing that turns a latency into the
capacity claim quoted in the docs -- "64 concurrent calls, p99 22 ms against a
250 ms budget" -- so an arithmetic slip here misstates the headline throughput
number rather than crashing.

The module exists because two figures commonly quoted for this kind of system
do not survive contact with it: an AASIST RTF measured without an SSL frontend,
and a transaction-streaming events/sec where an event is a 500-byte JSON record
rather than 16,000 audio samples. A test that only checked "it returns a float"
would not defend that.
"""
from __future__ import annotations

import time

import pytest
import torch

from vericall.bench.rtf import BenchResult, benchmark

SR = 16_000


def sleeper(ms: float):
    """A callable with a known, controllable cost, so the derived quantities
    can be checked against arithmetic rather than against hardware."""
    def fn(x):
        time.sleep(ms / 1000.0)
        return x
    return fn


def test_capacity_arithmetic_is_right():
    """The chain that produces the capacity claim: batch / p50 -> windows/s,
    x window_s -> audio-seconds per wall-second, / 2 -> calls, because the fast
    lane needs two windows per second per call.
    """
    r = benchmark(sleeper(10), batch=8, window_s=1.0,
                  device=torch.device("cpu"), name="t", n_iters=5, n_warmup=1)
    # ~10 ms for 8 windows -> ~800 windows/s -> ~800 audio-s/wall-s -> ~400 calls
    assert r.windows_per_s == pytest.approx(8 / (r.p50_ms / 1000.0), rel=1e-6)
    assert r.calls_at_2hz == pytest.approx(r.windows_per_s * 1.0 / 2.0, rel=1e-6)
    assert r.rtf == pytest.approx(1.0 / (r.windows_per_s * 1.0), rel=1e-6)


def test_rtf_is_inverted_so_lower_is_faster():
    """RTF here is wall-seconds per audio-second: below 1.0 means faster than
    real time. Getting the direction wrong would invert every capacity
    statement in the docs."""
    fast = benchmark(sleeper(2), batch=4, window_s=1.0,
                     device=torch.device("cpu"), name="f", n_iters=5, n_warmup=1)
    slow = benchmark(sleeper(20), batch=4, window_s=1.0,
                     device=torch.device("cpu"), name="s", n_iters=5, n_warmup=1)
    assert fast.rtf < slow.rtf
    assert fast.calls_at_2hz > slow.calls_at_2hz


def test_capacity_is_fast_lane_specific_and_window_length_does_not_change_it():
    """`calls_at_2hz` is windows/s divided by two, and stays so for a 3 s
    window. That is correct and worth pinning, because it is easy to misread.

    The figure answers "how many calls, if each needs 2 windows per second" --
    the fast lane, whose 1 s window at a 0.5 s hop emits exactly 2 Hz. It is
    NOT audio-throughput capacity. The slow lane emits 3 s windows at a 1 s
    hop, i.e. 1 Hz, so its capacity from the same measurement would be
    windows/s divided by one. Quoting calls_at_2hz for the slow lane would
    understate it by half.

    `rtf` is where window length does enter: it is wall-seconds per
    audio-second, so a longer window carries more audio per forward.
    """
    a = benchmark(sleeper(5), batch=4, window_s=1.0,
                  device=torch.device("cpu"), name="a", n_iters=5, n_warmup=1)
    b = benchmark(sleeper(5), batch=4, window_s=3.0,
                  device=torch.device("cpu"), name="b", n_iters=5, n_warmup=1)
    assert b.calls_at_2hz == pytest.approx(a.calls_at_2hz, rel=0.4), \
        "calls_at_2hz counts windows, not audio-seconds"
    # rtf does scale: three times the audio for the same wall time.
    assert b.rtf == pytest.approx(a.rtf / 3.0, rel=0.4)


def test_percentiles_are_ordered():
    r = benchmark(sleeper(3), batch=2, window_s=1.0,
                  device=torch.device("cpu"), name="p", n_iters=12, n_warmup=1)
    assert r.p50_ms <= r.p95_ms <= r.p99_ms


def test_warmup_iterations_are_excluded_from_the_timings():
    """Warmup exists because the first call compiles kernels and autotunes. If
    it were timed, every measurement would be pessimistic by a constant nobody
    could see."""
    calls = []
    def counting(x):
        calls.append(1)
        return x
    r = benchmark(counting, batch=1, window_s=1.0, device=torch.device("cpu"),
                  name="w", n_iters=7, n_warmup=4)
    assert len(calls) == 11, "should run warmup + iters"
    assert r.n_iters == 7, "but only report over the timed ones"


def test_the_input_is_the_shape_a_window_actually_is():
    """batch x window_s*16000 samples. A benchmark on the wrong shape measures
    the wrong thing and reads as a real number."""
    seen = {}
    def capture(x):
        seen["shape"] = tuple(x.shape)
        return x
    benchmark(capture, batch=5, window_s=1.0, device=torch.device("cpu"),
              name="s", n_iters=2, n_warmup=1)
    assert seen["shape"] == (5, SR)


def test_result_line_reports_the_numbers_it_measured():
    r = benchmark(sleeper(1), batch=2, window_s=1.0,
                  device=torch.device("cpu"), name="branch_A", n_iters=3,
                  n_warmup=1)
    line = r.line()
    assert "branch_A" in line and "RTF" in line and "calls" in line


def test_vram_is_none_off_cuda():
    """Reporting 0 MiB would read as "measured and tiny" rather than "not
    applicable on this device"."""
    r = benchmark(sleeper(1), batch=1, window_s=1.0,
                  device=torch.device("cpu"), name="v", n_iters=2, n_warmup=1)
    assert r.vram_mib is None
    assert isinstance(r, BenchResult)
