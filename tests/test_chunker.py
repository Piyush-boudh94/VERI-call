"""Independent cadence check for the dual-rate chunker.

Ported from a test written against an earlier stub API. Its behavioural
expectations were correct and are worth keeping as a second, differently-shaped
check on the lane cadences -- `test_audio.py` covers the same ground from the
other direction (ring buffer, tick alignment, irregular push sizes).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from vericall.audio.chunker import DualRateChunker

SR = 16_000


def test_dual_rate_cadence_over_two_pushes():
    chunker = DualRateChunker("c")

    # 1.5 s of audio: fast lane closes windows ending at 1.0 s and 1.5 s;
    # the slow lane needs 3 s and has nothing yet.
    out = chunker.push(torch.ones(int(1.5 * SR)))
    fast = [w for w in out if w.lane == "fast"]
    slow = [w for w in out if w.lane == "slow"]
    assert len(fast) == 2
    assert [w.t_end for w in fast] == pytest.approx([1.0, 1.5])
    assert all(w.samples.numel() == SR for w in fast)
    assert len(slow) == 0

    # 2 s more (3.5 s total): fast closes 2.0, 2.5, 3.0, 3.5; slow closes 3.0.
    out2 = chunker.push(torch.ones(int(2.0 * SR)))
    fast2 = [w for w in out2 if w.lane == "fast"]
    slow2 = [w for w in out2 if w.lane == "slow"]
    assert len(fast2) == 4
    assert [w.t_end for w in fast2] == pytest.approx([2.0, 2.5, 3.0, 3.5])
    assert len(slow2) == 1
    assert slow2[0].samples.numel() == 3 * SR
    assert slow2[0].t_end == pytest.approx(3.0)


# ------------------------------------------------- window CONTENT, not cadence
#
# The cadence test above passes whether or not the windows carry the right
# audio, and for a long time they did not: push() wrote the whole input into the
# ring buffer and then cut every due window with buffer.tail(n), which reads the
# END of the buffer. A push spanning several ticks therefore emitted N windows
# whose t_start advanced correctly and whose samples were byte-identical.
#
# It stayed hidden because the streaming callers push well under one hop at a
# time. It surfaced in eta: measured by pushing whole 14 s calls at once, every
# window in a sequence was identical, branch scores differed only by float
# non-determinism (median std 1.8e-15), and the reported eta = 0.080 was an
# autocorrelation of that noise. On windows that actually differ it is 0.367.
#
# These tests check what the audio IS, not just how many windows arrive.

def ramp(n: int) -> torch.Tensor:
    """A signal where every sample is distinguishable from every other, so a
    duplicated window is detectable by value rather than by statistics."""
    return torch.arange(n, dtype=torch.float32) / float(n)


def test_one_big_push_does_not_emit_duplicate_audio():
    ch = DualRateChunker("c")
    out = [w for w in ch.push(ramp(int(6.0 * SR))) if w.lane == "fast"]
    assert len(out) >= 8
    first = out[0].samples
    assert not all(torch.equal(first, w.samples) for w in out), (
        "every window carried identical audio: push() is cutting all windows "
        "from the end of the buffer instead of at each tick")
    seen = {tuple(w.samples[:8].tolist()) for w in out}
    assert len(seen) == len(out), "some windows duplicated another's audio"


def test_one_big_push_equals_many_small_pushes():
    """The strongest form of the guarantee: how the caller chose to slice its
    audio must not change a single emitted sample."""
    stream = ramp(int(7.3 * SR))

    ch_one = DualRateChunker("c")
    bulk = ch_one.push(stream)

    ch_many = DualRateChunker("c")
    step = int(0.1 * SR)
    incremental = []
    for s in range(0, stream.numel(), step):
        incremental += ch_many.push(stream[s:s + step])

    assert len(bulk) == len(incremental)
    for a, b in zip(bulk, incremental):
        assert a.lane == b.lane
        assert a.t_start == pytest.approx(b.t_start)
        assert a.t_end == pytest.approx(b.t_end)
        assert torch.equal(a.samples, b.samples), (
            f"{a.lane} window ending at {a.t_end}s differs between a bulk push "
            f"and an incremental one")


def test_window_content_matches_its_own_timestamps():
    """A window claiming [t_start, t_end) must contain exactly that audio."""
    stream = ramp(int(5.0 * SR))
    ch = DualRateChunker("c")
    for w in ch.push(stream):
        lo = int(round(w.t_start * SR))
        hi = int(round(w.t_end * SR))
        assert torch.equal(w.samples, stream[lo:hi]), (
            f"{w.lane} window [{w.t_start}, {w.t_end}) does not contain the "
            f"audio from that interval")


def test_slow_lane_windows_also_advance():
    ch = DualRateChunker("c")
    slow = [w for w in ch.push(ramp(int(10.0 * SR))) if w.lane == "slow"]
    assert len(slow) >= 5
    assert len({tuple(w.samples[:8].tolist()) for w in slow}) == len(slow)


def test_push_smaller_than_one_hop_still_accumulates():
    """The streaming path: many sub-hop pushes must still produce whole windows
    at the right ticks, with no partial or repeated audio."""
    stream = ramp(int(3.0 * SR))
    ch = DualRateChunker("c")
    got = []
    step = int(0.037 * SR)          # deliberately not a divisor of the hop
    for s in range(0, stream.numel(), step):
        got += ch.push(stream[s:s + step])
    fast = [w for w in got if w.lane == "fast"]
    assert [round(w.t_end, 3) for w in fast] == [
        round(1.0 + 0.5 * k, 3) for k in range(len(fast))]
    for w in fast:
        assert w.samples.numel() == SR
        assert torch.equal(w.samples,
                           stream[int(round(w.t_start * SR)):int(round(w.t_end * SR))])


def test_rtp_sized_pushes_match_bulk():
    """The serving path pushes RTP-sized frames (20 ms), which divide neither
    the hop nor the window. Slicing at tick boundaries must handle a push that
    is far smaller than a tick as cleanly as one that spans several."""
    stream = ramp(int(4.2 * SR))
    bulk = DualRateChunker("c").push(stream)

    ch = DualRateChunker("c")
    frame = int(0.02 * SR)
    rtp = []
    for s in range(0, stream.numel(), frame):
        rtp += ch.push(stream[s:s + frame])

    assert len(bulk) == len(rtp)
    for a, b in zip(bulk, rtp):
        assert a.lane == b.lane and a.t_end == pytest.approx(b.t_end)
        assert torch.equal(a.samples, b.samples)


def test_a_push_of_a_single_sample_is_safe():
    """max(1, ...) in the slicing loop must guarantee progress even when the
    computed step is zero or negative from float error at a tick boundary."""
    ch = DualRateChunker("c")
    out = []
    stream = ramp(int(1.2 * SR))
    for i in range(stream.numel()):
        out += ch.push(stream[i:i + 1])
    fast = [w for w in out if w.lane == "fast"]
    assert len(fast) == 1
    assert fast[0].t_end == pytest.approx(1.0)
    assert torch.equal(fast[0].samples, stream[:SR])
