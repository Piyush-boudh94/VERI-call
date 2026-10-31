"""Chunker, ring buffer and quality-gate tests."""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from vericall.audio.chunker import (
    ChunkerConfig, DualRateChunker, RingBuffer, speech_quality,
)

SR = 16_000


def noise(seconds: float, amp: float = 0.1):
    return torch.randn(int(seconds * SR)) * amp


# ------------------------------------------------------------ ring buffer

def test_ring_buffer_keeps_only_the_tail():
    rb = RingBuffer(1000)
    rb.push(torch.arange(600, dtype=torch.float32))
    rb.push(torch.arange(600, 1200, dtype=torch.float32))
    tail = rb.tail(1000)
    assert tail is not None
    assert float(tail[-1]) == 1199.0
    assert float(tail[0]) == 200.0          # 0..199 evicted
    assert rb.total_written == 1200


def test_ring_buffer_returns_none_until_enough_audio():
    rb = RingBuffer(1000)
    rb.push(torch.zeros(300))
    assert rb.tail(500) is None
    rb.push(torch.zeros(300))
    assert rb.tail(500) is not None


def test_ring_buffer_handles_a_push_larger_than_itself():
    rb = RingBuffer(100)
    rb.push(torch.arange(500, dtype=torch.float32))
    tail = rb.tail(100)
    assert float(tail[-1]) == 499.0 and float(tail[0]) == 400.0


# ------------------------------------------------------------ quality gate

def test_silence_scores_zero_quality():
    assert speech_quality(torch.zeros(SR)) == 0.0


def test_speech_like_audio_scores_high_quality():
    assert speech_quality(noise(1.0)) > 0.8


def test_clipping_is_penalised():
    clipped = torch.ones(SR)                      # fully saturated
    assert speech_quality(clipped) < 0.2


def test_mostly_silent_window_scores_low():
    x = torch.zeros(SR)
    x[: SR // 20] = torch.randn(SR // 20) * 0.5   # 5% speech
    assert speech_quality(x) < 0.35


# ------------------------------------------------------------ dual-rate

def test_lane_cadences_and_geometry():
    c = DualRateChunker("c")
    out = []
    for _ in range(20):                           # 10 s in 0.5 s pushes
        out += c.push(noise(0.5))
    fast = [w for w in out if w.lane == "fast"]
    slow = [w for w in out if w.lane == "slow"]

    assert all(w.window_s == pytest.approx(1.0) and w.hop_s == 0.5 for w in fast)
    assert all(w.window_s == pytest.approx(3.0) and w.hop_s == 1.0 for w in slow)
    assert fast[0].t_end == pytest.approx(1.0)    # first full 1 s window
    assert slow[0].t_end == pytest.approx(3.0)
    # 2 Hz vs 1 Hz
    assert [w.t_end for w in fast[:4]] == pytest.approx([1.0, 1.5, 2.0, 2.5])
    assert [w.t_end for w in slow[:3]] == pytest.approx([3.0, 4.0, 5.0])
    assert len(fast) > len(slow)


def test_windows_carry_exactly_their_declared_duration():
    c = DualRateChunker("c")
    out = []
    for _ in range(10):
        out += c.push(noise(0.5))
    for w in out:
        assert w.samples.numel() == int(round(w.window_s * SR))


def test_hop_never_exceeds_window():
    """The fusion engine credits hop and normalises by window; hop > window
    would let a single event claim audio it never saw."""
    c = DualRateChunker("c")
    out = []
    for _ in range(12):
        out += c.push(noise(0.5))
    assert out and all(w.hop_s <= w.window_s for w in out)


def test_tick_alignment_across_calls_with_different_start_phases():
    """The whole point of the global grid: calls that start at arbitrary times
    must still emit on the same instants, or a cross-call batcher collects ~2
    requests instead of a full batch."""
    grids = []
    for offset in (0.0, 0.13, 0.37, 0.49):
        c = DualRateChunker("c", start_offset_s=offset)
        out = []
        for _ in range(12):
            out += c.push(noise(0.5))
        fast = [w for w in out if w.lane == "fast"]
        grids.append([round(w.t_end + offset, 6) for w in fast[:6]])

    tick = ChunkerConfig().tick_s
    for g in grids:
        for t in g:
            assert abs(t / tick - round(t / tick)) < 1e-6, f"{t} is off-grid"


def test_irregular_push_sizes_do_not_shift_the_grid():
    """Network audio does not arrive in tidy hop-sized frames."""
    c = DualRateChunker("c")
    out = []
    for n in (0.02, 0.31, 0.07, 0.6, 0.15, 0.4, 0.9, 0.05, 0.5, 1.1, 0.3, 0.6, 0.8, 0.5):
        out += c.push(noise(n))
    fast = [w for w in out if w.lane == "fast"]
    assert len(fast) >= 8
    for w in fast:
        assert abs(w.t_end / 0.5 - round(w.t_end / 0.5)) < 1e-6


def test_low_quality_windows_are_still_emitted_and_flagged():
    """Emitted, not dropped: the branch is alive, it just has no evidence. The
    fusion engine needs the event to know the branch has not gone stale."""
    c = DualRateChunker("c")
    out = []
    for _ in range(8):
        out += c.push(torch.zeros(SR // 2))
    assert out
    assert all(w.quality == 0.0 for w in out)


def test_config_rejects_a_hop_off_the_tick_grid():
    with pytest.raises(ValueError, match="tick grid"):
        ChunkerConfig(tick_s=0.5, fast_hop_s=0.3)
