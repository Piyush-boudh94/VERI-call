"""Per-call ring buffering and dual-rate windowing.

Two lanes, because one chunk size cannot serve four branches with different
context needs:

  fast lane   1.0 s window / 0.5 s hop  -> Branches A, A2  (2 Hz)
  slow lane   3.0 s window / 1.0 s hop  -> Branches B, C   (1 Hz)

Windows are snapped to a **global tick grid** shared by every call. Calls start
at arbitrary times, so without this their window boundaries scatter uniformly
and a cross-call batcher collecting for 20 ms would gather ~2 requests rather
than 32 -- filling a batch would cost ~320 ms and blow the p99 budget. Naive
dynamic batching simply does not work on unsynchronised audio streams. Each call
pays a one-time 0-500 ms phase offset at setup (charged to time_to_first_score)
and thereafter every call produces its chunk on the same instant.

This layer is CPU-resident on purpose: it is I/O and memory movement, and there
is no GPU version of a socket. It is also not the bottleneck.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

import torch

from ..features.spectral import SAMPLE_RATE

Lane = Literal["fast", "slow"]


@dataclass(frozen=True)
class ChunkerConfig:
    sample_rate: int = SAMPLE_RATE
    tick_s: float = 0.5           # global grid; both hops are multiples of it
    fast_window_s: float = 1.0
    fast_hop_s: float = 0.5
    slow_window_s: float = 3.0
    slow_hop_s: float = 1.0

    def __post_init__(self) -> None:
        for hop in (self.fast_hop_s, self.slow_hop_s):
            k = hop / self.tick_s
            if abs(k - round(k)) > 1e-9:
                raise ValueError(
                    f"hop {hop}s is not a multiple of the tick grid {self.tick_s}s; "
                    "cross-call batching depends on every lane landing on the grid"
                )


@dataclass
class Window:
    """One analysis window, ready to score."""

    lane: Lane
    call_id: str
    t_start: float
    t_end: float
    hop_s: float
    samples: torch.Tensor          # (N,) float32, CPU
    quality: float

    @property
    def window_s(self) -> float:
        return self.t_end - self.t_start


class RingBuffer:
    """Fixed-capacity mono sample buffer.

    Holds only as much audio as the longest window needs. A 3 s buffer at 16 kHz
    is 192 kB per call, which is why per-call audio state stays in-process and is
    never externalised -- shipping it on every request would cost 384 kB/s/call
    and dominate the compute it was meant to distribute.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._buf = torch.zeros(capacity, dtype=torch.float32)
        self._filled = 0
        self.total_written = 0

    def push(self, samples: torch.Tensor) -> None:
        samples = samples.reshape(-1).to(torch.float32)
        n = samples.numel()
        self.total_written += n
        if n >= self.capacity:
            self._buf.copy_(samples[-self.capacity :])
            self._filled = self.capacity
            return
        # clone only the surviving prefix, not the whole buffer: source and
        # destination overlap, so some copy is unavoidable, but cloning
        # capacity bytes per push when capacity-n suffices is pure waste
        self._buf[:-n] = self._buf[n:].clone()
        self._buf[-n:] = samples
        self._filled = min(self.capacity, self._filled + n)

    def tail(self, n: int) -> Optional[torch.Tensor]:
        if n > self._filled:
            return None
        return self._buf[-n:].clone()

    @property
    def filled(self) -> int:
        return self._filled


def speech_quality(
    samples: torch.Tensor,
    frame: int = 400,
    hop: int = 160,
    floor_db: float = -55.0,
    clip_threshold: float = 0.99,
) -> float:
    """Quality weight in [0, 1] for one window: speech ratio x clipping penalty.

    Without this, silence, hold music and DTMF tones produce confident scores
    from every branch. The fusion engine treats a low-quality window as "the
    branch is alive but has no evidence to offer", which is the honest reading.

    Energy-based rather than a neural VAD so it costs nothing and adds no
    dependency; the interface is the part that matters, and silero-vad can be
    dropped in behind it unchanged.
    """
    if samples.numel() < frame:
        return 0.0
    frames = samples.unfold(0, frame, hop)
    rms = frames.pow(2.0).mean(dim=-1).clamp(min=1e-12).sqrt()
    db = 20.0 * torch.log10(rms)

    peak_db = float(db.max())
    if peak_db < floor_db:
        return 0.0                      # the whole window is silence
    # adaptive floor: speech is what sits within 25 dB of this window's peak
    active = (db > max(floor_db, peak_db - 25.0)).float().mean()

    clipped = (samples.abs() > clip_threshold).float().mean()
    return float((active * (1.0 - clipped)).clamp(0.0, 1.0))


class DualRateChunker:
    """Emits tick-aligned windows on both lanes for a single call."""

    def __init__(
        self,
        call_id: str,
        cfg: ChunkerConfig = ChunkerConfig(),
        start_offset_s: float = 0.0,
    ):
        self.call_id = call_id
        self.cfg = cfg
        # Where this call's audio clock sits relative to the global grid. Set
        # once at call setup and never adjusted -- re-phasing mid-call would
        # defeat the point.
        self.start_offset_s = start_offset_s

        self._sr = cfg.sample_rate
        capacity = int(max(cfg.fast_window_s, cfg.slow_window_s) * self._sr)
        self.buffer = RingBuffer(capacity)

        self._next_tick: dict[Lane, Optional[float]] = {"fast": None, "slow": None}
        self.windows_emitted: dict[Lane, int] = {"fast": 0, "slow": 0}

    @property
    def elapsed_s(self) -> float:
        return self.buffer.total_written / self._sr

    def _grid_time(self, t: float) -> float:
        """Audio-clock time expressed on the shared global grid."""
        return t + self.start_offset_s

    def _first_tick_after(self, window_s: float) -> float:
        """The earliest grid instant at which a full window exists."""
        tick = self.cfg.tick_s
        g = self._grid_time(window_s)
        k = -(-g // tick)                      # ceil division
        return k * tick - self.start_offset_s  # back to the audio clock

    def push(self, samples: torch.Tensor) -> list[Window]:
        """Append audio and return every window that just became ready.

        The audio is fed in slices that never span a tick. That is not an
        optimisation, it is the correctness condition: a window is cut with
        `buffer.tail(n)`, which reads the END of the buffer, so if a single push
        crossed several ticks every window it emitted would carry the same final
        second of audio while its `t_start` advanced correctly.

        That failure is silent and it already cost a headline number. eta -- the
        effective-sample-size deflation in docs/RESULTS.md -- was measured by
        pushing whole 14 s calls in one call to this method. Every window in a
        sequence was byte-identical, the branch scores differed only by float
        non-determinism (median std 1.8e-15), and the autocorrelation was
        computed on that noise. It reported eta = 0.080; measured on windows
        that actually differ it is 0.367.

        Streaming callers (the gateway, end_to_end.py) push well under one hop
        at a time and were never affected. The fix is here rather than in those
        callers so that a batch caller cannot reintroduce it.
        """
        samples = samples.reshape(-1).to(torch.float32)
        out: list[Window] = []
        i, total = 0, samples.numel()
        while i < total:
            step = max(1, min(self._samples_until_next_tick(), total - i))
            out.extend(self._push_slice(samples[i:i + step]))
            i += step
        out.sort(key=lambda w: (w.t_end, w.lane))
        return out

    def _samples_until_next_tick(self) -> int:
        """Samples that can be appended before some lane's next window is due."""
        now = self.elapsed_s
        best: Optional[int] = None
        for lane, window_s in (("fast", self.cfg.fast_window_s),
                               ("slow", self.cfg.slow_window_s)):
            if self._next_tick[lane] is None:
                self._next_tick[lane] = self._first_tick_after(window_s)
            # The epsilon is load-bearing, not defensive. now is a ratio of
            # integers, so (next_tick - now) * sr lands a few ulps above the
            # exact sample count -- ceil(16.000000000000004) is 17, and that one
            # extra sample shifts the window to [1, 16001) while its timestamps
            # still say [0, 16000). Subtracting a fraction of a sample before
            # the ceil absorbs the error without ever skipping a real sample.
            d = int(math.ceil((self._next_tick[lane] - now) * self._sr - 1e-6))
            best = d if best is None else min(best, d)
        return best if best is not None else 1

    def _push_slice(self, samples: torch.Tensor) -> list[Window]:
        self.buffer.push(samples)
        now = self.elapsed_s
        out: list[Window] = []

        for lane, window_s, hop_s in (
            ("fast", self.cfg.fast_window_s, self.cfg.fast_hop_s),
            ("slow", self.cfg.slow_window_s, self.cfg.slow_hop_s),
        ):
            if self._next_tick[lane] is None:
                self._next_tick[lane] = self._first_tick_after(window_s)

            n = int(round(window_s * self._sr))
            while now + 1e-9 >= self._next_tick[lane]:
                t_end = self._next_tick[lane]
                data = self.buffer.tail(n)
                if data is None:
                    break                       # not enough audio buffered yet
                out.append(
                    Window(
                        lane=lane,                     # type: ignore[arg-type]
                        call_id=self.call_id,
                        t_start=t_end - window_s,
                        t_end=t_end,
                        hop_s=hop_s,
                        samples=data,
                        quality=speech_quality(data),
                    )
                )
                self.windows_emitted[lane] += 1
                self._next_tick[lane] = t_end + hop_s

        return out


def assign_tick_offset(call_index: int, cfg: ChunkerConfig = ChunkerConfig()) -> float:
    """Phase offset for a new call.

    Real deployments derive this from the call's wall-clock arrival; the
    deterministic spread here is for load tests, so N simulated calls exercise
    the same grid alignment that real arrivals would produce.
    """
    return (call_index % 1) * cfg.tick_s
