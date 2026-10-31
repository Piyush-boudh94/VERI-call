"""Cross-call micro-batching for the fast lane.

The chunker snaps every call to a shared 500 ms grid, so N concurrent calls
present their windows at nearly the same instant. This collects those into one
GPU forward pass, which is the difference between serving 4 calls and serving 50
on a single MIG slice: the slice clocks down between sparse kernels, and a batch
of 1 varies 14x in latency where a batch of 32 varies 1.07x (docs/BENCHMARKS.md).

**It batches the FORWARD only, not the calibration.** An earlier version batched
`score_fast` whole, which made cross-call batching and per-call channel
calibration mutually exclusive -- windows from calls on different channels would
have shared one Platt mapping. The two halves have different shapes: the model
forward is expensive and identical for every call, the Platt mapping is two
multiplies specific to one call's detected channel. Batch what is shared, apply
what is not.

**Why not wait for the next tick.** An earlier version blocked until the next
500 ms boundary before flushing, adding up to 500 ms to every score -- worse
than not batching at all. The grid makes windows *arrive* together; the batcher
only has to catch them, which takes tens of milliseconds.

**Sizing the collect window.** This is Triton's `max_queue_delay_microseconds`
by another name, and the same trade: the delay is added to every request's
latency in exchange for amortising kernel-launch and memory-transfer overhead
across the batch. 30 ms against a 1.6 s time-to-first-score SLO is under 2% of
the budget, and it is a *soft* deadline -- a full batch flushes immediately.
"""
from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable, Optional, Sequence

import torch

from ..audio.chunker import Window

# 64: the same criterion the old 32 was chosen by -- "~96% of peak throughput
# with the best p99" -- re-measured after the MIG profile changed from
# 1g.18gb (16 SMs) to 2g.35gb (32 SMs) on 2026-10-31. Branch A + A2 together,
# which is what the fast lane actually runs:
#
#   batch    p99     % of peak throughput
#      32   26.4 ms         88%     <- the old constant, now off-peak
#      64   42.8 ms         96%     <- this
#     128   82.9 ms         99%
#     192  123.5 ms        100%
#
# 128 buys three points of throughput for forty milliseconds of p99, which is
# the wrong trade on a path with a 250 ms budget. 64 sits at 17% of budget.
#
# VRAM is not what bounds this and never was: batch 384 peaks at 7.4 GiB of
# 32.5. The slice upgrade helps through SMs, so the knob that matters is batch
# size, not model size.
BATCH_MAX = 64
COLLECT_MS = 30.0

# Refuse rather than queue without bound. An unbounded queue turns an overload
# into an out-of-memory kill and a latency spike that outlives the overload;
# a bounded one turns it into an immediate, attributable error. Sized well above
# what N=64 calls present in one collect window (64 calls x ~2 windows = 128).
QUEUE_MAX = 512


class BatcherOverloaded(RuntimeError):
    """The pending queue is full. Shed this request rather than queue it."""


class CrossCallBatcher:
    """Collects windows from concurrent calls into shared forward passes.

    `forward_fn(windows) -> dict[branch, Tensor]` must be the raw, uncalibrated
    forward. Callers get their own windows' raw scores back and apply their own
    calibration.
    """

    def __init__(self,
                 forward_fn: Callable[[Sequence[Window]], dict],
                 max_batch: int = BATCH_MAX,
                 collect_ms: float = COLLECT_MS,
                 queue_max: int = QUEUE_MAX):
        self.forward_fn = forward_fn
        self.max_batch = max_batch
        self.collect_s = collect_ms / 1000.0
        self.queue_max = queue_max
        self._pending: list[tuple[Window, asyncio.Future]] = []
        self._flush_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()        # guards the pending queue only
        # There is exactly one MIG slice. Concurrent flushes each spawn a thread
        # that contends for it, so they serialise inside CUDA anyway while
        # paying thread-pool and context-switch overhead on top. Measured
        # without this: p99 973 ms and the server falling behind real time.
        self._gpu = asyncio.Lock()
        self.batches = 0
        self.windows = 0
        self.rejected = 0
        self.sizes: list[int] = []
        self.wait_ms: list[float] = []

    async def submit(self, windows: Sequence[Window]) -> list[dict]:
        """Raw scores for these windows, batched with whatever else is in flight.

        Returns one {branch: float} dict per window, in the order given.
        Raises BatcherOverloaded if the queue is full.
        """
        if not windows:
            return []
        loop = asyncio.get_running_loop()
        t_submit = time.perf_counter()
        futures = []
        async with self._lock:
            if len(self._pending) + len(windows) > self.queue_max:
                self.rejected += len(windows)
                raise BatcherOverloaded(
                    f"inference queue full ({len(self._pending)}/"
                    f"{self.queue_max})")
            for w in windows:
                fut = loop.create_future()
                self._pending.append((w, fut))
                futures.append(fut)
            full = len(self._pending) >= self.max_batch
            if not full and (self._flush_task is None or self._flush_task.done()):
                self._flush_task = asyncio.create_task(self._flush_after())
        if full:
            await self._flush()

        out = await asyncio.gather(*futures)
        self.wait_ms.append((time.perf_counter() - t_submit) * 1000.0)
        if len(self.wait_ms) > 5000:
            del self.wait_ms[:2500]
        return list(out)

    async def _flush_after(self) -> None:
        await asyncio.sleep(self.collect_s)
        await self._flush()

    async def _take_batch(self) -> list[tuple[Window, asyncio.Future]]:
        async with self._lock:
            batch = self._pending[: self.max_batch]
            self._pending = self._pending[self.max_batch :]
            return batch

    async def _flush(self) -> None:
        batch = await self._take_batch()
        if not batch:
            return
        windows = [w for w, _ in batch]

        try:
            # Off the event loop, and NOT under the queue lock. forward_fn is a
            # synchronous GPU call; running it inline blocks the loop so every
            # other call's WebSocket reads and submits queue behind it. Measured
            # cost of getting this wrong: p99 101 ms -> 780 ms, while mean batch
            # size had correctly risen from 1.0 to 5.5. Batching was working;
            # holding the loop was not.
            async with self._gpu:
                raw = await asyncio.to_thread(self.forward_fn, windows)
        except Exception as exc:                              # noqa: BLE001
            for _, fut in batch:
                if not fut.done():
                    fut.set_exception(exc)
            return

        # Split the batched tensors back into per-window rows. Index order is
        # the batch order, which is why the forward must not reorder.
        cpu = {b: t.detach().to("cpu") for b, t in raw.items()}
        for i, (_w, fut) in enumerate(batch):
            if not fut.done():
                fut.set_result({b: t[i] for b, t in cpu.items()})

        self.batches += 1
        self.windows += len(windows)
        self.sizes.append(len(windows))
        if len(self.sizes) > 5000:
            del self.sizes[:2500]

        async with self._lock:
            more = bool(self._pending)
        if more:
            asyncio.create_task(self._flush())      # keep draining

    def stats(self) -> dict:
        if not self.sizes:
            return {"batches": 0, "windows": 0, "mean_batch": 0.0,
                    "max_batch_seen": 0, "rejected": self.rejected,
                    "queue_depth": len(self._pending)}
        waits = sorted(self.wait_ms)
        return {
            "batches": self.batches,
            "windows": self.windows,
            "mean_batch": round(sum(self.sizes) / len(self.sizes), 2),
            "max_batch_seen": max(self.sizes),
            "rejected": self.rejected,
            "queue_depth": len(self._pending),
            "submit_to_result_ms_p50": (round(waits[len(waits) // 2], 1)
                                        if waits else 0.0),
            "submit_to_result_ms_p99": (
                round(waits[min(len(waits) - 1, int(len(waits) * 0.99))], 1)
                if waits else 0.0),
        }
