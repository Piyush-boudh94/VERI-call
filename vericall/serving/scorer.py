"""Branch scoring with cross-call batching.

The throughput trick that makes this work is **global tick alignment**, applied
upstream in the chunker. Calls start at arbitrary times, so their window
boundaries scatter uniformly; a batcher collecting for 20 ms would gather ~2
requests rather than 32, and filling a batch would cost ~320 ms and blow the
p99 budget. Naive dynamic batching does not work on unsynchronised audio.

Measured on one mig-1g.18gb slice, batching also buys latency *stability*, not
just throughput: batch 1 varies 4-60 ms (14x spread) because a sparsely-fed slice
clocks down between kernels, while batch 32 varies 40.7-43.7 ms (1.07x).
"""
from __future__ import annotations

import contextlib
import os
import math
import time
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import torch

from ..audio.chunker import Window
from ..fusion.calibration import PlattCalibration
from ..fusion.types import BranchEvent

DEFAULT_VAR = 0.04          # calibration variance until a real fit exists


def _logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp(eps, 1.0 - eps)
    return torch.log(p / (1.0 - p))


@dataclass
class BranchCalibrations:
    """Raw score -> LLR, per branch.

    Identity by default so the pipeline runs before calibration is fit, but that
    is explicitly NOT a valid operating point: an uncalibrated logit is not an
    LLR, and every threshold in the risk engine assumes it is. `fitted` says
    whether the numbers can be believed.
    """

    A: Optional[PlattCalibration] = None
    A2: Optional[PlattCalibration] = None
    B_cohort: Optional[torch.Tensor] = None

    @property
    def fitted(self) -> bool:
        return self.A is not None

    def apply(self, branch: str, raw: torch.Tensor) -> torch.Tensor:
        cal = getattr(self, branch, None)
        z = _logit(raw)
        if cal is None:
            return z
        return cal.a * z + cal.b


@dataclass
class ScorerStats:
    batches: int = 0
    windows: int = 0
    # Counted rather than raised: a bad voiceprint degrades identity
    # escalation and must not take the spoof detector down with it. Silent
    # degradation is the thing to avoid, so it is reported in /health.
    identity_errors: int = 0
    latencies_ms: list[float] = field(default_factory=list)

    def observe(self, n: int, ms: float) -> None:
        self.batches += 1
        self.windows += n
        self.latencies_ms.append(ms)
        if len(self.latencies_ms) > 5000:
            del self.latencies_ms[:2500]

    def percentiles(self) -> dict[str, float]:
        if not self.latencies_ms:
            return {}
        s = sorted(self.latencies_ms)
        pick = lambda q: s[min(len(s) - 1, int(q * len(s)))]
        return {"p50": pick(0.50), "p95": pick(0.95), "p99": pick(0.99),
                "n": len(s)}


# Inference precision. fp16 halves the GPU time of the fast lane (measured 44.1
# -> 22.0 ms at batch 32, 24.9 -> 13.0 at batch 16) and the frozen SSL frontend
# dominates that cost, so this is the largest single efficiency lever available
# without changing the model.
#
# **fp16 rather than bf16, against the usual production advice, on measurement.**
# The standard recommendation is bf16 because it keeps fp32's exponent range and
# so cannot overflow. That reasoning is about large models with large
# activations. Measured here on real demo audio, the peak activation magnitude
# anywhere in either branch is 463.5 -- 141x of headroom to fp16's 65504 ceiling
# -- so overflow is not a live risk, and fp16's extra mantissa bits (10 vs 7)
# make it markedly more faithful:
#
#     vs fp32, 256 windows      max |delta P(spoof)|      Pearson r
#     bf16   Branch A                   2.09e-03           0.998836
#     fp16   Branch A                   2.21e-04           0.999925
#
# Fidelity is the binding constraint, not speed: every threshold in this system
# is a fitted LLR, so a precision change that moved the scores would silently
# invalidate the calibration rather than merely degrade it. A 2e-4 shift in
# P(spoof) is far below that. bf16 remains available for a deployment whose
# audio drives activations closer to the ceiling; VERICALL_PRECISION selects.
_PRECISION = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}


def _autocast_dtype():
    name = os.environ.get("VERICALL_PRECISION", "fp16").lower()
    if name not in _PRECISION:
        name = "fp16"
    return name, _PRECISION[name]


class BranchScorer:
    """Scores batches of windows and emits calibrated BranchEvents."""

    def __init__(self, branch_a, branch_a2=None, branch_b=None,
                 calib: BranchCalibrations = BranchCalibrations(),
                 device: torch.device | str = "cpu", branch_c=None):
        self.device = torch.device(device)
        self.a = branch_a
        self.a2 = branch_a2
        self.b = branch_b
        # Branch C rides the slow lane and emits an EMBEDDING, never a spoof
        # score: a good clone of an enrolled speaker raises identity confidence,
        # so letting identity into S would argue for the attacker. The fusion
        # engine uses the embedding for speaker-turn change points, and the
        # prevention policy uses the identity LLR for conflict detection.
        self.c = branch_c
        self.precision, self._dtype = _autocast_dtype()
        # CPU has no meaningful fp16 path and autocast there is a pessimisation.
        if self.device.type != "cuda":
            self.precision, self._dtype = "fp32", None
        self.calib = calib
        self.stats = ScorerStats()

    # ------------------------------------------------------------------ #

    @torch.inference_mode()
    def forward_fast(self, windows: Sequence[Window]) -> dict[str, torch.Tensor]:
        """Raw, UNCALIBRATED branch scores for a batch of fast-lane windows.

        Split out from score_fast so that cross-call batching and per-call
        calibration can coexist. They could not before: the batcher batched
        score_fast whole, calibration included, so windows from calls on
        different channels would have shared one Platt mapping -- and the code
        said so, disabling one feature to keep the other.

        The split is possible because the two halves have different shapes. The
        model forward is the expensive part, identical for every call, and wants
        to be batched. The Platt mapping is two multiplies, specific to the
        call's detected channel, and costs nothing to do per item afterwards.
        Batch what is shared; apply what is not.
        """
        if not windows:
            return {}
        t0 = time.perf_counter()
        x = torch.stack([w.samples for w in windows]).to(self.device)
        with self._cast():
            out = {"A": self.a.score(x).float()}
            if self.a2 is not None:
                out["A2"] = self.a2.score(x).float()
        self.stats.observe(len(windows), (time.perf_counter() - t0) * 1000.0)
        return out

    def _cast(self):
        """autocast when a half dtype is selected, a no-op otherwise."""
        if self._dtype is None:
            return contextlib.nullcontext()
        return torch.autocast("cuda", dtype=self._dtype)

    def events_from_raw(self, windows: Sequence[Window],
                        raw: dict[str, torch.Tensor],
                        calib: Optional[BranchCalibrations] = None
                        ) -> list[BranchEvent]:
        """Apply this call's calibration to raw scores. Cheap, CPU, per call."""
        if not windows or not raw:
            return []
        cal = calib if calib is not None else self.calib
        events: list[BranchEvent] = []
        for branch in ("A", "A2"):
            if branch not in raw:
                continue
            llr = cal.apply(branch, raw[branch]).cpu()
            for w, l in zip(windows, llr):
                events.append(self._event(branch, w, float(l)))
        return events

    @torch.inference_mode()
    def score_fast(self, windows: Sequence[Window],
                   calib: Optional[BranchCalibrations] = None) -> list[BranchEvent]:
        """forward + calibrate in one call, for offline scripts and tests.

        The serving path uses the two halves separately so the forward can be
        batched across calls.
        """
        return self.events_from_raw(windows, self.forward_fast(windows), calib)

    @torch.inference_mode()
    def score_slow(self, windows: Sequence[Window],
                   enroll_emb: Optional[torch.Tensor] = None) -> list[BranchEvent]:
        """Branch B and Branch C over slow-lane windows.

        Its output is an anomaly score with no negative distribution behind it,
        so it is mapped against a bona fide cohort and hard-clipped in the fusion
        engine rather than trusted at face value.

        `enroll_emb` is THIS call's enrolled voiceprint, or None. It is a
        per-call argument and not scorer state for the same reason `call_calib`
        is: one BranchScorer serves every concurrent call, so anything stored
        on it would leak one caller's identity into another's score.
        """
        if not windows or (self.b is None and self.c is None):
            return []
        x = torch.stack([w.samples for w in windows]).to(self.device)
        out = []

        if self.b is not None:
            with self._cast():
                raw = self.b.score(x).float().cpu()
            cohort = self.calib.B_cohort
            for w, r in zip(windows, raw):
                if cohort is not None and cohort.numel():
                    tail = float((cohort > float(r)).float().mean().clamp(min=1e-3))
                    llr = math.log(1.0 / tail) - math.log(1.0 / 0.05)
                else:
                    llr = 0.0            # no cohort yet: contribute nothing
                out.append(self._event("B", w, llr))

        if self.c is not None:
            # The embedding is always computed: the fusion engine diffs it
            # across windows to find speaker turns -- the signal that separates
            # "this call was synthetic from the first second" from "the caller
            # was swapped" -- and that works with no enrollment at all.
            with self._cast():
                emb = self.c.embed(x).float().cpu()

            # The LLR is 0.0 only when there is nothing to compare against.
            # It used to be hardcoded 0.0 unconditionally, which made
            # identity_conflict unreachable: the policy tests
            # `enrolled and identity_llr > 0` and 0.0 > 0 is False. Computed
            # from the embedding already in hand rather than by calling
            # llr_target(), which would run ECAPA a second time on the same
            # audio -- C is the most expensive branch per second here.
            #
            # This LLR never enters the spoof score S. The fusion engine routes
            # branch "C" to _speaker_turn and st.llr_target and returns before
            # any accumulation, because a convincing clone of an enrolled
            # speaker RAISES identity similarity: letting it into S would
            # argue for the attacker.
            ident = None
            if enroll_emb is not None:
                try:
                    ident = self.c.llr_target_from_emb(
                        enroll_emb.to(emb.device), emb).float().cpu()
                except Exception:                             # noqa: BLE001
                    # A malformed voiceprint must not kill a live call. The
                    # spoof branches are unaffected by identity.
                    ident = None
                    self.stats.identity_errors += 1

            for i, (w, e) in enumerate(zip(windows, emb)):
                llr = 0.0 if ident is None else float(ident[i])
                if not math.isfinite(llr):
                    llr = 0.0
                ev = self._event("C", w, llr)
                out.append(replace(ev, emb=e.tolist()))
        return out

    def _event(self, branch: str, w: Window, llr: float) -> BranchEvent:
        return BranchEvent(
            branch=branch, call_id=w.call_id, t_start=w.t_start, t_end=w.t_end,
            llr=llr, llr_var=DEFAULT_VAR, quality=w.quality, hop_s=w.hop_s,
            calib_id="identity" if not self.calib.fitted else "fitted",
        )
