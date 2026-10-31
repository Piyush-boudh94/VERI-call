"""Wire types for the fusion layer.

Everything crossing a branch boundary is ALREADY CALIBRATED to a log-likelihood
ratio in nats. The fusion engine never sees a raw model score -- that separation
is what lets the operating point move at runtime without retraining anything.
"""
from __future__ import annotations

import math

from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

# A, A2, B contribute to the spoof score. C is an IDENTITY signal and is
# deliberately kept out of it -- a good clone of an enrolled speaker RAISES
# voiceprint similarity, so adding C positively would reward the attack.
Branch = Literal["A", "A2", "B", "C"]
SPOOF_BRANCHES: tuple[Branch, ...] = ("A", "A2", "B")


@dataclass(frozen=True, slots=True)
class BranchEvent:
    """One scored window from one branch."""

    branch: Branch
    call_id: str
    t_start: float          # sec since call start, audio clock (not wall clock)
    t_end: float
    llr: float              # nats, spoof-vs-bonafide, for THIS window
    llr_var: float          # variance of that llr, from the calibration fit
    quality: float          # 0..1: speech_ratio * snr_gate * clip_gate
    hop_s: float            # non-overlapping duration this event may claim
    emb: Optional[Sequence[float]] = None   # branch C only: speaker embedding
    calib_id: str = ""
    model_id: str = ""

    @property
    def window_s(self) -> float:
        """Audio this LLR was actually computed over."""
        return self.t_end - self.t_start

    def __post_init__(self) -> None:
        if self.hop_s <= 0:
            raise ValueError(f"hop_s must be > 0, got {self.hop_s}")
        if self.t_end - self.t_start <= 0:
            raise ValueError("window_s must be > 0")
        if self.hop_s > self.t_end - self.t_start + 1e-9:
            raise ValueError("hop_s cannot exceed the window it credits")
        if not 0.0 <= self.quality <= 1.0:
            raise ValueError(f"quality must be in [0,1], got {self.quality}")
        if self.t_end < self.t_start:
            raise ValueError("t_end precedes t_start")
        # A negative or non-finite variance is a bug in whoever built the event,
        # not a runtime condition, so it fails loudly here. Left unchecked it
        # was ACCEPTED and produced S_lo == S -- i.e. maximum confidence from a
        # broken uncertainty estimate, which is the opposite of the safe
        # direction. Measured: llr_var=-0.5 gave the same S_lo as a perfectly
        # certain measurement.
        if not math.isfinite(self.llr_var) or self.llr_var < 0.0:
            raise ValueError(
                f"llr_var must be finite and >= 0, got {self.llr_var}")


@dataclass(frozen=True, slots=True)
class Confidence:
    level: Literal["LOW", "MEDIUM", "HIGH"]
    score: float
    audio_scored_s: float
    branches: tuple[Branch, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class RiskView:
    """What the UI and the policy layer consume."""

    S: float                # decision statistic, nats
    S_lo: float             # one-sided 95% lower bound -- decisions act on THIS
    S_fast: float
    S_slow: float
    L_rate: float           # nats/sec -- the rate-invariant UI meter
    confidence: Confidence
    subset: tuple[Branch, ...]
    llr_target: Optional[float]     # branch C, identity. NOT part of S.
    enrolled: bool
    T_eff: float
    contributions: dict[str, float] = field(default_factory=dict)
    changed_at: Optional[float] = None
    # True when the branch scores no longer resemble the distribution their
    # calibration was fitted on. Every threshold downstream assumes calibrated
    # LLRs, so when this is set the correct reading of the score is "unknown",
    # not "low risk". See vericall/fusion/domain_guard.py.
    calibration_suspect: bool = False
    calibration_notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TierChange:
    call_id: str
    tier: int
    direction: Literal["escalate", "de-escalate"]
    at: float
    S_lo: float
    reason: str
