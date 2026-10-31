"""Detecting that the calibration no longer applies.

The most dangerous result in this project was not a bad EER. It was that on
RTCFake the fitted fusion scored **worse than its own best single branch** --
72.70% against 28.72% on Mandarin -- and nothing anywhere said so. Branch A had
inverted (79% EER, i.e. 21% if flipped), the weights put the most weight on it
because they were fitted where it was strongest, and the system reported a
confident number the whole time.

Every threshold downstream assumes the branch scores are calibrated LLRs on the
distribution the calibration was fitted on. When that stops being true the
correct output is not a different number, it is **"I do not know"**. This module
detects the two signatures that were visible in the RTCFake data:

1. **Score drift.** Incoming scores sit far from the distribution the Platt
   parameters were fitted on. The mapping is an extrapolation, not a
   calibration.
2. **Branch disagreement.** A and A2 were correlated in-domain -- that is why
   their weights were fitted jointly. When one inverts and the other holds,
   they decorrelate. On RTCFake that was the clearest early signal: A at chance
   or worse while A2 retained real signal.

Neither detects domain shift in general; both detect the specific failure that
actually happened, which is worth more than a general claim that does not fire.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class BranchReference:
    """Where a branch's scores sat when its calibration was fitted."""

    branch: str
    mean: float
    sd: float

    def z(self, value: float) -> float:
        return abs(value - self.mean) / max(self.sd, 1e-6)


@dataclass
class DomainGuardConfig:
    # 3 sd is the conventional out-of-distribution line and is deliberately
    # loose: this should fire on a corpus change, not on an unusual caller.
    drift_z: float = 3.0
    # Sustained disagreement in nats between branches that were fitted as
    # correlated. One window disagreeing is normal; a call-long split is not.
    disagreement_nats: float = 4.0
    min_windows: int = 8            # do not judge a call on its first second
    confidence_penalty: float = 0.5  # multiplicative, applied when tripped


@dataclass
class DomainGuard:
    """Per-call running check that the calibration still applies."""

    cfg: DomainGuardConfig = field(default_factory=DomainGuardConfig)
    reference: dict[str, BranchReference] = field(default_factory=dict)
    _n: dict[str, int] = field(default_factory=dict)
    _sum: dict[str, float] = field(default_factory=dict)
    _last: dict[str, float] = field(default_factory=dict)
    tripped: bool = False
    reasons: list[str] = field(default_factory=list)

    def observe(self, branch: str, llr: float) -> None:
        self._n[branch] = self._n.get(branch, 0) + 1
        self._sum[branch] = self._sum.get(branch, 0.0) + llr
        self._last[branch] = llr

    def mean(self, branch: str) -> Optional[float]:
        n = self._n.get(branch, 0)
        return self._sum[branch] / n if n else None

    @property
    def windows(self) -> int:
        return max(self._n.values(), default=0)

    def evaluate(self) -> bool:
        """Re-check both signatures. Returns True if the calibration is suspect.

        Latching on purpose: once a call has shown that the calibration does not
        apply, later windows agreeing again does not restore trust in the
        numbers already accumulated.
        """
        if self.windows < self.cfg.min_windows:
            return self.tripped

        for branch, ref in self.reference.items():
            m = self.mean(branch)
            if m is None:
                continue
            z = ref.z(m)
            if z > self.cfg.drift_z:
                self._trip(
                    f"{branch} mean LLR {m:+.2f} sits {z:.1f} sd from the "
                    f"calibration distribution (mean {ref.mean:+.2f}, "
                    f"sd {ref.sd:.2f}) -- the mapping is extrapolating")

        a, a2 = self.mean("A"), self.mean("A2")
        if a is not None and a2 is not None:
            gap = abs(a - a2)
            if gap > self.cfg.disagreement_nats and (a * a2) < 0:
                self._trip(
                    f"A and A2 disagree by {gap:.2f} nats with opposite signs "
                    f"(A {a:+.2f}, A2 {a2:+.2f}). They were fitted as "
                    "correlated; opposite signs sustained over a call is the "
                    "signature of one branch inverting under domain shift")
        return self.tripped

    def _trip(self, reason: str) -> None:
        self.tripped = True
        if reason not in self.reasons:
            self.reasons.append(reason)

    def adjust_confidence(self, score: float) -> float:
        return score * self.cfg.confidence_penalty if self.tripped else score

    def to_json(self) -> dict:
        return {
            "calibration_suspect": self.tripped,
            "reasons": self.reasons,
            "windows": self.windows,
            "branch_mean_llr": {b: round(self.mean(b), 3)
                                for b in self._n if self.mean(b) is not None},
        }


def reference_from_summary(summary: dict) -> dict[str, BranchReference]:
    """Build references from a calibration summary, where recorded.

    Returns empty rather than inventing numbers when the summary predates
    distribution recording -- a guard with a fabricated reference is worse than
    no guard, because it fires or stays silent for reasons unrelated to the data.
    """
    out: dict[str, BranchReference] = {}
    for branch, entry in (summary.get("score_reference") or {}).items():
        try:
            out[branch] = BranchReference(branch, float(entry["mean"]),
                                          float(entry["sd"]))
        except (KeyError, TypeError, ValueError):
            continue
    return out
