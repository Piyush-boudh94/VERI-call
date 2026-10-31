"""Per-call session state and the branch-event pipeline.

Streaming detection is inherently stateful -- ring buffers, running fusion
state, per-call calibration -- so "stateless workers behind a load balancer" is
not achievable without shipping a 192 kB ring buffer on every request, which
would cost 384 kB/s/call and dominate the compute it was meant to distribute.

State is therefore split by cost and reconstructibility:

    audio ring buffer   in-process only. Too big to externalise, and cheap to
                        rebuild -- after a crash you re-buffer one second.
    FusionState (~1 KB) in-process, snapshotted. Small, and expensive to
                        recreate, because it IS the accumulated evidence.

The safety-critical invariant on recovery: a session must never report
confidence over a period it did not observe. A reconnect gap is charged against
T_eff, so it surfaces as a confidence drop rather than silently vanishing.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import torch

from ..audio.chunker import ChunkerConfig, DualRateChunker, Window
from ..fusion.config import FusionConfig, TierPolicy
from ..fusion.engine import FusionState, apply_hysteresis, update
from ..fusion.types import BranchEvent, RiskView, TierChange


@dataclass
class SessionConfig:
    chunker: ChunkerConfig = field(default_factory=ChunkerConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    policy: TierPolicy = field(default_factory=TierPolicy)
    tenant_id: str = "default"
    enrolled: bool = False


class CallSession:
    """One live call."""

    def __init__(self, call_id: str, cfg: SessionConfig = SessionConfig(),
                 tick_offset: float = 0.0):
        self.call_id = call_id
        self.cfg = cfg
        self.chunker = DualRateChunker(call_id, cfg.chunker, tick_offset)
        self.state = FusionState(call_id, cfg.tenant_id)
        self.state.enrolled = cfg.enrolled
        self.created = time.time()
        self.last_view: Optional[RiskView] = None
        self.changes: list[TierChange] = []
        self.samples_in = 0
        self.dropped = 0

    # ------------------------------------------------------------------ #

    def push_audio(self, pcm: torch.Tensor) -> list[Window]:
        self.samples_in += pcm.numel()
        return self.chunker.push(pcm)

    def apply_events(self, events: list[BranchEvent]) -> tuple[RiskView, list[TierChange]]:
        """Fold scored branch events into the running risk statistic."""
        view = None
        fired: list[TierChange] = []
        for ev in sorted(events, key=lambda e: e.t_end):
            view = update(self.state, ev, self.cfg.fusion)
            change = apply_hysteresis(self.state, view, self.cfg.fusion,
                                      self.cfg.policy)
            if change is not None:
                fired.append(change)
                self.changes.append(change)
        if view is not None:
            self.last_view = view
        return view, fired

    def record_drop(self, n: int = 1) -> None:
        """Every drop must be visible in the fusion state.

        If load shedding were invisible here, the UI could show a confident
        green on a call the system had largely stopped analysing -- which is the
        exact failure mode shedding exists to prevent.
        """
        self.dropped += n
        self.state.dropped_chunks += n

    def snapshot(self) -> dict:
        st = self.state
        return {
            "call_id": self.call_id,
            "t_now": st.t_now,
            "T_eff": st.T_eff,
            "tier": st.tier,
            "tier_latch": st.tier_latch,
            "tier_since": st.tier_since,
            "cusum_up": st.cusum_up,
            "dropped_chunks": st.dropped_chunks,
            "last_seen": dict(st.last_seen),
            "fast": {b: a.snapshot() for b, a in st.fast.items()},
            "slow": {b: a.snapshot() for b, a in st.slow.items()},
            "regime": st.regime.snapshot(),
            "snapshot_wall_t": time.time(),
        }

    @classmethod
    def restore(cls, snap: dict, cfg: SessionConfig = SessionConfig()) -> "CallSession":
        from ..fusion.engine import BranchAcc

        s = cls(snap["call_id"], cfg)
        st = s.state
        st.t_now = float(snap["t_now"])
        st.T_eff = float(snap["T_eff"])
        st.tier = int(snap["tier"])
        st.tier_latch = int(snap["tier_latch"])
        st.tier_since = float(snap["tier_since"])
        st.cusum_up = float(snap["cusum_up"])
        st.dropped_chunks = int(snap["dropped_chunks"])
        st.last_seen = dict(snap["last_seen"])
        st.fast = {b: BranchAcc.restore(v) for b, v in snap["fast"].items()}
        st.slow = {b: BranchAcc.restore(v) for b, v in snap["slow"].items()}
        st.regime = BranchAcc.restore(snap["regime"])

        # Charge the reconnect gap honestly: we did not observe that audio, so
        # it must not count toward confidence.
        gap = max(0.0, time.time() - float(snap.get("snapshot_wall_t", time.time())))
        st.T_eff = max(0.0, st.T_eff - gap)
        st.dropped_chunks += int(gap / 0.5)
        st.recovered = True
        return s

    def to_json(self) -> dict:
        v = self.last_view
        if v is None:
            return {"call_id": self.call_id, "state": "warming_up",
                    "tier": 0, "elapsed_s": self.chunker.elapsed_s}
        return {
            "call_id": self.call_id,
            "elapsed_s": round(self.chunker.elapsed_s, 2),
            "risk": round(v.S_lo, 3),          # decisions act on the lower bound
            "risk_raw": round(v.S, 3),
            "meter": round(v.L_rate, 3),       # rate-invariant, for the UI
            "tier": self.state.tier,
            "tier_latch": self.state.tier_latch,
            "action": self.cfg.policy.action.get(self.state.tier, "monitor"),
            "confidence": {
                "level": v.confidence.level,
                "score": round(v.confidence.score, 3),
                "audio_scored_s": round(v.confidence.audio_scored_s, 2),
                "reason": v.confidence.reason,
            },
            "branches": {b: round(c, 3) for b, c in v.contributions.items()},
            "active_branches": list(v.subset),
            "identity_llr": v.llr_target,
            "enrolled": v.enrolled,
            "changed_at": v.changed_at,
            "recovered": self.state.recovered,
            "dropped_chunks": self.state.dropped_chunks,
        }
