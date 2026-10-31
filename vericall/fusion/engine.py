"""Asynchronous multi-rate score fusion and the running risk statistic.

The problem this solves: branch A/A2 emit at 2 Hz, branch B at 1 Hz, branch C
often never. A fixed-width gating MLP structurally cannot consume that. The
mechanism here is linear logistic-regression fusion (the NIST SRE / ASVspoof
standard) composed with an exponentially-discounted sequential accumulator over
LLR *rates*, which makes multi-rate fusion fall out for free: the accumulator
integrates over TIME, not over events.

Pure float arithmetic -- no torch, no GPU, no cluster quota. Runs anywhere.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Iterable, Optional, Sequence

from .config import FusionConfig, TierPolicy
from .domain_guard import DomainGuard
from .types import (
    SPOOF_BRANCHES,
    Branch,
    BranchEvent,
    Confidence,
    RiskView,
    TierChange,
)

_Z95 = 1.6448536269514722   # one-sided 95%


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return num / (na * nb)


@dataclass(slots=True)
class BranchAcc:
    """Time-discounted weighted mean of LLR-rate, with its variance.

    Weighted mean:      num/den
    Variance of that:   num_var/den**2      (variance of a weighted mean)

    Discounting multiplies num and den by g, and num_var by g**2, so the
    variance stays consistent with the mean under decay.
    """

    num: float = 0.0
    den: float = 0.0
    num_var: float = 0.0
    t_last: float = -1e9
    n_events: int = 0

    def decay_to(self, t: float, tau: float) -> None:
        if self.den == 0.0:
            self.t_last = t
            return
        dt = t - self.t_last
        if dt <= 0.0:
            return
        g = math.exp(-dt / tau)
        self.num *= g
        self.den *= g
        self.num_var *= g * g
        self.t_last = t

    def observe(
        self, t: float, rate: float, rate_var: float, weight: float, tau: float
    ) -> None:
        self.decay_to(t, tau)
        self.num += weight * rate
        self.den += weight
        self.num_var += (weight ** 2) * rate_var
        self.n_events += 1

    @property
    def mean_rate(self) -> float:
        return self.num / self.den if self.den > 0.0 else 0.0

    @property
    def var_mean_rate(self) -> float:
        return self.num_var / (self.den ** 2) if self.den > 0.0 else 0.0

    def snapshot(self) -> dict[str, float]:
        return {
            "num": self.num, "den": self.den, "num_var": self.num_var,
            "t_last": self.t_last, "n_events": float(self.n_events),
        }

    @classmethod
    def restore(cls, d: dict[str, float]) -> "BranchAcc":
        return cls(
            num=float(d["num"]), den=float(d["den"]),
            num_var=float(d["num_var"]), t_last=float(d["t_last"]),
            n_events=int(d["n_events"]),
        )


@dataclass(slots=True)
class FusionState:
    """~40 floats. Snapshottable, recoverable, testable without a GPU."""

    call_id: str
    tenant_id: str = "default"
    t_now: float = 0.0                      # audio clock

    fast: dict[str, BranchAcc] = field(
        default_factory=lambda: {b: BranchAcc() for b in SPOOF_BRANCHES}
    )
    slow: dict[str, BranchAcc] = field(
        default_factory=lambda: {b: BranchAcc() for b in SPOOF_BRANCHES}
    )

    last_seen: dict[str, float] = field(default_factory=dict)
    T_eff: float = 0.0                      # credited, quality-weighted audio

    # Branch C -- identity. Deliberately NOT part of the spoof score.
    llr_target: Optional[float] = None
    enrolled: bool = False
    last_emb: Optional[list[float]] = None

    # regime: long-horizon view of the call, deliberately NOT reset by a
    # change point, so 'have we already established a spoof regime?' stays
    # answerable after the accumulators are wiped.
    regime: BranchAcc = field(default_factory=BranchAcc)

    # change-point
    cusum_up: float = 0.0
    last_change_t: Optional[float] = None
    change_events: list[dict] = field(default_factory=list)

    # decision / hysteresis
    tier: int = 0
    tier_since: float = 0.0
    tier_latch: int = 0                     # audit; never decreases
    consec_cross: int = 0
    last_escalation_t: Optional[float] = None   # cooldown is BETWEEN
    # escalations, not from call start -- gating on tier_since would make
    # an alert impossible during the first cooldown window of every call.

    # Optional domain guard. None means no reference distribution was
    # available, and a guard with a fabricated reference is worse than none.
    guard: Optional[DomainGuard] = None

    # health
    recovered: bool = False
    dropped_chunks: int = 0
    # Counted separately from load shedding: a shed chunk means the server was
    # busy, a non-finite score means a BRANCH is broken. Conflating them would
    # hide a model fault inside a capacity metric.
    nonfinite: int = 0
    clamped: int = 0
    degrade_level: int = 0

    # ------------------------------------------------------------------ #

    def available(self, cfg: FusionConfig) -> tuple[Branch, ...]:
        """Live branches that actually contribute.

        A zero-weight branch is excluded. Otherwise it would still select the
        subset calibration -- so with w[B]=0 a call would be scored against the
        A+A2+B table, whose (alpha, scale) were fitted while B carried 0.30.
        Contributing nothing but changing which scale is applied is the worst
        of both.
        """
        out: list[Branch] = []
        for b in SPOOF_BRANCHES:
            if cfg.weight_of(b) == 0.0:
                continue
            seen = self.last_seen.get(b)
            if seen is None:
                continue
            if self.t_now - seen <= cfg.stale_after.get(b, 2.0):
                out.append(b)
        return tuple(out)


def _speaker_turn(st: FusionState, ev: BranchEvent, cfg: FusionConfig) -> None:
    if ev.emb is None:
        return
    emb = list(ev.emb)
    if st.last_emb is not None:
        d = 1.0 - cosine(st.last_emb, emb)
        if d > cfg.spk_change_d:
            declare_change_point(st, ev.t_end, cause="speaker_turn", detail=d)
    st.last_emb = emb


def _change_allowed(
    st: FusionState, t: float, baseline_rate: float, cfg: FusionConfig
) -> bool:
    if st.regime.den <= 0.0 or st.regime.n_events < 4:
        return False        # no established baseline -- this is cold start
    if st.tier >= 1:
        return False        # already alerted; 'it changed' is no longer news
    if (
        st.last_change_t is not None
        and (t - st.last_change_t) < cfg.change_refractory_s
    ):
        return False        # already declared one recently
    # Judge against the persistent regime, not the (possibly just-wiped)
    # per-branch accumulator.
    return st.regime.mean_rate < cfg.change_baseline_max_rate


def declare_change_point(
    st: FusionState, t: float, cause: str, detail: float = 0.0
) -> None:
    """Discard pre-change evidence.

    Asymmetric by design: the score may rise immediately (the fast accumulator
    already carries post-change audio) but confidence DROPS, because T_eff is
    cut back. So a spurious change point makes the system cautious rather than
    alarmist. tier_latch is not reset -- the call stays flagged for audit even
    if the live banner clears.
    """
    st.change_events.append({"t": t, "cause": cause, "detail": detail})
    for b in st.slow:
        st.slow[b] = BranchAcc(t_last=t)
    st.T_eff = min(st.T_eff, 1.5)
    st.cusum_up = 0.0
    st.last_change_t = t
    st.tier_since = t        # restart dwell so a real swap can re-escalate fast


# Calibrated LLRs are single-digit nats. 50 is far outside any
# legitimate value and well inside float range.
LLR_CLAMP = 50.0


def update(st: FusionState, ev: BranchEvent, cfg: FusionConfig) -> RiskView:
    if ev.call_id != st.call_id:
        raise ValueError(f"event for {ev.call_id} applied to {st.call_id}")

    st.t_now = max(st.t_now, ev.t_end)

    # ---- non-finite scores: drop the event, never the call ---------------
    # A NaN or Inf LLR propagated straight through to S, S_lo and L_rate.
    # That is not merely untidy: every threshold test is `S_lo >= thr`, and a
    # NaN compares False against everything, so a single NaN silently made the
    # call permanently UNALERTABLE while the dashboard kept drawing a line.
    # Failing open is the worst direction for a fraud detector to fail in.
    #
    # It is a runtime numerical event rather than a programming error -- fp16
    # inference, a degenerate window, a model that diverges -- so the event is
    # dropped and counted rather than raised, which would kill a live call over
    # one bad window. The drop is visible on the RiskView through the same
    # mechanism as load shedding, so confidence cannot be built on evidence
    # that was discarded.
    if not math.isfinite(ev.llr):
        # dropped_chunks feeds the integrity term in render(), so a dropped
        # event lowers confidence exactly as a shed chunk does -- the UI cannot
        # show certainty built on evidence that never arrived.
        st.dropped_chunks += 1
        st.nonfinite += 1
        return render(st, cfg)

    # One window must not be able to dominate the accumulator. Calibrated LLRs
    # live in single digits of nats; anything past this is a broken branch, not
    # strong evidence, and clamping keeps the statistic interpretable instead
    # of letting 1e12 swamp every subsequent window.
    if abs(ev.llr) > LLR_CLAMP:
        ev = replace(ev, llr=math.copysign(LLR_CLAMP, ev.llr))
        st.clamped += 1

    # ---- Branch C: identity, handled entirely separately ----------------
    if ev.branch == "C":
        _speaker_turn(st, ev, cfg)
        if st.enrolled:
            st.llr_target = ev.llr
        st.last_seen["C"] = ev.t_end
        return render(st, cfg)

    # ---- zero-weight branches contribute no evidence at all -------------
    # Not just no LLR: no credited time either. S = T_banked * mean_rate, so a
    # branch that inflates T_eff moves the score even with weight 0. Caught by
    # test_zero_weight_branch_cannot_move_the_score, which measured 9.0 vs 3.0
    # for two otherwise-identical calls. It stays "alive" for staleness
    # bookkeeping so re-enabling it needs no special case.
    if cfg.weight_of(ev.branch) == 0.0:
        st.last_seen[ev.branch] = ev.t_end
        return render(st, cfg)

    # ---- quality gate: silence/hold-music must contribute nothing -------
    if ev.quality < cfg.quality_floor:
        st.last_seen[ev.branch] = ev.t_end      # alive, but no evidence
        return render(st, cfg)

    llr = ev.llr
    if ev.branch == "B":
        llr = max(cfg.floor_B, min(cfg.kappa_B, llr))

    # Credit only the NON-OVERLAPPING duration. Adjacent 1 s windows at a 0.5 s
    # hop share half their audio; crediting the full window would inflate
    # effective sample size by ~sqrt(2) and silently break every threshold.
    # TWO durations, and conflating them is the bug that makes overlapping
    # windows double-count. The LLR is normalised by the WINDOW it was
    # computed over (evidence density, nats per second of analysed audio),
    # while credited time advances only by the HOP (non-overlapping audio).
    # Normalising by the hop instead would cancel exactly and leave S as the
    # raw sum of LLRs -- i.e. no overlap correction at all.
    win = ev.window_s
    rate = llr / win
    # Effective-sample-size deflation belongs on the VARIANCE, not the
    # weight: a weighted mean and its variance are both scale-invariant in
    # the weights, so scaling them would be a silent no-op.
    eta = cfg.eta.get(ev.branch, 1.0)
    rate_var = ev.llr_var / (win ** 2) / max(eta, 1e-6)
    weight = ev.quality * ev.hop_s

    # Regime BEFORE this event, for the change-point check below.
    baseline_rate = st.regime.mean_rate

    st.fast[ev.branch].observe(ev.t_end, rate, rate_var, weight, cfg.tau_fast)
    st.slow[ev.branch].observe(ev.t_end, rate, rate_var, weight, cfg.tau_slow)

    if st.guard is not None:
        st.guard.observe(ev.branch, llr)

    st.regime.observe(ev.t_end, rate, rate_var, weight, cfg.tau_regime)

    st.last_seen[ev.branch] = ev.t_end
    st.T_eff += ev.quality * ev.hop_s

    # ---- CUSUM on the calibrated LLR stream (one-sided, upward) ---------
    st.cusum_up = max(0.0, st.cusum_up + (llr - cfg.cusum_nu))
    if st.cusum_up > cfg.cusum_h:
        if _change_allowed(st, ev.t_end, baseline_rate, cfg):
            declare_change_point(st, ev.t_end, cause="cusum", detail=st.cusum_up)
        else:
            # Alarm consumed: we are already in this regime, or too early to
            # have a baseline. Reset so it does not re-fire immediately.
            st.cusum_up = 0.0

    return render(st, cfg)


def _confidence(
    st: FusionState, K: tuple[Branch, ...], cfg: FusionConfig
) -> Confidence:
    # Breadth is measured against the branches that CAN report under this
    # config, not against every branch that exists. st.available() already
    # excludes zero-weight branches deliberately (see its docstring), so
    # dividing by len(SPOOF_BRANCHES) made the two disagree about the branch
    # universe -- and _confidence was the one that was wrong.
    #
    # The effect was not subtle. calib/weights.json fits only A and A2, so
    # w[B]=0, so B could never enter K, so breadth was pinned at 2/3 and
    # confidence at 1.0*(0.5+0.5*2/3) = 0.8333 for every call that ever ran.
    # That is the 0.833 in every recorded call in logs/vericall.db. It made a
    # fully-healthy call look 17% less certain than it was, and because
    # TierPolicy gates on min_confidence=0.4 it also delayed every action by
    # the time it took coverage to climb an extra 20% (~0.5s at 6s coverage).
    eligible = tuple(b for b in SPOOF_BRANCHES if cfg.weight_of(b) > 0.0)
    if not eligible:
        # Degenerate config: nothing is weighted, so nothing can report.
        # Report zero breadth rather than dividing by zero.
        eligible, breadth = SPOOF_BRANCHES, 0.0
    else:
        breadth = min(1.0, len(K) / float(len(eligible)))

    coverage = min(1.0, st.T_eff / 6.0)     # 6 s ~ every branch has fired once
    integrity = 1.0 - min(1.0, st.dropped_chunks / 20.0)
    score = coverage * (0.5 + 0.5 * breadth) * integrity
    level = "LOW" if score < 0.4 else ("MEDIUM" if score < 0.75 else "HIGH")

    missing = [b for b in eligible if b not in K]
    disabled = [b for b in SPOOF_BRANCHES if b not in eligible]
    if missing:
        reason = (
            f"{st.T_eff:.1f}s of speech analysed, {len(K)} of "
            f"{len(eligible)} detectors reporting "
            f"({', '.join(missing)} not yet available)"
        )
    else:
        reason = (f"{st.T_eff:.1f}s of speech analysed, all "
                  f"{len(eligible)} weighted detectors reporting")
    if disabled:
        # Say "zero-weighted", never "not yet available". The old message
        # promised B was coming; it was switched off by the fitted weights and
        # was never going to arrive.
        reason += f"; {', '.join(disabled)} zero-weighted"
    return Confidence(
        level=level, score=score, audio_scored_s=st.T_eff,
        branches=K, reason=reason,
    )


def render(st: FusionState, cfg: FusionConfig) -> RiskView:
    K = st.available(cfg)
    alpha_K, scale_K = cfg.subset_calib.get(frozenset(K), (0.0, 1.0))

    def combine(accs: dict[str, BranchAcc], tau: float) -> tuple[float, float]:
        total = alpha_K
        var = 0.0
        for b in K:
            acc = accs[b]
            acc.decay_to(st.t_now, tau)      # age stale branches to now
            wb = cfg.weight_of(b)
            total += wb * acc.mean_rate
            var += (wb ** 2) * acc.var_mean_rate
        return scale_K * total, (scale_K ** 2) * var

    L_fast, var_fast = combine(st.fast, cfg.tau_fast)
    L_slow, var_slow = combine(st.slow, cfg.tau_slow)

    T_banked = min(st.T_eff, cfg.T_cap)
    S_fast = T_banked * L_fast
    S_slow = T_banked * L_slow

    # The fast lane must be able to win, or a mid-call swap stays masked by the
    # slow accumulator's pre-swap history.
    if S_fast >= S_slow:
        S, var_S = S_fast, (T_banked ** 2) * var_fast
    else:
        S, var_S = S_slow, (T_banked ** 2) * var_slow

    sd = math.sqrt(max(0.0, var_S))
    conf = _confidence(st, K, cfg)
    suspect, notes = False, ()
    if st.guard is not None:
        suspect = st.guard.evaluate()
        if suspect:
            notes = tuple(st.guard.reasons)
            conf = Confidence(
                level="LOW", score=st.guard.adjust_confidence(conf.score),
                audio_scored_s=conf.audio_scored_s, branches=conf.branches,
                reason="calibration does not apply to this audio: "
                       + notes[0] if notes else conf.reason)

    contributions = {
        b: cfg.weight_of(b) * st.slow[b].mean_rate * T_banked * scale_K
        for b in K
    }

    return RiskView(
        S=S,
        S_lo=S - _Z95 * sd,          # decisions act on the lower bound
        S_fast=S_fast,
        S_slow=S_slow,
        L_rate=L_slow,               # rate-invariant meter for the UI
        confidence=conf,
        subset=K,
        llr_target=st.llr_target,
        enrolled=st.enrolled,
        T_eff=st.T_eff,
        contributions=contributions,
        changed_at=st.last_change_t,
        calibration_suspect=suspect,
        calibration_notes=notes,
    )


def _gates_met(st: FusionState, view: RiskView, tier: int, cfg: FusionConfig) -> bool:
    """A single branch, however confident, may never raise an agent alert."""
    if tier <= 1:
        return True
    return (
        st.T_eff >= cfg.T_min_escalate
        and len(view.subset) >= cfg.min_branches
    )


def apply_hysteresis(
    st: FusionState, view: RiskView, cfg: FusionConfig, policy: TierPolicy
) -> Optional[TierChange]:
    """Four anti-flap mechanisms, each doing a distinct job: the enter/exit gap
    kills boundary chatter, consecutive-crossing kills single-window spikes,
    dwell kills fast oscillation, and one-step-at-a-time de-escalation prevents
    a green flash after a real detection."""
    t = st.t_now
    target = policy.target_tier(view.S_lo)

    # Clamp to the highest tier whose gates are actually satisfied, rather than
    # aborting when the score qualifies for a tier the evidence does not
    # support. Aborting made a single-branch deployment silently unable to alert
    # AT ALL: with only Branch A live, a score deep into tier-3 territory failed
    # the two-branch gate and fell through to no escalation whatsoever, instead
    # of the tier-1 step-up it plainly warranted. Degrading to silence is the
    # one failure mode graceful degradation exists to prevent.
    while target > st.tier and not _gates_met(st, view, target, cfg):
        target -= 1

    if target > st.tier:
        st.consec_cross += 1
        ready = (
            st.consec_cross >= policy.consec_required
            and (
                st.last_escalation_t is None
                or (t - st.last_escalation_t) >= policy.escalation_cooldown
            )
            and view.confidence.score >= cfg.min_confidence_escalate
            and _gates_met(st, view, target, cfg)
        )
        if not ready:
            return None
        st.tier = target
        st.tier_since = t
        st.last_escalation_t = t
        st.consec_cross = 0
        st.tier_latch = max(st.tier_latch, st.tier)
        return TierChange(
            st.call_id, st.tier, "escalate", t, view.S_lo,
            policy.action.get(st.tier, "?"),
        )

    if st.tier > 0:
        exit_thr = policy.exit.get(st.tier, 0.0)
        dwell = policy.dwell.get(st.tier, 0.0)
        if view.S_lo < exit_thr and (t - st.tier_since) >= dwell:
            st.tier -= 1        # one step at a time, never straight to green
            st.tier_since = t
            st.consec_cross = 0
            return TierChange(
                st.call_id, st.tier, "de-escalate", t, view.S_lo,
                policy.action.get(st.tier, "?"),
            )

    st.consec_cross = 0
    return None


def replay(
    st: FusionState,
    events: Iterable[BranchEvent],
    cfg: FusionConfig,
    policy: Optional[TierPolicy] = None,
) -> list[tuple[RiskView, Optional[TierChange]]]:
    """Feed an ordered event stream through the engine. Used by the tests and
    the offline evaluator."""
    policy = policy or TierPolicy()
    out = []
    for ev in events:
        view = update(st, ev, cfg)
        change = apply_hysteresis(st, view, cfg, policy)
        out.append((view, change))
    return out
