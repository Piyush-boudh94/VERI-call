"""Fusion-engine behaviour tests.

These run on synthetic BranchEvent streams -- no audio, no model, no GPU. That
is deliberate: the hardest design work in this system is the fusion layer, and
it should be provable before anything is trained.
"""
from __future__ import annotations

import math
import pytest

from vericall.fusion.config import FusionConfig, TierPolicy
from vericall.fusion.engine import FusionState, apply_hysteresis, render, replay, update
from vericall.fusion.types import BranchEvent

CALL = "call-1"


def ev(branch, t_end, llr, *, quality=1.0, hop=None, var=0.04, emb=None):
    hop = hop if hop is not None else (0.5 if branch in ("A", "A2") else 1.0)
    win = 1.0 if branch in ("A", "A2") else 3.0
    return BranchEvent(
        branch=branch, call_id=CALL, t_start=max(0.0, t_end - win), t_end=t_end,
        llr=llr, llr_var=var, quality=quality, hop_s=hop, emb=emb,
    )


def fast_stream(llr, n, *, start=1.0, hop=0.5, branch="A", quality=1.0):
    return [ev(branch, start + i * hop, llr, hop=hop, quality=quality) for i in range(n)]


# --------------------------------------------------------------------- #
# cold start
# --------------------------------------------------------------------- #

def test_cold_start_single_event_is_low_confidence_and_tier_zero():
    cfg, pol = FusionConfig(), TierPolicy()
    st = FusionState(CALL)
    view = update(st, ev("A", 0.5, 3.0), cfg)

    assert view.subset == ("A",)
    assert view.confidence.level == "LOW"
    assert st.tier == 0
    assert apply_hysteresis(st, view, cfg, pol) is None
    # S is the sum of credited LLRs, so one window banks exactly its own LLR --
    # no more. Consecutive-crossing plus the escalation gates are what keep a
    # single confident window from alerting.
    assert view.S == pytest.approx(3.0)
    assert view.S_lo < view.S


def test_cold_start_reports_which_detectors_are_missing():
    cfg = FusionConfig()
    st = FusionState(CALL)
    view = update(st, ev("A", 0.5, 1.0), cfg)
    assert "A2" in view.confidence.reason and "B" in view.confidence.reason


# --------------------------------------------------------------------- #
# quality gate  (E17)
# --------------------------------------------------------------------- #

def test_silence_keeps_branch_alive_but_contributes_no_evidence():
    cfg = FusionConfig()
    st = FusionState(CALL)
    for e in [ev("A", 0.5 + i * 0.5, 5.0, quality=0.0) for i in range(10)]:
        view = update(st, e, cfg)
    assert st.T_eff == 0.0
    assert view.S == 0.0
    assert "A" in view.subset          # alive, just not contributing


# --------------------------------------------------------------------- #
# branch C is an identity signal, never a spoof score  (E4)
# --------------------------------------------------------------------- #

def test_branch_c_never_moves_the_spoof_score():
    cfg = FusionConfig()
    st = FusionState(CALL); st.enrolled = True
    for e in fast_stream(0.8, 6):
        update(st, e, cfg)
    before = render(st, cfg)

    # a *perfect* voiceprint match -- exactly the case a naive additive fusion
    # would treat as reassuring
    update(st, ev("C", 4.0, 9.0, hop=1.0, emb=[1.0, 0.0, 0.0]), cfg)
    after = render(st, cfg)

    assert after.S == pytest.approx(before.S, abs=1e-9)
    assert after.llr_target == 9.0
    assert "C" not in after.subset


def test_speaker_turn_is_detected_without_enrollment():
    cfg = FusionConfig()
    st = FusionState(CALL)                 # not enrolled
    update(st, ev("C", 3.0, 0.0, hop=1.0, emb=[1.0, 0.0, 0.0]), cfg)
    update(st, ev("C", 6.0, 0.0, hop=1.0, emb=[0.0, 1.0, 0.0]), cfg)
    assert any(c["cause"] == "speaker_turn" for c in st.change_events)


# --------------------------------------------------------------------- #
# staleness  (no hold-last-value)
# --------------------------------------------------------------------- #

def test_stale_branch_drops_out_of_the_available_set():
    cfg = FusionConfig()
    st = FusionState(CALL)
    for e in fast_stream(1.0, 4):
        update(st, e, cfg)
    assert "A" in render(st, cfg).subset

    update(st, ev("B", 12.0, 0.0, hop=1.0), cfg)
    view = render(st, cfg)
    assert "A" not in view.subset
    assert "B" in view.subset


def test_stale_branch_resumes_instantly_without_losing_its_accumulator():
    cfg = FusionConfig()
    st = FusionState(CALL)
    for e in fast_stream(1.5, 4):
        update(st, e, cfg)
    n_before = st.slow["A"].n_events
    update(st, ev("B", 12.0, 0.0, hop=1.0), cfg)     # A goes stale
    update(st, ev("A", 12.5, 1.5), cfg)              # A returns
    assert "A" in render(st, cfg).subset
    assert st.slow["A"].n_events == n_before + 1


# --------------------------------------------------------------------- #
# directionality
# --------------------------------------------------------------------- #

def test_bona_fide_call_stays_at_tier_zero():
    cfg, pol = FusionConfig(), TierPolicy()
    st = FusionState(CALL)
    for view, change in replay(st, fast_stream(-1.2, 40), cfg, pol):
        pass
    assert st.tier == 0
    assert view.S_lo < 0


def test_sustained_spoof_escalates():
    cfg, pol = FusionConfig(), TierPolicy()
    st = FusionState(CALL)
    events = []
    for i in range(40):
        t = 1.0 + i * 0.5
        events.append(ev("A", t, 2.2))
        events.append(ev("A2", t, 1.8))
        if i % 2 == 0:
            events.append(ev("B", t, 1.0, hop=1.0))
    events.sort(key=lambda e: e.t_end)
    replay(st, events, cfg, pol)
    assert st.tier >= 1, st.tier
    assert st.tier_latch >= st.tier


def test_single_branch_cannot_reach_an_agent_alert():
    """A single detector, however confident, may never raise tier > 1."""
    cfg, pol = FusionConfig(), TierPolicy()
    st = FusionState(CALL)
    replay(st, fast_stream(6.0, 60), cfg, pol)
    assert len(render(st, cfg).subset) == 1
    assert st.tier <= 1, st.tier


# --------------------------------------------------------------------- #
# mid-call swap
# --------------------------------------------------------------------- #

def test_mid_call_swap_triggers_a_change_point():
    cfg, pol = FusionConfig(), TierPolicy()
    st = FusionState(CALL)
    genuine = [ev("A", 1.0 + i * 0.5, -1.5) for i in range(30)]     # 15 s clean
    swapped = [ev("A", 16.0 + i * 0.5, 3.0) for i in range(20)]
    replay(st, genuine + swapped, cfg, pol)

    assert st.change_events, "no change point declared on a mid-call swap"
    t_change = st.change_events[0]["t"]
    assert 16.0 <= t_change <= 22.0, t_change      # detected within ~6 s


def test_change_point_drops_confidence_while_allowing_score_to_rise():
    cfg = FusionConfig()
    st = FusionState(CALL)
    for e in [ev("A", 1.0 + i * 0.5, -1.0) for i in range(30)]:
        update(st, e, cfg)
    conf_before = render(st, cfg).confidence.score

    at_change = None
    for e in [ev("A", 16.0 + i * 0.5, 3.5) for i in range(12)]:
        view = update(st, e, cfg)
        if st.change_events and at_change is None:
            at_change = view

    assert st.change_events
    # honest: right after a swap we know less than we did before it
    assert at_change.confidence.score < conf_before
    assert at_change.T_eff <= 2.0
    assert view.S > 0                    # but the score may rise at once


# --------------------------------------------------------------------- #
# hysteresis / anti-flap
# --------------------------------------------------------------------- #

def test_score_hovering_at_the_threshold_does_not_flap():
    cfg, pol = FusionConfig(), TierPolicy()
    st = FusionState(CALL)
    events = []
    for i in range(80):
        t = 1.0 + i * 0.5
        events.append(ev("A", t, 2.0 if i % 2 == 0 else -0.4))
        events.append(ev("A2", t, 1.6 if i % 2 == 0 else -0.3))
    events.sort(key=lambda e: e.t_end)
    changes = [c for _, c in replay(st, events, cfg, pol) if c is not None]
    assert len(changes) <= 3, [(c.tier, c.direction, round(c.at, 1)) for c in changes]


def test_de_escalation_is_one_step_at_a_time():
    cfg, pol = FusionConfig(), TierPolicy()
    st = FusionState(CALL)
    st.tier, st.tier_since, st.tier_latch = 3, 0.0, 3
    st.t_now = 60.0
    view = render(st, cfg)
    change = apply_hysteresis(st, view, cfg, pol)
    assert change is not None and change.direction == "de-escalate"
    assert st.tier == 2                          # not straight to 0


def test_tier_latch_never_decreases():
    cfg, pol = FusionConfig(), TierPolicy()
    st = FusionState(CALL)
    st.tier, st.tier_latch, st.tier_since, st.t_now = 2, 2, 0.0, 60.0
    apply_hysteresis(st, render(st, cfg), cfg, pol)
    assert st.tier < 2 and st.tier_latch == 2


# --------------------------------------------------------------------- #
# overlap accounting  (E16 / E20)
# --------------------------------------------------------------------- #

def test_hop_not_window_determines_credited_evidence():
    """Two streams covering the same span with the same evidence density must
    bank the same statistic, regardless of how densely they sampled it.

    Normalising the LLR by the hop instead of the window would cancel exactly
    and leave S as the raw sum of every window's LLR -- 40.0 vs 20.0 here, i.e.
    no overlap correction at all.
    """
    cfg = FusionConfig()
    a = FusionState("a"); b = FusionState("b")
    for i in range(20):                              # 0.5 s hop, 10 s span
        update(a, BranchEvent("A", "a", i * 0.5, 1.0 + i * 0.5, 2.0, 0.04, 1.0, 0.5), cfg)
    for i in range(10):                              # 1.0 s hop, same span
        update(b, BranchEvent("A", "b", i * 1.0, 1.0 + i * 1.0, 2.0, 0.04, 1.0, 1.0), cfg)

    assert a.T_eff == pytest.approx(10.0)
    assert b.T_eff == pytest.approx(10.0)
    va, vb = render(a, cfg), render(b, cfg)
    assert va.S == pytest.approx(vb.S, rel=0.05), (va.S, vb.S)
    assert va.S < 25.0


def test_eta_deflation_reduces_banked_confidence():
    """ESS deflation must actually widen the bound. Applied to the weights it
    would be a silent no-op: a weighted mean and its variance are both
    scale-invariant in the weights."""
    strict = FusionConfig(eta={"A": 0.6, "A2": 0.6, "B": 0.9})
    naive = FusionConfig(eta={"A": 1.0, "A2": 1.0, "B": 1.0})
    outs = []
    for cfg in (strict, naive):
        st = FusionState(CALL)
        for e in fast_stream(2.0, 20):
            update(st, e, cfg)
        outs.append(render(st, cfg))
    assert outs[0].S == pytest.approx(outs[1].S, rel=0.05)
    assert (outs[0].S - outs[0].S_lo) > (outs[1].S - outs[1].S_lo)


# --------------------------------------------------------------------- #
# branch B containment
# --------------------------------------------------------------------- #

def test_one_class_branch_b_cannot_dominate():
    cfg = FusionConfig()
    st = FusionState("cap")
    for i in range(20):
        update(st, BranchEvent("B", "cap", i, 3.0 + i, 50.0, 0.04, 1.0, 1.0), cfg)
    view = render(st, cfg)
    assert view.S < 7.0, view.S
    assert st.tier == 0


# --------------------------------------------------------------------- #
# change-point suppression  (E21)
# --------------------------------------------------------------------- #

def test_spoof_from_t_zero_declares_no_change_points():
    """There is no *change* on a call synthetic from the first window -- the
    first evidence IS the baseline. Without the regime gate the CUSUM re-fires
    every few seconds, wipes T_eff each time, and litters the UI timeline with
    speaker changes that never happened."""
    cfg, pol = FusionConfig(), TierPolicy()
    st = FusionState(CALL)
    events = []
    for i in range(60):
        t = 1.0 + i * 0.5
        events.append(ev("A", t, 2.4))
        events.append(ev("A2", t, 1.7))
    events.sort(key=lambda e: e.t_end)
    replay(st, events, cfg, pol)

    assert st.change_events == [], st.change_events
    assert st.tier >= 1                      # it still alerts, promptly


def test_regime_survives_a_change_point():
    cfg = FusionConfig()
    st = FusionState(CALL)
    for e in [ev("A", 1.0 + i * 0.5, -1.0) for i in range(30)]:
        update(st, e, cfg)
    for e in [ev("A", 16.0 + i * 0.5, 3.5) for i in range(6)]:
        update(st, e, cfg)
    assert st.change_events
    assert st.regime.n_events == 36
    assert st.slow["A"].n_events < 36


def test_time_to_alert_after_a_mid_call_swap_is_within_budget():
    """SLO: median time_to_alert <= 3 s, measured from the swap not from t=0.

    Regression guard for the cooldown bug: gating the escalation cooldown on
    tier_since (which initialises to 0.0) made an alert impossible during the
    first 10 s of every call.
    """
    cfg, pol = FusionConfig(), TierPolicy()
    st = FusionState(CALL)
    events = []
    for i in range(30):                      # 15 s genuine
        t = 1.0 + i * 0.5
        events += [ev("A", t, -1.1), ev("A2", t, -0.8)]
    for i in range(30):                      # then a clone
        t = 16.0 + i * 0.5
        events += [ev("A", t, 2.4), ev("A2", t, 1.7)]
    events.sort(key=lambda e: e.t_end)

    out = replay(st, events, cfg, pol)
    first = next(c.at for _, c in out if c and c.direction == "escalate")
    assert first - 16.0 <= 3.0, f"time_to_alert {first - 16.0:.1f}s exceeds budget"
