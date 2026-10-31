"""Fusion-engine behaviour tests.

These run on synthetic BranchEvent streams -- no audio, no model, no GPU. That
is deliberate: the hardest design work in this system is the fusion layer, and
it should be provable before anything is trained.
"""
from __future__ import annotations

import math

import pytest

from vericall.fusion.config import FusionConfig, TierPolicy
from vericall.fusion.engine import (
    FusionState, apply_hysteresis, render, replay, update,
)
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
    return [ev(branch, start + i * hop, llr, hop=hop, quality=quality)
            for i in range(n)]


# --------------------------------------------------------------------- cold start

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


# --------------------------------------------------------------------- quality gate

def test_silence_keeps_branch_alive_but_contributes_no_evidence():
    cfg = FusionConfig()
    st = FusionState(CALL)
    for e in [ev("A", 0.5 + i * 0.5, 5.0, quality=0.0) for i in range(10)]:
        view = update(st, e, cfg)
    assert st.T_eff == 0.0
    assert view.S == 0.0
    assert "A" in view.subset          # alive, just not contributing


# --------------------------------------------------------------------- branch C

def test_branch_c_never_moves_the_spoof_score():
    """A good clone of an enrolled person RAISES voiceprint similarity. If C fed
    the spoof score, the better the attack the safer the call would look."""
    cfg = FusionConfig()
    st = FusionState(CALL); st.enrolled = True
    for e in fast_stream(0.8, 6):
        update(st, e, cfg)
    before = render(st, cfg)

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


# --------------------------------------------------------------------- staleness

def test_stale_branch_drops_out_of_the_available_set():
    """Uses A2 as the fresh branch, not B: B now carries zero weight by
    measurement (see FusionConfig.w) and is excluded from the subset by
    design, which would make it useless for testing staleness."""
    cfg = FusionConfig()
    st = FusionState(CALL)
    for e in fast_stream(1.0, 4):
        update(st, e, cfg)
    assert "A" in render(st, cfg).subset

    update(st, ev("A2", 12.0, 1.0), cfg)
    view = render(st, cfg)
    assert "A" not in view.subset          # aged past stale_after
    assert "A2" in view.subset


def test_zero_weight_branch_is_excluded_from_the_subset():
    """A zero-weight branch contributing nothing must also not change WHICH
    subset calibration is applied. Otherwise w[B]=0 would still select the
    A+A2+B table, whose scale was fitted while B carried 0.30."""
    cfg = FusionConfig()
    assert cfg.weight_of("B") == 0.0
    st = FusionState(CALL)
    for i in range(6):
        t = 1.0 + i * 0.5
        update(st, ev("A", t, 1.0), cfg)
        update(st, ev("A2", t, 1.0), cfg)
        update(st, ev("B", t, 5.0, hop=1.0), cfg)
    view = render(st, cfg)
    assert "B" not in view.subset
    assert set(view.subset) == {"A", "A2"}
    assert "B" not in view.contributions


def test_zero_weight_branch_cannot_move_the_score():
    cfg = FusionConfig()
    a, b = FusionState(CALL), FusionState(CALL)
    for i in range(6):
        t = 1.0 + i * 0.5
        for st in (a, b):
            update(st, ev("A", t, 1.0), cfg)
        # only b receives Branch B, and it screams
        update(b, ev("B", t, 50.0, hop=1.0), cfg)
    assert render(a, cfg).S == pytest.approx(render(b, cfg).S)


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


# --------------------------------------------------------------------- direction

def test_bona_fide_call_stays_at_tier_zero():
    cfg, pol = FusionConfig(), TierPolicy()
    st = FusionState(CALL)
    for view, _ in replay(st, fast_stream(-1.2, 40), cfg, pol):
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


def test_single_branch_reaches_step_up_but_not_an_agent_alert():
    """A single detector may raise a step-up challenge but never an agent
    alert. An earlier form asserted `tier <= 1`, which passes on tier 0 -- and
    tier 0 was exactly the bug: the score qualified for tier 3, failed the
    two-branch gate, and escalated to nothing at all."""
    cfg, pol = FusionConfig(), TierPolicy()
    st = FusionState(CALL)
    replay(st, fast_stream(6.0, 60), cfg, pol)
    assert len(render(st, cfg).subset) == 1
    assert st.tier == 1, f"expected step-up, got tier {st.tier}"


def test_gates_clamp_the_tier_instead_of_suppressing_the_alert():
    cfg, pol = FusionConfig(), TierPolicy()
    st = FusionState(CALL)
    out = replay(st, fast_stream(8.0, 40), cfg, pol)
    escalations = [c for _, c in out if c and c.direction == "escalate"]
    assert escalations, "no escalation at all despite an overwhelming score"
    assert max(c.tier for c in escalations) == 1


# --------------------------------------------------------------------- mid-call swap

def test_mid_call_swap_triggers_a_change_point():
    cfg, pol = FusionConfig(), TierPolicy()
    st = FusionState(CALL)
    genuine = [ev("A", 1.0 + i * 0.5, -1.5) for i in range(30)]
    swapped = [ev("A", 16.0 + i * 0.5, 3.0) for i in range(20)]
    replay(st, genuine + swapped, cfg, pol)

    assert st.change_events, "no change point declared on a mid-call swap"
    t_change = st.change_events[0]["t"]
    assert 16.0 <= t_change <= 22.0, t_change


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
    assert at_change.confidence.score < conf_before
    assert at_change.T_eff <= 2.0
    assert view.S > 0                    # but the score may rise at once


def test_spoof_from_t_zero_declares_no_change_points():
    """There is no *change* on a call that was synthetic from the first window.
    Without the regime gate the CUSUM re-fires every few seconds, wipes T_eff
    each time, and litters the UI timeline with events that never happened."""
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
    assert st.tier >= 1


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
    """SLO: median time_to_alert <= 3 s, measured from the swap."""
    cfg, pol = FusionConfig(), TierPolicy()
    st = FusionState(CALL)
    events = []
    for i in range(30):
        t = 1.0 + i * 0.5
        events += [ev("A", t, -1.1), ev("A2", t, -0.8)]
    for i in range(30):
        t = 16.0 + i * 0.5
        events += [ev("A", t, 2.4), ev("A2", t, 1.7)]
    events.sort(key=lambda e: e.t_end)

    out = replay(st, events, cfg, pol)
    first = next(c.at for _, c in out if c and c.direction == "escalate")
    assert first - 16.0 <= 3.0, f"time_to_alert {first - 16.0:.1f}s exceeds budget"


# --------------------------------------------------------------------- hysteresis

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
    change = apply_hysteresis(st, render(st, cfg), cfg, pol)
    assert change is not None and change.direction == "de-escalate"
    assert st.tier == 2                          # not straight to 0


def test_tier_latch_never_decreases():
    cfg, pol = FusionConfig(), TierPolicy()
    st = FusionState(CALL)
    st.tier, st.tier_latch, st.tier_since, st.t_now = 2, 2, 0.0, 60.0
    apply_hysteresis(st, render(st, cfg), cfg, pol)
    assert st.tier < 2 and st.tier_latch == 2


# --------------------------------------------------------------------- overlap (E16)

def test_hop_not_window_determines_credited_evidence():
    """Two streams covering the same span with the same evidence density must
    bank the same statistic regardless of how densely they were sampled. The
    naive 'sum every window's LLR' answer would be 40 vs 20."""
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
    """ESS deflation must widen the bound. Applied to the weights instead of
    the variance it is a silent no-op: a weighted mean AND its variance are
    both scale-invariant in the weights."""
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


# --------------------------------------------------------------------- branch B cap

def test_one_class_branch_b_cannot_dominate():
    cfg = FusionConfig()
    st = FusionState("cap")
    for i in range(20):
        update(st, BranchEvent("B", "cap", i, 3.0 + i, 50.0, 0.04, 1.0, 1.0), cfg)
    view = render(st, cfg)
    assert view.S < 7.0, view.S
    assert st.tier == 0


# --------------------------------------------------------------------- subsets

def test_subset_calibration_rescales_a_degraded_configuration():
    base = FusionConfig()
    tuned = FusionConfig(subset_calib={frozenset({"A"}): (0.5, 2.0)})

    def run(cfg):
        st = FusionState(CALL)
        for e in fast_stream(1.0, 12):
            update(st, e, cfg)
        return render(st, cfg)

    plain, scaled = run(base), run(tuned)
    assert plain.subset == ("A",) and scaled.subset == ("A",)
    assert scaled.S != pytest.approx(plain.S)
    assert scaled.S > plain.S


def test_subset_table_roundtrips_through_json(tmp_path):
    import json as _json
    from vericall.fusion.config import load_subset_calib
    p = tmp_path / "subsets.json"
    p.write_text(_json.dumps({"A": [0.1, 1.2], "A+A2": [-0.3, 0.9],
                              "A+A2+B": [0.0, 1.0]}))
    table = load_subset_calib(p)
    assert frozenset({"A", "A2"}) in table
    assert table[frozenset({"A"})] == (0.1, 1.2)
    assert len(table) == 3


def test_unfitted_subset_falls_back_without_crashing():
    cfg = FusionConfig(subset_calib={frozenset({"A", "A2"}): (0.2, 1.5)})
    st = FusionState(CALL)
    for e in fast_stream(1.0, 8):
        view = update(st, e, cfg)
    assert view.subset == ("A",)
    assert math.isfinite(view.S)


# ------------------------------------------------- hostile / degenerate input
#
# Found by probing the engine with values a broken branch can actually emit,
# rather than by hitting them in a run. Each one FAILED OPEN before the fix,
# which is the worst direction for a fraud detector: the call kept streaming,
# the dashboard kept drawing, and nothing could ever alert.

def _ev(**kw):
    d = dict(branch="A", call_id="c", t_start=0.0, t_end=1.0, llr=1.0,
             llr_var=0.04, quality=1.0, hop_s=0.5)
    d.update(kw)
    return BranchEvent(**d)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_llr_cannot_make_a_call_unalertable(bad):
    """The bug this guards: every tier test is `S_lo >= threshold`, and NaN
    compares False against everything. One NaN LLR propagated into S_lo made
    the call permanently unalertable while looking completely healthy."""
    st = FusionState("c")
    view = update(st, _ev(llr=bad), FusionConfig())
    assert math.isfinite(view.S) and math.isfinite(view.S_lo)
    assert math.isfinite(view.L_rate)
    assert st.nonfinite == 1


def test_a_non_finite_event_lowers_confidence_rather_than_vanishing():
    """A dropped event must be visible. Silently discarding it would let the UI
    show confidence built on evidence that never arrived."""
    cfg = FusionConfig()
    clean = FusionState("c")
    for _ in range(4):
        update(clean, _ev(), cfg)
    dirty = FusionState("c")
    for _ in range(4):
        update(dirty, _ev(), cfg)
    for _ in range(6):
        update(dirty, _ev(llr=float("nan")), cfg)
    assert dirty.dropped_chunks >= 6
    assert (update(dirty, _ev(), cfg).confidence.score
            < update(clean, _ev(), cfg).confidence.score)


def test_one_absurd_window_cannot_dominate_the_accumulator():
    """Calibrated LLRs are single-digit nats. 1e12 is a broken branch, not
    strong evidence, and unclamped it swamps every later window."""
    st = FusionState("c")
    view = update(st, _ev(llr=1e12), FusionConfig())
    assert view.S <= 50.0 * 1.01
    assert st.clamped == 1


def test_clamping_preserves_the_sign():
    st = FusionState("c")
    view = update(st, _ev(llr=-1e12), FusionConfig())
    assert view.S < 0, "a huge bona-fide score must not become a spoof score"


@pytest.mark.parametrize("bad", [-0.5, float("nan"), float("inf")])
def test_invalid_variance_is_rejected_at_construction(bad):
    """A negative variance was ACCEPTED and produced S_lo == S -- maximum
    confidence from a broken uncertainty estimate, the opposite of safe."""
    with pytest.raises(ValueError, match="llr_var"):
        _ev(llr_var=bad)


def test_ordinary_events_are_untouched_by_the_guards():
    """The guards must not perturb the normal path."""
    st = FusionState("c")
    view = update(st, _ev(llr=2.5), FusionConfig())
    assert st.nonfinite == 0 and st.clamped == 0
    assert math.isfinite(view.S) and view.S > 0


# ------------------------------------------------------- confidence breadth

def _saturate(cfg, branches, n=40):
    """Run a long clean stream on `branches` so coverage and breadth saturate."""
    st = FusionState(CALL)
    view = None
    for i in range(n):
        for b in branches:
            view = update(st, ev(b, 1.0 + i * 0.5, 2.0), cfg)
    return view


def test_zero_weighted_branch_does_not_deflate_confidence():
    """A branch switched off by the fitted weights is not a missing detector.

    st.available() excludes zero-weight branches on purpose, so such a branch
    can never enter K. _confidence used to divide len(K) by len(SPOOF_BRANCHES)
    anyway, which pinned breadth at 2/3 and confidence at 0.8333 for every call
    this system has ever scored -- the default w has w[B]=0.0. A fully healthy
    call reported 17% less certainty than it had.
    """
    cfg = FusionConfig()                             # default w: B is 0.0
    assert cfg.weight_of("B") == 0.0, "precondition: B is zero-weighted"

    view = _saturate(cfg, ("A", "A2"))
    assert view is not None
    assert set(view.subset) == {"A", "A2"}
    assert view.confidence.score == pytest.approx(1.0), (
        "all weighted detectors reporting on a clean long call must be full "
        f"confidence, got {view.confidence.score}")
    assert view.confidence.level == "HIGH"


def test_confidence_reason_does_not_promise_a_zero_weighted_branch():
    """"B not yet available" was a promise that could never be kept."""
    cfg = FusionConfig()
    view = _saturate(cfg, ("A", "A2"))
    reason = view.confidence.reason
    assert "not yet available" not in reason, reason
    assert "zero-weighted" in reason, reason
    assert "B" in reason, "a disabled branch is still disclosed"


def test_a_genuinely_missing_weighted_branch_still_deflates_confidence():
    """The fix must not make every call look complete."""
    cfg = FusionConfig(w={"A": 1.0, "A2": 0.45, "B": 0.30})
    view = _saturate(cfg, ("A", "A2"))          # B weighted but silent
    assert view.confidence.score == pytest.approx(1.0 * (0.5 + 0.5 * 2 / 3))
    assert "not yet available" in view.confidence.reason
    assert "B" in view.confidence.reason


def test_all_branches_zero_weighted_does_not_divide_by_zero():
    cfg = FusionConfig(w={"A": 0.0, "A2": 0.0, "B": 0.0})
    view = _saturate(cfg, ("A", "A2"), n=10)
    assert view is not None
    assert view.confidence.score == pytest.approx(0.5 * min(1.0, view.confidence.audio_scored_s / 6.0))
