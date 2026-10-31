"""Domain guard: does it fire on the failure that actually happened?

A guard justified by reasoning alone is decoration. These tests replay the
RTCFake signature -- Branch A inverted while A2 held -- and the in-domain case,
and require the guard to distinguish them.
"""
from __future__ import annotations

import pytest

from vericall.fusion.domain_guard import (
    BranchReference, DomainGuard, DomainGuardConfig, reference_from_summary,
)


def in_domain_guard() -> DomainGuard:
    # ASVspoof-like: both branches negative on bona fide, correlated
    return DomainGuard(reference={
        "A": BranchReference("A", -2.0, 2.0),
        "A2": BranchReference("A2", -1.5, 2.0),
    })


def test_quiet_when_scores_match_the_calibration():
    g = in_domain_guard()
    for _ in range(12):
        g.observe("A", -2.2)
        g.observe("A2", -1.4)
    assert not g.evaluate()
    assert g.to_json()["calibration_suspect"] is False


def test_does_not_judge_a_call_too_early():
    """One window out of distribution is not a corpus change."""
    g = in_domain_guard()
    g.observe("A", +40.0)
    assert not g.evaluate()
    assert g.windows < g.cfg.min_windows


def test_fires_on_score_drift():
    g = in_domain_guard()
    for _ in range(12):
        g.observe("A", +25.0)       # far from mean -2.0, sd 2.0
        g.observe("A2", +24.0)
    assert g.evaluate()
    assert any("extrapolating" in r for r in g.reasons)


def test_fires_on_the_rtcfake_inversion_signature():
    """The case that motivated this module: A inverted (scoring bona fide as
    spoof) while A2 retained correct sign. On RTCFake this was A at 79% EER
    alongside A2 at 28.72%."""
    g = in_domain_guard()
    for _ in range(12):
        g.observe("A", +6.0)        # inverted: positive on bona fide audio
        g.observe("A2", -5.0)       # still correct
    assert g.evaluate()
    assert any("inverting" in r for r in g.reasons)


def test_same_sign_disagreement_is_not_flagged_as_inversion():
    """Branches differing in magnitude is ordinary; differing in SIGN over a
    whole call is the inversion signature. Only the latter should trip it."""
    g = DomainGuard()               # no reference, so drift cannot fire
    for _ in range(12):
        g.observe("A", 7.0)
        g.observe("A2", 1.0)        # 6 nats apart, same sign
    assert not g.evaluate()


def test_trip_latches():
    """Later agreement does not restore trust in evidence already banked."""
    g = in_domain_guard()
    for _ in range(12):
        g.observe("A", +6.0)
        g.observe("A2", -5.0)
    assert g.evaluate()
    for _ in range(40):
        g.observe("A", -2.0)
        g.observe("A2", -1.5)
    assert g.evaluate()             # still suspect


def test_confidence_is_penalised_only_when_tripped():
    g = in_domain_guard()
    assert g.adjust_confidence(0.8) == pytest.approx(0.8)
    for _ in range(12):
        g.observe("A", +6.0)
        g.observe("A2", -5.0)
    g.evaluate()
    assert g.adjust_confidence(0.8) == pytest.approx(0.4)


def test_reference_loader_refuses_to_invent_numbers():
    """A guard with a fabricated reference fires for reasons unrelated to the
    data, which is worse than no guard."""
    assert reference_from_summary({}) == {}
    assert reference_from_summary({"score_reference": {"A": {"mean": 1.0}}}) == {}
    good = reference_from_summary(
        {"score_reference": {"A": {"mean": -2.0, "sd": 1.5}}})
    assert good["A"].sd == 1.5


def test_json_is_reportable():
    g = in_domain_guard()
    for _ in range(12):
        g.observe("A", -2.0)
        g.observe("A2", -1.5)
    g.evaluate()
    j = g.to_json()
    assert j["windows"] == 12
    assert "A" in j["branch_mean_llr"]


# --------------------------------------------------------------------- #
# integration: the guard must actually suppress confidence in the engine
# --------------------------------------------------------------------- #

def test_engine_reports_and_penalises_a_suspect_calibration():
    """Wiring test. A guard that detects the problem but does not change the
    engine's output is decoration -- the RTCFake failure was precisely a
    confident number being reported while the calibration did not apply."""
    from vericall.fusion.config import FusionConfig
    from vericall.fusion.engine import FusionState, render, update
    from vericall.fusion.types import BranchEvent

    cfg = FusionConfig()
    ref = {"A": BranchReference("A", -5.0, 1.2),
           "A2": BranchReference("A2", -7.0, 2.8)}

    def run(with_guard: bool, a_llr: float, a2_llr: float):
        st = FusionState("c")
        if with_guard:
            st.guard = DomainGuard(reference=dict(ref))
        for i in range(14):
            t = 1.0 + i * 0.5
            update(st, BranchEvent("A", "c", t - 1, t, a_llr, 0.04, 1.0, 0.5), cfg)
            update(st, BranchEvent("A2", "c", t - 1, t, a2_llr, 0.04, 1.0, 0.5), cfg)
        return render(st, cfg)

    # in-domain scores: guard stays quiet, confidence untouched
    ok = run(True, -5.1, -7.2)
    assert not ok.calibration_suspect
    assert ok.calibration_notes == ()

    # the RTCFake zh signature: A inverted, far outside its distribution
    bad = run(True, +3.1, +0.2)
    assert bad.calibration_suspect
    assert bad.calibration_notes
    assert bad.confidence.level == "LOW"

    # and it genuinely lowers confidence rather than only labelling it
    unguarded = run(False, +3.1, +0.2)
    assert bad.confidence.score < unguarded.confidence.score


def test_no_guard_means_no_behaviour_change():
    """A state without a reference distribution must behave exactly as before;
    the guard is opt-in, not a silent global change."""
    from vericall.fusion.config import FusionConfig
    from vericall.fusion.engine import FusionState, render, update
    from vericall.fusion.types import BranchEvent

    cfg = FusionConfig()
    st = FusionState("c")
    for i in range(10):
        t = 1.0 + i * 0.5
        update(st, BranchEvent("A", "c", t - 1, t, 2.0, 0.04, 1.0, 0.5), cfg)
    v = render(st, cfg)
    assert st.guard is None
    assert not v.calibration_suspect
