"""Calibration and metric tests, on synthetic score distributions."""
from __future__ import annotations

import math
import numpy as np
import pytest

from vericall.fusion.calibration import (
    PlattCalibration, bootstrap_eer_ci, cllr, eer, estimate_eta, fit_platt,
    fit_subset_calibration, logit, min_cllr, one_class_to_llr,
)


def synth(n=4000, sep=2.0, seed=0):
    rng = np.random.default_rng(seed)
    non = rng.normal(-sep / 2, 1.0, n)      # bona fide
    tar = rng.normal(+sep / 2, 1.0, n)      # spoof
    return (np.concatenate([non, tar]),
            np.concatenate([np.zeros(n, int), np.ones(n, int)]))


# ------------------------------------------------------------------ EER

def test_eer_of_perfectly_separated_scores_is_zero():
    s = np.array([0., 0., 0., 1., 1., 1.])
    y = np.array([0, 0, 0, 1, 1, 1])
    assert eer(s, y)[0] == pytest.approx(0.0, abs=1e-9)


def test_eer_of_uninformative_scores_is_about_half():
    rng = np.random.default_rng(3)
    s = rng.normal(size=4000)
    y = rng.integers(0, 2, 4000)
    assert eer(s, y)[0] == pytest.approx(0.5, abs=0.05)


def test_eer_matches_closed_form_for_equal_variance_gaussians():
    """For N(+-d/2, 1) the EER is Phi(-d/2)."""
    s, y = synth(n=20000, sep=2.0, seed=1)
    expected = 0.5 * math.erfc((2.0 / 2) / math.sqrt(2))
    assert eer(s, y)[0] == pytest.approx(expected, abs=0.01)


# ------------------------------------------------------------------ Cllr

def test_cllr_of_uninformative_llrs_is_one():
    y = np.concatenate([np.zeros(2000, int), np.ones(2000, int)])
    assert cllr(np.zeros(4000), y) == pytest.approx(1.0, abs=1e-9)


def test_miscalibrated_scores_have_cllr_above_min_cllr():
    s, y = synth(sep=2.0, seed=2)
    bad = s * 12.0                       # correct ranking, wildly overconfident
    assert cllr(bad, y) > min_cllr(bad, y)
    # ranking is untouched, so the floor is unchanged
    assert min_cllr(bad, y) == pytest.approx(min_cllr(s, y), abs=0.02)


def test_platt_calibration_reduces_calibration_loss():
    s, y = synth(sep=2.0, seed=4)
    bad = s * 12.0
    cal = fit_platt(bad, y, branch="A")
    assert cal.calibration_loss < 0.05
    assert cllr(cal.apply(bad), y) < cllr(bad, y)


def test_platt_is_prior_free_under_class_imbalance():
    """ASVspoof LA is ~1:8.8. A prior-weighted fit must not absorb that skew
    into the intercept -- the operational prior is injected at decision time."""
    rng = np.random.default_rng(5)
    non = rng.normal(-1.0, 1.0, 9000)
    tar = rng.normal(+1.0, 1.0, 1000)          # 1:9 imbalance
    s = np.concatenate([non, tar])
    y = np.concatenate([np.zeros(9000, int), np.ones(1000, int)])
    cal = fit_platt(s, y)
    # a score exactly between the class means must map to ~0 nats
    assert cal.apply(0.0) == pytest.approx(0.0, abs=0.35)


def test_calibration_roundtrips_through_json(tmp_path):
    from vericall.fusion.calibration import load_calibration, save_calibration
    cal = PlattCalibration(a=1.5, b=-0.25, branch="A2", channel_bucket="g711_8k")
    p = tmp_path / "a2.json"
    save_calibration(p, cal)
    back = load_calibration(p)
    assert (back.a, back.b, back.branch) == (1.5, -0.25, "A2")


# ------------------------------------------------------------------ branch B

def test_one_class_llr_is_clipped_and_monotone():
    rng = np.random.default_rng(6)
    cohort = rng.normal(0.0, 1.0, 5000)          # bona fide anomaly scores
    raw = np.array([-5.0, 0.0, 2.0, 20.0, 1e6])
    llr = one_class_to_llr(raw, cohort, kappa=2.0, floor=-0.5)
    assert np.all(llr >= -0.5) and np.all(llr <= 2.0)
    assert np.all(np.diff(llr) >= -1e-9)         # monotone non-decreasing
    assert llr[-1] == pytest.approx(2.0)         # saturates, cannot dominate


# ------------------------------------------------------------------ subsets

def test_subset_calibration_covers_every_availability_subset():
    rng = np.random.default_rng(7)
    n = 3000
    y = np.concatenate([np.zeros(n, int), np.ones(n, int)])
    branch_llrs = {
        b: np.concatenate([rng.normal(-m, 1, n), rng.normal(+m, 1, n)])
        for b, m in (("A", 1.2), ("A2", 0.7), ("B", 0.4))
    }
    weights = {"A": 1.0, "A2": 0.45, "B": 0.30}
    subsets = [("A",), ("A", "A2"), ("A", "A2", "B"), ("A", "B"), ("A2",)]
    table = fit_subset_calibration(branch_llrs, y, weights, subsets)

    assert len(table) == 5
    for K, (alpha, scale) in table.items():
        assert math.isfinite(alpha) and math.isfinite(scale)
        assert scale > 0, (K, scale)          # more evidence -> higher score


# ------------------------------------------------------------------ eta

def test_eta_is_one_for_independent_sequences():
    rng = np.random.default_rng(8)
    seqs = [rng.normal(size=200) for _ in range(40)]
    assert estimate_eta(seqs) == pytest.approx(1.0, abs=0.15)


def test_eta_falls_below_one_for_overlapping_windows():
    """Simulate 50%-overlap: each window averages two independent half-hops, so
    consecutive windows share one of them."""
    rng = np.random.default_rng(9)
    seqs = []
    for _ in range(40):
        halves = rng.normal(size=201)
        seqs.append((halves[:-1] + halves[1:]) / 2.0)
    eta = estimate_eta(seqs)
    assert 0.3 < eta < 0.9, eta          # meaningfully deflated, not degenerate


# ------------------------------------------------------------------ CIs

def test_bootstrap_ci_brackets_the_point_estimate():
    s, y = synth(n=1500, sep=2.0, seed=10)
    groups = np.array([f"call{i % 60}" for i in range(s.size)])
    point, lo, hi = bootstrap_eer_ci(s, y, groups, n_boot=200, seed=1)
    assert lo <= point <= hi
    assert hi - lo > 0


def test_call_level_bootstrap_is_wider_than_window_level():
    """Resampling windows pretends correlated windows are independent and
    reports an interval that is too narrow. This is the reporting-side twin of
    the eta correction inside the engine."""
    rng = np.random.default_rng(11)
    scores, labels, groups = [], [], []
    for c in range(50):
        lab = c % 2
        base = rng.normal(1.0 if lab else -1.0, 1.0)   # per-call offset
        for _ in range(30):
            scores.append(base + rng.normal(0, 0.25))  # tight within a call
            labels.append(lab)
            groups.append(f"call{c}")
    scores = np.array(scores); labels = np.array(labels)
    groups = np.array(groups)

    _, lo_c, hi_c = bootstrap_eer_ci(scores, labels, groups, n_boot=300, seed=2)
    windows = np.array([f"w{i}" for i in range(scores.size)])
    _, lo_w, hi_w = bootstrap_eer_ci(scores, labels, windows, n_boot=300, seed=2)
    assert (hi_c - lo_c) > (hi_w - lo_w), ((lo_c, hi_c), (lo_w, hi_w))


def test_eer_handles_ties_correctly():
    """A score with no information must give ~50% EER regardless of ties.

    Regression test: without collapsing tied scores the ROC walks through a run
    of identical values as if they were distinct thresholds. Measured on real
    trimmed-silence durations (nearly all exactly the pad length) the untied
    version reported 95.67% EER for a feature whose class means differed by
    0.8 ms.
    """
    y = np.concatenate([np.zeros(500, int), np.ones(500, int)])
    # a wholly constant score has exactly one threshold and must give exactly 0.5
    assert eer(np.zeros(1000), y)[0] == pytest.approx(0.5, abs=1e-9)

    # Near-constant with negligible jitter. With only two distinct thresholds
    # the empirical EER can sit slightly either side of 0.5 through sampling
    # noise -- that is expected. What must NOT happen is the old behaviour,
    # where walking through the tied run reported ~0.96.
    rng = np.random.default_rng(0)
    s = np.full(1000, 0.03) + rng.choice([0.0, 1e-9], size=1000)
    e, _ = eer(s, y)
    assert 0.4 <= e <= 0.6, e


def test_eer_never_exceeds_one_half_for_uninformative_scores():
    rng = np.random.default_rng(7)
    y = np.concatenate([np.zeros(800, int), np.ones(800, int)])
    for n_levels in (1, 2, 5, 20):
        s = rng.integers(0, n_levels, size=1600).astype(float)
        e, _ = eer(s, y)
        assert e <= 0.62, (n_levels, e)
