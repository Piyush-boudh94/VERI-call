"""n-of-m aggregation and the independence assumption it rests on."""
from __future__ import annotations

import numpy as np
import pytest

from vericall.risk.aggregation import (
    binomial_at_least, effective_m, evaluate_aggregation, quality_discount, vote,
)


def test_binomial_matches_hand_computation():
    # P(>=1 of 2 at p=0.5) = 3/4
    assert binomial_at_least(1, 2, 0.5) == pytest.approx(0.75)
    # P(>=3 of 5 at p=0.5) = 16/32
    assert binomial_at_least(3, 5, 0.5) == pytest.approx(0.5)
    assert binomial_at_least(5, 5, 0.5) == pytest.approx(1 / 32)


def test_binomial_rejects_nonsense_arguments():
    with pytest.raises(ValueError):
        binomial_at_least(3, 5, 1.5)
    with pytest.raises(ValueError):
        binomial_at_least(6, 5, 0.5)


def test_vote_fires_only_on_a_full_window():
    flags = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    v = vote(flags, n=3, m=5)
    assert not v[:4].any()          # window not yet full
    assert bool(v[4])               # 3 of the first 5


def test_vote_needs_n_within_the_window_not_cumulatively():
    """Three hits spread across eight chunks must not trigger a 3-of-5 vote."""
    flags = np.array([1, 0, 0, 1, 0, 0, 1, 0])
    assert not vote(flags, n=3, m=5).any()


def test_vote_on_too_short_a_stream_is_all_false():
    assert not vote(np.array([1, 1, 1]), n=3, m=5).any()


def test_effective_m_reflects_measured_correlation():
    """At the measured eta = 0.080, five consecutive chunks are worth well
    under one independent observation -- which is why the binomial error model
    cannot be applied to them."""
    assert effective_m(5, 0.080) == pytest.approx(0.4)
    assert effective_m(5, 1.0) == 5.0
    with pytest.raises(ValueError):
        effective_m(5, 0.0)


def test_binomial_is_optimistic_on_correlated_streams():
    """The core finding behind conflict C11. Independent streams should track
    the binomial prediction; correlated streams must exceed it substantially."""
    rng = np.random.default_rng(0)

    independent = [rng.normal(0, 1, 60) for _ in range(150)]
    corr = []
    for _ in range(150):
        offset = rng.normal(0, 1.0)          # per-call level, as a real call has
        corr.append(offset + rng.normal(0, 0.4, 60))

    thr = 1.5
    ind = evaluate_aggregation(independent, thr, 3, 5, eta=1.0)
    dep = evaluate_aggregation(corr, thr, 3, 5, eta=0.08)

    assert ind.inflation < 3.0, ind.line()
    assert dep.inflation > ind.inflation, (ind.line(), dep.line())
    assert dep.empirical_call_fp > dep.predicted_call_fp_binomial


def test_evaluate_aggregation_rejects_streams_shorter_than_the_window():
    with pytest.raises(ValueError):
        evaluate_aggregation([np.zeros(3)], 0.0, 3, 5)


# ------------------------------------------------------------ FM-7

def test_quality_discount_zeroes_below_the_floor():
    assert quality_discount(0.10, 1.0) == 0.0


def test_quality_discount_is_monotone_in_quality_and_duration():
    assert quality_discount(0.9, 1.0) > quality_discount(0.5, 1.0)
    assert quality_discount(0.9, 1.0) > quality_discount(0.9, 0.4)


def test_quality_discount_saturates_at_one():
    assert quality_discount(1.0, 1.0) == pytest.approx(1.0)
    assert quality_discount(1.0, 5.0) == pytest.approx(1.0)


def test_short_chunks_contribute_proportionally_less():
    """FM-7: a 0.3 s chunk must not vote at the same weight as a full window."""
    full = quality_discount(0.8, 1.0)
    short = quality_discount(0.8, 0.3)
    assert short == pytest.approx(full * 0.3, rel=1e-6)
