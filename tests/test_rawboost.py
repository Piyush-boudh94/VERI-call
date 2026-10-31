"""RawBoost augmentation (Tak et al., ICASSP 2022).

Each test pins a property the augmentation must have to be useful rather than
merely noisy. The failure these guard against is subtle: an augmentation that
is broken still trains, still converges, and just does not help -- and "did not
help" is indistinguishable from "this idea does not work here" unless the
implementation is known good.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vericall.data.rawboost import (
    impulsive_additive, linear_nonlinear_convolutive, rawboost,
    stationary_additive,
)

SR = 16_000


def speech_like(seed=0):
    rng = np.random.default_rng(seed)
    t = torch.arange(SR, dtype=torch.float32) / SR
    x = 0.4 * torch.sin(2 * np.pi * 180 * t) + 0.2 * torch.sin(2 * np.pi * 900 * t)
    for k in range(0, SR, 1600):          # pauses, like speech
        x[k:k + 400] *= 0.02
    return x + 0.01 * torch.from_numpy(rng.standard_normal(SR).astype(np.float32))


@pytest.mark.parametrize("algo", ["1", "2", "3", "series", "parallel"])
def test_output_is_finite_and_bounded(algo):
    y = rawboost(speech_like(), np.random.default_rng(0), algo)
    assert torch.isfinite(y).all()
    assert y.abs().max() <= 1.0 + 1e-6, "must not clip the DAC"
    assert y.shape == (SR,)


@pytest.mark.parametrize("algo", ["1", "2", "3", "series"])
def test_augmentation_actually_changes_the_signal(algo):
    """A no-op augmentation is the quiet failure: training still runs."""
    x = speech_like()
    y = rawboost(x, np.random.default_rng(0), algo)
    assert not torch.allclose(x, y, atol=1e-4)


def test_unknown_algo_is_a_passthrough_not_a_crash():
    x = speech_like()
    assert torch.allclose(rawboost(x, np.random.default_rng(0), "nope"), x)


def test_it_is_random_across_calls():
    """Same audio twice must not give the same augmentation, or every epoch
    sees one fixed variant and the diversity is imaginary."""
    x = speech_like()
    rng = np.random.default_rng(0)
    a, b = rawboost(x, rng), rawboost(x, rng)
    assert not torch.allclose(a, b, atol=1e-4)


def test_it_is_reproducible_from_a_seed():
    x = speech_like()
    a = rawboost(x, np.random.default_rng(7), "series")
    b = rawboost(x, np.random.default_rng(7), "series")
    assert torch.allclose(a, b)


def test_impulsive_noise_is_sparse_not_broadband():
    """Algorithm 2 is clicks, not hiss. If it filled every sample it would be
    algorithm 3 with extra steps.

    Measured as a DIFFERENCE from the input rather than on silence: the noise
    is signal-dependent, so on an all-zero signal it correctly adds nothing and
    the test would be measuring that instead.
    """
    x = speech_like()
    y = impulsive_additive(x, np.random.default_rng(0))
    touched = ((y - x).abs() > 1e-9).float().mean().item()
    assert 0.0 < touched < 0.10, f"{touched:.3f} of samples touched"


def test_impulsive_noise_adds_nothing_to_silence():
    """The corollary, stated as its own fact: signal-dependent means a silent
    frame stays silent, which is right -- a click scaled to zero signal is zero."""
    y = impulsive_additive(torch.zeros(SR), np.random.default_rng(0))
    assert torch.allclose(y, torch.zeros(SR))


def test_impulsive_noise_scales_with_the_signal():
    """Signal-DEPENDENT is the point: a fixed-amplitude impulse is a constant
    cue a model can simply memorise."""
    quiet = impulsive_additive(speech_like() * 0.01, np.random.default_rng(3))
    loud = impulsive_additive(speech_like() * 1.00, np.random.default_rng(3))
    assert loud.abs().max() > quiet.abs().max() * 5


def test_stationary_noise_respects_its_snr_range():
    x = speech_like()
    y = stationary_additive(x, np.random.default_rng(0), snr_lo=30.0, snr_hi=30.0)
    noise = y - x
    snr = 10 * torch.log10(x.pow(2).mean() / noise.pow(2).mean().clamp(min=1e-12))
    assert 25.0 < float(snr) < 35.0, f"realised SNR {float(snr):.1f} dB"


def test_convolutive_filter_preserves_length_and_scale():
    x = speech_like()
    y = linear_nonlinear_convolutive(x, np.random.default_rng(0))
    assert y.shape == x.shape
    # rescaled to the input's peak, so the stage cannot smuggle in a level cue
    assert float(y.abs().max()) == pytest.approx(float(x.abs().max()), rel=0.2)


def test_series_is_not_dominated_by_one_stage():
    """All three should contribute. If the series output matched any single
    stage the composition would be broken."""
    x = speech_like()
    ser = rawboost(x, np.random.default_rng(1), "series")
    for algo in ("1", "2", "3"):
        one = rawboost(x, np.random.default_rng(1), algo)
        assert not torch.allclose(ser, one, atol=1e-3)
