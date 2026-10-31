"""Feature-extractor tests. Device-agnostic: run on the head node (CPU) or in
the pod (MIG GPU). Skipped entirely if torch is unavailable."""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from vericall.features.phase import (
    ModifiedGroupDelay, ModifiedRelativePhase, PhaseFeatures, cepstral_smooth,
)
from vericall.features.prosody import (
    F0Config, F0Tracker, IncrementalF0, frame_energy, prosody_statistics,
)
from vericall.features.spectral import LFCC, STFT, STFTConfig, delta

SR = 16_000
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tone(f0: float, seconds: float = 1.0, harmonics: int = 5, device=DEV):
    t = torch.arange(int(seconds * SR), device=device) / SR
    return sum(0.7 ** k * torch.sin(2 * math.pi * f0 * (k + 1) * t)
               for k in range(harmonics))


# ------------------------------------------------------------ spectral

def test_stft_and_lfcc_shapes_and_finiteness():
    cfg = STFTConfig()
    stft, lfcc = STFT(cfg).to(DEV), LFCC(cfg).to(DEV)
    x = torch.randn(4, SR, device=DEV)
    spec = stft(x)
    feats = lfcc(spec)
    assert spec.shape[0] == 4 and spec.shape[1] == cfg.n_freq
    assert feats.shape[1] == 60          # 20 cepstra + delta + delta-delta
    assert torch.isfinite(feats).all()


def test_features_stay_on_the_input_device():
    """The pipeline must never round-trip through the CPU mid-stream -- that is
    what makes cross-call GPU batching worth doing."""
    cfg = STFTConfig()
    lfcc, stft = LFCC(cfg).to(DEV), STFT(cfg).to(DEV)
    out = lfcc(stft(torch.randn(2, SR, device=DEV)))
    assert out.device.type == DEV.type


def test_delta_of_a_linear_ramp_is_constant():
    x = torch.arange(50, dtype=torch.float32).view(1, 1, 50).repeat(1, 3, 1)
    d = delta(x)[:, :, 5:-5]
    assert torch.allclose(d, torch.ones_like(d), atol=1e-4)


def test_lfcc_separates_two_different_spectra():
    cfg = STFTConfig()
    stft, lfcc = STFT(cfg).to(DEV), LFCC(cfg).to(DEV)
    a = lfcc(stft(tone(120.0).unsqueeze(0))).mean(-1)
    b = lfcc(stft(tone(300.0).unsqueeze(0))).mean(-1)
    assert (a - b).abs().max() > 0.5


# ------------------------------------------------------------ phase

def test_modified_relative_phase_is_shift_invariant():
    """Raw STFT phase depends on where the window landed, so it is useless as a
    feature. Normalising against a base bin removes exactly that dependence --
    a shift of one hop must reproduce the next frame identically."""
    cfg = STFTConfig()
    mrp = ModifiedRelativePhase(cfg).to(DEV)
    x = tone(150.0, seconds=1.0).unsqueeze(0)
    a = mrp(x[:, : SR - cfg.hop_length])
    b = mrp(x[:, cfg.hop_length : SR])
    n = min(a.shape[2] - 1, b.shape[2])
    assert (a[:, :, 1 : n + 1] - b[:, :, :n]).abs().mean() < 1e-3

    raw_a = torch.angle(torch.fft.rfft(x[0, :400]))
    raw_b = torch.angle(torch.fft.rfft(x[0, cfg.hop_length : cfg.hop_length + 400]))
    assert (raw_a - raw_b).abs().mean() > 0.5      # raw phase is not invariant


def test_cepstral_smoothing_removes_spectral_detail_but_keeps_the_envelope():
    mag = torch.rand(2, 257, 30, device=DEV) + 0.1
    sm = cepstral_smooth(mag)
    assert sm.shape == mag.shape and torch.isfinite(sm).all()
    assert sm.std(dim=1).mean() < mag.std(dim=1).mean()


def test_group_delay_is_finite_even_at_spectral_zeros():
    """The point of the *modified* group delay: dividing by the smoothed
    envelope instead of the raw magnitude keeps it finite where |X| -> 0."""
    mgd = ModifiedGroupDelay().to(DEV)
    x = torch.zeros(1, SR, device=DEV)
    x[0, ::100] = 1.0                    # comb -> many spectral nulls
    out = mgd.spectrum(x)
    assert torch.isfinite(out).all()


def test_phase_feature_stack_shape():
    pf = PhaseFeatures().to(DEV)
    out = pf(torch.randn(3, SR, device=DEV))
    assert out.shape[0] == 3 and out.shape[1] == 60 + 80
    assert torch.isfinite(out).all()


# ------------------------------------------------------------ prosody

@pytest.mark.parametrize("f0_hz", [110.0, 150.0, 220.0, 300.0])
def test_f0_tracker_is_accurate_on_synthetic_tones(f0_hz):
    tracker = F0Tracker().to(DEV)
    f0, _ = tracker(tone(f0_hz, seconds=1.0).unsqueeze(0))
    voiced = f0[f0 > 0]
    assert voiced.numel() > 0
    assert abs(float(voiced.median()) - f0_hz) / f0_hz < 0.02


def test_f0_threshold_picks_the_local_minimum_not_the_crossing():
    """Regression test. Taking the first sample below the YIN threshold instead
    of the first local minimum lands short of the true lag and biases pitch a
    few percent HIGH -- while voiced/unvoiced decisions still look perfect, so
    it is easy to ship unnoticed."""
    tracker = F0Tracker().to(DEV)
    f0, _ = tracker(tone(110.0, seconds=1.0).unsqueeze(0))
    est = float(f0[f0 > 0].median())
    assert abs(est - 110.0) < 1.5, f"estimated {est:.2f} Hz, expected ~110"
    assert est < 112.0, "systematic upward bias is back"


def test_unvoiced_noise_is_not_reported_as_pitched():
    tracker = F0Tracker().to(DEV)
    f0, _ = tracker(torch.randn(2, SR, device=DEV) * 0.1)
    assert float((f0 > 0).float().mean()) < 0.15


def test_f0_respects_configured_range():
    cfg = F0Config(f_min=80.0, f_max=300.0)
    tracker = F0Tracker(cfg).to(DEV)
    f0, _ = tracker(tone(400.0, seconds=1.0).unsqueeze(0))   # above f_max
    nz = f0[f0 > 0]
    if nz.numel():
        assert float(nz.max()) <= cfg.f_max + 1e-3


def test_incremental_f0_saves_most_of_the_recomputation():
    """3 s window at 1 s hop recomputes two thirds of every pitch track
    otherwise -- roughly a 3x cost on the most expensive slow-lane branch."""
    inc = IncrementalF0(F0Tracker().to(DEV), window_s=3.0, hop_s=1.0)
    steps = 6
    for _ in range(steps):
        f0, _ = inc.push(torch.randn(1, SR, device=DEV) * 0.05)
    naive = steps * inc._frames_per_window
    assert inc.frames_recomputed < naive * 0.45
    assert f0.shape[1] <= inc._frames_per_window


def test_prosody_statistics_are_finite_and_sized():
    tracker = F0Tracker().to(DEV)
    x = torch.stack([tone(120.0), tone(200.0)])
    f0, voiced = tracker(x)
    e = frame_energy(x)[:, : f0.shape[1]]
    stats = prosody_statistics(f0, voiced, e)
    assert stats.shape == (2, 10)
    assert torch.isfinite(stats).all()


def test_prosody_distinguishes_monotone_from_varying_pitch():
    """Branch B keys on contour dynamics, not mean pitch -- a vocoder matches
    average F0 easily and natural micro-variation much less so."""
    tracker = F0Tracker().to(DEV)
    t = torch.arange(2 * SR, device=DEV) / SR
    flat = torch.sin(2 * math.pi * 150.0 * t)
    swept = torch.sin(2 * math.pi * (120.0 + 60.0 * t) * t)
    x = torch.stack([flat, swept])
    f0, voiced = tracker(x)
    e = frame_energy(x)[:, : f0.shape[1]]
    stats = prosody_statistics(f0, voiced, e)
    jitter_flat, jitter_swept = float(stats[0, 3]), float(stats[1, 3])
    assert jitter_swept > jitter_flat
