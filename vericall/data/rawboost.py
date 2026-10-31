"""RawBoost: data augmentation designed for telephony anti-spoofing.

Tak, Kamble, Patino, Todisco & Evans, ICASSP 2022 (arXiv 2111.04433). Specified
in docs/PRODUCTION_READINESS.md and never implemented until now.

Why this one rather than generic audio augmentation: RawBoost needs **no
external noise corpus and no impulse-response database**. Everything is
synthesised from the signal itself, which matters here because the failure this
project keeps hitting is corpus-specific shortcut learning -- and an
augmentation that mixes in a fixed external noise set just adds one more
corpus-specific artifact to learn.

Three algorithms from the paper:

  1. **Linear and non-linear convolutive noise.** A randomly-parameterised FIR
     filter plus its harmonic distortions. Stands in for the frequency response
     of a handset, a codec, a room -- the things that differ between corpora.
  2. **Impulsive signal-dependent additive noise.** Sparse impulses scaled to
     the local signal level, i.e. clicks and crackle that scale with the audio
     rather than sitting at a fixed SNR.
  3. **Stationary signal-independent additive noise.** Coloured noise at a
     random SNR, with the colour itself randomised.

The paper's series configuration (1+2+3 applied in sequence) is the one that
performed best on ASVspoof 2021 LA, and is the default here.

**Applied to the waveform at training time only.** Never at inference: the
serving path must see the audio the caller actually sent, and an augmentation
in the scoring path would change the distribution the calibration was fitted
against.
"""
from __future__ import annotations

import numpy as np
import torch

SR = 16_000


def _rand(rng, lo, hi):
    return float(rng.uniform(lo, hi))


def linear_nonlinear_convolutive(x: torch.Tensor, rng, n_taps: int = 10,
                                 max_order: int = 4) -> torch.Tensor:
    """Algorithm 1: random FIR plus harmonic distortion.

    A random filter, not a measured one. The point is to make the model stop
    relying on ANY particular frequency response, not to imitate one device.
    """
    y = torch.zeros_like(x)
    order = rng.integers(1, max_order + 1)
    for p in range(1, order + 1):
        b = torch.from_numpy(rng.standard_normal(n_taps).astype(np.float32))
        b = b / (b.abs().max() + 1e-9)
        # depth falls off with harmonic order, so the fundamental dominates
        gain = 10.0 ** (-_rand(rng, 0.0, 20.0) * (p - 1) / 20.0)
        xp = torch.sign(x) * (x.abs() ** p)
        filt = torch.nn.functional.conv1d(
            xp.view(1, 1, -1),
            b.view(1, 1, -1).to(x.device),
            padding=n_taps // 2)[0, 0, : x.numel()]
        y = y + gain * filt
    peak = y.abs().max()
    return y / peak * x.abs().max() if peak > 1e-9 else x


def impulsive_additive(x: torch.Tensor, rng, max_pct: float = 0.05,
                       snr_lo: float = 5.0, snr_hi: float = 20.0) -> torch.Tensor:
    """Algorithm 2: sparse impulses scaled to the LOCAL signal level.

    Signal-dependent on purpose -- clicks on a real line are louder in loud
    passages, and a fixed-amplitude impulse is a cue a model can simply learn.
    """
    n = x.numel()
    k = max(1, int(n * _rand(rng, 0.0, max_pct)))
    idx = torch.from_numpy(rng.choice(n, size=k, replace=False))
    snr = _rand(rng, snr_lo, snr_hi)
    scale = (x.abs().mean() * 10.0 ** (-snr / 20.0))
    imp = torch.zeros_like(x)
    signs = torch.from_numpy(rng.choice([-1.0, 1.0], size=k).astype(np.float32))
    imp[idx] = signs.to(x.device) * scale * torch.from_numpy(
        rng.uniform(0.5, 1.5, size=k).astype(np.float32)).to(x.device)
    return x + imp


def stationary_additive(x: torch.Tensor, rng, snr_lo: float = 10.0,
                        snr_hi: float = 40.0) -> torch.Tensor:
    """Algorithm 3: coloured noise at a random SNR, colour also randomised."""
    n = x.numel()
    white = torch.from_numpy(rng.standard_normal(n).astype(np.float32)).to(x.device)
    # Shape it in the frequency domain: f^(-alpha) with alpha random, so the
    # model cannot key on one noise colour either.
    spec = torch.fft.rfft(white)
    freqs = torch.fft.rfftfreq(n, 1.0 / SR).to(x.device).clamp(min=1.0)
    alpha = _rand(rng, -1.0, 1.0)
    spec = spec * freqs.pow(-alpha)
    noise = torch.fft.irfft(spec, n=n)
    npow = noise.pow(2).mean().clamp(min=1e-12)
    spow = x.pow(2).mean().clamp(min=1e-12)
    snr = _rand(rng, snr_lo, snr_hi)
    noise = noise * (spow / npow / (10.0 ** (snr / 10.0))).sqrt()
    return x + noise


def rawboost(x: torch.Tensor, rng=None, algo: str = "series") -> torch.Tensor:
    """Apply RawBoost. `algo`: series (1+2+3), 1, 2, 3, or 'parallel'.

    Returns a peak-normalised waveform, because every stage can change the
    amplitude and a level shift is itself a cue a detector could learn.
    """
    rng = rng or np.random.default_rng()
    x = x.reshape(-1).float()
    if algo == "series":
        y = stationary_additive(
            impulsive_additive(linear_nonlinear_convolutive(x, rng), rng), rng)
    elif algo == "parallel":
        y = linear_nonlinear_convolutive(x, rng) + impulsive_additive(x, rng)
    elif algo == "1":
        y = linear_nonlinear_convolutive(x, rng)
    elif algo == "2":
        y = impulsive_additive(x, rng)
    elif algo == "3":
        y = stationary_additive(x, rng)
    else:
        return x
    peak = y.abs().max()
    return (y / peak).clamp(-1.0, 1.0) if peak > 1e-9 else x
