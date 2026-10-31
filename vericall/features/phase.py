"""Phase-domain features for Branch A2, batched on the GPU.

Why a separate branch at all: TTS and voice-conversion vocoders reconstruct a
waveform from a magnitude representation, so original phase structure is largely
destroyed and re-synthesised. SSL-based detectors (Branch A) key mostly on
magnitude-spectrum artifacts, which makes phase a genuinely complementary
signal rather than a redundant one -- and one that degrades differently under
codec compression, which is the point for telephony.

(The v3 blueprint quoted a 0.002% EER from ASVspoof 2015 as evidence here. That
benchmark is saturated and retired; treat phase as complementary information,
not as a source of headline numbers. See docs/CONFLICTS_RESOLVED.md E3.)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .spectral import STFTConfig, _dct_matrix, delta


def frame_signal(
    wav: torch.Tensor, cfg: STFTConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (X, Y): the windowed DFT and the DFT of the time-ramped frame.

    Group delay needs both, computed over *identical* frames, so we frame once
    here rather than calling torch.stft twice with different inputs.
    """
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    win = torch.hann_window(cfg.win_length, device=wav.device, dtype=wav.dtype)
    frames = wav.unfold(-1, cfg.win_length, cfg.hop_length)   # (B, T, win)
    windowed = frames * win
    ramp = torch.arange(cfg.win_length, device=wav.device, dtype=wav.dtype)

    X = torch.fft.rfft(windowed, n=cfg.n_fft, dim=-1)
    Y = torch.fft.rfft(windowed * ramp, n=cfg.n_fft, dim=-1)
    return X.transpose(1, 2), Y.transpose(1, 2)               # (B, F, T)


def cepstral_smooth(mag: torch.Tensor, lifter: int = 30) -> torch.Tensor:
    """Low-quefrency-liftered magnitude envelope.

    This is the 'modified' in modified group delay: dividing by the smoothed
    envelope instead of the raw magnitude suppresses the spurious spikes that
    appear wherever the magnitude spectrum approaches a zero, which is what made
    plain group delay unusable in practice.
    """
    log_mag = torch.log(mag + 1e-10)
    ceps = torch.fft.irfft(log_mag, dim=1)
    mask = torch.zeros(ceps.shape[1], device=ceps.device, dtype=ceps.dtype)
    mask[:lifter] = 1.0
    mask[-lifter:] = 1.0
    smoothed = torch.fft.rfft(ceps * mask[None, :, None], dim=1)
    return torch.exp(smoothed.real.clamp(-20.0, 20.0))


@dataclass(frozen=True)
class MGDConfig:
    alpha: float = 0.4        # compression of the group-delay magnitude
    gamma: float = 0.9        # strength of the envelope normalisation
    lifter: int = 30
    n_ceps: int = 20


class ModifiedGroupDelay(torch.nn.Module):
    """MGD spectrum and its cepstral coefficients (MGDCC)."""

    def __init__(self, cfg: STFTConfig = STFTConfig(), mgd: MGDConfig = MGDConfig()):
        super().__init__()
        self.cfg, self.mgd = cfg, mgd
        self.register_buffer(
            "dct", _dct_matrix(mgd.n_ceps, cfg.n_freq, "cpu", torch.float32),
            persistent=False,
        )

    def spectrum(self, wav: torch.Tensor) -> torch.Tensor:
        X, Y = frame_signal(wav, self.cfg)
        S = cepstral_smooth(X.abs(), self.mgd.lifter)
        tau = (X.real * Y.real + X.imag * Y.imag) / (
            S.pow(2.0 * self.mgd.gamma) + 1e-10
        )
        return torch.sign(tau) * tau.abs().pow(self.mgd.alpha)

    def forward(self, wav: torch.Tensor, deltas: bool = True) -> torch.Tensor:
        mgd = self.spectrum(wav)
        ceps = torch.matmul(self.dct, mgd)              # (B, n_ceps, T)
        if not deltas:
            return ceps
        d1 = delta(ceps)
        return torch.cat([ceps, d1, delta(d1)], dim=1)


class ModifiedRelativePhase(torch.nn.Module):
    """Phase normalised against a reference frequency bin.

    Raw STFT phase is useless as a feature: it depends on where the analysis
    window happened to land relative to the waveform, so it is effectively
    random across frames. Normalising every bin's phase against a base
    frequency removes that dependence and leaves the *structure*, which is what
    a vocoder fails to reproduce. Returned as (cos, sin) so the wrap at +-pi
    does not create a spurious discontinuity for the CNN.
    """

    def __init__(self, cfg: STFTConfig = STFTConfig(), base_hz: float = 1000.0):
        super().__init__()
        self.cfg = cfg
        self.base_bin = max(1, int(round(base_hz / (cfg.sample_rate / cfg.n_fft))))

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        X, _ = frame_signal(wav, self.cfg)
        phase = torch.angle(X)                                   # (B, F, T)
        base = phase[:, self.base_bin : self.base_bin + 1, :]     # (B, 1, T)
        k = torch.arange(X.shape[1], device=X.device, dtype=phase.dtype)
        ratio = (k / float(self.base_bin)).view(1, -1, 1)
        rel = phase - ratio * base
        rel = torch.remainder(rel + math.pi, 2 * math.pi) - math.pi
        return torch.cat([torch.cos(rel), torch.sin(rel)], dim=1)


class PhaseFeatures(torch.nn.Module):
    """MGDCC + MRP, concatenated -- the full Branch A2 front end."""

    def __init__(
        self,
        cfg: STFTConfig = STFTConfig(),
        mgd: MGDConfig = MGDConfig(),
        mrp_bins: int = 40,
    ):
        super().__init__()
        self.mgdcc = ModifiedGroupDelay(cfg, mgd)
        self.mrp = ModifiedRelativePhase(cfg)
        self.mrp_bins = mrp_bins       # keep low bins only; upper bins are noise

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        g = self.mgdcc(wav)
        p = self.mrp(wav)
        n = p.shape[1] // 2
        p = torch.cat([p[:, : self.mrp_bins], p[:, n : n + self.mrp_bins]], dim=1)
        return torch.cat([g, p], dim=1)
