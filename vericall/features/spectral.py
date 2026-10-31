"""Magnitude-domain features, batched on the GPU.

All extractors take (B, T) float32 waveforms on any device and return (B, C, F)
feature maps on the SAME device. Nothing here calls .cpu(), .numpy(), or librosa
-- keeping the whole pipeline resident on the accelerator is what makes
cross-call batching worthwhile, since feature extraction would otherwise
dominate the per-call CPU budget.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class STFTConfig:
    n_fft: int = 512
    hop_length: int = 160          # 10 ms at 16 kHz
    win_length: int = 400          # 25 ms
    center: bool = False           # causal: never look past the chunk edge
    sample_rate: int = SAMPLE_RATE

    @property
    def n_freq(self) -> int:
        return self.n_fft // 2 + 1


class STFT(torch.nn.Module):
    """Cached-window STFT. The window is a registered buffer so it moves with
    .to(device) and is never reallocated per call."""

    def __init__(self, cfg: STFTConfig = STFTConfig()):
        super().__init__()
        self.cfg = cfg
        self.register_buffer(
            "window", torch.hann_window(cfg.win_length), persistent=False
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        return torch.stft(
            wav,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            win_length=self.cfg.win_length,
            window=self.window,
            center=self.cfg.center,
            return_complex=True,
        )                                     # (B, n_freq, n_frames)


def _hz_to_mel(f: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + f / 700.0)


def _mel_to_hz(m: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _triangular_bank(
    n_filters: int, n_freq: int, sr: int, f_min: float, f_max: float,
    mel: bool, device, dtype,
) -> torch.Tensor:
    """Shared triangular-filterbank construction for both mel and linear
    spacing -- LFCC differs from MFCC only in that spacing."""
    all_f = torch.linspace(0.0, sr / 2.0, n_freq, device=device, dtype=dtype)
    if mel:
        pts = torch.linspace(
            float(_hz_to_mel(torch.tensor(f_min))),
            float(_hz_to_mel(torch.tensor(f_max))),
            n_filters + 2, device=device, dtype=dtype,
        )
        pts = _mel_to_hz(pts)
    else:
        pts = torch.linspace(f_min, f_max, n_filters + 2, device=device, dtype=dtype)

    lower, centre, upper = pts[:-2], pts[1:-1], pts[2:]
    up = (all_f[None, :] - lower[:, None]) / (centre - lower)[:, None]
    down = (upper[:, None] - all_f[None, :]) / (upper - centre)[:, None]
    return torch.clamp(torch.minimum(up, down), min=0.0)      # (n_filters, n_freq)


def _dct_matrix(n_out: int, n_in: int, device, dtype) -> torch.Tensor:
    """Orthonormal DCT-II."""
    n = torch.arange(n_in, device=device, dtype=dtype)
    k = torch.arange(n_out, device=device, dtype=dtype)
    m = torch.cos(math.pi / n_in * (n[None, :] + 0.5) * k[:, None])
    m[0] *= math.sqrt(0.5)
    return m * math.sqrt(2.0 / n_in)


class LFCC(torch.nn.Module):
    """Linear-frequency cepstral coefficients with delta and delta-delta.

    LFCC rather than MFCC on purpose: mel spacing compresses the high-frequency
    region, and that is exactly where vocoder and TTS artifacts concentrate.
    The anti-spoofing literature has favoured linear spacing for this reason.
    """

    def __init__(
        self,
        cfg: STFTConfig = STFTConfig(),
        n_filters: int = 70,
        n_ceps: int = 20,
        f_min: float = 0.0,
        f_max: float | None = None,
        deltas: bool = True,
        mel: bool = False,
    ):
        super().__init__()
        self.cfg, self.n_ceps, self.deltas = cfg, n_ceps, deltas
        f_max = f_max if f_max is not None else cfg.sample_rate / 2.0
        self.register_buffer(
            "bank",
            _triangular_bank(n_filters, cfg.n_freq, cfg.sample_rate, f_min,
                             f_max, mel, "cpu", torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "dct", _dct_matrix(n_ceps, n_filters, "cpu", torch.float32),
            persistent=False,
        )

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """spec: complex (B, n_freq, n_frames) -> (B, n_ceps*(1|3), n_frames)"""
        power = spec.abs().pow(2.0)
        filt = torch.matmul(self.bank, power)              # (B, n_filters, T)
        log = torch.log(filt + 1e-10)
        ceps = torch.matmul(self.dct, log)                 # (B, n_ceps, T)
        if not self.deltas:
            return ceps
        d1 = delta(ceps)
        return torch.cat([ceps, d1, delta(d1)], dim=1)


def delta(x: torch.Tensor, width: int = 2) -> torch.Tensor:
    """Regression-based temporal derivative, computed as a depthwise conv so it
    stays on the GPU and batches across calls."""
    b, c, t = x.shape
    taps = torch.arange(-width, width + 1, device=x.device, dtype=x.dtype)
    denom = (taps ** 2).sum()
    # No flip: torch conv1d is cross-correlation, out[t] = sum_k w[k] x[t+k],
    # which is already the regression form sum_n n*x[t+n]/sum(n^2). Flipping
    # the kernel here silently negates every delta coefficient.
    kernel = (taps / denom).view(1, 1, -1).repeat(c, 1, 1)
    padded = F.pad(x, (width, width), mode="replicate")
    return F.conv1d(padded, kernel, groups=c)


def log_power_spectrum(spec: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    return torch.log(spec.abs().pow(2.0) + eps)
