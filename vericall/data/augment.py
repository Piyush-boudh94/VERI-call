"""Channel augmentation: make training audio look like it came off a phone line.

The detector has to survive the telephony channel, so training on clean studio
audio teaches it to key on cues the channel destroys. Everything here is
torch-native so it runs batched on the GPU inside the extraction pass rather
than becoming a CPU bottleneck.

**A documented gap, stated rather than hidden:** the pod's ffmpeg has AMR-NB and
AMR-WB as *decode-only*, and there is no compiler or root to rebuild it. So AMR
cannot be synthesised. G.711 (mu-law/A-law) and narrowband band-limiting are
real and implemented here; Opus is available through ffmpeg but too slow to
apply per clip in a hot loop. The honest mitigation is RTCFake, which carries
genuine transmission through seven live platforms -- better evidence than any
simulation -- and which is currently blocked on a gated-repo access request.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch

SR = 16_000


# --------------------------------------------------------------------- #
# G.711 companding
# --------------------------------------------------------------------- #

def mu_law_roundtrip(x: torch.Tensor, mu: int = 255) -> torch.Tensor:
    """G.711 mu-law encode + decode: 8-bit companded quantisation.

    The quantisation noise is signal-dependent, which is exactly what makes it
    interesting -- it is not additive white noise and a model that has only seen
    clean audio has no reason to be robust to it.
    """
    m = float(mu)
    y = torch.sign(x) * torch.log1p(m * x.abs()) / math.log1p(m)
    q = torch.round((y + 1.0) * 127.5).clamp(0, 255)          # 8-bit
    y = q / 127.5 - 1.0
    return torch.sign(y) * ((1.0 + m) ** y.abs() - 1.0) / m


def a_law_roundtrip(x: torch.Tensor, A: float = 87.6) -> torch.Tensor:
    """G.711 A-law encode + decode (the European telephony variant)."""
    ax = x.abs().clamp(max=1.0)
    thresh = 1.0 / A
    lo = A * ax / (1.0 + math.log(A))
    hi = (1.0 + torch.log(A * ax.clamp(min=1e-12))) / (1.0 + math.log(A))
    y = torch.sign(x) * torch.where(ax < thresh, lo, hi)

    q = torch.round((y + 1.0) * 127.5).clamp(0, 255)
    y = q / 127.5 - 1.0

    ay = y.abs()
    t2 = 1.0 / (1.0 + math.log(A))
    lo_i = ay * (1.0 + math.log(A)) / A
    hi_i = torch.exp(ay * (1.0 + math.log(A)) - 1.0) / A
    return torch.sign(y) * torch.where(ay < t2, lo_i, hi_i)


# --------------------------------------------------------------------- #
# channel
# --------------------------------------------------------------------- #

def narrowband(x: torch.Tensor, sr: int = SR) -> torch.Tensor:
    """Band-limit to 300-3400 Hz: the actual passband of a telephone channel.

    This is the single most destructive thing the channel does, and it removes
    precisely the high-frequency detail Branch A keys on -- which is the entire
    argument for the prosody branch.
    """
    n = x.shape[-1]
    spec = torch.fft.rfft(x, dim=-1)
    freqs = torch.fft.rfftfreq(n, 1.0 / sr).to(x.device)
    band = ((freqs >= 300.0) & (freqs <= 3400.0)).to(x.dtype)
    # soften the edges so we band-limit rather than ring
    band = band + 0.05 * (~((freqs >= 300.0) & (freqs <= 3400.0))).to(x.dtype)
    return torch.fft.irfft(spec * band, n=n, dim=-1)


def add_noise(x: torch.Tensor, snr_db: float) -> torch.Tensor:
    sig = x.pow(2).mean(dim=-1, keepdim=True).clamp(min=1e-12)
    noise = torch.randn_like(x)
    npow = noise.pow(2).mean(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = (sig / npow / (10.0 ** (snr_db / 10.0))).sqrt()
    return x + noise * scale


def gain(x: torch.Tensor, db: float) -> torch.Tensor:
    return x * (10.0 ** (db / 20.0))


def packet_loss(x: torch.Tensor, rate: float, ms: float = 20.0,
                sr: int = SR) -> torch.Tensor:
    """Zero out random 20 ms frames -- what VoIP loss actually looks like."""
    if rate <= 0:
        return x
    frame = int(ms * sr / 1000)
    n_frames = x.shape[-1] // frame
    if n_frames == 0:
        return x
    mask = (torch.rand(*x.shape[:-1], n_frames, device=x.device) > rate).to(x.dtype)
    mask = mask.repeat_interleave(frame, dim=-1)
    if mask.shape[-1] < x.shape[-1]:
        pad = torch.ones(*x.shape[:-1], x.shape[-1] - mask.shape[-1],
                         device=x.device, dtype=x.dtype)
        mask = torch.cat([mask, pad], dim=-1)
    return x * mask


# --------------------------------------------------------------------- #
# named channel profiles
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class Channel:
    name: str
    codec_id: int          # index into Branch B's codec embedding
    fn: Callable[[torch.Tensor], torch.Tensor]


def _clean(x): return x


def _g711_mulaw(x):
    return mu_law_roundtrip(narrowband(x)).clamp(-1.0, 1.0)


def _g711_alaw(x):
    return a_law_roundtrip(narrowband(x)).clamp(-1.0, 1.0)


def _noisy_line(x):
    return mu_law_roundtrip(
        packet_loss(add_noise(narrowband(x), snr_db=15.0), rate=0.02)
    ).clamp(-1.0, 1.0)


def _opus_voip(x):
    """Real libopus at WebRTC's VoIP settings.

    Unlike every other channel here this one is not a DSP approximation: it
    encodes and decodes through libopus, because Opus artifacts are specific
    to its own transform and a hand-rolled imitation would measure the
    imitation. It is therefore the only channel with an optional dependency,
    and it degrades to a clear error rather than silently to something else.

    24 kbps mono is what a WebRTC audio-only call negotiates in practice.
    """
    from .opus import opus_roundtrip_batch      # local: PyAV is optional
    return opus_roundtrip_batch(x, bitrate=24_000).clamp(-1.0, 1.0)


CHANNELS: tuple[Channel, ...] = (
    Channel("clean", 0, _clean),
    Channel("g711_mulaw", 1, _g711_mulaw),
    Channel("g711_alaw", 1, _g711_alaw),
    Channel("noisy_line", 2, _noisy_line),
    # codec_id 3: a bucket Branch B's embedding has never been trained on.
    # Listed so the channel can be MEASURED; see docs/RESULTS.md on why it is
    # not yet a calibration bucket.
    Channel("opus_voip", 3, _opus_voip),
)
CHANNEL_BY_NAME = {c.name: c for c in CHANNELS}


def apply_channel(x: torch.Tensor, channel: Channel) -> torch.Tensor:
    out = channel.fn(x)
    peak = out.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
    return torch.where(peak > 1.0, out / peak, out)      # avoid clipping


def augment_batch(
    x: torch.Tensor, channels: Sequence[Channel], generator=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one randomly chosen channel per item. Returns (audio, codec_id)."""
    idx = torch.randint(len(channels), (x.shape[0],), generator=generator)
    out = torch.empty_like(x)
    codec = torch.empty(x.shape[0], dtype=torch.long, device=x.device)
    for i, ch in enumerate(channels):
        sel = idx == i
        if sel.any():
            out[sel] = apply_channel(x[sel], ch)
            codec[sel] = ch.codec_id
    return out, codec
