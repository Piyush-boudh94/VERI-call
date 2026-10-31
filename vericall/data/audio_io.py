"""Audio loading, resampling and silence trimming.

Two of the leakage/shortcut guards live here rather than in the split logic,
because they are properties of how a sample is turned into a tensor:

**Guard 6 -- one resampler, one code path.** ASVspoof is 16 kHz and WaveFake is
22.05 kHz, and the SSL frontend needs 16 kHz. If bona fide and spoof took
different resampling routes -- or if only one class were resampled at all -- the
*resampler* becomes the discriminative feature and the model learns nothing
transferable. Every item goes through `load_audio` regardless of source rate.

**The silence shortcut.** ASVspoof 2019 LA has systematically longer leading and
trailing silence on bona fide clips; a model trained on leading-silence duration
ALONE reaches 85% accuracy / 15.1% EER on it. Trimming is therefore mandatory,
not optional, and `scripts/train.py` also fits a duration-only control that must
be clearly beaten.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import torch

TARGET_SR = 16_000


@lru_cache(maxsize=8)
def _resampler(src: int, dst: int) -> torch.nn.Module:
    """Cached torchaudio resampler.

    Cached because constructing one builds a sinc kernel, and doing that per
    item would dominate loading. Keyed on (src, dst) so there is exactly one
    kernel per rate pair -- which is also what makes the single-path guarantee
    above true in practice and not merely intended.
    """
    import torchaudio

    return torchaudio.transforms.Resample(src, dst, dtype=torch.float32)


def decode(raw: bytes) -> tuple[torch.Tensor, int]:
    """Decode audio bytes to mono float32 in [-1, 1].

    soundfile first, PyAV as a fallback. The fallback exists for **m4a/AAC**,
    which libsndfile does not read and which Kathbath -- the real Hindi speech
    behind IndicSynth's clones -- ships in. Without it the only corpus that can
    give this project an honest Hindi bona fide class is unreadable.

    This keeps Guard 6 (one resampler, one code path) intact: both branches
    return raw PCM at the file's native rate and everything after this point --
    resampling, trimming, cropping -- is unchanged and shared. What the decoder
    must NOT do is differ *by class*, and it does not: the branch is chosen by
    container format, and the caller is responsible for making sure both
    classes arrive in comparable formats. See scripts/build_indic_paired.py on
    why that matters enough to be enforced there rather than trusted here.
    """
    import soundfile as sf

    try:
        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        wav = torch.from_numpy(data).mean(dim=1)      # mono
        return wav, int(sr)
    except Exception:                                  # noqa: BLE001
        return _decode_av(raw)


def _decode_av(raw: bytes) -> tuple[torch.Tensor, int]:
    """PyAV fallback: anything ffmpeg reads, including m4a/AAC."""
    import av
    import numpy as np

    with av.open(io.BytesIO(raw)) as container:
        stream = container.streams.audio[0]
        sr = int(stream.rate)
        # Decode to float32 mono at the file's own rate; resampling stays in
        # load_audio so there is exactly one resampler in the project.
        resampler = av.audio.resampler.AudioResampler(
            format="fltp", layout="mono", rate=sr)
        chunks = []
        for frame in container.decode(audio=0):
            for rf in resampler.resample(frame):
                chunks.append(rf.to_ndarray().reshape(-1))
    if not chunks:
        raise ValueError("no audio frames decoded")
    return torch.from_numpy(np.concatenate(chunks).astype("float32")), sr


def load_audio(
    raw: bytes,
    target_sr: int = TARGET_SR,
    trim: bool = True,
    trim_db: float = 35.0,
    pad_ms: float = 30.0,
) -> torch.Tensor:
    """Decode -> resample -> trim. The single path every sample takes."""
    wav, sr = decode(raw)
    if sr != target_sr:
        wav = _resampler(sr, target_sr)(wav)
    if trim:
        wav = trim_silence(wav, target_sr, top_db=trim_db, pad_ms=pad_ms)
    return wav


def trim_silence(
    wav: torch.Tensor,
    sample_rate: int = TARGET_SR,
    top_db: float = 35.0,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    pad_ms: float = 30.0,
) -> torch.Tensor:
    """Remove leading and trailing silence relative to the clip's own peak.

    Adaptive to the clip rather than an absolute threshold, so it behaves the
    same on a quiet recording and a loud one -- an absolute floor would trim
    quiet bona fide speech more aggressively than loud synthetic speech and
    reintroduce the very duration cue it exists to remove.

    A short pad is retained so the trim does not clip speech onsets, which carry
    real articulatory information.
    """
    frame = int(frame_ms * sample_rate / 1000)
    hop = int(hop_ms * sample_rate / 1000)
    if wav.numel() < frame:
        return wav

    frames = wav.unfold(0, frame, hop)
    rms = frames.pow(2).mean(dim=-1).clamp(min=1e-12).sqrt()
    db = 20.0 * torch.log10(rms)
    active = db > (db.max() - top_db)
    idx = torch.nonzero(active, as_tuple=False).flatten()
    if idx.numel() == 0:
        return wav

    pad = int(pad_ms * sample_rate / 1000)
    start = max(0, int(idx[0]) * hop - pad)
    end = min(wav.numel(), int(idx[-1]) * hop + frame + pad)
    return wav[start:end]


def fixed_length(wav: torch.Tensor, seconds: float, sample_rate: int = TARGET_SR,
                 random_crop: bool = False, generator: Optional[torch.Generator] = None):
    """Crop or tile to an exact duration.

    Tiling rather than zero-padding: a zero-padded tail is silence, and silence
    is precisely the artifact this pipeline works to remove. Padding short clips
    with zeros would hand the model back a duration cue through the side door.
    """
    n = int(round(seconds * sample_rate))
    if wav.numel() == n:
        return wav
    if wav.numel() > n:
        if random_crop:
            hi = wav.numel() - n
            start = int(torch.randint(0, hi + 1, (1,), generator=generator).item())
        else:
            start = (wav.numel() - n) // 2
        return wav[start:start + n]
    reps = -(-n // max(wav.numel(), 1))
    return wav.repeat(reps)[:n]


@dataclass(frozen=True)
class LoadStats:
    """Returned by the preprocessing pass so the trim can be audited rather
    than trusted -- if bona fide and spoof durations still differ sharply after
    trimming, the shortcut survived and the numbers are not to be believed."""

    n: int
    mean_seconds: float
    mean_trimmed_frac: float
