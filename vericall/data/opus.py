"""Real Opus round-trip, for measuring the channel WebRTC actually uses.

Every other channel in `augment.py` is a DSP model: mu-law is a closed-form
companding curve, the noisy line is additive noise plus frame dropping. Opus
is not amenable to that treatment. It is a transform codec with its own
psychoacoustic model, SILK/CELT hybrid switching and band folding, and the
artifacts it leaves are specific to those mechanisms. Approximating it would
measure the approximation, which for a question about *calibration validity*
is worse than not measuring at all.

So this shells into libopus through PyAV. That makes it the only channel here
with an optional dependency, and the import is deliberately local to the
function that needs it so the rest of the package still works without it.

**On alignment.** A codec that adds delay and then gets compared sample-wise
to its input produces nonsense, and Branch A2 reads phase, so this would be
the failure mode that looks like a finding. Measured: PyAV's decoder already
compensates Opus's pre-skip, giving a lag of 0 samples at 16/24/32/48 kbps
with input-output correlation 0.988-0.999 on speech. The length is still
normalised here, because a codec frame is 20 ms and the tail gets padded out
to a frame boundary.
"""
from __future__ import annotations

import io

import numpy as np
import torch

SR = 16_000


def _require_av():
    try:
        import av                                            # noqa: F401
    except ImportError as exc:                               # pragma: no cover
        raise RuntimeError(
            "the opus_voip channel needs PyAV (pip install av). It is optional "
            "because only the channel-measurement scripts use it; the served "
            "gateway does not."
        ) from exc
    return __import__("av")


def opus_roundtrip(x: np.ndarray, sr: int = SR, bitrate: int = 24_000,
                   application: str = "voip") -> np.ndarray:
    """Encode `x` to Opus and decode it back. float32 in, float32 out, same length."""
    av = _require_av()
    n = len(x)
    if n == 0:
        return x.astype(np.float32)

    buf = io.BytesIO()
    out = av.open(buf, mode="w", format="ogg")
    stream = out.add_stream("libopus", rate=sr, layout="mono")
    stream.bit_rate = int(bitrate)
    try:
        # VoIP mode is what a browser negotiates for a call: it favours
        # speech intelligibility over music fidelity, which is exactly the
        # transform whose artifacts we care about.
        stream.codec_context.options = {"application": application}
    except (AttributeError, TypeError):                      # pragma: no cover
        pass

    frame = av.AudioFrame.from_ndarray(
        (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16).reshape(1, -1),
        format="s16", layout="mono")
    frame.rate = sr
    for packet in stream.encode(frame):
        out.mux(packet)
    for packet in stream.encode(None):        # flush
        out.mux(packet)
    out.close()

    buf.seek(0)
    inp = av.open(buf, mode="r", format="ogg")
    resampler = av.audio.resampler.AudioResampler(
        format="s16", layout="mono", rate=sr)
    chunks = []
    for decoded in inp.decode(audio=0):
        for rf in resampler.resample(decoded):
            chunks.append(rf.to_ndarray().reshape(-1))
    inp.close()

    if not chunks:                                           # pragma: no cover
        return np.zeros(n, dtype=np.float32)
    y = np.concatenate(chunks).astype(np.float32) / 32768.0
    # Opus pads to a 20 ms frame boundary; trim or pad back to the input length
    # so callers can treat this as a pure channel and index it alongside x.
    if len(y) >= n:
        return y[:n]
    return np.pad(y, (0, n - len(y)))


def opus_roundtrip_batch(x: torch.Tensor, bitrate: int = 24_000,
                         sr: int = SR) -> torch.Tensor:
    """Apply `opus_roundtrip` per row of a (B, T) or (T,) tensor.

    Per row and not batched, because libopus is a stateful stream codec: one
    encoder instance carries prediction state across frames, so pushing
    several unrelated utterances through one stream would let each one's
    artifacts depend on the previous utterance. That is not a channel, it is a
    leak between examples.
    """
    single = x.dim() == 1
    rows = x.reshape(1, -1) if single else x
    dev, dt = rows.device, rows.dtype
    host = rows.detach().to("cpu", torch.float32).numpy()
    out = np.empty_like(host)
    for i in range(host.shape[0]):
        out[i] = opus_roundtrip(host[i], sr=sr, bitrate=bitrate)
    got = torch.from_numpy(out).to(dev, dt)
    return got.reshape(-1) if single else got
