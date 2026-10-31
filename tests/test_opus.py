"""The Opus channel.

Unlike every other channel in augment.py this one calls a real codec, so the
properties worth pinning are the ones that make it usable as a *channel*:
shape and length preservation, per-row independence, and determinism. A codec
that silently changed length or leaked state between rows would produce a
measurement of the harness rather than of Opus.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

av = pytest.importorskip("av", reason="the opus channel needs PyAV")

from vericall.data.augment import CHANNEL_BY_NAME, apply_channel
from vericall.data.opus import opus_roundtrip, opus_roundtrip_batch


def speech_like(n: int, seed: int = 0) -> torch.Tensor:
    """A harmonic stack with an envelope. Opus is a speech codec; white noise
    is its worst case and would understate what it preserves."""
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(n, dtype=torch.float32) / 16_000
    f0 = 120.0
    x = sum(torch.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 12))
    env = 0.5 + 0.5 * torch.sin(2 * np.pi * 3.0 * t)
    x = x * env + 0.01 * torch.randn(n, generator=g)
    return (x / x.abs().max() * 0.6).float()


def test_length_is_preserved_exactly():
    """Opus pads to a 20 ms frame boundary. A channel that returns a different
    length cannot be indexed alongside its input."""
    for n in (16_000, 16_000 + 7, 8_321, 32_000):
        x = speech_like(n).numpy()
        assert len(opus_roundtrip(x)) == n, n


def real_speech(seconds: float = 4.0) -> np.ndarray:
    """A bundled demo clip: real speech, and the actual input this channel is
    measured on. The synthetic harmonic stack above is fine for structural
    properties but is a harder case for a speech codec than speech is -- it
    round-trips at correlation 0.78 where the clip reaches 0.96 -- so a
    fidelity threshold set from it would be measuring the generator."""
    import soundfile as sf
    wav, sr = sf.read("web/demo/genuine.wav", dtype="float32", always_2d=True)
    assert sr == 16_000, sr
    return wav[:int(sr * seconds), 0]


def test_the_codec_preserves_speech_structure():
    """If this drops sharply, suspect alignment before suspecting the codec:
    PyAV compensates Opus's pre-skip, and if it ever stopped doing so the
    comparison would be off by ~6.5 ms and Branch A2's phase features would
    be measuring the offset."""
    x = real_speech()
    y = opus_roundtrip(x, bitrate=24_000)
    corr = float(np.corrcoef(x, y)[0, 1])
    assert corr > 0.9, f"correlation {corr} -- suspect a misalignment, not a codec"
    assert 0.9 < y.std() / x.std() < 1.1, "level moved more than a codec should"


def test_higher_bitrate_is_closer_to_the_input():
    """A sanity check on the knob itself: if this did not hold, the bitrate
    argument would not be reaching libopus."""
    x = real_speech()
    corrs = [float(np.corrcoef(x, opus_roundtrip(x, bitrate=b))[0, 1])
             for b in (12_000, 24_000, 48_000)]
    assert corrs[0] < corrs[2], corrs


def test_rows_are_independent():
    """libopus is a stateful stream codec. If the batch shared one encoder,
    row i's artifacts would depend on row i-1 -- a leak between examples
    dressed up as a channel."""
    a = speech_like(16_000, seed=1)
    b = speech_like(16_000, seed=2)
    alone = opus_roundtrip_batch(a.unsqueeze(0))[0]
    with_b = opus_roundtrip_batch(torch.stack([a, b]))[0]
    assert torch.allclose(alone, with_b, atol=1e-6), \
        "row 0 changed when row 1 was added: encoder state is shared"


def test_it_is_deterministic():
    x = speech_like(16_000).unsqueeze(0)
    assert torch.allclose(opus_roundtrip_batch(x), opus_roundtrip_batch(x))


def test_batch_shape_and_dtype_round_trip():
    x = speech_like(16_000).unsqueeze(0).repeat(3, 1)
    y = opus_roundtrip_batch(x)
    assert y.shape == x.shape and y.dtype == x.dtype

    one = speech_like(16_000)
    assert opus_roundtrip_batch(one).shape == one.shape, "1-D in, 1-D out"


def test_empty_input_does_not_raise():
    assert len(opus_roundtrip(np.zeros(0, dtype=np.float32))) == 0


def test_registered_as_a_channel_with_its_own_codec_id():
    ch = CHANNEL_BY_NAME["opus_voip"]
    assert ch.codec_id == 3, \
        "a distinct id: Branch B's codec embedding has never seen this channel"
    others = {c.codec_id for c in
              (CHANNEL_BY_NAME[n] for n in ("clean", "g711_mulaw", "noisy_line"))}
    assert ch.codec_id not in others

    x = speech_like(16_000).unsqueeze(0)
    y = apply_channel(x, ch)
    assert y.shape == x.shape
    assert y.abs().max() <= 1.0, "apply_channel must keep the peak in range"
