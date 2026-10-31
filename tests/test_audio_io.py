"""The single audio path every sample in this project takes.

`data/audio_io.py` had no test, which is the wrong place for that gap: it
carries two of the leakage guards, it is the only resampler in the project, and
a PyAV fallback was added to it so Kathbath's m4a -- the real Hindi speech the
entire Hindi result rests on -- could be read at all.
"""
from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf
import torch

from vericall.data.audio_io import TARGET_SR, decode, load_audio, trim_silence


def wav_bytes(x: np.ndarray, sr: int = TARGET_SR, fmt: str = "WAV") -> bytes:
    buf = io.BytesIO()
    sf.write(buf, x, sr, format=fmt, subtype="PCM_16")
    return buf.getvalue()


def speech_like(n: int, sr: int = TARGET_SR, seed: int = 0) -> np.ndarray:
    g = np.random.default_rng(seed)
    t = np.arange(n) / sr
    x = sum(np.sin(2 * np.pi * 130 * k * t) / k for k in range(1, 10))
    x = x * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)) + 0.01 * g.standard_normal(n)
    return (x / np.abs(x).max() * 0.6).astype(np.float32)


# ----------------------------------------------------------------- decode

def test_wav_and_flac_both_decode_to_mono_float():
    x = speech_like(TARGET_SR)
    for fmt in ("WAV", "FLAC"):
        w, sr = decode(wav_bytes(x, fmt=fmt))
        assert sr == TARGET_SR and w.dtype == torch.float32
        assert w.dim() == 1, "must be mono"
        assert float(w.abs().max()) <= 1.0


def test_stereo_is_averaged_not_truncated():
    """Averaging rather than taking channel 0: a caller who hands us a stereo
    file has not told us which side carries the speech."""
    left, right = speech_like(8000, seed=1), speech_like(8000, seed=2)
    w, _ = decode(wav_bytes(np.stack([left, right], axis=1)))
    assert w.numel() == 8000
    assert np.allclose(w.numpy(), (left + right) / 2, atol=2e-4)


def test_m4a_falls_back_to_pyav():
    """libsndfile cannot read m4a/AAC. Without the fallback, Kathbath -- the
    only corpus that can give this project an honest Hindi bona fide class --
    is unreadable, so this is the test that protects the Hindi result."""
    av = pytest.importorskip("av", reason="the m4a fallback needs PyAV")
    x = speech_like(TARGET_SR * 2)
    buf = io.BytesIO()
    out = av.open(buf, mode="w", format="ipod")          # m4a container
    st = out.add_stream("aac", rate=TARGET_SR, layout="mono")
    fr = av.AudioFrame.from_ndarray(
        (x * 32767).astype(np.int16).reshape(1, -1), format="s16", layout="mono")
    fr.rate = TARGET_SR
    for pkt in st.encode(fr):
        out.mux(pkt)
    for pkt in st.encode(None):
        out.mux(pkt)
    out.close()

    with pytest.raises(Exception):
        sf.read(io.BytesIO(buf.getvalue()))              # precondition
    w, sr = decode(buf.getvalue())
    assert sr == TARGET_SR and w.dim() == 1 and w.numel() > TARGET_SR


def test_undecodable_bytes_raise_rather_than_returning_silence():
    """Returning zeros on a bad file is the defect that was quarantined from
    vericall/audio/dataset.py: it kept the ORIGINAL LABEL, so a corrupt spoof
    became a silent 'spoof' the model could learn from."""
    with pytest.raises(Exception):
        decode(b"this is not audio at all, not even close")


# ------------------------------------------------------- the single path

def test_everything_is_resampled_through_one_path():
    """Guard 6. ASVspoof is 16 kHz and WaveFake 22.05 kHz; if the classes took
    different resampling routes the RESAMPLER becomes the discriminative
    feature and the model learns nothing transferable."""
    for src in (8_000, 22_050, 44_100, 48_000):
        n = int(src * 1.5)
        w = load_audio(wav_bytes(speech_like(n, sr=src), sr=src), trim=False)
        got = w.numel() / TARGET_SR
        assert abs(got - 1.5) < 0.05, f"{src} Hz -> {got:.3f}s"


def test_a_16k_file_is_not_resampled_at_all():
    x = speech_like(TARGET_SR)
    w = load_audio(wav_bytes(x), trim=False)
    assert w.numel() == TARGET_SR
    assert np.allclose(w.numpy(), x, atol=1e-4)


# -------------------------------------------------------------- trimming

def test_trim_removes_leading_and_trailing_silence():
    """The silence shortcut is not hypothetical: leading-silence duration ALONE
    reaches 15.1% EER on ASVspoof LA, because bona fide clips there have
    systematically longer leading silence."""
    core = speech_like(TARGET_SR)
    padded = np.concatenate([np.zeros(TARGET_SR, np.float32), core,
                             np.zeros(TARGET_SR, np.float32)])
    out = trim_silence(torch.from_numpy(padded), TARGET_SR)
    assert out.numel() < padded.size
    assert out.numel() >= core.size * 0.9, "must not eat the speech"


def test_trim_is_relative_to_the_clip_not_an_absolute_floor():
    """An absolute threshold would trim quiet bona fide speech harder than loud
    synthetic speech and reintroduce the very duration cue it removes."""
    core = speech_like(TARGET_SR)
    pad = np.zeros(TARGET_SR // 2, np.float32)
    loud = np.concatenate([pad, core, pad])
    quiet = np.concatenate([pad, core * 0.02, pad])
    a = trim_silence(torch.from_numpy(loud), TARGET_SR).numel()
    b = trim_silence(torch.from_numpy(quiet), TARGET_SR).numel()
    assert abs(a - b) < 0.15 * TARGET_SR, f"loud {a} vs quiet {b} samples"


def test_all_silence_is_returned_unchanged_rather_than_emptied():
    z = torch.zeros(TARGET_SR)
    assert trim_silence(z, TARGET_SR).numel() > 0


def test_a_clip_shorter_than_one_frame_survives():
    tiny = torch.from_numpy(speech_like(100))
    assert trim_silence(tiny, TARGET_SR).numel() == 100


def test_load_audio_trims_by_default():
    """The default matters: the live serving path and the training path must
    agree, and everything that reads a file goes through this."""
    core = speech_like(TARGET_SR)
    padded = np.concatenate([np.zeros(TARGET_SR, np.float32), core])
    raw = wav_bytes(padded)
    assert load_audio(raw).numel() < load_audio(raw, trim=False).numel()
