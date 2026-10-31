"""Runtime channel detection.

The thresholds here were fitted on labelled audio, so these tests do not re-check
the numbers -- scripts/eval_codec_detect.py does that. What they pin is the
*rules*, each of which was wrong at some point in a way the numbers alone did not
reveal:

  * noisy must be tested before the band-limit test, because added white noise
    lifts the noisy profile's high-band energy ABOVE clean speech's
  * packet loss produces CONSTANT frames, not silent ones, because mu-law
    companding has no exact-zero quantisation level
  * the call-level rule must be a vote, not a latch -- a latch made per-call
    accuracy WORSE than per-window accuracy
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from vericall.audio.codec_detect import (
    NARROWBAND_MAX_HIGH_RATIO, NOISY_MIN_FLOOR_RATIO, CodecDetector,
    classify_window, const_frame_ratio, high_band_ratio, noise_floor_ratio,
)

SR = 16_000


def tone(hz: float, n: int = SR, amp: float = 0.5) -> torch.Tensor:
    t = torch.arange(n, dtype=torch.float32) / SR
    return amp * torch.sin(2 * math.pi * hz * t)


def speechlike(x: torch.Tensor) -> torch.Tensor:
    """Punch quiet gaps into a signal.

    A steady tone has a flat envelope, so its p5 frame RMS equals its median and
    noise_floor_ratio reads ~1.0 -- i.e. "there is no quiet here", which is the
    noisy-line signature. That is the feature behaving correctly on an
    unrealistic input: real speech pauses between words. Every fixture below
    that is meant to stand for speech goes through this.
    """
    y = x.clone()
    for k in range(0, y.numel(), 800):
        y[k: k + 600] *= 0.001
    return y


def test_high_band_ratio_sees_energy_above_the_cutoff():
    assert high_band_ratio(tone(6000.0)) > 0.9
    assert high_band_ratio(tone(500.0)) < 0.01


def test_const_frame_ratio_finds_constant_frames_not_only_silent_ones():
    """mu-law maps a zeroed packet to a constant ~0.0039, not to silence. A
    detector testing for silence finds nothing at all."""
    x = tone(800.0).clone()
    x[SR // 2: SR // 2 + 320] = 0.0039        # one 20 ms 'lost' frame, companded
    r = const_frame_ratio(x)
    assert r > 0.0, "a constant non-zero frame must count as packet loss"
    assert r == pytest.approx(1 / 50, abs=1 / 50)

    x0 = tone(800.0).clone()
    x0[SR // 2: SR // 2 + 320] = 0.0          # exactly silent: also constant
    assert const_frame_ratio(x0) > 0.0


def test_const_frame_ratio_is_zero_on_ordinary_audio():
    assert const_frame_ratio(tone(800.0)) == 0.0


def test_noise_floor_ratio_is_high_when_there_is_no_quiet():
    g = torch.Generator().manual_seed(0)
    assert noise_floor_ratio(torch.randn(SR, generator=g) * 0.1) > 0.5
    # speech-like: bursts separated by near-silence
    x = tone(800.0).clone()
    for k in range(0, SR, 800):
        x[k: k + 600] *= 0.001
    assert noise_floor_ratio(x) < 0.3


def test_noisy_is_tested_before_the_band_limit_rule():
    """The ordering is part of the rule.

    A band-limited signal plus broadband noise has HIGH high-band energy, so the
    band-limit test alone reads it as clean. Testing noisy first is what makes
    the noisy profile reachable at all.
    """
    g = torch.Generator().manual_seed(1)
    noisy = tone(800.0) * 0.3 + torch.randn(SR, generator=g) * 0.15
    assert high_band_ratio(noisy) > NARROWBAND_MAX_HIGH_RATIO   # looks wideband
    assert noise_floor_ratio(noisy) >= NOISY_MIN_FLOOR_RATIO
    assert classify_window(noisy) == "noisy_line"


def test_band_limited_audio_reads_as_g711():
    x = speechlike(tone(800.0) + 0.3 * tone(1500.0))   # nothing above 4 kHz
    assert classify_window(x) == "g711"


def test_wideband_speechlike_audio_reads_as_clean():
    assert classify_window(speechlike(tone(800.0) + 0.25 * tone(5200.0))) == "clean"


def test_detector_reports_its_default_until_confident():
    det = CodecDetector(default="clean", min_windows=4)
    band_limited = speechlike(tone(800.0))
    for _ in range(3):
        det.update(band_limited)
        assert not det.confident
        assert det.bucket == "clean"           # the default, not a claim
    det.update(band_limited)
    assert det.confident
    assert det.bucket == "g711"


def test_one_stray_window_does_not_decide_the_call():
    """The regression this class was rewritten for.

    An earlier version latched: any single narrowband window fixed the call as
    narrowband permanently. Per-window accuracy was 94.5% but per-call accuracy
    was 88.4%, with clean calls at 53.9% -- an accumulator doing worse than its
    own inputs.
    """
    det = CodecDetector(min_windows=4)
    wide = speechlike(tone(800.0) + 0.25 * tone(5200.0))
    for _ in range(5):
        det.update(wide)
    det.update(speechlike(tone(800.0)))        # one band-limited misfire
    assert det.bucket == "clean"


def test_a_genuinely_narrowband_call_still_wins_the_vote():
    det = CodecDetector(min_windows=4)
    det.update(speechlike(tone(800.0) + 0.25 * tone(5200.0)))
    for _ in range(5):
        det.update(speechlike(tone(800.0)))
    assert det.bucket == "g711"


def test_explain_reports_what_the_decision_rested_on():
    det = CodecDetector(min_windows=2)
    det.update(speechlike(tone(800.0)))
    det.update(speechlike(tone(800.0)))
    e = det.explain()
    assert e["bucket"] == "g711"
    assert e["windows"] == 2
    assert e["narrowband_windows"] == 2
    assert set(e) >= {"mean_high_band_ratio", "mean_const_frame_ratio",
                      "mean_noise_floor_ratio", "confident"}


def test_a_flat_envelope_reads_as_noisy_and_that_is_correct():
    """Documented, not incidental.

    noise_floor_ratio asks "is there any quiet in this window". A signal with a
    flat envelope answers no, and is classified noisy_line whatever its
    bandwidth. That is the intended meaning of the feature, and it is also the
    most likely source of the measured clean -> noisy_line errors (135 of 4,000
    windows): a 1 s window of continuous speech with no pause in it.
    """
    assert classify_window(tone(800.0)) == "noisy_line"
    assert noise_floor_ratio(tone(800.0)) > NOISY_MIN_FLOOR_RATIO
