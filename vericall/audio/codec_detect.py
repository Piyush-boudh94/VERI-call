"""Runtime channel detection: pick the calibration bucket from the audio.

Branch A and A2 both carry per-channel Platt mappings whose slopes differ by
1.24x (A) and 1.67x (A2). Serving a G.711 call against the `clean` bucket
mis-scales its LLR by roughly that factor -- and A2 is the most channel-sensitive
component in the system (docs/RESULTS.md section 10: 10.54x EER degradation on a
noisy line). Until now nothing chose the bucket: the gateway read
VERICALL_CHANNEL and defaulted to `clean`. `/health` reported that honestly, but
a flag that names a gap is not a fix.

Three features, each measuring a thing the channel physically does. **All three
thresholds were fitted on labelled audio and checked on held-out clips**, not
asserted -- the first version of this file asserted them from physics and was
68% accurate, because the physics argument was wrong in two ways worth recording:

  * "Band-limiting removes high-band energy, so narrowband audio has less of it."
    True for G.711, false for the noisy line: `_noisy_line` adds 15 dB SNR white
    noise AFTER band-limiting, which repopulates the high band. Measured, the
    noisy profile carries MORE energy above 4 kHz than clean speech does
    (1.5e-2 vs 6.4e-3 median).
  * "Packet loss zeroes 20 ms frames, so look for silent frames." The loss does
    zero them, but mu-law companding runs afterwards and has no exact-zero
    quantisation level -- zero maps to ~0.0039. Lost packets come back as a
    CONSTANT frame, not a silent one, so the test is on variance, not level.

Features:

    high_band_ratio   energy above 4 kHz / total. Separates G.711 from clean:
                      measured, G.711's p95 (3.9e-4) sits below clean's p5
                      (5.2e-4). Measured above 4 kHz rather than at the 3400 Hz
                      passband edge, so it reads the stopband and not the
                      filter's roll-off.
    const_frame_ratio fraction of 20 ms frames with near-zero variance. A
                      precise packet-loss indicator -- it is never non-zero for
                      clean or G.711 audio -- but per window it is Poisson-thin
                      (2% loss over 50 frames is ~1 frame in expectation), which
                      is what the call-level accumulator is for.
    noise_floor_ratio p5 frame RMS / median frame RMS. Speech has near-silent
                      frames between words; a noisy line does not.

Held-out accuracy is reported by scripts/eval_codec_detect.py.

**The honest limit.** This detects the channel profiles *we simulate*. Real
telephony adds adaptive noise suppression and comfort noise, and a real
narrowband codec resamples to 8 kHz rather than low-passing at 16 kHz.
`high_band_ratio` reads both the same way -- absence is absence -- but the
noisy/clean split is tuned to this simulator's noise and loss model and must be
re-fitted against real transmitted audio before production. RTCFake would be the
instrument for that, and it has not been used for it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from ..features.spectral import SAMPLE_RATE

Bucket = Literal["clean", "g711", "noisy_line"]

HIGH_BAND_HZ = 4000.0
LOSS_FRAME_MS = 20.0

# Fitted on 8,000 labelled dev rows and checked on 8,000 held-out ones (split by
# CLIP -- the four crops of one clip cycle the four channel profiles, so a
# row-parity split lands exactly on that cycle and leaves the eval half with no
# clean rows at all). Fit 94.90%, held-out 94.84%: the thresholds are stable,
# not tuned to the half they were chosen on.
# Placed INSIDE the measured gap (G.711 p95 = 2.86e-4, clean p5 = 3.46e-4)
# rather than at the accuracy-maximising point. A grid search picked 4.98e-4,
# which is above clean's p5 -- it bought G.711 recall by calling ~5% of clean
# windows narrowband, and under the call-level vote those errors cost more than
# the recall was worth.
NARROWBAND_MAX_HIGH_RATIO = 3.15e-4
NOISY_MIN_FLOOR_RATIO = 0.2185
CONST_FRAME_VAR = 1e-9


def _frames(wav: torch.Tensor, sr: int, frame_ms: float) -> torch.Tensor:
    x = wav.reshape(-1).to(torch.float32)
    frame = int(frame_ms * sr / 1000.0)
    n = x.numel() // frame
    return x[: n * frame].reshape(n, frame) if n else x.new_zeros((0, frame))


def high_band_ratio(wav: torch.Tensor, sr: int = SAMPLE_RATE,
                    cutoff_hz: float = HIGH_BAND_HZ) -> float:
    """Fraction of spectral energy above `cutoff_hz`."""
    x = wav.reshape(-1).to(torch.float32)
    if x.numel() < 64:
        return 0.0
    spec = torch.fft.rfft(x)
    power = spec.real.pow(2) + spec.imag.pow(2)
    total = float(power.sum())
    if total <= 0.0:
        return 0.0
    freqs = torch.fft.rfftfreq(x.numel(), 1.0 / sr)
    return float(power[freqs >= cutoff_hz].sum()) / total


def const_frame_ratio(wav: torch.Tensor, sr: int = SAMPLE_RATE) -> float:
    """Fraction of 20 ms frames with near-zero variance -- the packet-loss tell.

    Variance, not level: mu-law companding after the loss maps the zeroed samples
    to a constant ~0.0039 rather than to silence.
    """
    f = _frames(wav, sr, LOSS_FRAME_MS)
    if f.shape[0] == 0:
        return 0.0
    return float((f.var(dim=-1) < CONST_FRAME_VAR).to(torch.float32).mean())


def noise_floor_ratio(wav: torch.Tensor, sr: int = SAMPLE_RATE) -> float:
    """p5 frame RMS over median frame RMS. High when there is no quiet."""
    f = _frames(wav, sr, LOSS_FRAME_MS)
    if f.shape[0] == 0:
        return 0.0
    rms = f.pow(2).mean(dim=-1).sqrt()
    med = float(rms.median())
    if med <= 0.0:
        return 0.0
    return float(torch.quantile(rms, 0.05)) / med


def classify_window(wav: torch.Tensor, sr: int = SAMPLE_RATE) -> Bucket:
    """Single-window verdict. Weak by design -- see CodecDetector.

    Noisy is tested FIRST because the noisy profile's added white noise lifts its
    high-band ratio above clean speech's, so the band-limit test alone calls it
    clean. Order is part of the rule, not an implementation detail.
    """
    if (const_frame_ratio(wav, sr) > 0.0
            or noise_floor_ratio(wav, sr) >= NOISY_MIN_FLOOR_RATIO):
        return "noisy_line"
    if high_band_ratio(wav, sr) < NARROWBAND_MAX_HIGH_RATIO:
        return "g711"
    return "clean"


@dataclass
class CodecDetector:
    """Call-level channel decision from per-window evidence.

    Per-call rather than per-window, for the same reason the fusion engine
    accumulates rather than re-deciding: one window of a lossy call can contain
    no lost frames, and flipping the calibration bucket mid-sentence would make
    the LLR scale jitter for no reason.

    Until `min_windows` is reached it reports `default` and says it is not
    confident, rather than silently asserting a channel it has not measured.
    """

    default: Bucket = "clean"
    min_windows: int = 4
    n: int = 0
    n_clean: int = 0
    n_narrowband: int = 0
    n_noisy: int = 0
    high_sum: float = 0.0
    floor_sum: float = 0.0
    const_sum: float = 0.0

    def update(self, wav: torch.Tensor, sr: int = SAMPLE_RATE) -> Bucket:
        h = high_band_ratio(wav, sr)
        c = const_frame_ratio(wav, sr)
        f = noise_floor_ratio(wav, sr)
        self.n += 1
        self.high_sum += h
        self.const_sum += c
        self.floor_sum += f
        if c > 0.0 or f >= NOISY_MIN_FLOOR_RATIO:
            self.n_noisy += 1
        elif h < NARROWBAND_MAX_HIGH_RATIO:
            self.n_narrowband += 1
        else:
            self.n_clean += 1
        return self.bucket

    @property
    def confident(self) -> bool:
        return self.n >= self.min_windows

    @property
    def bucket(self) -> Bucket:
        """Plain majority vote over per-window verdicts.

        An earlier version LATCHED: one window read as narrowband fixed the call
        as narrowband for good, on the argument that a real channel does not
        become wideband mid-call. The argument is sound and the rule was still
        wrong, because it converts any single per-window false positive into a
        permanent call-level error. Measured: per-window accuracy 94.5% but
        per-call accuracy 88.4%, with clean calls at 53.9% -- roughly
        1 - 0.9^6, exactly the probability that at least one of six windows
        misfires. Accumulation is supposed to beat a single window, and a rule
        that does worse than its own inputs is not an accumulator.
        """
        if not self.confident:
            return self.default
        counts = {"clean": self.n_clean, "g711": self.n_narrowband,
                  "noisy_line": self.n_noisy}
        return max(counts, key=lambda k: counts[k])  # type: ignore[return-value]

    def explain(self) -> dict:
        """What the decision rests on. Goes into /health and the audit record."""
        k = max(self.n, 1)
        return {
            "bucket": self.bucket,
            "confident": self.confident,
            "windows": self.n,
            "mean_high_band_ratio": self.high_sum / k,
            "mean_const_frame_ratio": self.const_sum / k,
            "mean_noise_floor_ratio": self.floor_sum / k,
            "clean_windows": self.n_clean,
            "narrowband_windows": self.n_narrowband,
            "noisy_windows": self.n_noisy,
        }
