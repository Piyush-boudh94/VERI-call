"""Prosody features for Branch B: F0 contour, energy, and pause structure.

Prosody is the compression-survival branch. Telephony codecs are built to
discard exactly the fine high-frequency spectral detail Branch A depends on, but
they preserve pitch and rhythm structure well -- so when Branch A degrades under
AMR/Opus/G.711, this is what still carries signal.

F0 is computed with a torch-native YIN so it runs batched on the GPU. That is
not just tidiness: pyworld and webrtcvad both need a source build, and the pod
image has no compiler and no root, so a CPU pitch tracker was never an option
here anyway.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .spectral import SAMPLE_RATE


@dataclass(frozen=True)
class F0Config:
    sample_rate: int = SAMPLE_RATE
    frame_length: int = 1024          # 64 ms -- needs >= 2 periods at f_min
    hop_length: int = 160             # 10 ms, matching the spectral front end
    f_min: float = 60.0               # low enough for deep male speech
    f_max: float = 400.0              # high enough for children/expressive speech
    threshold: float = 0.15           # YIN absolute threshold (aperiodicity)

    @property
    def tau_min(self) -> int:
        return max(2, int(self.sample_rate / self.f_max))

    @property
    def tau_max(self) -> int:
        return min(self.frame_length - 1, int(self.sample_rate / self.f_min))


def _difference_function(frames: torch.Tensor, tau_max: int) -> torch.Tensor:
    """YIN difference function via FFT autocorrelation.

    d(tau) = r(0) + r_tau(0) - 2 r(tau), where the two power terms are running
    sums of squares. Computing it directly would be O(W * tau_max) per frame;
    this is O(W log W) and batches across every frame of every call at once.
    """
    n, w = frames.shape
    n_fft = 1
    while n_fft < 2 * w:
        n_fft *= 2

    spec = torch.fft.rfft(frames, n=n_fft, dim=-1)
    acf = torch.fft.irfft(spec * spec.conj(), n=n_fft, dim=-1)[:, : tau_max + 1]

    # C[k] = sum of x[j]^2 for j < k, so that
    #   d(tau) = P(0, W-tau) + P(tau, W) - 2 r(tau)
    #          = C[W-tau]    + C[W]-C[tau] - 2 r(tau)
    # Both power terms must be tau-dependent: the naive version that uses a
    # single scalar r(0) for the first term silently mis-shapes and, where it
    # does broadcast, over-counts energy at long lags.
    cumsum = torch.cat(
        [torch.zeros(n, 1, device=frames.device, dtype=frames.dtype),
         frames.pow(2.0).cumsum(dim=-1)], dim=-1
    )                                                    # (n, w+1)
    taus = torch.arange(tau_max + 1, device=frames.device)
    d = cumsum[:, w - taus] + cumsum[:, w : w + 1] - cumsum[:, taus] - 2.0 * acf
    return d.clamp(min=0.0)


def _cumulative_mean_normalised(d: torch.Tensor) -> torch.Tensor:
    """d'(0)=1, d'(tau) = d(tau) / mean(d(1..tau)).

    This normalisation is what makes a single absolute threshold work across
    loud and quiet frames -- plain autocorrelation would need a per-frame one.
    """
    n, k = d.shape
    taus = torch.arange(1, k, device=d.device, dtype=d.dtype)
    running = d[:, 1:].cumsum(dim=-1) / taus
    out = torch.ones_like(d)
    out[:, 1:] = d[:, 1:] / (running + 1e-10)
    return out


class F0Tracker(torch.nn.Module):
    """Batched YIN. Returns (f0_hz, voiced_prob), zeros where unvoiced."""

    def __init__(self, cfg: F0Config = F0Config()):
        super().__init__()
        self.cfg = cfg

    def forward(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        b = wav.shape[0]
        frames = wav.unfold(-1, cfg.frame_length, cfg.hop_length)   # (B, T, W)
        t = frames.shape[1]
        flat = frames.reshape(b * t, cfg.frame_length)
        flat = flat - flat.mean(dim=-1, keepdim=True)

        d = _difference_function(flat, cfg.tau_max)
        dp = _cumulative_mean_normalised(d)

        search = dp[:, cfg.tau_min : cfg.tau_max + 1]

        # YIN's absolute-threshold step wants the first local MINIMUM below the
        # threshold, not the first sample below it. Descending into a dip
        # crosses the threshold several samples before reaching the bottom, so
        # taking the crossing yields a lag that is systematically too short and
        # a pitch that reads a few percent too high -- a bias that is easy to
        # miss because voiced/unvoiced decisions still look perfect.
        inf = torch.full_like(search[:, :1], float("inf"))
        left = torch.cat([inf, search[:, :-1]], dim=1)
        right = torch.cat([search[:, 1:], inf], dim=1)
        is_local_min = (search < left) & (search <= right)
        candidate = is_local_min & (search < cfg.threshold)

        first = torch.where(
            candidate.any(dim=-1),
            candidate.float().argmax(dim=-1),      # first qualifying minimum
            search.argmin(dim=-1),                 # else the global minimum
        )
        tau = first + cfg.tau_min

        # parabolic interpolation around the chosen lag, for sub-bin resolution
        idx = tau.clamp(1, cfg.tau_max - 1)
        y0 = dp.gather(1, (idx - 1).unsqueeze(1)).squeeze(1)
        y1 = dp.gather(1, idx.unsqueeze(1)).squeeze(1)
        y2 = dp.gather(1, (idx + 1).unsqueeze(1)).squeeze(1)
        denom = 2.0 * (2.0 * y1 - y0 - y2)
        shift = torch.where(denom.abs() > 1e-10, (y2 - y0) / denom, torch.zeros_like(denom))
        tau_ref = idx.to(wav.dtype) + shift.clamp(-1.0, 1.0)

        f0 = cfg.sample_rate / tau_ref.clamp(min=1.0)
        aperiodicity = y1.clamp(0.0, 1.0)
        voiced = (1.0 - aperiodicity).clamp(0.0, 1.0)
        f0 = torch.where(aperiodicity < cfg.threshold * 2.0, f0, torch.zeros_like(f0))
        f0 = torch.where(
            (f0 >= cfg.f_min) & (f0 <= cfg.f_max), f0, torch.zeros_like(f0)
        )
        return f0.view(b, t), voiced.view(b, t)


class IncrementalF0:
    """Per-call F0 cache for the slow lane.

    Branch B uses a 3 s window with a 1 s hop, so two thirds of every pitch
    track would be recomputed from audio already analysed. Caching per-frame F0
    and computing only the new second is roughly a 3x saving on the most
    expensive slow-lane branch -- large enough that skipping it would about
    halve total concurrent-call capacity.
    """

    def __init__(self, tracker: F0Tracker, window_s: float = 3.0, hop_s: float = 1.0):
        self.tracker = tracker
        self.cfg = tracker.cfg
        self.window_s, self.hop_s = window_s, hop_s
        self._f0: torch.Tensor | None = None
        self._voiced: torch.Tensor | None = None
        self.frames_recomputed = 0
        self.frames_reused = 0

    @property
    def _frames_per_window(self) -> int:
        return int(self.window_s * self.cfg.sample_rate / self.cfg.hop_length)

    def push(self, new_audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Append `new_audio` (one hop, shape (1, N)) and return the F0 track for
        the current window."""
        f0_new, v_new = self.tracker(new_audio)
        self.frames_recomputed += int(f0_new.shape[1])
        if self._f0 is None:
            self._f0, self._voiced = f0_new, v_new
        else:
            self.frames_reused += int(self._f0.shape[1])
            self._f0 = torch.cat([self._f0, f0_new], dim=1)
            self._voiced = torch.cat([self._voiced, v_new], dim=1)
        keep = self._frames_per_window
        self._f0, self._voiced = self._f0[:, -keep:], self._voiced[:, -keep:]
        return self._f0, self._voiced


def prosody_statistics(
    f0: torch.Tensor, voiced: torch.Tensor, energy: torch.Tensor
) -> torch.Tensor:
    """Summarise a window into the vector Branch B's encoder consumes.

    Deliberately trajectory-based (slopes, jitter, pause structure) rather than
    just means: a vocoder can match a speaker's average pitch easily, but
    reproducing natural micro-variation in the contour is much harder.
    """
    eps = 1e-8
    vmask = (f0 > 0).float()
    n_voiced = vmask.sum(dim=-1)
    denom = n_voiced.clamp(min=1.0)

    lf0 = torch.where(f0 > 0, torch.log(f0 + eps), torch.zeros_like(f0))
    mean = (lf0 * vmask).sum(-1) / denom
    var = ((lf0 - mean.unsqueeze(-1)) ** 2 * vmask).sum(-1) / denom
    d1 = lf0[:, 1:] - lf0[:, :-1]
    dmask = vmask[:, 1:] * vmask[:, :-1]
    slope = (d1 * dmask).sum(-1) / dmask.sum(-1).clamp(min=1.0)
    jitter = (d1.abs() * dmask).sum(-1) / dmask.sum(-1).clamp(min=1.0)

    log_e = torch.log(energy + eps)
    e_mean = log_e.mean(-1)
    e_var = log_e.var(-1, unbiased=False)
    e_d = (log_e[:, 1:] - log_e[:, :-1]).abs().mean(-1)

    voiced_ratio = n_voiced / f0.shape[-1]
    # longest unvoiced run == longest pause, as a fraction of the window
    unvoiced = 1.0 - vmask
    runs = []
    for i in range(unvoiced.shape[0]):
        row, best, cur = unvoiced[i], 0.0, 0.0
        for val in row.tolist():
            cur = cur + 1.0 if val > 0.5 else 0.0
            best = max(best, cur)
        runs.append(best / f0.shape[-1])
    longest_pause = torch.tensor(runs, device=f0.device, dtype=f0.dtype)

    return torch.stack(
        [mean, var.sqrt(), slope, jitter, e_mean, e_var.sqrt(), e_d,
         voiced_ratio, longest_pause, voiced.mean(-1)],
        dim=-1,
    )


def frame_energy(wav: torch.Tensor, frame_length: int = 400, hop_length: int = 160):
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    frames = wav.unfold(-1, frame_length, hop_length)
    return frames.pow(2.0).mean(dim=-1)
