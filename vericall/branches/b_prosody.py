"""Branch B -- one-class prosody detector.

The compression-survival branch. Telephony codecs are built to discard exactly
the fine high-frequency spectral detail Branch A depends on, but they preserve
pitch and rhythm structure well, so when Branch A degrades under AMR/Opus/G.711
this is what still carries signal.

Trained **one-class on bona fide speech only**, by design. A two-class model can
only recognise the spoofs it was shown; a one-class model flags anything that
does not look like natural pitch and rhythm, which is what a detector facing
unseen attack families needs.

The cost, which the fusion layer must absorb: the output is an unbounded anomaly
score with no probabilistic meaning and no negative distribution behind it. It
is mapped to the shared LLR scale by `fusion.calibration.one_class_to_llr` and
hard-clipped there, so an uncalibrated branch can never dominate a decision.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..features.prosody import (
    F0Config, F0Tracker, frame_energy, prosody_statistics,
)


@dataclass(frozen=True)
class BranchBConfig:
    contour_channels: int = 4      # log-F0, voiced prob, log-energy, delta-F0
    stat_dim: int = 10             # prosody_statistics() output width
    hidden: int = 96
    embed: int = 64
    n_codecs: int = 3              # clean / G.711 / noisy line
    codec_embed: int = 8
    dropout: float = 0.1


class ProsodyEncoder(nn.Module):
    """Encodes pitch/energy trajectories plus summary statistics.

    Conditioned on a codec embedding so the encoder can normalise for known
    channel distortion rather than mistaking it for anomaly -- without that, a
    G.711 call looks "abnormal" purely because it is narrowband.
    """

    def __init__(self, cfg: BranchBConfig = BranchBConfig()):
        super().__init__()
        self.cfg = cfg
        # bias=False and affine=False throughout, and this is a correctness
        # requirement rather than a style choice.
        #
        # Deep-SVDD (Ruff et al., ICML 2018) requires the network be bias-free:
        # any constant path lets the encoder map every input to the hypersphere
        # centre, driving the one-class loss to zero while learning nothing.
        # This project hit exactly that collapse -- embedding std 0.0057, dev
        # EER 49.8% -- and docs/RESEARCH.md has recorded the fix as applied
        # since it was written.
        #
        # It was not applied. The code carried default bias=True Conv1d/Linear
        # and affine=True BatchNorm until a test asserted otherwise
        # (test_b_encoder_is_bias_free). Most likely lost in the September
        # reconstruction that also overwrote a2_phase.py, whose docstring says
        # it was rebuilt from checkpoint shapes.
        #
        # Changing this invalidates any checkpoint trained against the old
        # shapes, which is why branch_B.pt is retrained alongside this edit
        # rather than after it.
        self.conv = nn.Sequential(
            nn.Conv1d(cfg.contour_channels, cfg.hidden, 5, padding=2,
                      bias=False),
            nn.BatchNorm1d(cfg.hidden, affine=False), nn.GELU(),
            nn.Conv1d(cfg.hidden, cfg.hidden, 5, padding=2, groups=4,
                      bias=False),
            nn.BatchNorm1d(cfg.hidden, affine=False), nn.GELU(),
        )
        self.codec = nn.Embedding(cfg.n_codecs, cfg.codec_embed)
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden * 2 + cfg.stat_dim + cfg.codec_embed,
                      cfg.hidden, bias=False),
            nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.embed, bias=False),
        )

    def forward(self, contours: torch.Tensor, stats: torch.Tensor,
                codec_id: torch.Tensor) -> torch.Tensor:
        h = self.conv(contours)
        pooled = torch.cat([h.mean(dim=-1), h.max(dim=-1).values], dim=-1)
        return self.head(torch.cat([pooled, stats, self.codec(codec_id)], dim=-1))


class BranchB(nn.Module):
    """Deep-SVDD style one-class detector over prosody.

    Bona fide examples are pulled toward a fixed centre during training; the
    anomaly score is squared distance from it. The centre is a buffer rather
    than a parameter -- if it were learnable the objective would collapse to
    mapping everything to a single point, which is the classic failure mode of
    one-class deep models.
    """

    def __init__(self, cfg: BranchBConfig = BranchBConfig(),
                 f0: F0Config = F0Config()):
        super().__init__()
        self.cfg = cfg
        self.tracker = F0Tracker(f0)
        self.encoder = ProsodyEncoder(cfg)
        self.register_buffer("center", torch.zeros(cfg.embed))
        self.register_buffer("center_initialised", torch.zeros(1))

    def _inputs(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        f0, voiced = self.tracker(wav)
        energy = frame_energy(wav)[:, : f0.shape[1]]
        stats = prosody_statistics(f0, voiced, energy)

        log_f0 = torch.where(f0 > 0, torch.log(f0 + 1e-8), torch.zeros_like(f0))
        d_f0 = torch.cat([torch.zeros_like(log_f0[:, :1]),
                          log_f0[:, 1:] - log_f0[:, :-1]], dim=1)
        contours = torch.stack(
            [log_f0, voiced, torch.log(energy + 1e-8), d_f0], dim=1
        )
        return contours, stats

    def embed(self, wav: torch.Tensor, codec_id: torch.Tensor | None = None):
        contours, stats = self._inputs(wav)
        if codec_id is None:
            codec_id = torch.zeros(contours.shape[0], dtype=torch.long,
                                   device=contours.device)
        return self.encoder(contours, stats, codec_id.long())

    def forward(self, wav: torch.Tensor, codec_id: torch.Tensor | None = None):
        z = self.embed(wav, codec_id)
        return ((z - self.center) ** 2).sum(dim=-1)

    @torch.inference_mode()
    def score(self, wav: torch.Tensor, codec_id: torch.Tensor | None = None):
        """Anomaly score in [0, inf). NOT a probability and NOT comparable to
        Branch A/A2 output -- `fusion.calibration.one_class_to_llr` maps it onto
        the shared LLR scale against a bona fide cohort."""
        return self(wav, codec_id)

    @torch.inference_mode()
    def init_center(self, wav: torch.Tensor, eps: float = 0.1) -> torch.Tensor:
        """Set the centre to the mean embedding of a bona fide batch.

        Components very close to zero are nudged away from it: a near-zero
        coordinate lets the encoder drive that dimension to zero and win on the
        objective without learning anything, which is the documented Deep-SVDD
        collapse.
        """
        z = self.embed(wav).mean(dim=0)
        z[(z.abs() < eps) & (z < 0)] = -eps
        z[(z.abs() < eps) & (z >= 0)] = eps
        self.center.copy_(z)
        self.center_initialised.fill_(1.0)
        return self.center
