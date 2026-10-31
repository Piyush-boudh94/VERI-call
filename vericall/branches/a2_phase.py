"""Branch A2 -- phase / group-delay detector.

Why a separate branch: TTS and voice-conversion vocoders reconstruct a waveform
from a magnitude representation, so original phase structure is largely
destroyed and re-synthesised. Branch A keys mostly on magnitude-spectrum
artifacts, which makes phase genuinely complementary rather than redundant --
and it degrades differently under codec compression, which is the point for
telephony.

Architecture: 140-channel phase front end (60 MGDCC + 80 MRP) -> depthwise-
separable 1D convolution stack with squeeze-excitation -> mean+max pooling ->
classifier. Depthwise-separable rather than dense convolution because the branch
runs on the fast lane at 2 Hz per call, so its cost is multiplied by every
concurrent call; SE blocks let it weight the phase channels that matter without
a full attention stack.

NOTE: this module was reconstructed from the shapes in
`checkpoints/branch_A2_seed0.pt` after the original source was overwritten. The
parameterised layers match that checkpoint exactly, but activation choices are
not recoverable from a state_dict, so the branch is RETRAINED against this code
rather than trusting the old weights. Retraining costs ~3 minutes; guessing does
not come with a correctness guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..features.phase import PhaseFeatures
from ..features.spectral import STFTConfig


@dataclass(frozen=True)
class BranchA2Config:
    in_channels: int = 140         # 60 MGDCC (20 x delta x delta-delta) + 80 MRP
    width: int = 128
    se_reduction: int = 8
    dropout: float = 0.2
    kernel: int = 5


class SEBlock(nn.Module):
    """Squeeze-and-excitation over phase-feature channels.

    Group delay is noisy in the channels near spectral nulls and informative in
    others; a per-channel gate is a cheap way to learn that without the cost of
    attention over the time axis.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x.mean(dim=-1))            # squeeze over time
        return x * w.unsqueeze(-1)


def _ds_block(in_ch: int, out_ch: int, kernel: int) -> nn.Sequential:
    """Depthwise-separable conv: depthwise over time, then pointwise mixing.

    The nonlinearity lives INSIDE the block. Stacking these blocks without one
    collapses the whole stack to a single linear map -- a mistake that still
    loads a checkpoint cleanly (activations carry no parameters, so they are
    invisible in a state_dict) while scoring at chance. Verified: without it,
    dev EER was 49.84%.
    """
    return nn.Sequential(
        nn.Conv1d(in_ch, in_ch, kernel, padding=kernel // 2, groups=in_ch),
        nn.Conv1d(in_ch, out_ch, 1),
        nn.BatchNorm1d(out_ch),
        nn.GELU(),
    )


class BranchA2(nn.Module):
    """Phase-domain spoof detector. Consumes raw waveform."""

    def __init__(self, cfg: BranchA2Config = BranchA2Config(),
                 stft: STFTConfig = STFTConfig()):
        super().__init__()
        self.cfg = cfg
        self.features = PhaseFeatures(stft)
        w, k = cfg.width, cfg.kernel
        self.net = nn.Sequential(
            nn.BatchNorm1d(cfg.in_channels),
            _ds_block(cfg.in_channels, w, k),
            _ds_block(w, w, k),
            SEBlock(w, cfg.se_reduction),
            _ds_block(w, w, k),
            _ds_block(w, w, k),
            SEBlock(w, cfg.se_reduction),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(w * 2, w),           # mean + max pooling concatenated
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(w, 2),
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        with torch.no_grad():                  # the front end has no parameters
            x = self.features(wav)
        h = self.net(x)
        pooled = torch.cat([h.mean(dim=-1), h.max(dim=-1).values], dim=-1)
        return self.classifier(pooled)

    @torch.inference_mode()
    def score(self, wav: torch.Tensor) -> torch.Tensor:
        """Spoof posterior. Uncalibrated -- the fusion engine never sees it raw."""
        return torch.softmax(self(wav), dim=-1)[:, 1]
