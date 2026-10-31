"""One-class softmax (OC-Softmax) for spoof detection.

Zhang, Jiang & Duan, "One-Class Learning Towards Synthetic Voice Spoofing
Detection", IEEE Signal Processing Letters 2021 (arXiv 2010.13995).

Branch A trains with ordinary two-class cross-entropy, which asks the model to
separate *these* bona fide clips from *these* spoofs. That is the wrong question
for the failure this project keeps measuring. Cross-entropy is free to place the
boundary anywhere that separates the two training classes, including along a
corpus artifact -- and docs/RESULTS.md section 16 measured exactly that
happening: a model at 11.35% in-domain and at chance, sometimes INVERTED, on
another corpus.

OC-Softmax asks a different question: put all bona fide speech inside a compact
cone around a single learned direction, and push everything else outside it by a
margin. The asymmetry matters because the two classes are not symmetric in
reality. "Bona fide human speech" is one coherent thing; "synthetic speech" is
an open, growing set whose next member has not been invented yet. A boundary
built to enclose the closed class generalises to unseen members of the open one
in a way a boundary built to separate two closed sets does not.

    L = mean( softplus( alpha * (m_y - cos(w, x)) * (-1)^y ) )

with y = 0 bona fide, y = 1 spoof, and m_0 > m_1 so the target margin is
tighter than the exclusion margin. Both w and x are L2-normalised, so the score
is a cosine and the geometry is a cone rather than a half-space.

**The score this yields is a cosine, not a logit**, and everything downstream of
Branch A expects a two-class posterior it can turn into an LLR. `to_logits`
does that conversion explicitly rather than letting a cosine be silently
consumed as a logit -- which would not crash, and would put every fitted
threshold in the system onto a different scale.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class OCSoftmax(nn.Module):
    """One-class softmax head. Replaces the 2-way classifier, not the encoder.

    Defaults are the paper's: alpha 20, m_0 0.9, m_1 0.2. m_0 > m_1 is the
    whole point and is asserted rather than assumed -- swapping them inverts the
    objective into "scatter bona fide, cluster spoofs", which trains perfectly
    happily and produces a detector that is exactly backwards.
    """

    def __init__(self, feat_dim: int, alpha: float = 20.0,
                 m_real: float = 0.9, m_fake: float = 0.2):
        super().__init__()
        if not m_real > m_fake:
            raise ValueError(
                f"m_real ({m_real}) must exceed m_fake ({m_fake}); reversing "
                f"them inverts the objective into clustering spoofs and "
                f"scattering bona fide")
        self.alpha = alpha
        self.m_real = m_real
        self.m_fake = m_fake
        # The single direction bona fide speech is gathered around.
        self.center = nn.Parameter(torch.randn(1, feat_dim))
        nn.init.kaiming_uniform_(self.center, a=5 ** 0.5)

    def cosine(self, x: torch.Tensor) -> torch.Tensor:
        w = F.normalize(self.center, dim=1)
        return F.normalize(x, dim=1) @ w.t()          # (B, 1) in [-1, 1]

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple:
        """Returns (loss, score) where score is higher => more spoof.

        Sign is flipped from the cosine so that the score agrees with every
        other branch's convention. Getting this backwards is silent: the EER
        comes out near 1 - EER, which looks like a bad model rather than a
        flipped one.
        """
        cos = self.cosine(x).squeeze(1)
        m = torch.where(y == 0, self.m_real, self.m_fake)
        sign = torch.where(y == 0, 1.0, -1.0)
        loss = F.softplus(self.alpha * (m - cos) * sign).mean()
        return loss, -cos

    def to_logits(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 2) logits, so the rest of the system is unchanged.

        **The cosine is NOT scaled by alpha here, and that is deliberate.**
        alpha is 20, the cosine lies in [-1, 1], and softmax(20 * cos) saturates
        to exactly 0.0 and 1.0 across most of that range -- measured while
        writing tests/test_ocsoftmax.py, where it collapsed distinct scores into
        ties. Saturated posteriors are not a cosmetic problem here: Platt
        calibration is fitted on these values, and a probability pinned at 1.0
        carries no information for it to fit, which is the same failure that
        made every codec-detection quantile land on the same threshold.

        alpha belongs to the LOSS, where its job is to sharpen the gradient. The
        scale of the served score is the calibration layer's job, and handing it
        an unsaturated cosine is what lets it do that job.
        """
        cos = self.cosine(x).squeeze(1)
        return torch.stack([cos, -cos], dim=1)
