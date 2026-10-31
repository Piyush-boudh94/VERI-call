"""Domain-adversarial training, for the failure this project could not fix by
adding data.

docs/RESULTS.md §16 measured the problem and named three untested fixes. The
problem: a model fitted on one corpus is at chance on another, and **naive
concatenation does not combine, it trades** -- mixing ASVspoof with RTCFake gave
the best in-domain number in the project and destroyed WaveFake outright
(30.08% -> 49.89%), even with ASVspoof at 64% of the training rows. The
mechanism offered there was that a naive mixture lets the model take whichever
cue is cheapest across the mixture, and "which corpus is this" is extremely
cheap.

So make that cue unusable. A domain classifier is attached to the pooled
representation through a **gradient reversal layer**: the domain head is trained
to predict the corpus, and the gradient flowing back into the encoder is
negated, so the encoder is trained to make the corpus *unpredictable* while the
spoof classifier still needs it to separate bona fide from spoof. The
equilibrium is a representation that carries synthesis evidence and not corpus
identity (Ganin & Lempitsky, "Unsupervised Domain Adaptation by
Backpropagation", ICML 2015).

**Why this is the right shape of fix here rather than more data.** The corpora
disagree about what bona fide *sounds like* -- studio ASVspoof, single-speaker
LJSpeech, RTC-transmitted telephony -- so more of any one of them sharpens a
corpus-specific boundary. Adversarial training attacks the boundary itself.

**The honest risk, which is why lambda is ramped and measured rather than set.**
If corpus membership is genuinely correlated with the label, forcing
corpus-invariance destroys signal along with the shortcut. That is why lambda
starts at zero and grows: the spoof head gets to learn something real before the
adversary starts removing information, and the eval matrix says whether the
trade paid.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class _GradientReversal(torch.autograd.Function):
    """Identity forward, negated-and-scaled gradient backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        # view, not clone: forward is an identity and copying the batch here
        # would be pure waste at every step.
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):                                  # type: ignore[override]
        return -ctx.lambd * grad, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return _GradientReversal.apply(x, lambd)


class DomainHead(nn.Module):
    """Predicts which corpus a representation came from.

    Deliberately small, and the balance matters in both directions. A large
    adversary wins early, saturates its loss and stops producing useful
    gradient, so training silently reduces to the naive mixture this exists to
    avoid. Measured while writing tests/test_dann.py: a two-layer adversary
    against a one-layer encoder with no competing objective drove domain loss
    from 0.75 down to 0.01 -- the adversary winning outright, which is the
    opposite of what the reversal is for. Keep it small, ramp lambda, and judge
    the run by cross-corpus EER rather than by the domain loss curve.
    """

    def __init__(self, in_dim: int, n_domains: int, hidden: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_domains),
        )

    def forward(self, pooled: torch.Tensor, lambd: float) -> torch.Tensor:
        return self.net(grad_reverse(pooled, lambd))


def dann_lambda(step: int, total: int, max_lambda: float = 1.0,
                gamma: float = 10.0) -> float:
    """Ganin's schedule: 2/(1+exp(-gamma*p)) - 1, ramped 0 -> max_lambda.

    Zero at the start on purpose. An adversary that starts at full strength
    removes information before the spoof head has learned to use any, and the
    run converges to a representation that is invariant and useless -- which
    looks like successful adaptation right up until you read the EER.
    """
    if total <= 0:
        return max_lambda
    p = min(1.0, max(0.0, step / total))
    return max_lambda * (2.0 / (1.0 + math.exp(-gamma * p)) - 1.0)
