"""n-of-m chunk aggregation, and the independence assumption underneath it.

The pattern comes from cryptographic call authentication, where one-second
speech digests are compared against a known-good reference and an alert is
raised when 3 of 5 exceed threshold, with the combined error rate computed from
the binomial distribution over Bernoulli trials.

That math is sound **there** because each digest is an independent comparison
against a cryptographic reference. VeriCall has no reference: each chunk score
is a model output on audio that overlaps its neighbour and shares a speaker, a
channel and a generator with every other chunk in the call. We measured that
correlation directly -- lag-1 autocorrelation +0.824, decaying only to +0.630 by
lag 8 (docs/RESULTS.md). Trials that correlated are not Bernoulli.

This module implements the vote, and implements the binomial prediction
alongside it, so the gap between predicted and empirical error can be measured
rather than assumed. `effective_m` gives the honest window size to use once the
correlation is accounted for.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def binomial_at_least(n: int, m: int, p: float) -> float:
    """P(at least n of m independent trials fire), each with probability p."""
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be a probability, got {p}")
    if not 1 <= n <= m:
        raise ValueError(f"need 1 <= n <= m, got n={n} m={m}")
    return float(sum(math.comb(m, k) * p ** k * (1 - p) ** (m - k)
                     for k in range(n, m + 1)))


def effective_m(m: int, eta: float) -> float:
    """The number of *independent* observations m correlated chunks are worth.

    eta is the effective-sample-size ratio measured in calibration. At the
    measured eta = 0.080, five consecutive chunks carry 0.4 independent
    observations -- so a '3 of 5' vote is not drawing on five looks at the call,
    it is drawing on well under one.
    """
    if not 0.0 < eta <= 1.0:
        raise ValueError(f"eta must be in (0,1], got {eta}")
    return m * eta


def vote(flags: np.ndarray, n: int = 3, m: int = 5) -> np.ndarray:
    """Sliding n-of-m vote over a boolean per-chunk decision stream.

    Returns a boolean array aligned to the END of each window; the first m-1
    positions are False because the window is not yet full.
    """
    flags = np.asarray(flags).astype(int)
    out = np.zeros(flags.shape, dtype=bool)
    if flags.size < m:
        return out
    csum = np.concatenate([[0], np.cumsum(flags)])
    counts = csum[m:] - csum[:-m]
    out[m - 1:] = counts >= n
    return out


@dataclass(frozen=True)
class AggregationReport:
    n: int
    m: int
    eta: float
    chunk_fp_rate: float
    predicted_call_fp_binomial: float
    empirical_call_fp: float
    inflation: float          # empirical / predicted
    effective_m: float

    def line(self) -> str:
        return (
            f"{self.n}-of-{self.m}: chunk FP {self.chunk_fp_rate*100:6.2f}%  "
            f"binomial predicts {self.predicted_call_fp_binomial*100:7.4f}%  "
            f"empirical {self.empirical_call_fp*100:7.4f}%  "
            f"({self.inflation:6.1f}x optimistic)  "
            f"eff. m = {self.effective_m:.2f}"
        )


def evaluate_aggregation(
    sequences: list[np.ndarray],
    threshold: float,
    n: int = 3,
    m: int = 5,
    eta: float = 1.0,
) -> AggregationReport:
    """Compare the binomial prediction against the empirical rate.

    `sequences` are per-call chunk score streams from BONA FIDE calls, so any
    vote that fires is a false positive.
    """
    all_flags = [np.asarray(s) > threshold for s in sequences if len(s) >= m]
    if not all_flags:
        raise ValueError("no sequence long enough for the window")

    chunk_fp = float(np.mean(np.concatenate(all_flags)))
    predicted = binomial_at_least(n, m, chunk_fp)
    fired = [bool(vote(f, n, m).any()) for f in all_flags]
    # per-window empirical rate, comparable to the per-window binomial figure
    windows = np.concatenate([vote(f, n, m)[m - 1:] for f in all_flags])
    empirical = float(np.mean(windows)) if windows.size else 0.0

    inflation = (empirical / predicted) if predicted > 0 else float("inf")
    return AggregationReport(
        n=n, m=m, eta=eta, chunk_fp_rate=chunk_fp,
        predicted_call_fp_binomial=predicted, empirical_call_fp=empirical,
        inflation=inflation, effective_m=effective_m(m, eta),
    )


def quality_discount(quality: float, duration_s: float,
                     min_quality: float = 0.15,
                     full_duration_s: float = 1.0) -> float:
    """Confidence discount for short or low-quality chunks (FM-7).

    Detection reliability degrades with both utterance length and speech
    quality, and a 1 s fast-lane window is the most exposed configuration in the
    design. A discounted chunk contributes proportionally less rather than
    voting at full weight.
    """
    if quality < min_quality:
        return 0.0
    q = (quality - min_quality) / (1.0 - min_quality)
    d = min(1.0, max(0.0, duration_s / full_duration_s))
    return float(q * d)
