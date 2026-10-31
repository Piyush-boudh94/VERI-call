"""Score calibration and the metrics that make the fusion thresholds mean
something.

Branches emit incompatible quantities: A/A2 give supervised posteriors, B gives
a one-class anomaly score with no negative distribution behind it, C gives a
cosine similarity. Feeding those into one fusion stage unconverted is a real
bug, not a nuisance -- their scales differ by orders of magnitude and the
"weights" would silently absorb the mismatch.

Everything here maps a branch's raw score to a log-likelihood ratio in nats, on
a split that is disjoint from training. Pure numpy; runs on the head node.
"""
from __future__ import annotations

import json
import math
import pathlib
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

LOG2 = math.log(2.0)


# ===================================================================== #
# metrics
# ===================================================================== #

def eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Equal error rate. labels: 1 = spoof (positive), 0 = bona fide.

    Returns (eer, threshold). Higher score must mean *more likely spoof*.

    **Ties are collapsed.** A run of identical scores is ONE threshold, not many.
    Walking through tied values as if they were distinct thresholds fabricates
    an ROC curve the data does not support: measured on trimmed-silence
    durations -- nearly all exactly the padding length -- the untied form
    reported 95.67% EER for a feature whose class means differed by 0.8 ms. A
    wholly uninformative score must give exactly 0.5, and it now does.

    The crossing is linearly interpolated rather than snapped to the nearest
    operating point, which matters when few distinct thresholds exist.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    if scores.size == 0 or len(np.unique(labels)) < 2:
        return float("nan"), float("nan")

    order = np.argsort(-scores, kind="mergesort")
    s, y = scores[order], labels[order]
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())

    tp = np.concatenate([[0], np.cumsum(y)])
    fp = np.concatenate([[0], np.cumsum(1 - y)])
    # operating points: the empty prefix, then the end of each run of equal
    # scores, then everything accepted
    keep = np.concatenate([[0], np.flatnonzero(np.diff(s)) + 1, [s.size]])

    fnr = 1.0 - tp[keep] / n_pos
    fpr = fp[keep] / n_neg
    diff = fnr - fpr

    sign_change = np.flatnonzero(diff[:-1] * diff[1:] <= 0)
    if sign_change.size:
        i = int(sign_change[0])
        d0, d1 = diff[i], diff[i + 1]
        t = 0.0 if d0 == d1 else d0 / (d0 - d1)
        value = float(fnr[i] + t * (fnr[i + 1] - fnr[i]))
        j = min(keep[i + 1], s.size - 1)
        return value, float(s[j])

    i = int(np.nanargmin(np.abs(diff)))
    j = min(keep[i], s.size - 1)
    return float((fnr[i] + fpr[i]) / 2.0), float(s[j])


def cllr(llrs: np.ndarray, labels: np.ndarray) -> float:
    """Application-independent cost of the log-likelihood-ratio.

    Cllr = 1.0 means the scores carry no usable information; below 1.0 they do.
    Unlike EER this is sensitive to *calibration*, not just ranking -- which is
    the whole reason the fusion layer needs it.
    """
    llrs = np.asarray(llrs, dtype=float)
    labels = np.asarray(labels).astype(int)
    tar, non = llrs[labels == 1], llrs[labels == 0]
    if tar.size == 0 or non.size == 0:
        return float("nan")
    c_tar = np.mean(np.logaddexp(0.0, -tar)) / LOG2
    c_non = np.mean(np.logaddexp(0.0, non)) / LOG2
    return float((c_tar + c_non) / 2.0)


def min_cllr(scores: np.ndarray, labels: np.ndarray) -> float:
    """Cllr after optimal monotone recalibration (PAV). The gap
    `cllr - min_cllr` is calibration loss: how much is lost to miscalibration
    rather than to the scores being genuinely uninformative."""
    from sklearn.isotonic import IsotonicRegression

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    if len(np.unique(labels)) < 2:
        return float("nan")

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    p = iso.fit_transform(scores, labels)
    eps = 1e-6
    p = np.clip(p, eps, 1.0 - eps)

    # PAV gives posteriors under the empirical prior; divide it out so the
    # result is a prior-free LLR.
    prior = labels.mean()
    prior_odds = prior / (1.0 - prior)
    llr = np.log(p / (1.0 - p)) - math.log(prior_odds)
    return cllr(llr, labels)


def bootstrap_eer_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: Sequence[str],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """EER with a confidence interval, resampling whole CALLS.

    Resampling windows would treat 20 correlated windows from one call as 20
    independent observations and report an interval several times too narrow --
    the same error `eta` corrects for inside the fusion engine.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    groups = np.asarray(groups)

    point, _ = eer(scores, labels)
    uniq = np.unique(groups)
    if uniq.size < 2:
        return point, float("nan"), float("nan")

    index = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        picked = rng.choice(uniq, size=uniq.size, replace=True)
        sel = np.concatenate([index[g] for g in picked])
        e, _ = eer(scores[sel], labels[sel])
        if not math.isnan(e):
            draws.append(e)
    if not draws:
        return point, float("nan"), float("nan")
    lo, hi = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return point, float(lo), float(hi)


# ===================================================================== #
# calibration artifacts
# ===================================================================== #

@dataclass
class PlattCalibration:
    """llr = a * raw + b, with `raw` already in a logit-like domain.

    Fit with prior-weighted logistic regression so the output is prior-free:
    the operational prior is injected once, at decision time, rather than baked
    into every branch.
    """

    a: float
    b: float
    branch: str = ""
    channel_bucket: str = "clean"
    calib_id: str = ""
    n_fit: int = 0
    cllr: float = float("nan")
    min_cllr: float = float("nan")

    def apply(self, raw: float | np.ndarray) -> float | np.ndarray:
        return self.a * np.asarray(raw, dtype=float) + self.b

    @property
    def calibration_loss(self) -> float:
        return self.cllr - self.min_cllr


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def fit_platt(
    raw: np.ndarray,
    labels: np.ndarray,
    *,
    branch: str = "",
    channel_bucket: str = "clean",
    calib_id: str = "",
) -> PlattCalibration:
    """Prior-weighted logistic regression of label on raw score.

    Each class is reweighted to a synthetic 0.5 prior, which is what makes the
    output a prior-free LLR rather than a posterior tied to the training mix
    (ASVspoof LA is ~1:8.8, so this matters a lot).
    """
    from sklearn.linear_model import LogisticRegression

    raw = np.asarray(raw, dtype=float).reshape(-1, 1)
    labels = np.asarray(labels).astype(int)
    if len(np.unique(labels)) < 2:
        raise ValueError("need both classes to fit a calibration")

    lr = LogisticRegression(
        class_weight="balanced", C=1e6, solver="lbfgs", max_iter=1000
    )
    lr.fit(raw, labels)
    a = float(lr.coef_[0][0])
    b = float(lr.intercept_[0])

    llr = a * raw.ravel() + b
    return PlattCalibration(
        a=a, b=b, branch=branch, channel_bucket=channel_bucket,
        calib_id=calib_id, n_fit=int(labels.size),
        cllr=cllr(llr, labels), min_cllr=min_cllr(raw.ravel(), labels),
    )


def one_class_to_llr(
    raw: np.ndarray,
    cohort: np.ndarray,
    *,
    alpha: float = 0.05,
    u_floor: float = 1e-3,
    kappa: float = 2.0,
    floor: float = -0.5,
) -> np.ndarray:
    """Branch B fallback when no spoof labels are available.

    Converts an anomaly score to a bona-fide tail probability against a cohort,
    then to a one-sided surprise measure offset so llr = 0 at the target tail
    `alpha`. Hard-clipped: a one-class branch with no negative distribution
    behind it must never be able to dominate the decision.
    """
    raw = np.asarray(raw, dtype=float)
    cohort = np.sort(np.asarray(cohort, dtype=float))
    # empirical bona-fide tail probability
    u = 1.0 - np.searchsorted(cohort, raw, side="right") / max(cohort.size, 1)
    u = np.maximum(u, u_floor)
    llr = np.log(1.0 / u) - math.log(1.0 / alpha)
    return np.clip(llr, floor, kappa)


# ===================================================================== #
# subset calibration and ESS deflation
# ===================================================================== #

def fit_subset_calibration(
    branch_llrs: Mapping[str, np.ndarray],
    labels: np.ndarray,
    weights: Mapping[str, float],
    subsets: Iterable[Iterable[str]],
) -> dict[frozenset[str], tuple[float, float]]:
    """Fit (alpha_K, scale_K) for each branch-availability subset.

    Two scalars per subset, fit by logistic regression on CACHED dev-set branch
    outputs. No model is retrained and no GPU is touched -- which is what makes
    "a missing branch is a dropped term, not a different model" true in practice
    rather than just on a slide.
    """
    from sklearn.linear_model import LogisticRegression

    labels = np.asarray(labels).astype(int)
    out: dict[frozenset[str], tuple[float, float]] = {}
    for subset in subsets:
        K = frozenset(subset)
        if not K:
            continue
        combined = np.zeros_like(labels, dtype=float)
        for b in K:
            combined = combined + weights.get(b, 0.0) * np.asarray(
                branch_llrs[b], dtype=float
            )
        lr = LogisticRegression(
            class_weight="balanced", C=1e6, solver="lbfgs", max_iter=1000
        )
        lr.fit(combined.reshape(-1, 1), labels)
        scale = float(lr.coef_[0][0])
        alpha = float(lr.intercept_[0]) / scale if scale != 0.0 else 0.0
        out[K] = (alpha, scale)
    return out


def estimate_eta(sequences: Sequence[Sequence[float]], max_lag: int = 8) -> float:
    """Effective-sample-size deflation from within-call autocorrelation.

    ESS = N / (1 + 2 * sum_k rho_k), so eta = ESS/N in (0, 1]. Overlapping
    windows are positively autocorrelated, which drives eta below 1 and widens
    the confidence bound accordingly.

    Must be fit over whole CALLS -- estimating it from pooled windows destroys
    the very correlation structure being measured.
    """
    rhos: list[np.ndarray] = []
    for seq in sequences:
        x = np.asarray(seq, dtype=float)
        if x.size < max_lag + 2:
            continue
        x = x - x.mean()
        denom = float(np.dot(x, x))
        if denom <= 0:
            continue
        rhos.append(
            np.array([float(np.dot(x[:-k], x[k:])) / denom for k in range(1, max_lag + 1)])
        )
    if not rhos:
        return 1.0

    rho = np.mean(np.stack(rhos), axis=0)
    # sum only the leading positive run: once autocorrelation crosses zero the
    # remaining terms are noise (Geyer's initial-positive-sequence rule)
    total = 0.0
    for r in rho:
        if r <= 0:
            break
        total += r
    return float(min(1.0, max(1e-3, 1.0 / (1.0 + 2.0 * total))))


# ===================================================================== #
# persistence
# ===================================================================== #

def save_calibration(path: str | pathlib.Path, cal: PlattCalibration) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(cal), indent=2))


def load_calibration(path: str | pathlib.Path) -> PlattCalibration:
    return PlattCalibration(**json.loads(pathlib.Path(path).read_text()))
