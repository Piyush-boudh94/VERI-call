"""Fusion and tier-policy configuration.

Thresholds live here as data, not code, because the whole point of calibrating
branches to a common LLR scale is that the operating point becomes arithmetic on
tenant cost and prior -- changeable at runtime, no retraining.
"""
from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass, field

# Calibration error of +-0.5..1 nat is realistic, so a threshold outside this
# band is not meaningful. See docs/EVALUATION.md on the base-rate analysis: the
# cost-optimal threshold for voice fraud lands BELOW the floor in every realistic
# banking context, so the real constraint on alerting is agent capacity, not
# decision theory. We clamp and say so rather than presenting a fake-precise
# theta*.
THRESHOLD_CLAMP: tuple[float, float] = (1.0, 8.0)


@dataclass(frozen=True, slots=True)
class FusionConfig:
    # --- accumulator time constants -------------------------------------
    tau_fast: float = 1.5      # s; ~N=3 events at a 0.5 s hop
    tau_slow: float = 4.0      # s; ~N=8
    T_cap: float = 10.0        # s; max credited audio the statistic may bank

    # --- branch bookkeeping ---------------------------------------------
    # A branch older than this is dropped from the available set K. Its
    # accumulator is NOT zeroed (so it resumes instantly), it just stops
    # contributing. Hold-last-value past this point is forbidden: it would turn
    # "we have not looked in 4 s" into "we looked and it was fine".
    stale_after: dict[str, float] = field(
        default_factory=lambda: {"A": 2.0, "A2": 2.0, "B": 6.0, "C": 8.0}
    )
    # Jointly fit on dev-set outputs, so correlation between A and A2 is
    # absorbed into the weights rather than double-counted.
    # Branch B carries ZERO weight, on measurement rather than on principle.
    # It was tested in every condition available and reached chance in all of
    # them: ~39% supervised ceiling on clean ASVspoof (RESULTS section 6),
    # no change to fused EER when added to A+A2 (2.52% -> 2.52%, section 8),
    # flat-but-useless under simulated codec degradation (section 10), and
    # 42-53% EER on real RTC transmission including cross-lingual (section 13).
    # The code stays -- the prosody front end is sound and a future corpus may
    # vindicate it -- but it must not contribute to a decision until something
    # measures it doing so. A zero-weight branch is also skipped by
    # FusionState.available(), so it does not perturb subset calibration.
    w: dict[str, float] = field(
        default_factory=lambda: {"A": 1.0, "A2": 0.45, "B": 0.0}
    )
    # Effective-sample-size deflation. Overlapping windows are not independent;
    # eta is fit on held-out CALLS so accumulated variance matches theory.
    #
    # MEASURED, per branch, on 800 synthetic calls from the calib fold -- each
    # one speaker, one label, one generator, scored on the lane that branch
    # actually runs on (scripts/measure_eta_branches.py, docs/eta_branches.json).
    # These replace the unmeasured 0.6/0.6/0.9 that shipped before, and the
    # three do not move together:
    #
    #   A   0.207  fast lane. Lag profile decays slowly (+0.38 -> +0.23 over
    #              eight lags), which is call-level consistency -- one speaker,
    #              one channel, one generator -- not window overlap. Overlap
    #              alone couples adjacent windows only: a 1 s window at a 0.5 s
    #              hop shares no samples with its neighbour twice removed.
    #   A2  0.928  near-independent. Phase residual is a local property, so it
    #              decorrelates within one hop (lag-1 is +0.03).
    #   B   0.565  slow lane, 3 s window at a 1 s hop. Its profile (+0.34,
    #              +0.03, -0.15) tracks the overlap fractions 2/3, 1/3, 0
    #              almost exactly, which is a good sign the measurement is
    #              reading signal rather than an artifact.
    #
    # An earlier measurement reported A = 0.080. It was wrong: the chunker
    # emitted byte-identical windows for a push spanning several ticks, so the
    # autocorrelation was computed on float non-determinism. See
    # vericall/audio/chunker.py.
    eta: dict[str, float] = field(
        default_factory=lambda: {"A": 0.207, "A2": 0.928, "B": 0.565}
    )

    # Branch B is trained one-class on bona fide only, so it has no negative
    # distribution to calibrate against and must never dominate the decision.
    # A prosody model has no business asserting more than ~7:1 odds alone.
    # This is a deliberate safety cap, not a tuning knob.
    kappa_B: float = 2.0
    floor_B: float = -0.5

    # Windows below this are silence / hold music / DTMF: the branch is alive
    # but contributes no evidence.
    quality_floor: float = 0.15

    # (alpha_K, scale_K) per available-branch subset, fit by logistic regression
    # on CACHED dev-set branch outputs. ~6 rows, ~2 KB of JSON, refits in
    # seconds, no GPU. This is what makes a missing branch a dropped term rather
    # than a different model.
    subset_calib: dict[frozenset[str], tuple[float, float]] = field(
        default_factory=dict
    )

    # --- change-point detection -----------------------------------------
    cusum_nu: float = 0.30     # nats, drift/reference value
    cusum_h: float = 4.0       # nats, alarm threshold
    # A change point means "the character of this call CHANGED". On a call
    # that is spoofed from t=0 there is no change -- the first evidence IS
    # the baseline. Without these three gates the CUSUM re-fires every few
    # windows on a sustained spoof and keeps resetting T_eff, so the call can
    # never bank enough evidence to alert. That is the opposite of intended.
    change_min_baseline_s: float = 4.0    # need an established baseline first
    change_refractory_s: float = 8.0      # min gap between change points
    change_baseline_max_rate: float = 0.5  # only fire when we believed it was clean
    tau_regime: float = 20.0   # long-horizon regime estimate, never reset by a
    # change point -- otherwise wiping the accumulators re-arms the baseline
    # check and a sustained spoof declares a bogus 'speaker change' every few
    # seconds, littering the UI timeline with events that did not happen.
    spk_change_d: float = 0.55  # cosine distance for a speaker-turn candidate

    # --- escalation gates -----------------------------------------------
    T_min_escalate: float = 3.0   # s of credited audio before tier > 1
    min_branches: int = 2         # a single branch may never raise an agent alert
    min_confidence_escalate: float = 0.4

    def weight_of(self, branch: str) -> float:
        return self.w.get(branch, 0.0)


@dataclass(frozen=True, slots=True)
class TierPolicy:
    """Tiered response. A tier is just a different C_fa on the same calibrated
    LLR, which is why adding one is a config change and not a code change."""

    enter: dict[int, float] = field(
        default_factory=lambda: {1: 2.5, 2: 4.0, 3: 6.0}
    )
    # ~1.2 nat gap (about 3.3:1 odds of hysteresis) kills boundary chatter.
    exit: dict[int, float] = field(
        default_factory=lambda: {1: 1.5, 2: 2.8, 3: 4.5}
    )
    dwell: dict[int, float] = field(
        default_factory=lambda: {1: 5.0, 2: 8.0, 3: 10.0}
    )
    consec_required: int = 2       # kills single-window spikes
    escalation_cooldown: float = 10.0
    action: dict[int, str] = field(
        default_factory=lambda: {
            0: "monitor",
            1: "step_up",       # in-flow KBA/OTP; cheap false positive
            2: "agent_banner",  # advisory, hold high-value action
            3: "fraud_desk",    # transfer, freeze action
        }
    )

    def target_tier(self, s_lo: float) -> int:
        return max((k for k, thr in self.enter.items() if s_lo >= thr), default=0)


def load_weights(path: str | pathlib.Path,
                 default: dict[str, float] | None = None) -> dict[str, float]:
    """Branch weights for one deployment profile.

    Weights live beside the branch calibrations they were fitted against, not in
    FusionConfig, because they are only meaningful against those calibrations.
    Measured: with A2 served through a single pooled mapping the hand-set 0.45
    was about right, and once A2 gained per-channel buckets (calibration loss
    -93%) the fitted weight nearly doubled to 0.87. Applying one profile's
    weights to another profile's LLRs is the same class of error as applying one
    head's Platt mapping to another head's scores.

    Missing file -> the FusionConfig default, which is what a profile that has
    never been fitted should get.
    """
    p = pathlib.Path(path)
    base = dict(default or {"A": 1.0, "A2": 0.45, "B": 0.0})
    if not p.exists():
        return base
    raw = json.loads(p.read_text()).get("w", {})
    base.update({k: float(v) for k, v in raw.items()})
    return base


def load_subset_calib(
    path: str | pathlib.Path,
) -> dict[frozenset[str], tuple[float, float]]:
    """Read the fitted per-subset table produced by scripts/fit_subsets.py.

    Keys are stored as "A+A2+B" for JSON's sake and rehydrated to frozensets,
    which is what the fusion engine looks up. An unfitted subset falls back to
    (0.0, 1.0) inside `render`, which is correct only for whichever subset the
    branch weights happened to be tuned on -- so shipping without this table
    silently mis-scales every degraded configuration.
    """
    raw = json.loads(pathlib.Path(path).read_text())
    return {frozenset(k.split("+")): (float(v[0]), float(v[1]))
            for k, v in raw.items()}


def cost_optimal_threshold(
    prior: float, c_miss: float, c_fa: float, clamp: bool = True
) -> float:
    """theta* = log[ C_fa (1-pi) / (C_miss pi) ], the LLR threshold minimising
    normalised detection cost.

    Returns the clamped value by default. With realistic voice-fraud costs the
    raw optimum is almost always below the floor -- that finding is the point,
    not a bug to hide.
    """
    if not 0.0 < prior < 1.0:
        raise ValueError(f"prior must be in (0,1), got {prior}")
    if c_miss <= 0 or c_fa <= 0:
        raise ValueError("costs must be positive")
    theta = math.log((c_fa * (1.0 - prior)) / (c_miss * prior))
    if clamp:
        lo, hi = THRESHOLD_CLAMP
        return min(max(theta, lo), hi)
    return theta
