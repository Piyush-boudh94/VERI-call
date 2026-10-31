"""Prevention: turning a risk score into an intervention.

PS 26104 asks for detection **and prevention**. Everything upstream of this
module answers "is this synthetic?"; nothing acted on the answer. The fusion
engine emits a tier and an action string, but a string is not an intervention.

The design point: **the same risk score should produce different actions
depending on what the caller is trying to do.** A 4-nat score on a balance
enquiry is not worth interrupting anyone over; the same score on a large
transfer is. That is not a detector parameter, it is a workflow decision, and it
is the only lever that actually fixes the base-rate problem -- at 1e-4 prevalence
no achievable detector FPR makes an unconditioned alert precise, but conditioning
on transaction context changes the prior by two orders of magnitude.

What this module does NOT do, stated plainly: it does not send an SMS, place a
call, or freeze an account. Those are integrations owned by the deploying bank.
It decides *what should happen* and produces an auditable record saying so. The
boundary is deliberate -- a hackathon build that claimed to move money would be
lying.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ..fusion.config import cost_optimal_threshold


class Action(str, Enum):
    """What the call flow should actually do."""

    ALLOW = "allow"                      # nothing; log only
    SILENT_FLAG = "silent_flag"          # record for review, caller unaware
    CHALLENGE = "challenge"              # liveness challenge-response in-band
    STEP_UP = "step_up"                  # out-of-band: OTP, app push, callback
    HOLD = "hold"                        # block the action, keep the call
    FRAUD_DESK = "fraud_desk"            # transfer to a human specialist


@dataclass(frozen=True)
class TransactionContext:
    """What the caller is asking for, and what being wrong costs.

    `prevalence` is the per-context prior probability that a call attempting
    this action is an impersonation. It is deliberately per-context: fraud does
    not distribute itself evenly across balance enquiries and large transfers.
    """

    name: str
    prevalence: float           # P(impersonation | this action attempted)
    cost_miss: float            # currency cost of allowing a real attack
    cost_false_alarm: float     # cost of challenging a genuine customer
    reversible: bool = True     # can the action be undone after the fact?
    requires_enrollment: bool = False


# Defaults are illustrative and MUST be re-fit per deployment. They encode a
# real asymmetry rather than a guess at magnitude: irreversible, high-value
# actions carry a miss cost orders of magnitude above the cost of a challenge.
CONTEXTS: dict[str, TransactionContext] = {
    "balance_enquiry": TransactionContext("balance_enquiry", 1e-5, 0.0, 35.0),
    "statement_request": TransactionContext("statement_request", 1e-5, 100.0, 35.0),
    "contact_change": TransactionContext("contact_change", 1e-3, 50_000.0, 35.0,
                                         reversible=False),
    "payee_add": TransactionContext("payee_add", 2e-3, 100_000.0, 35.0,
                                    reversible=False),
    "transfer_small": TransactionContext("transfer_small", 5e-4, 50_000.0, 35.0,
                                         reversible=False),
    "transfer_large": TransactionContext("transfer_large", 1e-2, 500_000.0, 35.0,
                                         reversible=False),
    "inbound_unclassified": TransactionContext("inbound_unclassified", 1e-4,
                                               200_000.0, 35.0),
}


@dataclass
class Decision:
    action: Action
    reason: str
    context: str
    tier: int
    risk: float
    threshold: float
    identity_conflict: bool = False
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "action": self.action.value, "reason": self.reason,
            "context": self.context, "tier": self.tier,
            "risk_nats": round(self.risk, 3),
            "threshold_nats": round(self.threshold, 3),
            "identity_conflict": self.identity_conflict,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class PreventionPolicy:
    """Maps (risk, identity, context) to an intervention."""

    challenge_margin: float = 1.0    # nats below threshold where a challenge is
                                     # cheaper than a step-up
    min_confidence: float = 0.4      # below this, do not act on the score alone

    def decide(
        self,
        *,
        risk_nats: float,
        tier: int,
        confidence: float,
        context: str = "inbound_unclassified",
        identity_llr: Optional[float] = None,
        enrolled: bool = False,
    ) -> Decision:
        ctx = CONTEXTS.get(context, CONTEXTS["inbound_unclassified"])
        notes: list[str] = []

        # Cost-optimal threshold for THIS action, not a global one. Clamped --
        # for high-value contexts the raw optimum sits below the floor where
        # calibration error (+-0.5-1 nat) makes a threshold meaningless, so the
        # real constraint there is agent capacity rather than decision theory.
        threshold = cost_optimal_threshold(
            ctx.prevalence, max(ctx.cost_miss, 1.0), ctx.cost_false_alarm)

        # A convincing clone of an ENROLLED person is the worst cell in the
        # table, not the safest: identity matching and a high spoof score
        # together mean someone is successfully impersonating a known customer.
        identity_conflict = bool(
            enrolled and identity_llr is not None
            and identity_llr > 0 and risk_nats >= threshold)
        if identity_conflict:
            notes.append("spoof-high AND identity-match-high: a convincing "
                         "clone of an enrolled customer")

        if confidence < self.min_confidence:
            notes.append(f"confidence {confidence:.2f} below {self.min_confidence}"
                         " -- not acting on the score alone")
            # Still refuse irreversible high-value actions on low confidence:
            # the safe failure here is friction, not an unrecoverable transfer.
            if not ctx.reversible and ctx.cost_miss >= 100_000:
                return Decision(Action.STEP_UP,
                                "low confidence on an irreversible high-value "
                                "action -- verify out of band", context, tier,
                                risk_nats, threshold, identity_conflict, notes)
            return Decision(Action.SILENT_FLAG, "insufficient confidence",
                            context, tier, risk_nats, threshold,
                            identity_conflict, notes)

        if ctx.cost_miss <= 0.0:
            return Decision(Action.ALLOW, "nothing at stake in this context",
                            context, tier, risk_nats, threshold,
                            identity_conflict, notes)

        if risk_nats < threshold - self.challenge_margin:
            return Decision(Action.ALLOW, "risk below threshold", context, tier,
                            risk_nats, threshold, identity_conflict, notes)

        if risk_nats < threshold:
            # Near the boundary: an in-band challenge costs the caller seconds
            # and needs no second channel, so it is the cheapest way to buy
            # evidence rather than guessing.
            return Decision(Action.CHALLENGE,
                            "near threshold -- buy evidence in-band before "
                            "escalating", context, tier, risk_nats, threshold,
                            identity_conflict, notes)

        if identity_conflict or (not ctx.reversible and ctx.cost_miss >= 500_000):
            return Decision(Action.FRAUD_DESK,
                            "clone of a known customer" if identity_conflict
                            else "irreversible action at the highest value band",
                            context, tier, risk_nats, threshold,
                            identity_conflict, notes)

        if not ctx.reversible:
            return Decision(Action.HOLD,
                            "risk above threshold on an irreversible action",
                            context, tier, risk_nats, threshold,
                            identity_conflict, notes)

        return Decision(Action.STEP_UP, "risk above threshold -- verify out of "
                        "band", context, tier, risk_nats, threshold,
                        identity_conflict, notes)
