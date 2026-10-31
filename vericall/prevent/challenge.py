"""Challenge-response liveness.

The detection branches ask "does this audio look synthetic?". A challenge asks a
different and complementary question: **can whoever is on this call produce an
unpredictable utterance, right now, at conversational latency?**

That difference is what makes it a *prevention* mechanism rather than another
detector. It changes the attacker's problem:

  - A **pre-recorded or pre-generated** clone cannot answer at all. The content
    was fixed before the challenge existed.
  - A **real-time** clone can answer, but it must run TTS in the loop, which
    costs latency on top of human reaction time. The response arriving is
    therefore not sufficient -- *when* it arrives carries the signal.
  - A **genuine caller** answers within human reaction time.

So the decision is a joint test on content and timing, and the timing half is
the part an improving generator does not automatically defeat: faster TTS
narrows the gap but the attacker still pays the round trip that a human does
not.

Honest limits, stated because a challenge is easy to oversell:
  - An attacker with a low-latency pipeline and a cooperative victim on another
    line (a relay attack) defeats the timing test. Challenge-response raises
    cost; it does not close the channel.
  - The latency threshold is a deployment parameter. It is set here from human
    reaction-time literature, NOT fitted on data we have, and is marked as such.
  - Verifying the *content* of the response needs ASR, which VeriCall
    deliberately does not run (no transcript is ever produced). The interface
    below therefore takes a caller-supplied verification result rather than
    pretending to transcribe.
"""
from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# Simple human reaction to an unexpected spoken prompt runs ~0.2-0.4 s; framing
# a short spoken answer takes longer. 2.5 s is a deliberately generous ceiling
# so that a nervous or elderly caller is not treated as an attacker. It is a
# STARTING POINT for per-deployment tuning, not a measured value.
HUMAN_CEILING_S = 2.5
SUSPICIOUS_FLOOR_S = 0.25       # faster than human reaction: machine-in-the-loop


class ChallengeKind(str, Enum):
    DIGITS = "digits"            # "read back 4 7 1 9"
    PHRASE = "phrase"            # an unusual word pair, hard to pre-generate
    ARITHMETIC = "arithmetic"    # "what is six plus three" -- needs comprehension


class ChallengeResult(str, Enum):
    PASSED = "passed"
    FAILED_CONTENT = "failed_content"
    FAILED_TIMEOUT = "failed_timeout"
    SUSPICIOUS_LATENCY = "suspicious_latency"
    PENDING = "pending"


_WORDS = ("harbour", "velvet", "cinnamon", "lantern", "pebble", "thunder",
          "marigold", "compass", "walnut", "quartz", "meadow", "saffron")


@dataclass
class Challenge:
    """One issued challenge, awaiting a response."""

    call_id: str
    kind: ChallengeKind
    prompt: str
    expected: str
    issued_at: float
    timeout_s: float = 10.0
    result: ChallengeResult = ChallengeResult.PENDING
    latency_s: Optional[float] = None
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "call_id": self.call_id, "kind": self.kind.value,
            "prompt": self.prompt, "result": self.result.value,
            "latency_s": None if self.latency_s is None else round(self.latency_s, 3),
            "notes": self.notes,
        }


def issue(call_id: str, kind: ChallengeKind = ChallengeKind.DIGITS,
          now: Optional[float] = None) -> Challenge:
    """Create an unpredictable challenge.

    `secrets` rather than `random`: the whole mechanism rests on the attacker
    being unable to predict or pre-generate the prompt, so a seedable PRNG would
    quietly undermine the one property that matters.
    """
    now = time.monotonic() if now is None else now
    if kind is ChallengeKind.DIGITS:
        digits = "".join(secrets.choice("0123456789") for _ in range(4))
        return Challenge(call_id, kind,
                         f"Please read back these digits: {' '.join(digits)}",
                         digits, now)
    if kind is ChallengeKind.PHRASE:
        a, b = secrets.choice(_WORDS), secrets.choice(_WORDS)
        return Challenge(call_id, kind, f'Please say the words "{a} {b}".',
                         f"{a} {b}", now)
    x, y = secrets.randbelow(8) + 2, secrets.randbelow(8) + 2
    return Challenge(call_id, kind, f"What is {x} plus {y}?", str(x + y), now)


def verify(ch: Challenge, content_matches: bool,
           responded_at: Optional[float] = None) -> Challenge:
    """Score a response on content AND timing.

    `content_matches` is supplied by the caller's own verification path (DTMF
    keypad entry, an agent confirming by ear, or an external ASR service).
    VeriCall does not transcribe: producing a transcript would break the
    no-content-retention property the privacy design rests on.
    """
    responded_at = time.monotonic() if responded_at is None else responded_at
    ch.latency_s = max(0.0, responded_at - ch.issued_at)

    if ch.latency_s > ch.timeout_s:
        ch.result = ChallengeResult.FAILED_TIMEOUT
        ch.notes.append(
            f"no usable response within {ch.timeout_s:.0f}s -- consistent with "
            "pre-recorded or pre-generated audio, which cannot answer a prompt "
            "that did not exist when it was made")
        return ch

    if not content_matches:
        ch.result = ChallengeResult.FAILED_CONTENT
        ch.notes.append("response did not match the challenge")
        return ch

    if ch.latency_s < SUSPICIOUS_FLOOR_S:
        ch.result = ChallengeResult.SUSPICIOUS_LATENCY
        ch.notes.append(
            f"correct but answered in {ch.latency_s:.2f}s, below the "
            f"{SUSPICIOUS_FLOOR_S}s floor for human reaction -- machine in the loop")
        return ch

    ch.result = ChallengeResult.PASSED
    if ch.latency_s > HUMAN_CEILING_S:
        ch.notes.append(
            f"passed, but {ch.latency_s:.2f}s exceeds the {HUMAN_CEILING_S}s "
            "human ceiling; consistent with real-time synthesis in the loop. "
            "Threshold is a deployment parameter, not a measured value.")
    return ch


def risk_adjustment(ch: Challenge) -> float:
    """Evidence in nats, to fold into the running risk score.

    Deliberately asymmetric. A pass is weak evidence of innocence -- a genuine
    caller passes, but so does a competent real-time attacker -- while a
    timeout is strong evidence that nobody is actually there to answer.
    Treating them symmetrically would let an attacker buy a large innocence
    bonus for the price of one successful response.
    """
    return {
        ChallengeResult.PASSED: -1.5,
        ChallengeResult.SUSPICIOUS_LATENCY: +2.0,
        ChallengeResult.FAILED_CONTENT: +3.0,
        ChallengeResult.FAILED_TIMEOUT: +4.0,
        ChallengeResult.PENDING: 0.0,
    }[ch.result]
