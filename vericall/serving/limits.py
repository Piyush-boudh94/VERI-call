"""Admission control, authentication and input limits for the gateway.

Everything here exists because the WebSocket endpoint had none of it: no auth,
no cap on concurrent calls, no frame-size limit, no idle timeout, and a
client-supplied `call_id` used directly as a dictionary key. That is the
standard list from the OWASP WebSocket Security cheat sheet, and each item below
maps to a specific way this service could be taken down or misused by one
client.

The design rule throughout: **refuse early and say why**. An unbounded queue or
an unbounded session table converts a load spike into an out-of-memory kill,
which takes down every call including the ones that were fine. A bounded one
converts it into an immediate, attributable rejection for the marginal caller.

Configuration is environment-driven so a deployment can tighten it without a
code change, and every effective value is reported by /health so the operating
configuration is never a guess.
"""
from __future__ import annotations

import asyncio
import hmac
import os
import re
from dataclasses import dataclass, field
from typing import Container, Optional

# A call id is an opaque handle chosen by the caller, and it lands in a dict
# key, a log line and an audit record. Constrain it rather than sanitise at each
# use: unbounded length is a memory amplifier, and control characters in a log
# line are how log injection works.
# \Z, not $. In Python `$` also matches just BEFORE a trailing newline, so
# "^...{1,64}$" happily accepts "aaa...\n" -- a call id with a newline in it,
# which is precisely the log-injection case this pattern exists to stop. A
# newline in the middle was rejected and one at the end was not, which is the
# kind of hole that survives review. Caught by tests/test_limits.py.
CALL_ID_RE = re.compile(r"\A[A-Za-z0-9._:-]{1,64}\Z")

# 2 s of 16-bit PCM at 16 kHz is 64,000 bytes. A client streaming in real time
# sends far less per frame; anything much larger is either a bug or an attempt
# to make the server allocate on demand.
DEFAULT_MAX_FRAME_BYTES = 64 * 1024

# A call that has sent nothing for this long is gone, whatever TCP thinks. A
# half-open socket that is never collected is the documented WebSocket session
# leak: the object stays in a global table and memory grows until the process
# dies.
DEFAULT_IDLE_TIMEOUT_S = 30.0

# Above the 50-concurrent-call target with headroom, and low enough that the
# refusal happens before the GPU queue saturates.
DEFAULT_MAX_CALLS = 64

# Per-client cap, as a fraction of the total. The global cap bounds the damage
# one *population* of callers can do; without a per-client cap one client can
# still occupy every slot and deny the service to everyone else -- which is
# indistinguishable from an outage to every other tenant, and needs no exploit,
# only a loop. A quarter leaves room for a busy legitimate integrator while
# guaranteeing at least four distinct clients can always get in.
DEFAULT_CLIENT_FRACTION = 0.25

# How long shutdown waits for live calls to end before giving up on them. Must
# be comfortably under the orchestrator's terminationGracePeriodSeconds (30 s by
# default in Kubernetes) or the drain is cut short by SIGKILL and the whole
# exercise is theatre.
DEFAULT_DRAIN_TIMEOUT_S = 20.0


def _int_env(name: str, default: int) -> int:
    try:
        v = int(os.environ.get(name, "") or default)
        return v if v > 0 else default
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        v = float(os.environ.get(name, "") or default)
        return v if v > 0 else default
    except ValueError:
        return default


@dataclass
class Limits:
    max_calls: int = field(default_factory=lambda: _int_env("VERICALL_MAX_CALLS", DEFAULT_MAX_CALLS))
    max_frame_bytes: int = field(default_factory=lambda: _int_env("VERICALL_MAX_FRAME_BYTES", DEFAULT_MAX_FRAME_BYTES))
    idle_timeout_s: float = field(default_factory=lambda: _float_env("VERICALL_IDLE_TIMEOUT_S", DEFAULT_IDLE_TIMEOUT_S))
    drain_timeout_s: float = field(default_factory=lambda: _float_env("VERICALL_DRAIN_TIMEOUT_S", DEFAULT_DRAIN_TIMEOUT_S))

    @property
    def max_per_client(self) -> int:
        explicit = _int_env("VERICALL_MAX_CALLS_PER_CLIENT", 0)
        if explicit:
            return explicit
        return max(1, int(self.max_calls * DEFAULT_CLIENT_FRACTION))

    def as_dict(self) -> dict:
        return {"max_calls": self.max_calls,
                "max_calls_per_client": self.max_per_client,
                "max_frame_bytes": self.max_frame_bytes,
                "idle_timeout_s": self.idle_timeout_s,
                "drain_timeout_s": self.drain_timeout_s}


class Admission:
    """Caps concurrent calls, and refuses the marginal one immediately.

    A semaphore rather than a queue, deliberately. Queueing a 51st call behind
    50 live ones does not help anybody: the caller waits without knowing why,
    the audio it is buffering goes stale, and when it finally runs it is scoring
    a conversation that has moved on. Refusing lets the client fail over.
    """

    def __init__(self, limits: Limits):
        self.limits = limits
        self._active = 0
        self._per_client: dict[str, int] = {}
        self._lock = asyncio.Lock()
        self.rejected = 0
        self.rejected_per_client = 0
        self.peak = 0

    async def acquire(self, client: Optional[str] = None) -> bool:
        """Take a slot. `client` is the identity the per-client cap applies to.

        The two checks are deliberately separate counters: "we are full" and
        "you personally are using too much of it" are different operational
        situations. The first says add capacity; the second says one caller is
        misbehaving, and conflating them hides the second inside the first.
        """
        async with self._lock:
            if self._active >= self.limits.max_calls:
                self.rejected += 1
                return False
            if client is not None:
                used = self._per_client.get(client, 0)
                if used >= self.limits.max_per_client:
                    self.rejected_per_client += 1
                    return False
                self._per_client[client] = used + 1
            self._active += 1
            self.peak = max(self.peak, self._active)
            return True

    async def release(self, client: Optional[str] = None) -> None:
        async with self._lock:
            self._active = max(0, self._active - 1)
            if client is not None and client in self._per_client:
                # Delete at zero rather than leaving a 0 behind: this dict is
                # keyed by client-supplied identity, so an entry that outlives
                # its calls is an unbounded memory leak with an attacker
                # choosing the keys.
                left = self._per_client[client] - 1
                if left > 0:
                    self._per_client[client] = left
                else:
                    del self._per_client[client]

    @property
    def active(self) -> int:
        return self._active

    def stats(self) -> dict:
        return {"active": self._active, "peak": self.peak,
                "rejected": self.rejected,
                "rejected_per_client": self.rejected_per_client,
                "distinct_clients": len(self._per_client),
                "limit": self.limits.max_calls,
                "limit_per_client": self.limits.max_per_client}


def valid_call_id(call_id: str) -> bool:
    return bool(CALL_ID_RE.match(call_id or ""))


def auth_token() -> Optional[str]:
    """The shared secret, or None if the deployment has not set one."""
    tok = os.environ.get("VERICALL_AUTH_TOKEN", "").strip()
    return tok or None


def check_auth(presented: Optional[str]) -> bool:
    """Constant-time comparison against the configured token.

    `hmac.compare_digest` rather than `==` because a plain string comparison
    returns as soon as it finds a differing byte, which leaks the length of the
    matching prefix through timing. That is a real attack against a fixed shared
    secret, and the fix costs nothing.

    A deployment with no token configured accepts everything. That is the wrong
    default for production and the right one for a laptop, so it is allowed and
    reported loudly in /health rather than silently assumed either way.
    """
    expected = auth_token()
    if expected is None:
        return True
    return bool(presented) and hmac.compare_digest(presented, expected)


# Channels a client may declare. Deliberately a separate concept from the
# channels the runtime detector can RECOGNISE.
UNDETECTABLE_CHANNELS = frozenset({"opus"})


def resolve_channel(
    declared: Optional[str], available: Container[str]
) -> tuple[Optional[str], Optional[str]]:
    """Validate a client-declared channel against the loaded buckets.

    Returns (channel, complaint). `channel` is None when nothing usable was
    declared, and the caller then falls back to detection as before.

    Most channels are a property of the audio and are measured, not declared --
    that is what `vericall/audio/codec_detect.py` is for, and taking a client's
    word over a measurement would be strictly worse. Opus is the exception that
    forces this function to exist: it is wideband and loses no frames, so it is
    indistinguishable from clean audio by measurement (97-98% "clean",
    docs/RESULTS.md section 23) while being the channel that costs Branch A2
    nine tenths of its information. It cannot be detected, so it is declared,
    and the browser placing the WebRTC call is the one component that knows.

    An unknown value is **ignored, not rejected**, and the distinction is the
    point. This string is attacker-controlled, and the worst a wrong bucket can
    do is mis-scale an LLR -- while refusing the call outright would let anyone
    hang up anyone else's call by appending a query parameter. A scale error is
    recoverable; a denial of service is not. The complaint is returned so the
    caller can log it without echoing the attacker's string.
    """
    if declared is None:
        return None, None
    if not isinstance(declared, str) or not declared:
        return None, "empty channel"
    if declared not in available:
        # Never echo the value: it is attacker-controlled and this is exactly
        # how log injection works. Length only, as with call_id.
        return None, f"unknown channel ({len(declared)} bytes)"
    return declared, None
