"""Admission control, auth and input limits.

Each test here corresponds to a way one client could degrade or take over the
service for everybody else. They are cheap, they run without a GPU, and they
exist because the endpoint originally had none of these checks: no auth, no cap
on concurrent calls, no frame-size limit, no idle timeout, and a client-supplied
call_id used directly as a dict key.
"""
from __future__ import annotations

import asyncio
import os

import pytest

from vericall.serving.limits import (
    Admission, Limits, check_auth, valid_call_id,
)


# ------------------------------------------------------------------ call ids

@pytest.mark.parametrize("cid", ["abc", "call-1", "a.b:c_d", "A" * 64, "0"])
def test_reasonable_call_ids_are_accepted(cid):
    assert valid_call_id(cid)


@pytest.mark.parametrize("cid", [
    "",                      # empty -> would key the session table on ""
    "A" * 65,                # unbounded length is a memory amplifier
    "../../etc/passwd",      # path-ish, in case an id ever reaches a filename
    "a b",                   # whitespace
    "call\nid",              # newline -> log injection
    "call\x00id",            # NUL
    "id;DROP TABLE",         # punctuation we have no reason to allow
    "callé",                 # non-ASCII: no use case, plenty of encoding edges
])
def test_hostile_call_ids_are_rejected(cid):
    assert not valid_call_id(cid)


def test_call_id_rejection_does_not_depend_on_length_alone():
    """A 64-char id is fine; a 64-char id with a newline is not."""
    assert valid_call_id("a" * 64)
    assert not valid_call_id("a" * 63 + "\n")


# ----------------------------------------------------------------- auth

def test_no_configured_token_accepts_everything(monkeypatch):
    """The laptop default. Wrong for production, which is why /health says so."""
    monkeypatch.delenv("VERICALL_AUTH_TOKEN", raising=False)
    assert check_auth(None)
    assert check_auth("anything")


def test_configured_token_is_required_and_must_match(monkeypatch):
    monkeypatch.setenv("VERICALL_AUTH_TOKEN", "s3cret")
    assert check_auth("s3cret")
    assert not check_auth("s3crev")
    assert not check_auth("s3cre")          # prefix must not pass
    assert not check_auth("s3secret")
    assert not check_auth(None)
    assert not check_auth("")


def test_blank_token_env_is_treated_as_unset(monkeypatch):
    """A deployment that exports VERICALL_AUTH_TOKEN= would otherwise require
    the empty string as a password, which no client sends -- a lockout that
    looks like a bug in the client."""
    monkeypatch.setenv("VERICALL_AUTH_TOKEN", "   ")
    assert check_auth(None)


# ------------------------------------------------------------- admission

def test_admission_caps_concurrency_and_refuses_the_marginal_call():
    async def go():
        adm = Admission(Limits(max_calls=3))
        assert all([await adm.acquire() for _ in range(3)])
        assert not await adm.acquire()
        assert adm.active == 3
        assert adm.rejected == 1
        await adm.release()
        assert await adm.acquire()           # a slot freed is a slot reusable
    asyncio.run(go())


def test_release_without_acquire_cannot_drive_active_negative():
    """A double-release in a finally block would otherwise manufacture capacity
    the service does not have."""
    async def go():
        adm = Admission(Limits(max_calls=2))
        await adm.release()
        await adm.release()
        assert adm.active == 0
        assert await adm.acquire()
        assert await adm.acquire()
        assert not await adm.acquire()
    asyncio.run(go())


def test_admission_is_safe_under_concurrent_acquire():
    """The check and the increment must be atomic together. Without the lock,
    N coroutines can each read active < limit before any of them increments,
    and all N get in."""
    async def go():
        adm = Admission(Limits(max_calls=10))
        got = await asyncio.gather(*[adm.acquire() for _ in range(50)])
        assert sum(got) == 10
        assert adm.active == 10
        assert adm.rejected == 40
    asyncio.run(go())


def test_admission_peak_is_tracked_for_capacity_planning():
    async def go():
        adm = Admission(Limits(max_calls=8))
        for _ in range(5):
            await adm.acquire()
        for _ in range(3):
            await adm.release()
        assert adm.active == 2
        assert adm.peak == 5           # peak survives the drain
        assert adm.stats()["limit"] == 8
    asyncio.run(go())


# ------------------------------------------------------------------ limits

def test_limits_read_the_environment(monkeypatch):
    monkeypatch.setenv("VERICALL_MAX_CALLS", "99")
    monkeypatch.setenv("VERICALL_MAX_FRAME_BYTES", "4096")
    monkeypatch.setenv("VERICALL_IDLE_TIMEOUT_S", "12.5")
    lim = Limits()
    assert lim.max_calls == 99
    assert lim.max_frame_bytes == 4096
    assert lim.idle_timeout_s == 12.5


@pytest.mark.parametrize("bad", ["0", "-5", "not-a-number", ""])
def test_nonsense_limits_fall_back_to_the_default(monkeypatch, bad):
    """A max_calls of 0 or -1 would refuse every call, and a typo in a
    deployment manifest should not be able to do that silently."""
    monkeypatch.setenv("VERICALL_MAX_CALLS", bad)
    assert Limits().max_calls == 64


# ------------------------------------------------- per-client fair share

def test_one_client_cannot_occupy_every_slot():
    """The global cap bounds what a population can do. Without a per-client cap
    a single client still denies the service to everyone else, which needs no
    exploit -- only a loop -- and is indistinguishable from an outage to every
    other tenant."""
    async def go():
        adm = Admission(Limits(max_calls=8))          # per-client = 2
        assert adm.limits.max_per_client == 2
        assert await adm.acquire("noisy")
        assert await adm.acquire("noisy")
        assert not await adm.acquire("noisy")
        assert adm.rejected_per_client == 1
        assert adm.rejected == 0, "this is not 'server full'"
        assert await adm.acquire("quiet"), "another client must still get in"
    asyncio.run(go())


def test_releasing_frees_the_clients_own_slot():
    async def go():
        adm = Admission(Limits(max_calls=8))
        await adm.acquire("a"); await adm.acquire("a")
        assert not await adm.acquire("a")
        await adm.release("a")
        assert await adm.acquire("a")
    asyncio.run(go())


def test_client_table_does_not_grow_without_bound():
    """The key is client-supplied identity, so an entry that outlives its calls
    is a memory leak with an attacker choosing the keys."""
    async def go():
        adm = Admission(Limits(max_calls=64))
        for i in range(500):
            assert await adm.acquire(f"client-{i}")
            await adm.release(f"client-{i}")
        assert adm.stats()["distinct_clients"] == 0
        assert adm.active == 0
    asyncio.run(go())


def test_server_full_still_beats_the_per_client_cap():
    """When the server is full nobody gets in, including a client well under
    its own share -- and it must be counted as 'full', not as their fault."""
    async def go():
        adm = Admission(Limits(max_calls=4))          # per-client = 1
        for c in ("a", "b", "c", "d"):
            assert await adm.acquire(c)
        assert not await adm.acquire("e")
        assert adm.rejected == 1
        assert adm.rejected_per_client == 0
    asyncio.run(go())


def test_explicit_per_client_limit_overrides_the_fraction(monkeypatch):
    monkeypatch.setenv("VERICALL_MAX_CALLS_PER_CLIENT", "3")
    assert Limits(max_calls=64).max_per_client == 3


# ------------------------------------------------------- declared channel

from vericall.serving.limits import UNDETECTABLE_CHANNELS, resolve_channel

BUCKETS = {"clean", "g711", "noisy_line", "opus", "pooled"}


def test_no_declaration_falls_back_to_detection():
    """Most channels are a property of the audio and should be MEASURED.
    Declaring nothing must leave the detector in charge."""
    assert resolve_channel(None, BUCKETS) == (None, None)


def test_a_known_channel_is_accepted():
    assert resolve_channel("opus", BUCKETS) == ("opus", None)
    assert resolve_channel("g711", BUCKETS) == ("g711", None)


def test_an_unknown_channel_is_ignored_not_rejected():
    """The distinction is the point. The string is attacker-controlled: the
    worst a wrong bucket does is mis-scale an LLR, while refusing the call
    would let anyone hang up anyone else's call with a query parameter. A
    scale error is recoverable; a denial of service is not."""
    chan, complaint = resolve_channel("bogus", BUCKETS)
    assert chan is None
    assert complaint and "unknown channel" in complaint


def test_the_complaint_never_echoes_the_value():
    """Attacker-controlled strings in log lines are how log injection works.
    Length only, the same rule call_id follows."""
    evil = "clean\ninjected log line"
    chan, complaint = resolve_channel(evil, BUCKETS)
    assert chan is None
    assert "injected" not in complaint
    assert "clean" not in complaint
    assert str(len(evil)) in complaint


def test_empty_string_is_not_a_declaration():
    assert resolve_channel("", BUCKETS) == (None, "empty channel")


def test_a_channel_with_no_loaded_bucket_is_refused():
    """Selecting a bucket that was never loaded would fall through to the
    default silently, which is the outcome the declaration exists to avoid."""
    chan, complaint = resolve_channel("opus", {"clean", "pooled"})
    assert chan is None and "unknown channel" in complaint


def test_opus_is_marked_undetectable():
    """Recorded in code, not only in prose: this is the channel that cannot be
    measured from the audio and therefore has to be declared."""
    assert "opus" in UNDETECTABLE_CHANNELS
    for detectable in ("clean", "g711", "noisy_line"):
        assert detectable not in UNDETECTABLE_CHANNELS
