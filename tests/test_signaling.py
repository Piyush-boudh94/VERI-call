"""Signalling-hub behaviour.

No sockets and no browser: the hub is the part with the invariants worth
proving -- exactly two peers per room, rooms deleted when empty, a bounded
table, and a relay that admits four verbs and nothing else.
"""
from __future__ import annotations

import asyncio

import pytest

from vericall.serving.signaling import (
    MAX_SIGNAL_BYTES, RELAY_TYPES, SignalingHub, validate_signal,
)


class FakeSocket:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_json(self, payload: dict) -> None:
        self.sent.append(payload)


def run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------- occupancy

def test_first_peer_is_caller_and_second_is_callee():
    """Roles are assigned server-side to remove the glare case.

    If both browsers decided for themselves, both could create an offer at
    once and the handshake has to be restarted.
    """
    async def go():
        hub = SignalingHub()
        a, err = await hub.join("room", FakeSocket())
        assert err == ""
        b, err = await hub.join("room", FakeSocket())
        assert err == ""
        return a.role, b.role
    assert run(go()) == ("caller", "callee")


def test_third_peer_is_refused_rather_than_joined():
    async def go():
        hub = SignalingHub()
        await hub.join("room", FakeSocket())
        await hub.join("room", FakeSocket())
        peer, err = await hub.join("room", FakeSocket())
        return peer, err, hub.stats()
    peer, err, stats = run(go())
    assert peer is None
    assert err == "room_full"
    assert stats["rejected_room_full"] == 1
    assert stats["peers"] == 2, "the refusal must not have joined anybody"


def test_room_is_deleted_when_the_last_peer_leaves():
    """A room that lingers is a slot nobody can use and a key nobody removes."""
    async def go():
        hub = SignalingHub()
        a, _ = await hub.join("room", FakeSocket())
        b, _ = await hub.join("room", FakeSocket())
        left = await hub.leave("room", a)
        assert [p.role for p in left] == ["callee"]
        assert hub.stats()["rooms"] == 1, "one peer remains, keep the room"
        await hub.leave("room", b)
        return hub.stats()["rooms"]
    assert run(go()) == 0


def test_leaving_frees_the_seat_for_a_new_peer():
    async def go():
        hub = SignalingHub()
        a, _ = await hub.join("room", FakeSocket())
        await hub.join("room", FakeSocket())
        assert (await hub.join("room", FakeSocket()))[1] == "room_full"
        await hub.leave("room", a)
        peer, err = await hub.join("room", FakeSocket())
        return peer is not None, err, peer.role if peer else None
    joined, err, role = run(go())
    assert joined and err == ""
    # The seat is free, and whoever takes it is the callee: someone is already
    # in the room, so they must not also try to offer.
    assert role == "callee"


def test_room_table_is_bounded():
    """Rooms are created by whoever connects, so an unbounded dict is a memory
    leak with a public trigger."""
    async def go():
        hub = SignalingHub(max_rooms=3)
        for i in range(3):
            assert (await hub.join(f"r{i}", FakeSocket()))[1] == ""
        peer, err = await hub.join("r3", FakeSocket())
        return peer, err, hub.stats()
    peer, err, stats = run(go())
    assert peer is None and err == "too_many_rooms"
    assert stats["rooms"] == 3
    assert stats["rejected_too_many_rooms"] == 1


def test_a_full_table_still_admits_a_second_peer_to_an_existing_room():
    """The cap is on ROOMS, not on joins. Capping joins would make the last
    room created un-callable."""
    async def go():
        hub = SignalingHub(max_rooms=1)
        await hub.join("only", FakeSocket())
        return await hub.join("only", FakeSocket())
    peer, err = run(go())
    assert peer is not None and err == ""


def test_leave_on_an_unknown_room_is_not_an_error():
    """The endpoint's finally block runs even when join failed."""
    async def go():
        hub = SignalingHub()
        peer, _ = await hub.join("room", FakeSocket())
        await hub.leave("room", peer)
        return await hub.leave("room", peer)      # again, now gone
    assert run(go()) == []


def test_others_excludes_the_asking_peer():
    async def go():
        hub = SignalingHub()
        a, _ = await hub.join("room", FakeSocket())
        b, _ = await hub.join("room", FakeSocket())
        return [p.role for p in await hub.others("room", a)]
    assert run(go()) == ["callee"]


def test_rooms_are_isolated_from_each_other():
    async def go():
        hub = SignalingHub()
        a, _ = await hub.join("one", FakeSocket())
        await hub.join("two", FakeSocket())
        return await hub.others("one", a)
    assert run(go()) == []


# ------------------------------------------------------------ message gate

@pytest.mark.parametrize("kind", sorted(RELAY_TYPES))
def test_the_four_handshake_verbs_are_admitted(kind):
    msg, err = validate_signal('{"type": "%s"}' % kind)
    assert err == "" and msg["type"] == kind


@pytest.mark.parametrize("raw,reason", [
    ('{"type":"subscribe"}', "unsupported_type"),
    ('{"type":123}', "unsupported_type"),
    ('{"no_type":1}', "unsupported_type"),
    ("not json at all", "not_json"),
    ("[1,2,3]", "not_object"),
    ('"a string"', "not_object"),
    ("null", "not_object"),
])
def test_anything_else_is_refused(raw, reason):
    """This must not be repurposable as a general message bus between two
    arbitrary clients."""
    msg, err = validate_signal(raw)
    assert msg is None and err == reason


def test_oversize_payloads_are_refused_before_parsing():
    """An unbounded relay is an amplifier pointed at the other peer."""
    raw = '{"type":"offer","sdp":"' + "x" * (MAX_SIGNAL_BYTES + 1) + '"}'
    msg, err = validate_signal(raw)
    assert msg is None and err == "oversize"


def test_a_realistic_sdp_offer_fits_comfortably():
    """The cap has to be generous enough for a real offer with candidates.
    A cap that rejects ordinary traffic is an outage, not a control."""
    sdp = "v=0\r\n" + "a=candidate:1 1 udp 2130706431 192.168.1.5 5000 typ host\r\n" * 60
    raw = '{"type":"offer","sdp":%s}' % __import__("json").dumps(sdp)
    msg, err = validate_signal(raw)
    assert err == "" and msg["type"] == "offer"


def test_payload_fields_are_forwarded_untouched():
    """The relay is deliberately incurious: only the far end's WebRTC stack
    can interpret an SDP, and a half-hearted parse here is a second
    implementation to get wrong."""
    msg, err = validate_signal(
        '{"type":"ice","candidate":{"candidate":"a=x","sdpMid":"0"},"extra":7}')
    assert err == ""
    assert msg["candidate"] == {"candidate": "a=x", "sdpMid": "0"}
    assert msg["extra"] == 7
