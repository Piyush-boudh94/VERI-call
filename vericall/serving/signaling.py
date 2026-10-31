"""WebRTC signalling: the smallest relay that lets two browsers place a call.

Why this exists at all. Everything else in this service scores audio that
something else already captured -- a bundled clip, a file, one browser's
microphone. The problem statement is about *calls*, and a detector that has
never seen two-party audio has not been tested on its own use case. This
module carries the offer/answer/ICE exchange between exactly two peers so the
browsers can build a peer connection; the media itself never touches the
server, which is the point of WebRTC and also the reason this file is short.

**This is a relay, not a participant.** It does not parse SDP, does not
inspect ICE candidates, and does not decrypt anything -- WebRTC media is
SRTP-encrypted end to end and the server has no key. The detector therefore
cannot tap the wire; each browser taps its own decoded audio and streams it to
/ws/{call_id} itself. That is a deliberate consequence of the transport, not a
shortcut, and it is worth stating because it bounds what this system can claim:
VeriCall scores what a *participant* hears, not what a network operator sees.

What the relay must nonetheless do carefully, because it is reachable:

*Treat payloads as opaque but bounded.* SDP is attacker-controlled text from
the peer's point of view. It is never parsed here, only forwarded, and only
after a size check -- an unbounded relay is an amplifier pointed at the other
peer. Offers run a few kilobytes; the cap is generous and still finite.

*Allow only the four verbs the handshake needs.* Anything else is dropped
rather than forwarded, so this cannot be repurposed as a general message bus
between two arbitrary clients.

*Bound the room table.* Rooms are created by whoever connects, so an unbounded
dict is a memory leak with a public trigger. Both the room count and the
occupancy per room are capped, and rooms are deleted when they empty.

*Never let a third party into a two-party call.* The third connection is
refused with a distinct close code rather than silently joining, because
"someone else is already in this room" is something the user needs to be told.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("vericall.signaling")

# The handshake needs exactly these. "bye" is a courtesy so the far end can
# tear down immediately instead of waiting for an ICE timeout.
RELAY_TYPES = frozenset({"offer", "answer", "ice", "bye"})

MAX_SIGNAL_BYTES = 32 * 1024     # an SDP offer with many candidates is ~8 KB
MAX_ROOMS = 64
ROOM_CAPACITY = 2

# Close codes. 4400-4499 is the application range; these are reported by the
# page so a failed join says why rather than just disconnecting.
CLOSE_BAD_ROOM = 4400
CLOSE_UNAUTHORIZED = 4401
CLOSE_ROOM_FULL = 4409
CLOSE_TOO_MANY_ROOMS = 4429


@dataclass
class Peer:
    role: str                    # "caller" (joined first) or "callee"
    socket: object               # the WebSocket; typed loosely to keep this
                                 # module importable without fastapi
    def __hash__(self) -> int:
        return id(self)


@dataclass
class Room:
    name: str
    peers: list[Peer] = field(default_factory=list)


class SignalingHub:
    """Room registry for two-party signalling.

    One lock guards the whole table. The critical sections are dict mutations
    measured in microseconds, and a per-room lock would buy nothing while
    making the join/leave race between two peers much harder to reason about.
    """

    def __init__(self, max_rooms: int = MAX_ROOMS,
                 capacity: int = ROOM_CAPACITY) -> None:
        self.rooms: dict[str, Room] = {}
        self.max_rooms = max_rooms
        self.capacity = capacity
        self._lock = asyncio.Lock()
        self.rejected_full = 0
        self.rejected_capacity = 0

    async def join(self, name: str, socket: object) -> tuple[Optional[Peer], str]:
        """Add `socket` to room `name`.

        Returns (peer, "") on success or (None, reason) on refusal. The caller
        closes the socket with the matching code -- doing it here would make
        this module depend on the web framework for no gain.
        """
        async with self._lock:
            room = self.rooms.get(name)
            if room is None:
                if len(self.rooms) >= self.max_rooms:
                    self.rejected_capacity += 1
                    return None, "too_many_rooms"
                room = Room(name)
                self.rooms[name] = room
            if len(room.peers) >= self.capacity:
                self.rejected_full += 1
                return None, "room_full"
            # First in the room offers. Deciding it server-side removes the
            # classic glare case where both peers create an offer at once.
            role = "caller" if not room.peers else "callee"
            peer = Peer(role=role, socket=socket)
            room.peers.append(peer)
            return peer, ""

    async def leave(self, name: str, peer: Peer) -> list[Peer]:
        """Remove `peer`; return whoever is left so they can be notified."""
        async with self._lock:
            room = self.rooms.get(name)
            if room is None:
                return []
            room.peers = [p for p in room.peers if p is not peer]
            others = list(room.peers)
            if not room.peers:
                # Delete on empty. A room that lingers is a slot nobody can
                # use and a key nobody removes.
                self.rooms.pop(name, None)
            return others

    async def others(self, name: str, peer: Peer) -> list[Peer]:
        async with self._lock:
            room = self.rooms.get(name)
            return [p for p in room.peers if p is not peer] if room else []

    async def occupancy(self, name: str) -> int:
        async with self._lock:
            room = self.rooms.get(name)
            return len(room.peers) if room else 0

    def stats(self) -> dict:
        return {
            "rooms": len(self.rooms),
            "peers": sum(len(r.peers) for r in self.rooms.values()),
            "max_rooms": self.max_rooms,
            "room_capacity": self.capacity,
            "rejected_room_full": self.rejected_full,
            "rejected_too_many_rooms": self.rejected_capacity,
        }


def validate_signal(raw: str) -> tuple[Optional[dict], str]:
    """Parse and admit one signalling message.

    Returns (message, "") or (None, reason). Deliberately strict about the
    envelope and deliberately incurious about the payload: `sdp` and
    `candidate` are forwarded untouched because only the far end's WebRTC
    stack can meaningfully interpret them, and a half-hearted parse here would
    be a second implementation to get wrong.
    """
    if len(raw.encode("utf-8", "ignore")) > MAX_SIGNAL_BYTES:
        return None, "oversize"
    try:
        msg = json.loads(raw)
    except ValueError:
        return None, "not_json"
    if not isinstance(msg, dict):
        return None, "not_object"
    kind = msg.get("type")
    if not isinstance(kind, str) or kind not in RELAY_TYPES:
        return None, "unsupported_type"
    return msg, ""
