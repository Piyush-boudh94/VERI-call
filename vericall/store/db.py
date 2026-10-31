"""Call history: the record of what the detector decided, and why.

Until now the system forgot every call the moment it ended. The audit log
(`vericall/prevent/audit.py`) records *prevention decisions*, which is a legal
artefact and deliberately narrow -- one record per action change, no scores, no
timeline. It cannot answer "show me yesterday's flagged calls", which is the
first thing anyone operating this would ask.

**SQLite, on purpose.** It needs no server, the file lives beside the audit log,
and a single process serving 64 concurrent calls writes a few rows per call --
about two orders of magnitude below where SQLite starts to care. Postgres would
be the right answer for several replicas sharing one history; this is one
process, and pretending otherwise would add an operational dependency to a
project whose scratch disk vanishes twice a day.

**What is NOT stored, and why.** No audio, no waveform, no features, no
transcript. Voice is biometric data under the DPDP Act 2023
(docs/THREAT_MODEL.md H4) and a history table is exactly the kind of thing that
quietly becomes a biometric database. The same FORBIDDEN_KEYS discipline the
audit log uses applies here, enforced on write rather than trusted.

Retention is a real setting rather than a promise: `prune()` deletes calls older
than a configured age, and it is the caller's job to run it.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

# Mirrors vericall/prevent/audit.py. A history row must never become a place
# where audio or text lands "just for debugging".
FORBIDDEN_KEYS = {"audio", "pcm", "waveform", "samples", "transcript", "text",
                  "features", "feat", "embedding", "emb", "mfcc", "spectrogram"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    call_id      TEXT PRIMARY KEY,
    started_at   REAL NOT NULL,
    ended_at     REAL,
    context      TEXT,
    client       TEXT,
    max_risk     REAL,
    final_tier   INTEGER,
    alerted      INTEGER NOT NULL DEFAULT 0,
    alert_at     REAL,
    final_action TEXT,
    channel      TEXT,
    branches     TEXT,
    calibrated   INTEGER,
    notes        TEXT
);
CREATE INDEX IF NOT EXISTS calls_started ON calls(started_at DESC);
CREATE INDEX IF NOT EXISTS calls_alerted ON calls(alerted, started_at DESC);

CREATE TABLE IF NOT EXISTS events (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id  TEXT NOT NULL,
    at       REAL NOT NULL,
    kind     TEXT NOT NULL,
    tier     INTEGER,
    risk     REAL,
    detail   TEXT,
    FOREIGN KEY (call_id) REFERENCES calls(call_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS events_call ON events(call_id, at);
"""


class ForbiddenContent(ValueError):
    """Something that looks like speech content was about to be stored."""


def _check(payload: dict[str, Any]) -> None:
    for k in payload:
        if k.lower() in FORBIDDEN_KEYS:
            raise ForbiddenContent(
                f"refusing to store {k!r}: call history must not become a "
                f"biometric or transcript database")


def _resolve_db_path() -> Path:
    """Where the call history lives, resolved against the PROJECT not the CWD.

    This was `Path(os.environ.get("VERICALL_DB") or "logs/vericall.db")` -- a
    relative default, so the database landed wherever the gateway happened to
    be started from. Start it from your home directory and yesterday's call
    history is invisible, in a second file, with no error: /health would report
    `calls: 0` and be telling the truth about the wrong file.

    Same class of defect as the checkpoint path in serving/app.py, where a
    relative override silently resolved past `checkpoints/` and served an
    untrained head. A relative path in configuration should be anchored to
    something stable, and the only stable thing here is the project root.
    """
    root = Path(__file__).resolve().parents[2]
    raw = os.environ.get("VERICALL_DB")
    p = Path(raw) if raw else Path("logs/vericall.db")
    return p if p.is_absolute() else root / p


@dataclass
class CallStore:
    """Thread-safe SQLite call history.

    One connection guarded by a lock rather than a connection pool. Writes here
    are a few rows per call and always happen off the event loop, so contention
    is not the problem; two threads sharing one sqlite3 connection without a
    lock IS a problem, and it surfaces as an intermittent
    "recursive use of cursors" rather than as anything legible.
    """

    path: Path = field(default_factory=lambda: _resolve_db_path())
    _conn: Optional[sqlite3.Connection] = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # WAL so a reader (the history endpoint) never blocks a writer (a live
        # call). Without it a dashboard refresh can stall a call's write.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    # ---------------------------------------------------------------- write

    def open_call(self, call_id: str, context: str = "", client: str = "",
                  channel: str = "", branches: Iterable[str] = (),
                  calibrated: bool = False) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO calls "
                "(call_id, started_at, context, client, channel, branches, "
                " calibrated, alerted) VALUES (?,?,?,?,?,?,?,0)",
                (call_id, time.time(), context, client, channel,
                 json.dumps(sorted(branches)), int(calibrated)))
            self._conn.commit()

    def add_event(self, call_id: str, kind: str, at: float,
                  tier: Optional[int] = None, risk: Optional[float] = None,
                  detail: Optional[dict] = None) -> None:
        """Record a tier change, an alert, or a prevention action.

        Deliberately NOT called per chunk. At 2 Hz per call and 64 calls that
        would be 128 rows/second of mostly identical state; the interesting
        events are the transitions, and a table of transitions is what anyone
        reviewing a call actually reads.
        """
        if detail:
            _check(detail)
        with self._lock:
            self._conn.execute(
                "INSERT INTO events (call_id, at, kind, tier, risk, detail) "
                "VALUES (?,?,?,?,?,?)",
                (call_id, at, kind, tier, risk,
                 json.dumps(detail) if detail else None))
            self._conn.commit()

    def close_call(self, call_id: str, max_risk: Optional[float] = None,
                   final_tier: Optional[int] = None,
                   alerted: bool = False, alert_at: Optional[float] = None,
                   final_action: str = "", notes: Iterable[str] = ()) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE calls SET ended_at=?, max_risk=?, final_tier=?, "
                "alerted=?, alert_at=?, final_action=?, notes=? "
                "WHERE call_id=?",
                (time.time(), max_risk, final_tier, int(alerted), alert_at,
                 final_action, json.dumps(list(notes)), call_id))
            self._conn.commit()

    # ----------------------------------------------------------------- read

    def recent(self, limit: int = 50, alerted_only: bool = False) -> list[dict]:
        q = ("SELECT * FROM calls "
             + ("WHERE alerted=1 " if alerted_only else "")
             + "ORDER BY started_at DESC LIMIT ?")
        with self._lock:
            rows = self._conn.execute(q, (limit,)).fetchall()
        return [self._row(r) for r in rows]

    def call(self, call_id: str) -> Optional[dict]:
        with self._lock:
            r = self._conn.execute(
                "SELECT * FROM calls WHERE call_id=?", (call_id,)).fetchone()
            if r is None:
                return None
            evs = self._conn.execute(
                "SELECT at, kind, tier, risk, detail FROM events "
                "WHERE call_id=? ORDER BY at", (call_id,)).fetchall()
        out = self._row(r)
        out["events"] = [
            {"at": e["at"], "kind": e["kind"], "tier": e["tier"],
             "risk": e["risk"],
             "detail": json.loads(e["detail"]) if e["detail"] else None}
            for e in evs]
        return out

    def stats(self) -> dict:
        with self._lock:
            c = self._conn.execute(
                "SELECT COUNT(*) n, SUM(alerted) a FROM calls").fetchone()
            e = self._conn.execute("SELECT COUNT(*) n FROM events").fetchone()
        total, alerted = c["n"] or 0, c["a"] or 0
        return {"calls": total, "alerted": alerted,
                "alert_rate": round(alerted / total, 4) if total else 0.0,
                "events": e["n"] or 0, "path": str(self.path)}

    # ------------------------------------------------------------ retention

    def prune(self, older_than_days: float = 30.0) -> int:
        """Delete calls older than `older_than_days`. Returns rows removed.

        Retention as a callable rather than a sentence in a policy document.
        Events cascade via the foreign key, which is why PRAGMA foreign_keys is
        on -- SQLite ignores the constraint otherwise and the events table would
        grow forever behind a calls table that looked pruned.
        """
        cutoff = time.time() - older_than_days * 86400.0
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM calls WHERE started_at < ?", (cutoff,))
            self._conn.commit()
            return cur.rowcount

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    @staticmethod
    def _row(r: sqlite3.Row) -> dict:
        d = dict(r)
        for k in ("branches", "notes"):
            if d.get(k):
                try:
                    d[k] = json.loads(d[k])
                except ValueError:
                    pass
        d["alerted"] = bool(d.get("alerted"))
        # `action` means two different things in this system and the names did
        # not distinguish them. The live stream's payload["action"] is a static
        # tier->action lookup (tier 3 -> "fraud_desk"); payload["prevention"]
        # ["action"] is the cost-optimal decision for what the caller is
        # attempting ("step_up"). close_call stores the SECOND one, so the same
        # call read "tier 3 · fraud_desk" live and "step_up" in a history
        # column also headed "action". Both are correct; neither said which it
        # was. The column keeps its name for compatibility with existing rows;
        # this alias is what new readers should use.
        d["final_prevention_action"] = d.get("final_action") or ""
        return d
