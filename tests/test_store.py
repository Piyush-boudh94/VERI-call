"""Call history store.

The tests that matter here are not "does SQLite work" but the three properties
that make this table safe to keep: it must never hold speech content, it must
actually delete on retention, and concurrent writers must not corrupt it.
"""
from __future__ import annotations

import threading
import time

import pytest

from vericall.store.db import CallStore, ForbiddenContent


@pytest.fixture
def store(tmp_path):
    s = CallStore(path=tmp_path / "t.db")
    yield s
    s.close()


def test_a_call_round_trips(store):
    store.open_call("c1", context="transfer_large", client="1.2.3.4",
                    channel="clean", branches=["A", "A2"], calibrated=True)
    store.add_event("c1", "tier", at=3.0, tier=1, risk=4.2)
    store.add_event("c1", "alert", at=3.5, tier=3, risk=9.1,
                    detail={"reason": "sustained"})
    store.close_call("c1", max_risk=9.1, final_tier=3, alerted=True,
                     alert_at=3.5, final_action="fraud_desk")

    got = store.call("c1")
    assert got["alerted"] is True
    assert got["final_tier"] == 3
    assert got["branches"] == ["A", "A2"]
    assert [e["kind"] for e in got["events"]] == ["tier", "alert"]
    assert got["events"][1]["detail"]["reason"] == "sustained"


def test_speech_content_is_refused(store):
    """Voice is biometric data. A history table is exactly the thing that
    quietly becomes a biometric database, so the check is on write."""
    store.open_call("c2")
    for key in ("audio", "pcm", "waveform", "transcript", "embedding", "mfcc"):
        with pytest.raises(ForbiddenContent):
            store.add_event("c2", "tier", at=1.0, detail={key: [1, 2, 3]})


def test_forbidden_check_is_case_insensitive(store):
    store.open_call("c3")
    with pytest.raises(ForbiddenContent):
        store.add_event("c3", "tier", at=1.0, detail={"Transcript": "hello"})


def test_harmless_detail_is_allowed(store):
    store.open_call("c4")
    store.add_event("c4", "alert", at=1.0, tier=2,
                    detail={"reason": "cusum", "direction": "up"})
    assert store.call("c4")["events"][0]["detail"]["reason"] == "cusum"


def test_recent_is_newest_first_and_can_filter_alerts(store):
    for i in range(5):
        store.open_call(f"c{i}")
        store.close_call(f"c{i}", alerted=(i % 2 == 0), max_risk=float(i))
        time.sleep(0.01)
    rec = store.recent(limit=10)
    assert [r["call_id"] for r in rec] == ["c4", "c3", "c2", "c1", "c0"]
    assert all(r["alerted"] for r in store.recent(alerted_only=True))
    assert len(store.recent(alerted_only=True)) == 3


def test_prune_actually_deletes_and_cascades(store):
    """Retention as a callable, not a sentence in a policy document. The
    cascade matters: without PRAGMA foreign_keys the events table grows
    forever behind a calls table that looks pruned."""
    store.open_call("old")
    store.add_event("old", "tier", at=1.0, tier=1)
    # backdate it past the cutoff
    with store._lock:
        store._conn.execute("UPDATE calls SET started_at=? WHERE call_id=?",
                            (time.time() - 60 * 86400, "old"))
        store._conn.commit()
    store.open_call("new")
    store.add_event("new", "tier", at=1.0, tier=1)

    assert store.prune(older_than_days=30) == 1
    assert store.call("old") is None
    assert store.call("new") is not None
    assert store.stats()["events"] == 1, "events did not cascade"


def test_stats_reports_alert_rate(store):
    for i in range(4):
        store.open_call(f"s{i}")
        store.close_call(f"s{i}", alerted=(i == 0))
    st = store.stats()
    assert st["calls"] == 4 and st["alerted"] == 1
    assert st["alert_rate"] == 0.25


def test_stats_on_an_empty_store_does_not_divide_by_zero(store):
    assert store.stats() == {"calls": 0, "alerted": 0, "alert_rate": 0.0,
                             "events": 0, "path": str(store.path)}


def test_concurrent_writers_do_not_corrupt_it(store):
    """One sqlite3 connection shared by threads without a lock fails as an
    intermittent 'recursive use of cursors', which is not a legible symptom."""
    def worker(n):
        for i in range(20):
            store.open_call(f"w{n}_{i}")
            store.add_event(f"w{n}_{i}", "tier", at=float(i), tier=1)
            store.close_call(f"w{n}_{i}", alerted=False)

    ts = [threading.Thread(target=worker, args=(n,)) for n in range(6)]
    for t in ts: t.start()
    for t in ts: t.join()
    st = store.stats()
    assert st["calls"] == 120
    assert st["events"] == 120


def test_reopening_the_same_call_id_does_not_duplicate(store):
    store.open_call("dup")
    store.open_call("dup")
    assert store.stats()["calls"] == 1
