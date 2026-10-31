"""Enrolled voiceprints, and the identity path they unlock.

`identity_conflict` was unreachable in the served system for two independent
reasons -- nothing set `enrolled`, and C events carried a hardcoded llr=0.0 --
so these tests pin both halves plus the handling of what is, legally, biometric
personal data.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from vericall.prevent.policy import CONTEXTS, PreventionPolicy
from vericall.store.enroll import (
    EMBED_DIM, EnrollmentStore, valid_speaker_id,
)


def emb(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(EMBED_DIM)


# ------------------------------------------------------------------- store

def test_embeddings_are_stored_normalised(tmp_path):
    """Stored normalised so a cosine is a cosine regardless of what was
    handed in -- ECAPA already normalises, but an enrolment path that trusts
    its caller for that is one bad client away from a meaningless score."""
    s = EnrollmentStore(tmp_path / "v.npz")
    s.enroll("alice", emb() * 37.0)
    assert float(np.linalg.norm(s.get("alice"))) == pytest.approx(1.0)


def test_it_survives_a_restart(tmp_path):
    s = EnrollmentStore(tmp_path / "v.npz")
    s.enroll("alice", emb(1), seconds=4.0)
    again = EnrollmentStore(tmp_path / "v.npz")
    assert again.get("alice") is not None
    assert np.allclose(again.get("alice"), s.get("alice"))
    assert again.list()[0]["seconds_of_audio"] == 4.0


def test_the_write_is_atomic_and_leaves_no_temp_files(tmp_path):
    """np.savez APPENDS .npz to a path that lacks it, so np.savez("v.npz.tmp")
    writes "v.npz.tmp.npz" and the os.replace then fails on a file that was
    never created. Caught by this exact assertion the first time."""
    s = EnrollmentStore(tmp_path / "v.npz")
    s.enroll("alice", emb())
    assert sorted(p.name for p in tmp_path.iterdir()) == ["v.json", "v.npz"]


def test_erasure_actually_erases(tmp_path):
    """The right to erasure applies to biometric data, so delete() has to
    reach the disk and not just the in-memory dict."""
    s = EnrollmentStore(tmp_path / "v.npz")
    s.enroll("alice", emb())
    s.enroll("bob", emb(2))
    assert s.delete("alice") is True
    assert s.delete("alice") is False, "deleting twice is not an error"
    assert EnrollmentStore(tmp_path / "v.npz").get("alice") is None
    assert EnrollmentStore(tmp_path / "v.npz").get("bob") is not None


def test_listing_never_exposes_an_embedding(tmp_path):
    """A leaked voiceprint is not revocable the way a password is."""
    s = EnrollmentStore(tmp_path / "v.npz")
    s.enroll("alice", emb())
    blob = json.dumps(s.list())
    assert "alice" in blob
    for v in s.get("alice")[:8]:
        assert f"{v:.6f}" not in blob


def test_no_audio_is_retained(tmp_path):
    """The store takes an embedding, never a waveform. Keeping the audio would
    turn a detector into a corpus of customer recordings worth stealing."""
    import inspect
    src = inspect.getsource(EnrollmentStore)
    for word in ("wav", "audio", "waveform", "pcm"):
        assert word not in src.lower().replace("seconds_of_audio", ""), word


def test_a_wrong_sized_embedding_is_refused(tmp_path):
    s = EnrollmentStore(tmp_path / "v.npz")
    with pytest.raises(ValueError, match="192"):
        s.enroll("alice", np.zeros(64))


def test_a_degenerate_embedding_is_refused(tmp_path):
    """All-zeros would normalise to NaN and then poison every comparison."""
    s = EnrollmentStore(tmp_path / "v.npz")
    with pytest.raises(ValueError):
        s.enroll("alice", np.zeros(EMBED_DIM))
    with pytest.raises(ValueError):
        s.enroll("alice", np.full(EMBED_DIM, np.nan))


def test_a_corrupt_store_does_not_stop_startup(tmp_path):
    """Identity escalation degrades; spoof detection is untouched by it."""
    (tmp_path / "v.npz").write_bytes(b"not an npz at all")
    s = EnrollmentStore(tmp_path / "v.npz")
    assert s.stats()["enrolled"] == 0


@pytest.mark.parametrize("bad", ["", "alice\n", "a" * 65, "al ice", "../etc"])
def test_speaker_ids_follow_the_call_id_rule(bad):
    """\\A..\\Z and not ^..$: Python's $ also matches before a trailing
    newline, so "alice\\n" would become a second speaker whose id prints
    identically in every log line."""
    assert not valid_speaker_id(bad)


def test_ordinary_ids_are_accepted():
    for good in ("alice", "customer-9931", "acct.123", "a:b", "A_1"):
        assert valid_speaker_id(good)


# -------------------------------------------------------- the policy cell

def test_identity_conflict_needs_both_halves():
    """A clone of an enrolled customer is the WORST cell, not the safest:
    identity match and a high spoof score together mean someone is
    successfully impersonating a known person."""
    p = PreventionPolicy()
    both = p.decide(risk_nats=6.0, tier=2, confidence=0.9,
                    context="payee_add", identity_llr=1.0, enrolled=True)
    assert both.identity_conflict is True

    for kwargs in (
        {"identity_llr": 1.0, "enrolled": False},      # no enrollment
        {"identity_llr": None, "enrolled": True},      # no comparison made
        {"identity_llr": -1.0, "enrolled": True},      # identity did NOT match
    ):
        d = p.decide(risk_nats=6.0, tier=2, confidence=0.9,
                     context="payee_add", **kwargs)
        assert d.identity_conflict is False, kwargs


def test_a_clean_call_from_the_real_customer_is_not_a_conflict():
    """High identity match with a LOW spoof score is the good case, and it
    must not escalate -- that would punish enrolled customers for enrolling."""
    p = PreventionPolicy()
    d = p.decide(risk_nats=-8.0, tier=0, confidence=0.9,
                 context="payee_add", identity_llr=13.5, enrolled=True)
    assert d.identity_conflict is False


def test_the_conflict_actually_changes_decisions():
    """Otherwise the escalation is decorative. It must move the action in the
    contexts where something irreversible is at stake, and leave the
    read-only ones alone."""
    p = PreventionPolicy()
    moved = []
    for ctx in sorted(CONTEXTS):
        a = p.decide(risk_nats=6.0, tier=2, confidence=0.9, context=ctx,
                     identity_llr=None, enrolled=False)
        b = p.decide(risk_nats=6.0, tier=2, confidence=0.9, context=ctx,
                     identity_llr=1.0, enrolled=True)
        if a.action is not b.action:
            moved.append(ctx)
    assert "payee_add" in moved and "contact_change" in moved
    # Read-only contexts have nothing to escalate about.
    assert "balance_enquiry" not in moved
    assert "statement_request" not in moved


# ------------------------------------------------------- the branch C half

def test_branch_c_llr_from_embedding_matches_llr_from_audio():
    """The serving path uses llr_target_from_emb to avoid a second ECAPA
    forward per window. If the two ever disagreed, the served identity score
    would silently differ from the one the offline evaluation measured."""
    from vericall.branches.c_speaker import ASNorm, BranchC, BranchCConfig

    c = BranchC.__new__(BranchC)          # no ECAPA download in a unit test
    torch.nn.Module.__init__(c)
    c.cfg = BranchCConfig()
    c.asnorm = None
    c.register_buffer("calib_a", torch.tensor(1.0219))
    c.register_buffer("calib_b", torch.tensor(0.4283))

    enroll = torch.nn.functional.normalize(torch.randn(192), dim=-1)
    test = torch.nn.functional.normalize(torch.randn(3, 192), dim=-1)
    c.embed = lambda wav: test                     # stand in for the encoder

    assert torch.allclose(c.llr_target_from_emb(enroll, test),
                          c.llr_target(enroll, torch.zeros(3, 16000)))


def test_load_calibration_reaches_the_buffers():
    """These defaulted to 1.0/0.0 and nothing loaded the fitted values,
    because their only consumer had no call site. An identity LLR on the
    wrong scale is worse than none: the policy tests `identity_llr > 0`."""
    from vericall.branches.c_speaker import BranchC, BranchCConfig

    c = BranchC.__new__(BranchC)
    torch.nn.Module.__init__(c)
    c.cfg = BranchCConfig()
    c.asnorm = None
    c.register_buffer("calib_a", torch.tensor(1.0))
    c.register_buffer("calib_b", torch.tensor(0.0))

    c.load_calibration(1.0219377424155327, 0.42834739349610695)
    assert float(c.calib_a) == pytest.approx(1.0219377, abs=1e-6)
    assert float(c.calib_b) == pytest.approx(0.4283474, abs=1e-6)

    e = torch.nn.functional.normalize(torch.ones(192), dim=-1)
    got = float(c.llr_target_from_emb(e, e.unsqueeze(0))[0])
    assert got == pytest.approx(1.0219377 * 1.0 + 0.4283474, abs=1e-5)
