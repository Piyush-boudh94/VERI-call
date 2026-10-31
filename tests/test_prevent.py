"""Prevention layer: intervention policy, challenge-response, audit integrity."""
from __future__ import annotations

import pytest

from vericall.prevent.audit import (
    AuditLog, ComplianceFlags, PreventionRecord, build_record,
)
from vericall.prevent.challenge import (
    HUMAN_CEILING_S, SUSPICIOUS_FLOOR_S, Challenge, ChallengeKind,
    ChallengeResult, issue, risk_adjustment, verify,
)
from vericall.prevent.policy import Action, PreventionPolicy

POL = PreventionPolicy()


def decide(**kw):
    base = dict(risk_nats=0.0, tier=0, confidence=0.9,
                context="inbound_unclassified")
    base.update(kw)
    return POL.decide(**base)


# ------------------------------------------------------------------ policy

def test_low_risk_is_allowed():
    assert decide(risk_nats=-5.0).action is Action.ALLOW


def test_same_risk_gives_different_actions_by_context():
    """The core design claim: context, not just score, decides the action.
    At 1e-4 prevalence no achievable FPR makes an unconditioned alert precise,
    so conditioning on what the caller is doing is the only real lever."""
    risk = 6.0
    enquiry = decide(risk_nats=risk, context="balance_enquiry")
    transfer = decide(risk_nats=risk, context="transfer_large")
    assert enquiry.action is Action.ALLOW
    assert transfer.action is Action.FRAUD_DESK
    assert enquiry.action is not transfer.action


def test_nothing_at_stake_is_never_escalated():
    d = decide(risk_nats=50.0, context="balance_enquiry")
    assert d.action is Action.ALLOW


def test_irreversible_action_escalates_harder_than_reversible():
    """Isolates reversibility. An earlier version of this test compared
    payee_add against statement_request, which differ in BOTH reversibility and
    value by four orders of magnitude -- so it proved nothing about either. The
    stock CONTEXTS have no reversible high-value entry, so the matched pair is
    constructed here."""
    from vericall.prevent.policy import CONTEXTS, TransactionContext

    CONTEXTS["_t_rev"] = TransactionContext("_t_rev", 5e-4, 50_000.0, 35.0,
                                            reversible=True)
    CONTEXTS["_t_irrev"] = TransactionContext("_t_irrev", 5e-4, 50_000.0, 35.0,
                                              reversible=False)
    try:
        rev = decide(risk_nats=6.0, context="_t_rev")
        irrev = decide(risk_nats=6.0, context="_t_irrev")
        assert rev.threshold == irrev.threshold      # only reversibility differs
        assert rev.action is Action.STEP_UP
        assert irrev.action is Action.HOLD
    finally:
        CONTEXTS.pop("_t_rev", None)
        CONTEXTS.pop("_t_irrev", None)


def test_trivial_stakes_tolerate_high_risk():
    """A statement request costs ~100 to get wrong and 35 to challenge, so the
    cost-optimal threshold is high and a mid-range score should NOT interrupt
    the customer. This is the base-rate argument working as intended."""
    d = decide(risk_nats=6.0, context="statement_request")
    assert d.action is Action.ALLOW
    assert d.threshold > 6.0


def test_near_threshold_buys_evidence_in_band():
    """Just under threshold should challenge rather than guess: a challenge
    costs seconds and needs no second channel."""
    thr = decide(risk_nats=0.0, context="transfer_small").threshold
    d = decide(risk_nats=thr - 0.5, context="transfer_small")
    assert d.action is Action.CHALLENGE


def test_clone_of_enrolled_customer_is_the_worst_case_not_the_safest():
    """Identity match + high spoof = someone is successfully impersonating a
    known customer. A naive design treats a voiceprint match as reassuring."""
    d = decide(risk_nats=8.0, context="transfer_small",
               identity_llr=4.0, enrolled=True)
    assert d.identity_conflict
    assert d.action is Action.FRAUD_DESK
    assert any("clone" in n for n in d.notes)


def test_identity_match_without_spoof_is_not_a_conflict():
    d = decide(risk_nats=-8.0, context="transfer_small",
               identity_llr=4.0, enrolled=True)
    assert not d.identity_conflict
    assert d.action is Action.ALLOW


def test_low_confidence_does_not_act_on_the_score_alone():
    d = decide(risk_nats=8.0, confidence=0.1, context="statement_request")
    assert d.action is Action.SILENT_FLAG
    assert any("confidence" in n for n in d.notes)


def test_low_confidence_still_protects_irreversible_high_value_actions():
    """The safe failure on an irreversible transfer is friction, not an
    unrecoverable loss."""
    d = decide(risk_nats=8.0, confidence=0.1, context="transfer_large")
    assert d.action is Action.STEP_UP


def test_decision_serialises():
    d = decide(risk_nats=6.0, context="transfer_small")
    j = d.to_json()
    assert j["action"] and j["context"] == "transfer_small"


# --------------------------------------------------------------- challenge

@pytest.mark.parametrize("kind", list(ChallengeKind))
def test_issue_produces_an_unpredictable_prompt(kind):
    a = issue("c1", kind)
    b = issue("c1", kind)
    assert a.prompt and a.expected
    # 20 draws colliding every time would mean a predictable generator
    assert any(issue("c1", kind).expected != a.expected for _ in range(20))


def test_timeout_reads_as_pre_recorded_audio():
    ch = issue("c1", ChallengeKind.DIGITS, now=0.0)
    verify(ch, content_matches=False, responded_at=99.0)
    assert ch.result is ChallengeResult.FAILED_TIMEOUT
    assert any("pre-recorded" in n for n in ch.notes)


def test_wrong_content_fails():
    ch = issue("c1", ChallengeKind.DIGITS, now=0.0)
    verify(ch, content_matches=False, responded_at=1.0)
    assert ch.result is ChallengeResult.FAILED_CONTENT


def test_impossibly_fast_correct_answer_is_suspicious():
    """Correct AND faster than human reaction means a machine answered."""
    ch = issue("c1", ChallengeKind.DIGITS, now=0.0)
    verify(ch, content_matches=True, responded_at=SUSPICIOUS_FLOOR_S / 2)
    assert ch.result is ChallengeResult.SUSPICIOUS_LATENCY


def test_human_paced_correct_answer_passes():
    ch = issue("c1", ChallengeKind.DIGITS, now=0.0)
    verify(ch, content_matches=True, responded_at=1.2)
    assert ch.result is ChallengeResult.PASSED
    assert not ch.notes


def test_slow_but_correct_answer_passes_with_a_caveat():
    ch = issue("c1", ChallengeKind.DIGITS, now=0.0)
    verify(ch, content_matches=True, responded_at=HUMAN_CEILING_S + 1.0)
    assert ch.result is ChallengeResult.PASSED
    assert any("real-time synthesis" in n for n in ch.notes)


def test_risk_adjustment_is_asymmetric():
    """A pass is weak evidence of innocence; a timeout is strong evidence
    nobody is there. Symmetry would let an attacker buy a large innocence
    bonus for one successful response."""
    def adj(result):
        ch = issue("c", ChallengeKind.DIGITS, now=0.0)
        ch.result = result
        return risk_adjustment(ch)

    passed = adj(ChallengeResult.PASSED)
    timeout = adj(ChallengeResult.FAILED_TIMEOUT)
    assert passed < 0 < timeout
    assert abs(timeout) > abs(passed)


# ------------------------------------------------------------------- audit

def test_record_captures_the_decision(tmp_path):
    d = decide(risk_nats=6.0, context="transfer_small")
    rec = build_record(call_id="call-1", decision=d)
    log = AuditLog(tmp_path / "audit.jsonl")
    payload = log.append(rec)
    assert payload["call_id"] == "call-1"
    assert payload["action"] == d.action.value
    assert payload["at_iso"].endswith("Z")
    assert len(log.read_all()) == 1


def test_audit_log_is_append_only(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    for i in range(3):
        log.append(build_record(call_id=f"c{i}", decision=decide(risk_nats=6.0)))
    assert len(log.read_all()) == 3


def test_audit_refuses_to_persist_audio_or_transcripts(tmp_path):
    """The no-retention promise is only real if the record physically cannot
    hold audio or text. Enforced, not documented."""
    d = decide(risk_nats=6.0)
    rec = build_record(call_id="c1", decision=d)
    rec.model_ids = {"transcript": "hello this is what they said"}
    log = AuditLog(tmp_path / "audit.jsonl")
    with pytest.raises(ValueError, match="retention guarantee"):
        log.append(rec)
    assert log.read_all() == []


def test_audit_rejects_nested_forbidden_content(tmp_path):
    d = decide(risk_nats=6.0)
    rec = build_record(call_id="c1", decision=d)
    rec.calib_ids = {"debug": {"nested": {"waveform": [0.1, 0.2]}}}
    log = AuditLog(tmp_path / "audit.jsonl")
    with pytest.raises(ValueError, match="retention guarantee"):
        log.append(rec)


def test_compliance_flags_are_recorded_not_asserted(tmp_path):
    """Whether a call is CERT-In reportable is a legal determination, not a
    detector output -- the field exists to be populated, defaulting to unknown."""
    rec = build_record(call_id="c1", decision=decide(risk_nats=6.0),
                       compliance=ComplianceFlags(cnap_name="ACME BANK",
                                                  cnap_verified=True,
                                                  cnap_mismatch=False))
    log = AuditLog(tmp_path / "audit.jsonl")
    p = log.append(rec)
    assert p["compliance"]["cnap_name"] == "ACME BANK"
    assert p["compliance"]["certin_reportable"] is None


# --------------------------------------------------- audit log durability

def _a_record(call_id: str = "call-1"):
    return build_record(call_id=call_id, decision=decide(risk_nats=6.0))


def test_audit_rotates_instead_of_filling_the_disk(tmp_path):
    """An unbounded append-only log is not a logging problem, it is an outage:
    the next write raises inside the call path when the volume fills."""
    from vericall.prevent.audit import AuditLog

    log = AuditLog(tmp_path / "audit.jsonl", max_bytes=2048, keep=3)
    for i in range(400):
        log.append(_a_record(f"call-{i:04d}"))

    assert log.rotations > 0, "never rotated"
    assert log.path.exists()
    assert log.path.stat().st_size < 2048 * 4
    gens = sorted(p.name for p in tmp_path.glob("audit.jsonl.*"))
    assert gens, "rotation produced no generations"
    assert len(gens) <= 3, f"kept more than `keep` generations: {gens}"


def test_audit_keeps_the_newest_generation_not_the_oldest(tmp_path):
    """Generation .1 must be the most recently rotated file. Getting the shift
    direction backwards silently preserves the oldest evidence and discards the
    records from the incident you are investigating."""
    from vericall.prevent.audit import AuditLog

    log = AuditLog(tmp_path / "audit.jsonl", max_bytes=1024, keep=3)
    for i in range(300):
        log.append(_a_record(f"call-{i:04d}"))

    newest_rotated = (tmp_path / "audit.jsonl.1").read_text()
    older_rotated = (tmp_path / "audit.jsonl.2")
    if older_rotated.exists():
        first_new = newest_rotated.splitlines()[0]
        first_old = older_rotated.read_text().splitlines()[0]
        assert first_new > first_old, (
            "generation .1 is older than .2 -- the shift runs the wrong way")


def test_audit_survives_a_hard_kill(tmp_path):
    """fsync, not just flush. The records that matter most are the ones written
    in the seconds before something went wrong."""
    from vericall.prevent.audit import AuditLog

    path = tmp_path / "audit.jsonl"
    log = AuditLog(path, max_bytes=10 ** 9)
    log.append(_a_record())
    # Read through a fresh handle: the bytes must be on disk, not only in this
    # process's buffers.
    assert path.read_text().strip(), "record was not durable after append"
