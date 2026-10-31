"""Auditable prevention records, and the India-specific compliance surface.

Two jobs, both boring and both load-bearing:

1. **An evidence trail that survives the call.** A prevention decision that
   cannot be reconstructed afterwards is not defensible to a regulator, a
   customer disputing a blocked transaction, or an internal reviewer asking why
   a genuine caller was challenged.

2. **Carrying the retention constraint into the record itself.** The privacy
   design says raw audio is never persisted and no transcript is produced. That
   promise is only real if the audit record physically cannot hold either, so
   this module stores decisions, scores and identifiers -- never samples, never
   text of what was said.

The compliance references below are the ones E14 named. They are recorded as
*fields to populate*, not as integrations: whether a given call is reportable
under CERT-In's advisory or the IT Amendment Rules is a legal determination the
deploying institution makes, and a hackathon build asserting otherwise would be
overstepping.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

SCHEMA_VERSION = "1.0"


@dataclass
class ComplianceFlags:
    """Fields a deploying institution populates; none are auto-asserted.

    - **CNAP** (TRAI Calling Name Presentation): whether the network-verified
      caller name matched the claimed identity. A mismatch alongside a high
      spoof score is materially stronger evidence than either alone, because
      CNAP is bound to KYC rather than to the audio.
    - **CERT-In**: the 2024 deepfake advisory sets expectations for incident
      handling; `reportable` is a determination, not a detection output.
    - **IT Amendment Rules 2026**: synthetic-media labelling and the 36-hour
      takedown window apply to intermediaries, so this records the clock start
      rather than claiming compliance.
    """

    cnap_name: Optional[str] = None
    cnap_verified: Optional[bool] = None
    cnap_mismatch: Optional[bool] = None
    certin_reportable: Optional[bool] = None
    it_rules_2026_clock_started_at: Optional[float] = None
    notes: list[str] = field(default_factory=list)


@dataclass
class PreventionRecord:
    """One decision, reconstructable after the fact. Contains no audio, no text."""

    record_id: str
    call_id: str
    tenant_id: str
    at: float
    action: str
    reason: str
    context: str
    tier: int
    risk_nats: float
    threshold_nats: float
    confidence: float
    identity_conflict: bool
    active_branches: list[str]
    challenge: Optional[dict] = None
    compliance: ComplianceFlags = field(default_factory=ComplianceFlags)
    schema_version: str = SCHEMA_VERSION
    model_ids: dict = field(default_factory=dict)
    calib_ids: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        d = asdict(self)
        d["at_iso"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.at))
        return d


# Keys that must never appear in a record. Enforced rather than documented,
# because "we promise not to log audio" is worth exactly as much as the check
# that proves it.
FORBIDDEN_KEYS = {"audio", "pcm", "waveform", "samples", "transcript", "text",
                  "utterance", "embedding", "voiceprint"}


def _assert_no_content(payload: dict) -> None:
    def walk(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                if k.lower() in FORBIDDEN_KEYS:
                    raise ValueError(
                        f"audit record would contain '{path}{k}', which breaks "
                        "the no-audio/no-transcript retention guarantee")
                walk(v, f"{path}{k}.")
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item, path)
    walk(payload)


def build_record(*, call_id: str, decision, view=None, challenge=None,
                 tenant_id: str = "default", compliance: Optional[ComplianceFlags] = None,
                 model_ids: Optional[dict] = None,
                 calib_ids: Optional[dict] = None) -> PreventionRecord:
    conf = 0.0
    branches: list[str] = []
    if view is not None:
        conf = getattr(getattr(view, "confidence", None), "score", 0.0) or 0.0
        branches = list(getattr(view, "subset", ()) or ())
    return PreventionRecord(
        record_id=uuid.uuid4().hex,
        call_id=call_id,
        tenant_id=tenant_id,
        at=time.time(),
        action=decision.action.value,
        reason=decision.reason,
        context=decision.context,
        tier=decision.tier,
        risk_nats=round(decision.risk, 3),
        threshold_nats=round(decision.threshold, 3),
        confidence=round(conf, 3),
        identity_conflict=decision.identity_conflict,
        active_branches=branches,
        challenge=challenge.to_json() if challenge is not None else None,
        compliance=compliance or ComplianceFlags(),
        model_ids=model_ids or {},
        calib_ids=calib_ids or {},
    )


class AuditLog:
    """Append-only JSONL sink.

    Append-only on purpose: a prevention log that can be edited after an
    incident is not evidence.

    **Rotation is not the deployer's problem, whatever the previous version of
    this docstring said.** An unbounded append-only file on the same volume as
    everything else fills the disk, and a full disk is not a logging outage --
    it is a total outage, because the next write raises inside the call path.
    Rotation happens here, with a bounded number of generations, so the failure
    mode is "oldest evidence is gone" rather than "the service is down".

    Size-based rather than time-based: a quiet day and a busy day should not
    produce files of wildly different sizes, and the bound that matters is disk.
    """

    def __init__(self, path: str | Path = "logs/prevention_audit.jsonl",
                 max_bytes: int = 64 * 1024 * 1024, keep: int = 5):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(os.environ.get("VERICALL_AUDIT_MAX_BYTES", 0)
                             or max_bytes)
        self.keep = int(os.environ.get("VERICALL_AUDIT_KEEP", 0) or keep)
        self.written = 0
        self.rotations = 0

    def _rotate_if_needed(self) -> None:
        try:
            if not self.path.exists() or self.path.stat().st_size < self.max_bytes:
                return
        except OSError:
            return
        # Shift generations down, dropping the oldest. Renames are atomic within
        # a filesystem, so a reader either sees the old file or the new one and
        # never a half-written one.
        oldest = self.path.with_suffix(self.path.suffix + f".{self.keep}")
        if oldest.exists():
            oldest.unlink()
        for i in range(self.keep - 1, 0, -1):
            src = self.path.with_suffix(self.path.suffix + f".{i}")
            if src.exists():
                src.rename(self.path.with_suffix(self.path.suffix + f".{i + 1}"))
        self.path.rename(self.path.with_suffix(self.path.suffix + ".1"))
        self.rotations += 1

    def append(self, record: PreventionRecord) -> dict:
        payload = record.to_json()
        _assert_no_content(payload)
        self._rotate_if_needed()
        with self.path.open("a") as fh:
            fh.write(json.dumps(payload, sort_keys=True) + "\n")
            # The point of an audit record is that it survives the incident it
            # documents. Without a flush+fsync the most recent records live in
            # the page cache, and a hard kill loses exactly the evidence from
            # the moments that mattered most.
            fh.flush()
            os.fsync(fh.fileno())
        self.written += 1
        return payload

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]
