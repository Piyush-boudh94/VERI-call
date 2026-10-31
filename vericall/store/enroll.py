"""Enrolled voiceprints, so Branch C's identity path is not dead code.

`prevent/policy.py` treats "high spoof score AND high identity match" as the
worst cell in the table -- a convincing clone of a known customer -- and
escalates on it. That escalation has never been reachable in the served system.
It needs `enrolled and identity_llr > 0`, and neither was ever true: no
endpoint set `enrolled`, and the scorer emitted C events with a hardcoded
`llr=0.0`, so `0.0 > 0` was False regardless. `BranchC.llr_target()`, the
method that computes a real one, had no call site anywhere in the package.
This module is the half that was missing.

**What is stored, and what is deliberately not.** One 192-dimensional ECAPA
embedding per speaker, and nothing else. No audio, ever -- not the enrolment
clip, not a hash of it. That is not a convenience: a voiceprint is biometric
data under the DPDP Act 2023 and under GDPR Art. 9, an embedding cannot be
played back as the person's voice, and keeping the audio would turn a
detection system into a corpus of customer recordings that is worth stealing.

**What an embedding still is.** It is not anonymous. It identifies the person
it came from -- that is its entire function -- and two systems holding the same
embedding can link their records. So it is personal data, it needs a lawful
basis and a retention policy like any other, and `delete()` exists because the
right to erasure applies to it. Saying "we only store a vector" as though that
settled the question would be wrong.

**Storage format.** One `.npz` of float32 rows plus a JSON sidecar of metadata,
rewritten whole on each change. At the scale this is for -- tens to low
thousands of enrolled customers -- a rewrite costs microseconds and buys atomic
replacement via `os.replace`, so a crash mid-write cannot leave a half-written
voiceprint file. A real deployment with millions of customers wants a database;
this is honest about being the small case.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Same rule as call_id: \A..\Z and not ^..$, because Python's $ also matches
# before a trailing newline, so "alice\n" would pass a ^..$ check and become a
# second, distinct speaker whose id prints identically in every log line.
SPEAKER_ID_RE = re.compile(r"\A[A-Za-z0-9._:-]{1,64}\Z")

EMBED_DIM = 192


def valid_speaker_id(s: str) -> bool:
    return bool(s) and SPEAKER_ID_RE.match(s) is not None


class EnrollmentStore:
    """Speaker id -> one L2-normalised ECAPA embedding."""

    def __init__(self, path: str | Path = "enroll/voiceprints.npz") -> None:
        self.path = Path(path)
        self.meta_path = self.path.with_suffix(".json")
        self._lock = threading.Lock()
        self._emb: dict[str, np.ndarray] = {}
        self._meta: dict[str, dict] = {}
        self._load()

    # ------------------------------------------------------------------ #

    def _load(self) -> None:
        if self.path.exists():
            try:
                with np.load(self.path) as z:
                    self._emb = {k: z[k].astype(np.float32) for k in z.files}
            except (OSError, ValueError):
                # A corrupt store must not stop the gateway starting. Identity
                # escalation degrades; spoof detection is untouched by it.
                self._emb = {}
        if self.meta_path.exists():
            try:
                self._meta = json.loads(self.meta_path.read_text())
            except (OSError, ValueError):
                self._meta = {}

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".npz.tmp")
        # A file handle, not a path. np.savez APPENDS ".npz" to a path that
        # does not already end in it, so np.savez("v.npz.tmp", ...) silently
        # writes "v.npz.tmp.npz" and the os.replace below then fails on a file
        # that was never created. Handing it an open file disables that.
        with tmp.open("wb") as fh:
            np.savez(fh, **self._emb)
        os.replace(tmp, self.path)          # atomic
        tmp_m = self.meta_path.with_suffix(".json.tmp")
        tmp_m.write_text(json.dumps(self._meta, indent=2) + "\n")
        os.replace(tmp_m, self.meta_path)

    # ------------------------------------------------------------------ #

    def enroll(self, speaker_id: str, embedding: np.ndarray,
               seconds: float = 0.0, note: str = "") -> dict:
        if not valid_speaker_id(speaker_id):
            raise ValueError("invalid speaker id")
        e = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if e.shape[0] != EMBED_DIM:
            raise ValueError(f"expected {EMBED_DIM}-d embedding, got {e.shape[0]}")
        n = float(np.linalg.norm(e))
        if not np.isfinite(n) or n == 0.0:
            raise ValueError("embedding is not finite or is all zeros")
        e = e / n                            # store normalised, always
        with self._lock:
            self._emb[speaker_id] = e
            self._meta[speaker_id] = {
                "enrolled_at": time.time(),
                "seconds_of_audio": round(float(seconds), 2),
                "note": note[:200],
            }
            self._flush()
            return {"speaker_id": speaker_id, **self._meta[speaker_id]}

    def get(self, speaker_id: str) -> Optional[np.ndarray]:
        with self._lock:
            e = self._emb.get(speaker_id)
            return None if e is None else e.copy()

    def delete(self, speaker_id: str) -> bool:
        """Erasure. Present because the right to it applies to biometric data."""
        with self._lock:
            had = self._emb.pop(speaker_id, None) is not None
            self._meta.pop(speaker_id, None)
            if had:
                self._flush()
            return had

    def list(self) -> list[dict]:
        """Metadata only. The embeddings themselves are never returned over an
        API: a leaked voiceprint is not revocable the way a password is."""
        with self._lock:
            return [{"speaker_id": k, **self._meta.get(k, {})}
                    for k in sorted(self._emb)]

    def stats(self) -> dict:
        with self._lock:
            return {"enrolled": len(self._emb), "path": str(self.path)}
