"""Checkpoint and calibration integrity.

docs/THREAT_MODEL.md A5 listed this as **unknown**: nothing verified that the
weights being served are the weights that were evaluated. Fixing
`weights_only=True` stopped a tampered checkpoint from executing code, which is
the acute problem. It does not stop a tampered checkpoint from *scoring*.

That matters more here than in most services, because the failure is silent by
construction. A model with altered weights does not crash and does not look
wrong: it produces confident numbers that no downstream check can distinguish
from correct ones. The same is true of an altered Platt mapping -- three edited
digits in a JSON file move every threshold in the system, and `/health` would
still say `calibrated: true`.

So: a manifest of SHA-256 digests, written deliberately by an operator who has
decided these are the right artifacts, and verified at load.

**What this is and is not.** It is a tripwire for accidental drift and for an
attacker who can write one file but not all of them -- the common case, and the
one that otherwise goes unnoticed for months. It is **not** a signature: anyone
who can rewrite a checkpoint can rewrite the manifest beside it. Real supply-
chain integrity needs the manifest signed with a key the serving host does not
hold. Stating that here rather than letting a hash file imply more than it does.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

MANIFEST_NAME = "MANIFEST.sha256"
CHUNK = 1 << 20


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build_manifest(paths: Iterable[Path], root: Path) -> dict:
    out = {}
    for p in sorted(paths):
        if p.is_file():
            out[str(p.relative_to(root))] = digest(p)
    return out


@dataclass
class IntegrityReport:
    checked: int = 0
    ok: int = 0
    mismatched: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    unsealed: list[str] = field(default_factory=list)
    manifest_present: bool = False

    @property
    def clean(self) -> bool:
        return (self.manifest_present and not self.mismatched
                and not self.missing and not self.unsealed)

    def as_dict(self) -> dict:
        return {
            "manifest_present": self.manifest_present,
            "checked": self.checked, "ok": self.ok,
            "mismatched": self.mismatched, "missing": self.missing,
            "unsealed": self.unsealed, "clean": self.clean,
        }


def verify(root: Path, must_cover: Iterable[Path]) -> IntegrityReport:
    """Check `must_cover` against root/MANIFEST.sha256.

    `unsealed` is reported separately from `mismatched` on purpose. A file the
    manifest never mentioned is not a tampering alarm -- it is usually a new
    artifact somebody forgot to seal -- but it is still a file being served
    that nobody vouched for, and silently treating "absent from the manifest"
    as "fine" would let an attacker bypass the check by adding a file rather
    than editing one.
    """
    rep = IntegrityReport()
    mpath = root / MANIFEST_NAME
    if not mpath.exists():
        rep.unsealed = [str(Path(p).name) for p in must_cover]
        return rep
    rep.manifest_present = True
    try:
        manifest = json.loads(mpath.read_text())
    except (OSError, ValueError):
        rep.mismatched = ["<unreadable manifest>"]
        return rep

    for p in must_cover:
        p = Path(p)
        rel = str(p.relative_to(root)) if p.is_relative_to(root) else p.name
        if not p.exists():
            rep.missing.append(rel)
            continue
        want = manifest.get(rel)
        if want is None:
            rep.unsealed.append(rel)
            continue
        rep.checked += 1
        if digest(p) == want:
            rep.ok += 1
        else:
            rep.mismatched.append(rel)
    return rep


def enforcement_required() -> bool:
    """Whether a failed check should stop the service.

    Defaults to off. A hard failure on a laptop or in the middle of a demo is
    worse than the risk it prevents there; in production it is the opposite, so
    a deployment sets VERICALL_REQUIRE_SEALED=1 and gets a refusal to serve
    unverified weights. Either way /health reports the result, so the state is
    never a guess.
    """
    return os.environ.get("VERICALL_REQUIRE_SEALED", "0") == "1"
