"""Artifact integrity.

weights_only=True stopped a tampered checkpoint from executing code. It does not
stop one from SCORING, and that failure is silent by construction: altered
weights do not crash and do not look wrong, they produce confident numbers
nothing downstream can distinguish from correct ones.

These tests pin the tripwire, including the case that would otherwise let an
attacker bypass it entirely -- adding a file rather than editing one.
"""
from __future__ import annotations

import json

import pytest

from vericall.serving.integrity import (
    MANIFEST_NAME, build_manifest, digest, verify,
)


def seed(tmp_path, names=("a.pt", "b.json")):
    files = []
    for i, n in enumerate(names):
        p = tmp_path / n
        p.write_bytes(b"artifact-" + bytes([i]) * 32)
        files.append(p)
    return files


def seal(tmp_path, files):
    (tmp_path / MANIFEST_NAME).write_text(
        json.dumps(build_manifest(files, tmp_path), indent=2))


def test_sealed_artifacts_verify_clean(tmp_path):
    files = seed(tmp_path)
    seal(tmp_path, files)
    rep = verify(tmp_path, files)
    assert rep.clean
    assert rep.checked == 2 and rep.ok == 2
    assert not (rep.mismatched or rep.missing or rep.unsealed)


def test_a_single_changed_byte_is_caught(tmp_path):
    files = seed(tmp_path)
    seal(tmp_path, files)
    original = files[0].read_bytes()
    files[0].write_bytes(original[:-1] + bytes([original[-1] ^ 0x01]))

    rep = verify(tmp_path, files)
    assert not rep.clean
    assert rep.mismatched == ["a.pt"]
    assert rep.ok == 1, "the untouched file should still verify"


def test_an_edited_calibration_digit_is_caught(tmp_path):
    """The quiet one. Three digits in a Platt JSON move every threshold in the
    system, and /health would still report calibrated: true."""
    p = tmp_path / "A@clean.json"
    p.write_text(json.dumps({"a": 0.447, "b": -0.905}))
    seal(tmp_path, [p])
    p.write_text(json.dumps({"a": 0.947, "b": -0.905}))
    assert verify(tmp_path, [p]).mismatched == ["A@clean.json"]


def test_a_file_missing_from_the_manifest_is_reported_not_ignored(tmp_path):
    """Otherwise the check is bypassed by ADDING a file rather than editing one:
    'absent from the manifest' must not silently mean 'fine'."""
    files = seed(tmp_path)
    seal(tmp_path, files[:1])                 # seal only a.pt
    rep = verify(tmp_path, files)
    assert not rep.clean
    assert rep.unsealed == ["b.json"]
    assert not rep.mismatched, "unsealed is not the same alarm as tampered"


def test_a_deleted_artifact_is_reported_as_missing(tmp_path):
    files = seed(tmp_path)
    seal(tmp_path, files)
    files[1].unlink()
    rep = verify(tmp_path, files)
    assert rep.missing == ["b.json"]
    assert not rep.clean


def test_no_manifest_means_unsealed_not_clean(tmp_path):
    """An unsealed directory must never read as verified. A default that treats
    'no manifest' as success makes the whole control decorative."""
    files = seed(tmp_path)
    rep = verify(tmp_path, files)
    assert not rep.clean
    assert not rep.manifest_present
    assert sorted(rep.unsealed) == ["a.pt", "b.json"]


def test_an_unreadable_manifest_fails_closed(tmp_path):
    files = seed(tmp_path)
    (tmp_path / MANIFEST_NAME).write_text("{not json")
    rep = verify(tmp_path, files)
    assert not rep.clean


def test_digest_is_stable_and_content_addressed(tmp_path):
    p = tmp_path / "x.pt"
    p.write_bytes(b"same")
    first = digest(p)
    p.write_bytes(b"same")
    assert digest(p) == first
    p.write_bytes(b"different")
    assert digest(p) != first


def test_manifest_covers_large_files_in_chunks(tmp_path):
    """Checkpoints here are ~3.6 MB and the reader is chunked; a bug in the
    loop would hash only the first chunk and miss tampering after it."""
    p = tmp_path / "big.pt"
    p.write_bytes(b"\x00" * (3 << 20) + b"TAIL")
    seal(tmp_path, [p])
    p.write_bytes(b"\x00" * (3 << 20) + b"tail")
    assert verify(tmp_path, [p]).mismatched == ["big.pt"]
