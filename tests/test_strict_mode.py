"""Strict mode, and the marker list it depends on.

The gateway has nineteen places where it loses a capability, says so in
/health notes, and carries on answering status "ok". That is right for a
laptop and wrong for a deployment. VERICALL_STRICT=1 inverts it.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from vericall.serving.app import DEGRADATION_MARKERS, strict_mode

APP = Path(__file__).resolve().parents[1] / "vericall" / "serving" / "app.py"


def test_strict_mode_is_off_by_default(monkeypatch):
    """A hard failure on a laptop mid-demo is worse than the risk it
    prevents there."""
    monkeypatch.delenv("VERICALL_STRICT", raising=False)
    assert strict_mode() is False


@pytest.mark.parametrize("val,want", [("1", True), ("0", False),
                                      ("", False), ("true", False)])
def test_only_an_exact_1_enables_it(monkeypatch, val, want):
    """Same convention as VERICALL_REQUIRE_SEALED. 'true' not being accepted
    is deliberate: a setting that half-works is worse than one that does not,
    because it reads as enabled in a config file while being off."""
    monkeypatch.setenv("VERICALL_STRICT", val)
    assert strict_mode() is want


def test_every_marker_still_matches_a_real_note():
    """The safeguard that makes a marker list safe.

    Markers are substrings matched against note text, so a reworded note would
    silently retire its marker and strict mode would stop catching that
    degradation -- with nothing failing. This asserts each marker still occurs
    in the module, so a rename breaks the build instead.
    """
    src = APP.read_text()
    dead = [m for m in DEGRADATION_MARKERS if m not in src]
    assert not dead, f"markers no longer matching any note in app.py: {dead}"


def test_markers_cover_the_load_bearing_failures():
    """Named explicitly so a future edit cannot quietly drop one. These are the
    failures that change what the numbers MEAN rather than how fast they
    arrive: a missing head, an unfitted calibration, a dead audit trail, an
    open door."""
    joined = " | ".join(DEGRADATION_MARKERS)
    for essential in ("checkpoint missing", "NOT CALIBRATED", "INTEGRITY:",
                      "cohort missing", "call history unavailable",
                      "NO AUTH TOKEN"):
        assert essential in joined, essential


def test_a_note_that_is_merely_informational_is_not_a_degradation():
    """Strict mode must not fire on observations, or nobody will enable it."""
    informational = [
        "runtime codec detection is on",
        "branch_C AS-Norm cohort present",
        "loaded 4 branches",
    ]
    for n in informational:
        assert not any(m in n for m in DEGRADATION_MARKERS), n


def test_every_marker_appears_in_code_not_only_in_prose():
    """A marker must match a real note, not a comment mentioning it.

    Without this, test_every_marker_still_matches_a_real_note would pass on
    the marker list's own explanatory comment and strict mode would guard
    nothing.

    Comments are stripped with tokenize and each marker must then appear at
    least TWICE: once in the DEGRADATION_MARKERS declaration and once in the
    note it guards. An earlier version of this test collected STRING tokens
    instead and failed on five markers that were perfectly fine -- Python 3.12
    tokenises an f-string as FSTRING_START/MIDDLE/END rather than STRING, and
    every one of those five notes is an f-string.
    """
    import io
    import tokenize

    src = APP.read_text()
    spans = [(t.start, t.end) for t in
             tokenize.generate_tokens(io.StringIO(src).readline)
             if t.type == tokenize.COMMENT]
    lines = src.split("\n")
    for (sr, sc), (er, ec) in reversed(spans):
        assert sr == er, "a comment should not span lines"
        lines[sr - 1] = lines[sr - 1][:sc] + lines[sr - 1][ec:]
    code = "\n".join(lines)

    thin = [m for m in DEGRADATION_MARKERS if code.count(m) < 2]
    assert not thin, (
        "markers appearing only in their own declaration, so they guard no "
        f"actual note: {thin}")
