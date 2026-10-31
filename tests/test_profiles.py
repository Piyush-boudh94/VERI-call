"""Deployment profiles: heads and the calibration fitted for them, as one unit.

Three variables had to agree by hand -- VERICALL_CKPT_A, VERICALL_CKPT_A2 and
VERICALL_CALIB_DIR -- and when they disagreed nothing failed. A Platt mapping
fitted for one model applied to another produces confident numbers, so the
error surfaces as a bad result days later rather than as an exception. These
tests pin the properties that make a profile safe to trust.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from vericall import profiles

ROOT = Path(__file__).resolve().parents[1]


def test_registry_parses_and_has_a_default():
    ps = profiles.all_profiles()
    assert ps, "profiles.json defines no profiles"
    assert profiles.default_id() in ps, "the default names a profile that does not exist"


def test_the_default_is_servable():
    assert profiles.get(profiles.default_id()).servable


@pytest.mark.parametrize("pid", sorted(profiles.all_profiles()))
def test_ready_profiles_point_at_files_that_exist(pid):
    """A profile marked ready whose checkpoint is missing is the exact failure
    that served an untrained head for a day."""
    p = profiles.get(pid)
    if p.status != "ready":
        return
    assert p.calib_dir, f"{pid} is ready but names no calibration"
    assert (ROOT / p.calib_dir).is_dir(), f"{pid}: {p.calib_dir} is not a directory"
    for name in (p.ckpt_a, p.ckpt_a2):
        if name:
            assert (ROOT / "checkpoints" / name).exists(), f"{pid}: {name} missing"


@pytest.mark.parametrize("pid", sorted(profiles.all_profiles()))
def test_calibration_was_fitted_for_these_heads(pid):
    """provenance.json records which heads the mapping was fitted for. If the
    profile names different ones, the profile is the thing that is wrong."""
    p = profiles.get(pid)
    if p.status != "ready":
        return
    prov = ROOT / p.calib_dir / "provenance.json"
    if not prov.exists():
        pytest.skip(f"{p.calib_dir} records no provenance")
    fitted = (json.loads(prov.read_text()).get("fitted_for") or {})
    for branch, named in (("A", p.ckpt_a), ("A2", p.ckpt_a2)):
        if named and branch in fitted:
            assert Path(fitted[branch]).name == named, (
                f"{pid} serves {named} under a calibration fitted for "
                f"{fitted[branch]} -- these must not be mixed")


def test_experimental_profiles_are_refused():
    """An uncalibrated head still ranks, so it looks fine. Refuse, do not warn."""
    exp = [p for p, v in profiles.all_profiles().items() if not v.servable]
    assert exp, "expected at least one experimental profile (indic5)"
    for pid in exp:
        env: dict = {}
        with pytest.raises(RuntimeError, match="not servable"):
            profiles.apply_to_env(pid, env)
        assert env == {}, "a refused profile must not leave variables set"


def test_apply_sets_all_three_together():
    env: dict = {}
    profiles.apply_to_env("hindi-telephony", env)
    assert env["VERICALL_CALIB_DIR"] == "calib_hindi_ch"
    assert env["VERICALL_CKPT_A"] == "branch_Ahindich_seed0.pt"
    assert env["VERICALL_CKPT_A2"] == "branch_A2hindich_seed0.pt"


def test_an_explicit_variable_wins():
    """Existing scripts and one-off experiments must keep working."""
    env = {"VERICALL_CKPT_A": "branch_A_seed0.pt"}
    applied = profiles.apply_to_env("hindi-telephony", env)
    assert env["VERICALL_CKPT_A"] == "branch_A_seed0.pt"
    assert "VERICALL_CKPT_A" not in applied
    assert env["VERICALL_CALIB_DIR"] == "calib_hindi_ch"


def test_unknown_profile_lists_the_known_ones():
    with pytest.raises(KeyError, match="hindi-telephony"):
        profiles.get("no-such-language")


def test_every_language_named_is_one_we_can_build():
    """A profile claiming a language with no corpus is a claim, not a feature."""
    from vericall.data.manifest import INDIC_LANGUAGES
    allowed = set(INDIC_LANGUAGES) | {"english", "mandarin"}
    for pid, p in profiles.all_profiles().items():
        unknown = set(p.languages) - allowed
        assert not unknown, f"{pid} claims {unknown}, which nothing can build"
