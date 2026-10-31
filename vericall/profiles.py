"""Deployment profiles: one trained head set plus the calibration fitted for it.

A profile exists because three things have to agree and nothing used to make
them agree. Serving a Hindi head needed VERICALL_CKPT_A, VERICALL_CKPT_A2 and
VERICALL_CALIB_DIR all set consistently by hand, and the failure mode when they
disagreed was not an error -- it was confident numbers from a Platt mapping
fitted for a different model. That has already cost this project a day: a bare
VERICALL_CKPT_A resolved to a path that did not exist, the gateway served an
untrained head, and /health reported the configured filename as though it had
loaded (docs/RESULTS.md, and the _ckpt docstring in serving/app.py).

So the three move together, named once:

    VERICALL_PROFILE=hindi-telephony python run_local.py

Explicit variables still win over the profile. That is deliberate: it keeps
every existing script, test and deployment working unchanged, and leaves an
escape hatch for a one-off experiment without inventing a profile for it.

Adding a language is an entry in profiles.json and no code change at all. See
docs/ADDING_A_LANGUAGE.md.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "profiles.json"


@dataclass(frozen=True)
class Profile:
    id: str
    display: str
    languages: tuple[str, ...]
    calib_dir: Optional[str]
    ckpt_a: Optional[str]
    ckpt_a2: Optional[str]
    status: str
    measured: dict
    note: str

    @property
    def servable(self) -> bool:
        """A head without calibration has no likelihood-ratio scale.

        It can still rank, but every number this system reports downstream --
        the accumulated LLR, the risk score, the escalation threshold -- is in
        nats and assumes calibration. Serving an uncalibrated head would not
        look broken, which is the reason to refuse it rather than warn.
        """
        return self.status == "ready" and bool(self.calib_dir)


def _load() -> dict:
    if not REGISTRY.exists():
        return {"default": None, "profiles": {}}
    return json.loads(REGISTRY.read_text())


def all_profiles() -> dict[str, Profile]:
    raw = _load()
    out = {}
    for pid, p in raw.get("profiles", {}).items():
        out[pid] = Profile(
            id=pid,
            display=p.get("display", pid),
            languages=tuple(p.get("languages", ())),
            calib_dir=p.get("calib_dir"),
            ckpt_a=p.get("ckpt_a"),
            ckpt_a2=p.get("ckpt_a2"),
            status=p.get("status", "experimental"),
            measured=p.get("measured", {}),
            note=p.get("note", ""),
        )
    return out


def default_id() -> Optional[str]:
    return _load().get("default")


def get(pid: str) -> Profile:
    profiles = all_profiles()
    if pid not in profiles:
        known = ", ".join(sorted(profiles)) or "(none)"
        raise KeyError(f"unknown profile {pid!r}. Known profiles: {known}")
    return profiles[pid]


def apply_to_env(pid: str, env: Optional[dict] = None) -> dict[str, str]:
    """Set the three variables from a profile, without overriding explicit ones.

    Returns what it set, so a caller can report it rather than guess. Called
    before the serving module reads its configuration.
    """
    env = os.environ if env is None else env
    p = get(pid)
    if not p.servable:
        raise RuntimeError(
            f"profile {pid!r} is not servable: status={p.status}, "
            f"calibration={p.calib_dir or 'none'}. {p.note}")
    applied = {}
    for var, val in (("VERICALL_CALIB_DIR", p.calib_dir),
                     ("VERICALL_CKPT_A", p.ckpt_a),
                     ("VERICALL_CKPT_A2", p.ckpt_a2)):
        if val and not env.get(var):
            env[var] = val
            applied[var] = val
    return applied


def describe() -> str:
    """A table for --list-profiles. Status first: it is the thing that decides."""
    ps = all_profiles()
    dflt = default_id()
    lines = [f"{'profile':<18}{'status':<14}{'languages':<34}note"]
    lines.append("-" * 100)
    for pid in sorted(ps, key=lambda k: (ps[k].status != "ready", k)):
        p = ps[pid]
        mark = " *" if pid == dflt else ""
        langs = ", ".join(p.languages)
        lines.append(f"{pid + mark:<18}{p.status:<14}{langs[:32]:<34}"
                     f"{p.note[:46]}")
    lines.append("")
    lines.append(f"* default. Select with VERICALL_PROFILE=<id> or "
                 f"run_local.py --profile <id>.")
    return "\n".join(lines)
