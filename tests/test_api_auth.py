"""Every mutating or data-returning REST endpoint must honour the token.

Three /api/calls endpoints were unauthenticated while /ws/ and /api/enroll/
both required a token -- so the intent was never in doubt, the REST surface was
simply never wired to it. GET leaked per-call risk scores, timings and client
addresses; POST /api/calls/prune DELETED call history, which is the record an
incident review starts from.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from vericall.serving import app as appmod

APP = Path(__file__).resolve().parents[1] / "vericall" / "serving" / "app.py"

# Endpoints that return caller data or change state, and must refuse outright
# without a token. /health and /metrics are deliberately absent, for different
# reasons: a monitoring scrape that needs a secret tends to end up with the
# secret in a config file everyone can read, and /health answers 200 to an
# unauthenticated probe but narrows what it says -- graduated rather than
# guarded, pinned by the tests at the end of this file.
GUARDED = (
    "/api/calls",
    "/api/calls/{call_id}",
    "/api/calls/prune",
    "/api/enroll",
    "/api/enroll/{speaker_id}",
)


def route_handlers():
    out = {}
    for r in appmod.app.routes:
        path = getattr(r, "path", None)
        fn = getattr(r, "endpoint", None)
        if path and fn:
            out.setdefault(path, []).append(fn)
    return out


@pytest.mark.parametrize("path", GUARDED)
def test_the_endpoint_exists(path):
    assert path in route_handlers(), f"{path} is not registered"


@pytest.mark.parametrize("path", GUARDED)
def test_every_data_endpoint_takes_a_request_and_checks_auth(path):
    """It must accept `request` -- without it there is nothing to read a token
    from -- and its body must call _require_auth."""
    for fn in route_handlers()[path]:
        sig = inspect.signature(fn)
        assert "request" in sig.parameters, (
            f"{path} ({fn.__name__}) cannot see the token: no `request` param")
        src = inspect.getsource(fn)
        assert "_require_auth" in src, (
            f"{path} ({fn.__name__}) never calls _require_auth")


def test_auth_returns_a_refusal_rather_than_raising(monkeypatch):
    """A refusal is a JSON body the page can render. An exception would be a
    500, which reads as a broken server rather than a locked door."""
    monkeypatch.setenv("VERICALL_AUTH_TOKEN", "sekrit")

    class Req:
        query_params: dict = {}
        headers: dict = {}

    bad = appmod._require_auth(Req())
    assert bad == {"error": "unauthorized"}


def test_the_right_token_passes_by_header_or_query(monkeypatch):
    monkeypatch.setenv("VERICALL_AUTH_TOKEN", "sekrit")

    class ByHeader:
        query_params: dict = {}
        headers = {"x-vericall-token": "sekrit"}

    class ByQuery:
        query_params = {"token": "sekrit"}
        headers: dict = {}

    assert appmod._require_auth(ByHeader()) is None
    assert appmod._require_auth(ByQuery()) is None


def test_no_token_configured_means_open_and_that_is_reported(monkeypatch):
    """Correct for a laptop, wrong for anything reachable. The gateway is
    allowed to run open and must say so -- NO AUTH TOKEN is in
    DEGRADATION_MARKERS, so VERICALL_STRICT=1 refuses to start that way."""
    monkeypatch.delenv("VERICALL_AUTH_TOKEN", raising=False)

    class Req:
        query_params: dict = {}
        headers: dict = {}

    assert appmod._require_auth(Req()) is None
    assert any("NO AUTH TOKEN" in m for m in appmod.DEGRADATION_MARKERS)


def test_health_and_metrics_stay_open():
    """Asserted so nobody 'fixes' them into needing a secret: a liveness probe
    that requires a credential is a liveness probe that stops being used."""
    handlers = route_handlers()
    for path in ("/health", "/metrics"):
        assert path in handlers
        for fn in handlers[path]:
            assert "_require_auth" not in inspect.getsource(fn), path


# --- /health: graduated disclosure -----------------------------------------
#
# /health is the endpoint most likely to be reachable, and it was returning
# checkpoint names, calibration provenance, the ICE server list, the enrolled
# voiceprint count and the exact admission limits to anyone who asked. That is
# reconnaissance, not liveness. It is graduated now, and these tests pin both
# halves: a probe with no token still works (a probe that needs a secret is a
# probe that gets switched off) and the configuration detail is gone from it.
#
# Called directly rather than over a TestClient, like the rest of this file:
# app startup loads checkpoints, and none of what is under test needs them.

_RECON = ("checkpoints", "calib_dir", "fusion_weights", "limits", "webrtc",
          "identity", "notes", "integrity", "calibration_fitted_for",
          "admission", "call_history", "branches_loaded")


class _Req:
    """Only what health() reads: query params and headers."""

    def __init__(self, token=None, header=None):
        self.query_params = {"token": token} if token else {}
        self.headers = {"x-vericall-token": header} if header else {}


def test_health_is_full_when_no_token_configured(monkeypatch):
    """The laptop case. No configured token means nothing to protect, and the
    demo tooling reads these fields -- it has to keep working untouched."""
    monkeypatch.delenv("VERICALL_AUTH_TOKEN", raising=False)
    body = appmod.health(_Req())
    assert body["auth_required"] is False
    for key in _RECON:
        assert key in body, f"{key} should be visible with no token configured"


def test_health_without_token_is_liveness_only(monkeypatch):
    monkeypatch.setenv("VERICALL_AUTH_TOKEN", "s3cret")
    body = appmod.health(_Req())
    assert body["auth_required"] is True
    assert set(body) == {"status", "active_calls", "degraded",
                         "auth_required", "detail"}
    for key in _RECON:
        assert key not in body, f"{key} leaked to an unauthenticated caller"


def test_health_unauthenticated_still_answers_liveness(monkeypatch):
    """No exception and a real status: a 401 here reads as an unhealthy pod
    and would take the service out rather than protect it."""
    monkeypatch.setenv("VERICALL_AUTH_TOKEN", "s3cret")
    body = appmod.health(_Req())
    assert body["status"] in {"ok", "loading", "draining"}
    assert isinstance(body["active_calls"], int)


@pytest.mark.parametrize("req", [_Req(token="s3cret"), _Req(header="s3cret")])
def test_health_with_token_is_full(monkeypatch, req):
    """Both carriers work -- a browser can only put it in the query string."""
    monkeypatch.setenv("VERICALL_AUTH_TOKEN", "s3cret")
    body = appmod.health(req)
    for key in _RECON:
        assert key in body, f"{key} missing for an authenticated operator"


def test_health_wrong_token_gets_the_thin_payload(monkeypatch):
    monkeypatch.setenv("VERICALL_AUTH_TOKEN", "s3cret")
    body = appmod.health(_Req(token="wrong"))
    assert "checkpoints" not in body


def test_health_degraded_is_a_bool_not_the_notes(monkeypatch):
    """The thin payload says whether anything is lost, never what."""
    monkeypatch.setenv("VERICALL_AUTH_TOKEN", "s3cret")
    assert isinstance(appmod.health(_Req())["degraded"], bool)
