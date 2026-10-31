#!/usr/bin/env bash
# Run the VeriCall gateway.
#
# `python -m vericall.serving.app` does NOT work and never did: app.py defines
# the FastAPI app at module scope and has no __main__ block, so running it as a
# module imports it and exits 0 -- no error, no server, exit status success.
# That is the worst shape a demo-path bug can have, and it survived because the
# load tests (sweep_concurrency.sh, ab_batching.sh) invoke uvicorn directly and
# never used this script.
set -euo pipefail
# Resolve the project root from this script's own location, so the
# repo can be moved or exported to another machine unchanged.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${VERICALL_PORT:-8001}"
HOST="${VERICALL_HOST:-0.0.0.0}"
# Prefer the project venv, fall back to whatever python is on PATH. A hard
# .venv/bin/python breaks on Windows (Scripts/python.exe) and on any machine
# that installed the deps globally.
if [ -x .venv/bin/python ]; then PY=.venv/bin/python
elif [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe
else PY="$(command -v python3 || command -v python)"; fi
exec "$PY" -m uvicorn vericall.serving.app:app \
    --host "$HOST" --port "$PORT" "$@"
