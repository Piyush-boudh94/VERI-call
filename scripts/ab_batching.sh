#!/usr/bin/env bash
# A/B the serving path with and without cross-call batching at identical load.
# docs/BENCHMARKS.md argues for batching from OFFLINE microbenchmarks of the raw
# forward pass. Whether that transfers to ONLINE serving at this arrival rate is
# an empirical question; this answers it instead of assuming.
set -uo pipefail
# Resolve the project root from this script's own location, so the
# repo can be moved or exported to another machine unchanged.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=.venv/bin/python
CALLS=${CALLS:-8}
SECONDS_=${SECONDS_:-15}

for mode in 0 1; do
    label=$([ "$mode" = "1" ] && echo "BATCHED" || echo "DIRECT ")
    pkill -f "uvicorn vericall" 2>/dev/null || true
    sleep 3
    VERICALL_BATCHING=$mode setsid nohup $PY -m uvicorn vericall.serving.app:app \
        --host 0.0.0.0 --port 8001 > logs/serve_ab_$mode.log 2>&1 < /dev/null &
    sleep 35
    echo "=== $label (VERICALL_BATCHING=$mode), $CALLS calls x ${SECONDS_}s ==="
    $PY -m vericall.bench.load --calls "$CALLS" --seconds "$SECONDS_" --port 8001 \
        2>&1 | grep -E "wall clock|completed|first_score|chunk_inference|windows scored"
    echo
done
pkill -f "uvicorn vericall" 2>/dev/null || true
echo "=== A/B COMPLETE ==="
