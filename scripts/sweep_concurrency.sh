#!/usr/bin/env bash
# Find the largest N meeting every SLO simultaneously. Criterion declared before
# the run, per the project's own gate discipline:
#   chunk_inference p99 <= 250 ms  AND  all calls complete  AND
#   the server keeps up with real time (wall clock ~ audio duration)
set -uo pipefail
# Resolve the project root from this script's own location, so the
# repo can be moved or exported to another machine unchanged.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=.venv/bin/python
pkill -f "uvicorn vericall" 2>/dev/null || true
sleep 3
setsid nohup $PY -m uvicorn vericall.serving.app:app --host 0.0.0.0 --port 8001 \
    > logs/serve_sweep.log 2>&1 < /dev/null &
sleep 35
for n in ${NS:-1 4 16 32 50 64}; do
    echo "=== N=$n ==="
    $PY -m vericall.bench.load --calls "$n" --seconds 12 --port 8001 2>&1 \
        | grep -E "wall clock|completed|failures|first_score|chunk_inference|windows scored"
    $PY - <<'EOPY'
import json, urllib.request
d = json.load(urllib.request.urlopen("http://127.0.0.1:8001/metrics", timeout=10))
b = d.get("cross_call_batching") or {}
print(f"  mean_batch {b.get('mean_batch', 0)}  max {b.get('max_batch_seen', 0)}"
      f"  rejected {b.get('rejected', 0)}  queue_depth {b.get('queue_depth', 0)}"
      f"  batch_wait_p99 {b.get('submit_to_result_ms_p99', 0)} ms")
EOPY
    echo
    sleep 3
done
pkill -f "uvicorn vericall" 2>/dev/null || true
echo "=== SWEEP COMPLETE ==="
