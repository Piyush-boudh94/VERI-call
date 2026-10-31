#!/usr/bin/env bash
# Branch A2 and B are already trained (checkpoints survived the pod wipe on
# NFS) -- only their cache-dependent fits (C's cohort, the subset table) need
# redoing after a scratch wipe. Both only touch the train and calib folds, not
# dev/eval, so this does not wait for the full extract_asvspoof.sh pipeline.
#
# Waits on the actual completion marker (info.json), not process existence --
# a `pgrep` check right after launching a sibling process races the process
# actually starting and can pass through empty. Cost one wasted run already.
set -euo pipefail
# Resolve the project root from this script's own location, so the
# repo can be moved or exported to another machine unchanged.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=.venv/bin/python
CACHE=/tmp/vericall-cache/asvspoof2019_la

echo "=== waiting for train + calib folds ==="
while [ ! -f "$CACHE/train/info.json" ] || [ ! -f "$CACHE/calib/info.json" ]; do
    sleep 15
done
echo "folds ready"

echo; echo "=== Branch C: AS-Norm cohort + identity calibration ==="
$PY scripts/fit_branch_c.py --cohort 1500 --trials 4000

echo; echo "=== per-subset calibration table ==="
$PY scripts/fit_subsets.py

echo; echo "=== C + SUBSETS COMPLETE ==="
