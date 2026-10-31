#!/usr/bin/env bash
# Close the remaining branch gaps, in order, after Branch A2 finishes.
#
# Sequential rather than parallel on purpose: there is exactly one
# mig-1g.18gb slice, and two jobs contending for 16 SMs finish later than the
# same two run back to back, while making every timing measurement meaningless.
set -euo pipefail
# Resolve the project root from this script's own location, so the
# repo can be moved or exported to another machine unchanged.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=.venv/bin/python

echo "=== waiting for Branch A2 to finish ==="
while pgrep -f "train.py --branch A2" > /dev/null; do sleep 20; done
echo "A2 done"

echo; echo "=== Branch B: one-class prosody (Deep-SVDD) ==="
$PY scripts/train_branch_b.py --epochs 8 --batch 64

echo; echo "=== Branch C: AS-Norm cohort + identity calibration ==="
$PY scripts/fit_branch_c.py --cohort 1500 --trials 4000

echo; echo "=== per-subset calibration table ==="
$PY scripts/fit_subsets.py

echo; echo "=== ALL BRANCH GAPS CLOSED ==="
