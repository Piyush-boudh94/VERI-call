#!/usr/bin/env bash
# Build the full training cache.
#
# A script file rather than an inlined command chain: this runs through
# kubectl exec -> bash -lc -> bash -c, and nested quoting across those layers
# silently mangles && chains -- which cost one whole launch that appeared to
# start and did nothing.
set -euo pipefail
# Resolve the project root from this script's own location, so the
# repo can be moved or exported to another machine unchanged.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=.venv/bin/python
W=${WORKERS:-14}

echo "=== asvspoof train (4 crops) ==="
$PY scripts/extract_features.py --dataset asvspoof2019_la --folds train  --crops 4 --workers "$W"
echo "=== asvspoof dev/calib (2 crops) ==="
$PY scripts/extract_features.py --dataset asvspoof2019_la --folds dev calib --crops 2 --workers "$W"
echo "=== asvspoof eval (1 crop) ==="
$PY scripts/extract_features.py --dataset asvspoof2019_la --folds eval_in_domain --crops 1 --workers "$W"
echo "=== wavefake train (4 crops) ==="
$PY scripts/extract_features.py --dataset wavefake --folds train --crops 4 --workers "$W"
echo "=== wavefake dev/calib (2 crops) ==="
$PY scripts/extract_features.py --dataset wavefake --folds dev calib --crops 2 --workers "$W"
echo "=== wavefake eval folds (1 crop) ==="
$PY scripts/extract_features.py --dataset wavefake --folds eval_seen_vocoder eval_unseen_vocoder --crops 1 --workers "$W"
echo "=== ALL EXTRACTION COMPLETE ==="
