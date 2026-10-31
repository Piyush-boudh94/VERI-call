#!/usr/bin/env bash
# Rebuild only the ASVspoof folds. The full extract_all.sh also covers WaveFake,
# whose corpus was lost in the pod wipe; this is what can be rebuilt right now.
set -euo pipefail
# Resolve the project root from this script's own location, so the
# repo can be moved or exported to another machine unchanged.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=.venv/bin/python
W=${WORKERS:-14}
echo "=== train (4 crops) ==="
$PY scripts/extract_features.py --dataset asvspoof2019_la --folds train --crops 4 --workers "$W"
echo "=== dev/calib (2 crops) ==="
$PY scripts/extract_features.py --dataset asvspoof2019_la --folds dev calib --crops 2 --workers "$W"
echo "=== eval (1 crop) ==="
$PY scripts/extract_features.py --dataset asvspoof2019_la --folds eval_in_domain --crops 1 --workers "$W"
echo "=== ASVSPOOF EXTRACTION COMPLETE ==="
