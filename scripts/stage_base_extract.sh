#!/usr/bin/env bash
# Rebuild the ASVspoof + WaveFake caches after a scratch wipe, waiting for the
# fetch to finish first.
#
# A script rather than an inlined chain: this runs through kubectl exec ->
# bash -lc -> bash -c, and nested quoting across those layers silently mangles
# && chains. That cost one whole launch that appeared to start and did nothing.
set -euo pipefail
# Resolve the project root from this script's own location, so the
# repo can be moved or exported to another machine unchanged.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=.venv/bin/python
W=${WORKERS:-12}

echo "=== waiting for fetch to finish ==="
# The fetch writes its own completion marker; polling for the process is racy
# (a pgrep can pass before the sibling has started), so wait on the log line.
for _ in $(seq 1 360); do
    if grep -q "=== summary ===" logs/fetch_base.log 2>/dev/null; then break; fi
    sleep 10
done
grep -q "=== summary ===" logs/fetch_base.log || { echo "fetch did not finish"; exit 1; }
tail -5 logs/fetch_base.log

echo "=== manifests ==="
$PY scripts/build_manifests.py --datasets asvspoof2019_la wavefake

echo "=== asvspoof train (4 crops) ==="
$PY scripts/extract_features.py --dataset asvspoof2019_la --folds train --crops 4 --workers "$W"
echo "=== asvspoof dev/calib (2 crops) ==="
$PY scripts/extract_features.py --dataset asvspoof2019_la --folds dev calib --crops 2 --workers "$W"
echo "=== asvspoof eval (1 crop) ==="
$PY scripts/extract_features.py --dataset asvspoof2019_la --folds eval_in_domain --crops 1 --workers "$W"
echo "=== wavefake eval folds (1 crop) ==="
$PY scripts/extract_features.py --dataset wavefake --folds eval_seen_vocoder eval_unseen_vocoder --crops 1 --workers "$W"
echo "=== BASE EXTRACTION COMPLETE ==="
