#!/usr/bin/env bash
# Rebuild every cache needed for cross-corpus training after a scratch wipe.
# Reproducible from NFS-resident manifests and splits; nothing here is precious.
set -euo pipefail
# Resolve the project root from this script's own location, so the
# repo can be moved or exported to another machine unchanged.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=.venv/bin/python
W=${WORKERS:-12}

$PY scripts/fetch_datasets.py --datasets asvspoof2019_la wavefake rtcfake --workers 8
cd /tmp/vericall-data/rtcfake/data/wav && tar xzf train.tar.gz && tar xzf dev.tar.gz && cd -
$PY scripts/build_manifests.py --datasets asvspoof2019_la wavefake rtcfake

echo "=== asvspoof ==="
$PY scripts/extract_features.py --dataset asvspoof2019_la --folds train --crops 4 --workers "$W"
$PY scripts/extract_features.py --dataset asvspoof2019_la --folds dev calib --crops 2 --workers "$W"
$PY scripts/extract_features.py --dataset asvspoof2019_la --folds eval_in_domain --crops 1 --workers "$W"
echo "=== wavefake ==="
$PY scripts/extract_features.py --dataset wavefake --folds train --crops 2 --workers "$W"
$PY scripts/extract_features.py --dataset wavefake --folds eval_unseen_vocoder --crops 1 --workers "$W"
echo "=== rtcfake (clean only: its channel effects are already real) ==="
$PY scripts/extract_features.py --dataset rtcfake --folds train --crops 4 --channels clean --workers "$W"
$PY scripts/extract_features.py --dataset rtcfake --folds dev calib --crops 2 --channels clean --workers "$W"
$PY scripts/extract_features.py --dataset rtcfake --folds eval_rtc_en eval_rtc_zh --crops 1 --channels clean --workers "$W"
echo "=== STAGE ALL COMPLETE ==="
