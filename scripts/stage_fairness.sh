#!/usr/bin/env bash
# Re-fetch RTCFake and rebuild only the eval folds needed for the per-language
# false-positive measurement. Scratch is ephemeral and has been wiped 7 times;
# everything here is reproducible from NFS-resident manifests and splits.
set -euo pipefail
# Resolve the project root from this script's own location, so the
# repo can be moved or exported to another machine unchanged.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=.venv/bin/python
$PY scripts/fetch_datasets.py --datasets rtcfake --workers 8
cd /tmp/vericall-data/rtcfake/data/wav && tar xzf dev.tar.gz && cd -
$PY scripts/build_manifests.py --datasets rtcfake
$PY scripts/extract_features.py --dataset rtcfake --folds eval_rtc_en eval_rtc_zh \
    --crops 1 --channels clean --workers 12
echo "=== FAIRNESS DATA READY ==="
