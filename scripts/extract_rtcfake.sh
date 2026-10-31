#!/usr/bin/env bash
# Build the RTCFake feature cache.
#
# --channels clean everywhere, deliberately. RTCFake's value is that its channel
# effects are REAL -- the same utterance offline and online through a live RTC
# platform. Stacking the G.711/noisy-line simulator on top would train the model
# on a distribution that exists nowhere: simulated degradation applied to
# already-transmitted speech. docs/RESULTS.md section 10 measured the simulator
# and explicitly recorded that it is not the deployment. The offline/online
# pairing IS the channel augmentation here.
#
# Multiple crops are still used on train/dev/calib: those are different random
# 1 s windows of the same clip, which adds time diversity without inventing
# channel conditions.
set -euo pipefail
# Resolve the project root from this script's own location, so the
# repo can be moved or exported to another machine unchanged.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=.venv/bin/python
W=${WORKERS:-14}
echo "=== rtcfake train (4 crops, clean) ==="
$PY scripts/extract_features.py --dataset rtcfake --folds train --crops 4 --channels clean --workers "$W"
echo "=== rtcfake dev/calib (2 crops, clean) ==="
$PY scripts/extract_features.py --dataset rtcfake --folds dev calib --crops 2 --channels clean --workers "$W"
echo "=== rtcfake eval folds (1 crop, clean) ==="
$PY scripts/extract_features.py --dataset rtcfake --folds eval_rtc_en eval_rtc_zh --crops 1 --channels clean --workers "$W"
echo "=== RTCFAKE EXTRACTION COMPLETE ==="
