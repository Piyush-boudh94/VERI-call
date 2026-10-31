#!/usr/bin/env bash
# Which SSL layer generalises best? The 2026 literature rates the readout layer
# as a dominant factor in out-of-domain performance; this project picked layer 5
# from a general argument and never measured it.
#
# All layers come from ONE encoder pass, so the extraction costs the same as a
# single-layer run. Folds are kept small on purpose: the question is which layer
# RANKS best cross-corpus, and that does not need the full train set.
set -euo pipefail
# Resolve the project root from this script's own location, so the
# repo can be moved or exported to another machine unchanged.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=.venv/bin/python
W=${WORKERS:-12}
LAYERS="3 5 7 9 11"

# Scratch has been wiped ten times in this project, twice mid-job, so the
# sweep fetches what it needs rather than assuming it is there. Everything
# below is reproducible from NFS-resident manifests and splits.
echo "=== fetching corpora (scratch is ephemeral) ==="
$PY scripts/fetch_datasets.py --datasets asvspoof2019_la wavefake rtcfake --workers 8
if [ ! -d /tmp/vericall-data/rtcfake/data/wav/dev ]; then
  cd /tmp/vericall-data/rtcfake/data/wav && tar xzf dev.tar.gz && cd -
fi
$PY scripts/build_manifests.py --datasets asvspoof2019_la wavefake rtcfake

echo "=== extracting layers $LAYERS (one pass) ==="
$PY scripts/extract_features.py --dataset asvspoof2019_la --folds train \
    --crops 2 --layers $LAYERS --workers "$W"
$PY scripts/extract_features.py --dataset asvspoof2019_la --folds dev \
    --crops 1 --layers $LAYERS --workers "$W"
$PY scripts/extract_features.py --dataset wavefake --folds eval_unseen_vocoder \
    --crops 1 --layers $LAYERS --workers "$W"
$PY scripts/extract_features.py --dataset rtcfake --folds eval_rtc_en eval_rtc_zh \
    --crops 1 --channels clean --layers $LAYERS --workers "$W"

for L in $LAYERS; do
  echo ""
  echo "############ layer $L ############"
  $PY scripts/train_crossdomain.py --datasets asvspoof2019_la \
      --eval-on "wavefake:eval_unseen_vocoder@L$L" \
                "rtcfake:eval_rtc_en@L$L" "rtcfake:eval_rtc_zh@L$L" \
      --train-fold "train@L$L" --dev-fold "dev@L$L" \
      --epochs 12 --tag "_L$L" --seeds 0
done
echo "=== LAYER SWEEP COMPLETE ==="
