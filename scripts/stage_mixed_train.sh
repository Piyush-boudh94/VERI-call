#!/usr/bin/env bash
# Train Branch A on ASVspoof + RTCFake-en together, then score every checkpoint
# against every fold through one code path.
#
# The question this answers: docs/RESULTS.md section 13 found Branch A trained
# on ASVspoof INVERTING on RTCFake Mandarin (79.08% EER -- anti-correlated, not
# merely uninformative). Three configurations, one comparison matrix:
#
#   branch_A_seed0        ASVspoof only      the baseline that inverted
#   branch_A_rtc_seed0    RTCFake-en only    does the corpus fix it?
#   branch_A_mix_seed0    both               or does mixing just relocate it?
#
# Mandarin is held out of training in BOTH new configurations, so eval_rtc_zh
# stays a genuine cross-lingual test in every row of the matrix.
set -euo pipefail
# Resolve the project root from this script's own location, so the
# repo can be moved or exported to another machine unchanged.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=.venv/bin/python

echo "=== waiting for base extraction ==="
for _ in $(seq 1 360); do
    if grep -q "BASE EXTRACTION COMPLETE" logs/stage_base.log 2>/dev/null; then break; fi
    if grep -qi "Traceback" logs/stage_base.log 2>/dev/null; then
        echo "base extraction failed:"; tail -20 logs/stage_base.log; exit 1
    fi
    sleep 10
done
grep -q "BASE EXTRACTION COMPLETE" logs/stage_base.log || { echo "timed out"; exit 1; }

echo "=== waiting for the rtcfake-only run to release the GPU ==="
for _ in $(seq 1 180); do
    if grep -q "duration-only baseline" logs/train_A_rtc.log 2>/dev/null; then break; fi
    sleep 10
done

echo "=== train Branch A on asvspoof + rtcfake ==="
$PY scripts/train.py --branch A --dataset asvspoof2019_la rtcfake --tag _mix \
    --eval-on asvspoof2019_la:eval_in_domain wavefake:eval_unseen_vocoder \
              rtcfake:eval_rtc_en rtcfake:eval_rtc_zh \
    --seeds 0 1 2

echo "=== comparison matrix ==="
$PY scripts/eval_folds.py \
    --ckpt checkpoints/branch_A_seed0.pt \
           checkpoints/branch_A_rtc_seed0.pt \
           checkpoints/branch_A_mix_seed0.pt \
    --folds asvspoof2019_la:eval_in_domain wavefake:eval_unseen_vocoder \
            rtcfake:eval_rtc_en rtcfake:eval_rtc_zh
echo "=== MIXED STAGE COMPLETE ==="
