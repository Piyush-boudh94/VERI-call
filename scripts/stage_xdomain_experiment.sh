#!/usr/bin/env bash
# The section 16 experiment: does anything fix cross-corpus generalisation?
#
# Four configurations on identical folds, so the rows are comparable to the
# published matrix. The two single-corpus baselines are already measured.
set -uo pipefail
# Resolve the project root from this script's own location, so the
# repo can be moved or exported to another machine unchanged.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=.venv/bin/python

echo "=== waiting for caches ==="
for _ in $(seq 1 540); do
    grep -q "STAGE ALL COMPLETE" logs/stage_all.log 2>/dev/null && break
    grep -qi "Traceback\|No space left" logs/stage_all.log 2>/dev/null && {
        echo "staging failed:"; tail -20 logs/stage_all.log; exit 1; }
    sleep 10
done
grep -q "STAGE ALL COMPLETE" logs/stage_all.log || { echo "timed out"; exit 1; }

EV="asvspoof2019_la:eval_in_domain wavefake:eval_unseen_vocoder rtcfake:eval_rtc_en rtcfake:eval_rtc_zh"

echo ""; echo "############ 1/4  naive mix (reproduce the section 16 baseline) ############"
$PY scripts/train_crossdomain.py --datasets asvspoof2019_la rtcfake \
    --eval-on $EV --tag _naive --seeds 0

echo ""; echo "############ 2/4  balanced sampling only ############"
$PY scripts/train_crossdomain.py --datasets asvspoof2019_la rtcfake \
    --eval-on $EV --balanced --tag _bal --seeds 0

echo ""; echo "############ 3/4  balanced + domain adversary ############"
$PY scripts/train_crossdomain.py --datasets asvspoof2019_la rtcfake \
    --eval-on $EV --balanced --dann --tag _dann --seeds 0

echo ""; echo "############ 4/4  balanced + adversary + OC-Softmax ############"
$PY scripts/train_crossdomain.py --datasets asvspoof2019_la rtcfake \
    --eval-on $EV --balanced --dann --ocsoftmax --tag _oc --seeds 0

echo ""; echo "=== comparison matrix, one code path ==="
$PY scripts/eval_folds.py \
    --ckpt checkpoints/branch_A_seed0.pt \
           checkpoints/branch_A_rtc_seed0.pt \
           checkpoints/branch_A_naive_seed0.pt \
           checkpoints/branch_A_bal_seed0.pt \
           checkpoints/branch_A_dann_seed0.pt \
    --folds asvspoof2019_la:eval_in_domain wavefake:eval_unseen_vocoder \
            rtcfake:eval_rtc_en rtcfake:eval_rtc_zh \
    --out docs/eval_matrix_xdomain.json
echo "=== XDOMAIN EXPERIMENT COMPLETE ==="
