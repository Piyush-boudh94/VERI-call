#!/usr/bin/env bash
# Re-run the section 13 paired protocol against the retrained Branch A heads.
#
# The comparison matrix (eval_folds.py) scores 1 s cached crops; section 13
# scored 4 s centre crops over 1,200 paired utterances. Both are valid, but only
# one of them is subtractable from the numbers already in docs/RESULTS.md -- and
# "did retraining fix the 79.08% inversion" is a question about THAT number.
# Same script, same crops, same utterances, only the checkpoint differs.
set -euo pipefail
# Resolve the project root from this script's own location, so the
# repo can be moved or exported to another machine unchanged.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=.venv/bin/python

echo "=== waiting for the mixed stage ==="
for _ in $(seq 1 540); do
    if grep -q "MIXED STAGE COMPLETE" logs/stage_mixed.log 2>/dev/null; then break; fi
    if grep -qi "Traceback" logs/stage_mixed.log 2>/dev/null; then
        echo "mixed stage failed:"; tail -30 logs/stage_mixed.log; exit 1
    fi
    sleep 10
done
grep -q "MIXED STAGE COMPLETE" logs/stage_mixed.log || { echo "timed out"; exit 1; }

for cfg in "asvspoof:checkpoints/branch_A_seed0.pt:docs/rtcfake_paired_asvspoof.json" \
           "rtc:checkpoints/branch_A_rtc_seed0.pt:docs/rtcfake_paired_rtc.json" \
           "mix:checkpoints/branch_A_mix_seed0.pt:docs/rtcfake_paired_mix.json"; do
    name="${cfg%%:*}"; rest="${cfg#*:}"; ck="${rest%%:*}"; out="${rest#*:}"
    echo ""
    echo "############ paired protocol, Branch A trained on: $name ############"
    $PY scripts/eval_rtcfake.py --branch-a "$ck" --out "$out"
done
echo "=== PAIRED RERUN COMPLETE ==="
