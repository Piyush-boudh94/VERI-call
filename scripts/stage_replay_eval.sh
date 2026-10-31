#!/usr/bin/env bash
# Replay evaluation -- the threat THREAT_MODEL.md A4 has listed as untested.
#
# eval_replay and eval_synth_echo share the same bona fide clips and differ
# only in attack type, so the gap between their EERs is a measurement of
# "does physical replay evade this detector" rather than two unrelated numbers.
set -euo pipefail
# Resolve the project root from this script's own location, so the
# repo can be moved or exported to another machine unchanged.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=.venv/bin/python
W=${WORKERS:-12}

$PY scripts/extract_features.py --dataset echofake \
    --folds eval_replay eval_synth_echo --crops 1 --workers "$W"

$PY scripts/eval_folds.py \
    --ckpt checkpoints/branch_A_seed0.pt checkpoints/branch_A_rtc_seed0.pt \
           checkpoints/branch_A_dann_seed0.pt \
    --folds echofake:eval_replay echofake:eval_synth_echo \
    --out docs/eval_replay.json
echo "=== REPLAY EVAL COMPLETE ==="
