#!/usr/bin/env bash
# The whole Hindi measurement, on the GPU, in one pass.
#
# Run it inside the pod (scripts/pod.sh exec scripts/hindi_gpu_pipeline.sh).
# Measured 309 clips/s here against 5 clips/s on bmu-headnode, which has no
# GPU at all -- so this is ~60x, and the difference between minutes and hours.
#
# The feature cache goes to /scratch (1.8 TB node-local) and not to NFS: NFS
# home is 98% full, and the project's own notes record the pod serving stale
# reads of files the head node just wrote. Checkpoints and calibration are
# small and do land on NFS, because those are the artifacts that must survive
# the pod.
set -euo pipefail
ROOT="/home/dgx-s-bmu-cse-240577/SIH final"
CACHE="${VERICALL_CACHE:-/scratch/vericall-cache}"
cd "$ROOT"

say() { printf "\n\033[1m== %s\033[0m\n" "$*"; }

say "GPU"
python - <<'PY'
import torch
assert torch.cuda.is_available(), "no CUDA -- are you running this on the head node?"
p = torch.cuda.get_device_properties(0)
print(f"  {p.name}  {p.total_memory/2**30:.1f} GiB  {p.multi_processor_count} SMs")
PY

say "features: matched (train/dev/calib/eval) + unmatched (eval)"
python -u scripts/extract_features.py --dataset hindi_paired \
    --folds train dev calib eval_hindi --crops 4 --channels clean \
    --workers 8 --out "$CACHE"
python -u scripts/extract_features.py --dataset hindi_paired_unmatched \
    --folds eval_hindi --crops 4 --channels clean --workers 8 --out "$CACHE"

say "baseline: what the existing heads do on Hindi"
python -u scripts/eval_folds.py --cache "$CACHE" \
    --ckpt checkpoints/branch_A_seed0.pt checkpoints/branch_A_rtc_seed0.pt \
           checkpoints/branch_A_dann_seed0.pt checkpoints/branch_A_mix_seed0.pt \
    --folds hindi_paired:eval_hindi --out docs/eval_hindi_baseline.json

say "train the head on Hindi -- the experiment that had never been run"
python -u scripts/train_crossdomain.py --cache "$CACHE" \
    --datasets hindi_paired --train-fold train --dev-fold dev \
    --eval-on hindi_paired:eval_hindi --epochs 16 --seeds 0 1 2 \
    --balanced --tag hindi

say "the same head on the UNMATCHED arm -- how much of it is codec, not synthesis"
python -u scripts/eval_folds.py --cache "$CACHE" \
    --ckpt checkpoints/branch_Ahindi_seed0.pt \
    --folds hindi_paired:eval_hindi hindi_paired_unmatched:eval_hindi \
    --out docs/eval_hindi_codec_arms.json

printf "\n\033[32mDONE\033[0m -- cache in %s, results in docs/\n\n" "$CACHE"
