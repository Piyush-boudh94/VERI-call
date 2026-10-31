#!/usr/bin/env bash
# Make the Hindi detector channel-robust, and fit the buckets it was missing.
#
# Run inside the GPU pod. VERICALL_STRICT=1 refused to start the Hindi profile
# with two complaints, and this closes the second one:
#
#   "codec detection is on but these buckets are missing: g711, noisy_line
#    -- calls detected as those keep the startup bucket"
#
# calib_hindi held only `clean` and `pooled`, because the Hindi corpus was
# extracted with --channels clean. A Hindi call arriving over G.711 therefore
# got the clean mapping, and docs/RESULTS.md section 10 measured the cost of
# exactly that: fitted slopes differ 1.24x for A and 1.67x for A2, and A2
# degrades 10.54x on a noisy line. The problem statement is about telephony and
# VoIP, so the narrowband case is not an edge case here -- it is the main one.
#
# Channels are cycled across the 4 crops per clip, so each clip appears clean,
# narrowband and lossy. That is augmentation for training and, at the same
# time, the labelled material fit_channels.py needs per bucket.
set -euo pipefail
ROOT="/home/dgx-s-bmu-cse-240577/SIH final"
CACHE="${VERICALL_CACHE:-/scratch/vericall-cache-ch}"
cd "$ROOT"
say() { printf "\n\033[1m== %s\033[0m\n" "$*"; }

say "GPU"
python -c "import torch;p=torch.cuda.get_device_properties(0);print(f'  {p.name} {p.total_memory/2**30:.1f}GiB')"

say "features, three channels cycled across four crops"
python -u scripts/extract_features.py --dataset hindi_paired \
    --folds train dev calib eval_hindi --crops 4 \
    --channels clean g711_mulaw noisy_line --workers 8 --out "$CACHE"

say "retrain A on channel-augmented Hindi"
python -u scripts/train_crossdomain.py --cache "$CACHE" --datasets hindi_paired \
    --train-fold train --dev-fold dev --eval-on hindi_paired:eval_hindi \
    --epochs 18 --seeds 0 1 2 --balanced --tag hindich

say "retrain A2 on channel-augmented Hindi"
python -u scripts/train.py --branch A2 --cache "$CACHE" --dataset hindi_paired \
    --eval-on hindi_paired:eval_hindi --epochs 18 --seeds 0 1 2 --tag hindich

say "per-channel calibration for both branches"
python -u scripts/fit_channels.py --branch A  --ckpt checkpoints/branch_Ahindich_seed0.pt \
    --cache "$CACHE" --dataset hindi_paired --fold calib --out calib_hindi_ch
python -u scripts/fit_channels.py --branch A2 --ckpt checkpoints/branch_A2hindich_seed0.pt \
    --cache "$CACHE" --dataset hindi_paired --fold calib --out calib_hindi_ch

say "subset table for these heads"
python -u scripts/fit_subsets.py --cache "$CACHE" --dataset hindi_paired --fold calib \
    --a-ckpt checkpoints/branch_Ahindich_seed0.pt \
    --a2-ckpt checkpoints/branch_A2hindich_seed0.pt \
    --calib-dir calib_hindi_ch --out calib_hindi_ch/subsets.json --per-channel

say "eval: per channel, so the bucket claim is measured not assumed"
python -u scripts/eval_folds.py --cache "$CACHE" \
    --ckpt checkpoints/branch_Ahindich_seed0.pt checkpoints/branch_Ahindi_seed0.pt \
    --folds hindi_paired:eval_hindi --out docs/eval_hindi_channels.json

printf "\n\033[32mDONE\033[0m -- calib_hindi_ch/, cache %s\n\n" "$CACHE"
