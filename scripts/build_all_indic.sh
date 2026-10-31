#!/usr/bin/env bash
# Build a paired bona-fide/spoof corpus for each Indic language.
#
# The same construction that produced the Hindi result in docs/RESULTS.md
# section 25, applied to the rest: Kathbath supplies the real speech,
# IndicSynth supplies clones of THOSE SAME SPEAKERS, and the two are reunited
# rather than borrowed across corpora.
#
# GUJARATI IS BUILT AND NOT SCORED. It is the frozen holdout -- the project no
# longer has an untouched one, because In-the-Wild has been scored by six
# checkpoints and CodecFake by five. Building it now costs nothing and keeps
# one language that has never influenced a decision.
set -euo pipefail
ROOT="/home/dgx-s-bmu-cse-240577/SIH final"
DATA="${VERICALL_DATA:-/scratch/vericall-data}"
cd "$ROOT"

LANGS="${1:-bengali marathi tamil telugu gujarati}"
declare -A CODE=( [bengali]=bn [marathi]=mr [tamil]=ta [telugu]=te [gujarati]=gu )
declare -A DIR=(  [bengali]=Bengali [marathi]=Marathi [tamil]=Tamil \
                  [telugu]=Telugu [gujarati]=Gujarati )

for lang in $LANGS; do
    code="${CODE[$lang]}"; dirname="${DIR[$lang]}"
    printf "\n\033[1m===== %s =====\033[0m\n" "$lang"

    # IndicSynth manifest for this language, if not already built.
    if [ ! -s "data/manifests/indicsynth_${code}.jsonl" ]; then
        python - "$code" "$DATA/indicsynth_${code}" <<'PY'
import json, sys
sys.path.insert(0, "/home/dgx-s-bmu-cse-240577/SIH final")
from vericall.data.manifest import build_indicsynth
code, root = sys.argv[1], sys.argv[2]
m = build_indicsynth(root, dataset=f"indicsynth_{code}")
out = f"data/manifests/indicsynth_{code}.jsonl"
with open(out, "w") as fh:
    for it in m:
        fh.write(json.dumps(it.__dict__ if hasattr(it, "__dict__")
                            else {k: getattr(it, k) for k in it.__slots__}) + "\n")
print(f"  manifest: {len(m)} rows -> {out}")
PY
    fi

    python -u scripts/build_indic_paired.py \
        --language "$lang" \
        --kathbath "$DATA/kathbath" \
        --indicsynth-manifest "data/manifests/indicsynth_${code}.jsonl" \
        --out "$DATA/paired_${lang}" \
        --per-generator 800 \
        --report "docs/paired_${lang}_build.json" || {
            echo "  (skipped: no shared speakers or no material)"; continue; }

    python scripts/build_manifests.py --datasets "${lang}_paired" \
        "${lang}_paired_unmatched" --data-root "$DATA" || true
done
printf "\n\033[32mDONE\033[0m\n"
