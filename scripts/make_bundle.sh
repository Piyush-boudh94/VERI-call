#!/usr/bin/env bash
# Pack the whole project into ONE portable file.
#
# Two problems, one answer.
#
# 1. There is no git remote. Every commit exists only on a shared NFS volume
#    that is 98% full, on a cluster whose /tmp has been wiped a dozen times
#    and whose namespace was emptied under us mid-project. A `git bundle` is a
#    single file containing the complete history, verifiable, and restorable
#    with `git clone <file>` -- so it can be copied anywhere, including off
#    this cluster entirely.
# 2. The project is meant to be exported to a laptop and run there. The bundle
#    plus the untracked corpora are exactly what that needs.
#
# A bundle is NOT a backup of the working tree: untracked files (corpora/,
# enroll/, .tls/, logs/) are not in it, by design -- corpora are regenerable,
# and voiceprints and private keys must not travel in something people copy
# around.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$HOME/vericall-$(date +%Y%m%d-%H%M).bundle}"
cd "$ROOT"

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "WARNING: uncommitted changes will NOT be in the bundle:" >&2
    git status --short >&2
fi

git bundle create "$OUT" --all
git bundle verify "$OUT" >/dev/null
printf "\n\033[32mwrote %s\033[0m  (%s, %s commits)\n" \
    "$OUT" "$(du -h "$OUT" | cut -f1)" "$(git rev-list --all --count)"

cat <<EOF

To restore it anywhere -- another machine, your laptop, a fresh cluster:

    git clone $(basename "$OUT") vericall
    cd vericall
    pip install -r requirements.txt
    python run_local.py --check

What the bundle does NOT contain, and how to rebuild each:

    corpora/            1.2 GB of paired Hindi wav
                        scripts/fetch_datasets.py --datasets kathbath_valid \\
                            kathbath_testkn indicsynth_hi
                        scripts/build_indic_paired.py
    enroll/             enrolled voiceprints -- biometric data, deliberately
                        excluded; re-enrol via POST /api/enroll/<id>
    .tls/               self-signed key; run_local.py --tls regenerates it
    logs/               call history and run logs

The checkpoints, all three calibration profiles, the demo clips and the
vendored ECAPA weights ARE in it, so a clone can serve immediately --
scripts/verify_export.sh proves exactly that.
EOF
