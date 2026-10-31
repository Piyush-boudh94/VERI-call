#!/usr/bin/env bash
# Restore a wiped pod to a working state.
#
# The pod's container filesystem is ephemeral and has been recreated at least
# TWICE now (Sep 4->5, and again by Sep 11 -- age 4h42m on inspection), each
# time taking every raw corpus and the feature cache with it. Everything
# derived (checkpoints, calibration, splits, manifests) lives on NFS and
# survives, so recovery is a re-fetch, not a redo -- but it must be one
# command, not a rediscovery each time. This file was itself overwritten by a
# stub once already (2026-09-05 incident, missed during recovery) and is
# committed to git now specifically so that can be diffed and caught.
#
#   ./scripts/rehydrate.sh                 # check state, fetch the default set
#   ./scripts/rehydrate.sh --check         # report only, change nothing
#   ./scripts/rehydrate.sh asvspoof2019_la wavefake
#
# Idempotent: already-present datasets are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
POD=./scripts/pod.sh
DEFAULT_SETS=(asvspoof2019_la)

check_only=0
sets=()
for arg in "$@"; do
    case "$arg" in
        --check) check_only=1 ;;
        -*) echo "unknown flag: $arg" >&2; exit 2 ;;
        *) sets+=("$arg") ;;
    esac
done
[ ${#sets[@]} -eq 0 ] && sets=("${DEFAULT_SETS[@]}")

echo "=== NFS-resident artifacts (should survive any pod wipe) ==="
missing=0
for p in checkpoints calib data/splits data/manifests; do
    if [ -e "$p" ]; then
        printf '  %-18s %s\n' "$p" "$(du -sh "$p" | cut -f1)"
    else
        printf '  %-18s MISSING\n' "$p"; missing=1
    fi
done
if [ "$missing" = 1 ]; then
    echo
    echo "  WARNING: a derived artifact is missing from NFS. That is not a pod"
    echo "  wipe -- rehydrate cannot restore it. Retrain or refit as needed."
fi

echo
echo "=== pod scratch ==="
$POD exec 'du -sh /tmp/vericall-data/* /tmp/vericall-cache 2>/dev/null || echo "  (empty)"' || true

if [ "$check_only" = 1 ]; then
    echo; echo "--check given; nothing fetched."
    exit 0
fi

echo
for ds in "${sets[@]}"; do
    if $POD exec "test -d /tmp/vericall-data/$ds" 2>/dev/null; then
        echo "=== $ds already present, skipping ==="
        continue
    fi
    echo "=== fetching $ds ==="
    $POD exec ".venv/bin/python scripts/fetch_datasets.py --datasets $ds"
done

echo
echo "=== done ==="
echo "Feature cache is NOT rebuilt automatically -- it is tens of GB and"
echo "specific to the folds you need. Rebuild what the current task requires:"
echo "    ./scripts/pod.sh exec 'bash scripts/extract_asvspoof.sh'"
