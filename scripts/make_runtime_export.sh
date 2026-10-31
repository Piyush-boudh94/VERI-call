#!/usr/bin/env bash
# A laptop-sized export: everything needed to RUN, nothing needed to retrain.
#
# scripts/make_bundle.sh is the other export and the two answer different
# questions. A bundle is the complete git history -- the right thing if you
# intend to keep developing, and currently 2.4 GB, because eight fine-tuning
# checkpoints of 363 MB each are in that history and only a history rewrite
# would take them out.
#
# This one is for the demo machine. The working tree carries 3.0 GB of
# checkpoints and the profiles in profiles.json name 19 MB of them; the other
# 36 files are experiment artefacts that nothing serves. Dropping those and
# .git turns a 2.4 GB transfer into about 35 MB, which fits on anything and
# copies in seconds.
#
# What it deliberately leaves out, beyond the unused checkpoints:
#   corpora/  -- regenerable, and only needed for training
#   enroll/   -- voiceprints are biometric data under the DPDP Act 2023; a
#                leaked one is not revocable the way a password is
#   .tls/     -- private keys. A key that travels in a tarball is a key you
#                have to assume is public
#   logs/     -- call history, which is caller data
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$HOME/vericall-runtime-$(date +%Y%m%d-%H%M).tar.gz}"
cd "$ROOT"

# The checkpoints any profile names, plus Branch B, which is loaded
# unconditionally and is not in profiles.json because it has no per-language
# variant. Asking the registry rather than hardcoding a list means adding a
# language updates this export with no change here.
mapfile -t KEEP < <(python3 - <<'PY'
import sys
sys.path.insert(0, ".")
from vericall import profiles
need = {"branch_B.pt"}
for p in profiles.all_profiles().values():
    if not p.servable:
        continue
    for c in (p.ckpt_a, p.ckpt_a2):
        if c:
            need.add(c)
print("\n".join(sorted(need)))
PY
)
echo "keeping ${#KEEP[@]} checkpoints of $(ls checkpoints/*.pt | wc -l):"
printf '  %s\n' "${KEEP[@]}"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
DEST="$STAGE/vericall"
mkdir -p "$DEST/checkpoints"

# git ls-files, so the export contains exactly what is tracked -- no stray
# scratch file, no half-finished experiment, nothing untracked by accident.
git ls-files -z \
  | grep -zv '^checkpoints/' \
  | tar --null -T - -cf - \
  | tar -xf - -C "$DEST"

for f in "${KEEP[@]}"; do
    [ -f "checkpoints/$f" ] && cp "checkpoints/$f" "$DEST/checkpoints/$f"
done

# The commit this came from, so a bug report can name a revision even though
# the history itself is not in here.
git rev-parse HEAD > "$DEST/EXPORTED_FROM.txt"
git log -1 --format='%s (%ad)' --date=short >> "$DEST/EXPORTED_FROM.txt"
{
  echo ""
  echo "Runtime export: no git history, and only the checkpoints the profiles"
  echo "in profiles.json name. To develop rather than demo, use the git bundle"
  echo "from scripts/make_bundle.sh instead."
} >> "$DEST/EXPORTED_FROM.txt"

tar -C "$STAGE" -czf "$OUT" vericall
echo
echo "wrote $OUT  ($(du -h "$OUT" | cut -f1))"
echo
echo "On the demo machine:"
echo "  tar xzf $(basename "$OUT") && cd vericall"
echo "  python -m venv .venv && . .venv/bin/activate"
echo "  pip install torch --index-url https://download.pytorch.org/whl/cu121   # match your driver"
echo "  pip install -r requirements.txt"
echo "  python run_local.py --check"
echo "  python run_local.py --profile hindi-telephony"
