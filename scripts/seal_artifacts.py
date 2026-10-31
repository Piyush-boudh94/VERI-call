#!/usr/bin/env python
"""Write the integrity manifest for the artifacts this service will serve.

Deliberate by design: sealing is an operator saying "these are the weights and
the calibration I have evaluated, serve exactly these". Nothing regenerates the
manifest automatically, because a manifest that updates itself whenever a file
changes records history rather than intent, and detects nothing.

    scripts/seal_artifacts.py                  # seal checkpoints/ and calib/
    scripts/seal_artifacts.py --check          # verify without writing
    scripts/seal_artifacts.py --dir calib_rtc  # seal one profile

After sealing, set VERICALL_REQUIRE_SEALED=1 in the deployment so the gateway
refuses to serve anything that does not match.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vericall.serving.integrity import MANIFEST_NAME, build_manifest, verify

PATTERNS = ("*.pt", "*.pth", "*.json", "*.npy")


def targets(root: Path) -> list[Path]:
    out: list[Path] = []
    for pat in PATTERNS:
        out += [p for p in root.glob(pat)
                if p.is_file() and p.name != MANIFEST_NAME]
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", nargs="+", default=["checkpoints", "calib"])
    ap.add_argument("--check", action="store_true",
                    help="verify against the existing manifest, write nothing")
    args = ap.parse_args()

    failed = False
    for d in args.dir:
        root = Path(d)
        if not root.is_dir():
            print(f"[{d}] not a directory; skipping")
            continue
        files = targets(root)
        if not files:
            print(f"[{d}] nothing to seal")
            continue

        if args.check:
            rep = verify(root, files)
            state = "CLEAN" if rep.clean else "FAILED"
            print(f"[{d}] {state}  checked {rep.checked}  ok {rep.ok}")
            for label, items in (("MISMATCHED", rep.mismatched),
                                 ("MISSING", rep.missing),
                                 ("unsealed", rep.unsealed)):
                for i in items:
                    print(f"    {label}: {i}")
            if not rep.manifest_present:
                print(f"    no {MANIFEST_NAME} -- run without --check to create it")
            failed = failed or not rep.clean
            continue

        man = build_manifest(files, root)
        (root / MANIFEST_NAME).write_text(json.dumps(man, indent=2,
                                                     sort_keys=True) + "\n")
        print(f"[{d}] sealed {len(man)} file(s) -> {root / MANIFEST_NAME}")
        for name in sorted(man):
            print(f"    {man[name][:16]}...  {name}")

    if args.check and failed:
        print("\nIntegrity check FAILED. Either an artifact changed since it "
              "was sealed,\nor a new one was added and never sealed. Re-seal "
              "deliberately; do not\nassume the difference is benign.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
