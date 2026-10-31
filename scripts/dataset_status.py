#!/usr/bin/env python
"""Live status of every corpus: what is on disk vs what is expected.

Reads actual bytes on scratch rather than trusting the fetch ledger, because
scratch is ephemeral and the ledger lives on NFS -- after a pod wipe the ledger
will happily still say "complete" for data that is gone.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.fetch_datasets import SOURCES  # noqa: E402

ROOT = Path("/tmp/vericall-data")
LEDGERS = ["data/fetch_state.json", "data/fetch_state_hf.json",
           "data/fetch_state_itw.json"]


def du(p: Path) -> int:
    if not p.exists():
        return 0
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def bar(frac: float, width: int = 22) -> str:
    frac = max(0.0, min(1.0, frac))
    n = int(round(frac * width))
    return "#" * n + "." * (width - n)


def main() -> int:
    ledger: dict = {}
    for lf in LEDGERS:
        p = Path(lf)
        if p.exists():
            try:
                ledger.update(json.loads(p.read_text()))
            except json.JSONDecodeError:
                pass

    print(f"{'corpus':<20}{'on disk':>10}{'expected':>10}  {'progress':<24}status")
    print("-" * 86)
    total_have = total_want = 0.0

    for name, src in SOURCES.items():
        if name == "in_the_wild_origin":
            continue
        # In-the-Wild lands as a nested archive; the inner one is the real corpus
        candidates = [ROOT / name, ROOT / f"{name}.zip", ROOT / f"{name}_inner.zip"]
        have = max((du(c) for c in candidates), default=0)
        want = src.approx_gb * 2 ** 30
        frac = have / want if want else 0.0
        state = ledger.get(name, {}).get("status", "-")

        if name == "rtcfake" and have < 1 << 20:
            state = "GATED - awaiting access"
            frac = 0.0
        elif frac >= 0.97:
            state = "complete"
        elif have > 0 and state not in ("complete",):
            state = state if state != "-" else "downloading"

        total_have += have
        total_want += want
        print(f"{name:<20}{have/2**30:>9.2f}G{src.approx_gb:>9.2f}G  "
              f"[{bar(frac)}] {min(frac,1.0)*100:5.1f}%  {state}")

    print("-" * 86)
    print(f"{'TOTAL':<20}{total_have/2**30:>9.2f}G{total_want/2**30:>9.2f}G  "
          f"[{bar(total_have/total_want)}] {total_have/total_want*100:5.1f}%")

    import shutil
    free = shutil.disk_usage(ROOT if ROOT.exists() else "/").free
    print(f"\nscratch free: {free/2**40:.1f} TB   (ephemeral - lost on pod restart)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
