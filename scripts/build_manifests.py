#!/usr/bin/env python
"""Build dataset manifests and leakage-verified split plans.

Manifests and split plans are generated ONCE and written to NFS. They are never
recomputed on the fly: a split that changes between runs makes every number
irreproducible, and a split that silently leaks makes every number wrong.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vericall.data.manifest import INDIC_LANGUAGES as _INDIC, build
from vericall.data.splits import (
    FOLDS, plan_asvspoof, plan_codecfake, plan_echofake, plan_hindi_paired,
    plan_in_the_wild, plan_rtcfake, plan_wavefake, verify,
)

PLANNERS = {"asvspoof2019_la": plan_asvspoof, "wavefake": plan_wavefake,
            "rtcfake": plan_rtcfake, "echofake": plan_echofake,
            "codecfake": plan_codecfake, "in_the_wild": plan_in_the_wild,
            **{f"{_l}_paired{_s}": plan_hindi_paired
               for _l in _INDIC for _s in ("", "_unmatched")}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/tmp/vericall-data")
    ap.add_argument("--out", default="data")
    ap.add_argument("--datasets", nargs="+", default=list(PLANNERS))
    args = ap.parse_args()

    out = Path(args.out)
    (out / "manifests").mkdir(parents=True, exist_ok=True)
    (out / "splits").mkdir(parents=True, exist_ok=True)

    failed = False
    # Both codec arms of the paired Hindi corpus live in one directory; the
    # dataset name selects the arm, not the path.
    from vericall.data.manifest import INDIC_LANGUAGES
    # Both codec arms of each paired corpus live in one directory; the dataset
    # name selects the arm, not the path.
    ROOT_ALIAS = {}
    for _lang in INDIC_LANGUAGES:
        ROOT_ALIAS[f"{_lang}_paired"] = ("hindi_paired" if _lang == "hindi"
                                         else f"paired_{_lang}")
        ROOT_ALIAS[f"{_lang}_paired_unmatched"] = ROOT_ALIAS[f"{_lang}_paired"]

    for name in args.datasets:
        root = Path(args.data_root) / ROOT_ALIAS.get(name, name)
        if not root.exists():
            print(f"[{name}] SKIP (not downloaded: {root})")
            continue

        print(f"\n=== {name} ===")
        man = build(name, root)
        man.save(out / "manifests" / f"{name}.jsonl")
        c = man.counts()
        print(f"  manifest: {c['n']} items, {c['bonafide']} bona fide, "
              f"{c['spoof']} spoof, {c['speakers']} speakers, "
              f"{c['source_utts']} source utts, {c['attacks']} generators")

        if name not in PLANNERS:
            # An eval-only corpus has no split plan, and inventing one would be
            # worse than having none: IndicSynth publishes spoof only, so any
            # "fold" carved from it is single-class and every threshold fitted
            # on it is undefined. See scripts/eval_indic.py for what it CAN
            # measure.
            print(f"  no split plan: {name} is eval-only "
                  f"({c['bonafide']} bona fide / {c['spoof']} spoof). "
                  f"Manifest written; no folds, no thresholds.")
            continue

        plan = PLANNERS[name](man)
        (out / "splits" / f"{name}.json").write_text(
            json.dumps({k: list(v) for k, v in plan.assignment.items()},
                       separators=(",", ":"))
        )

        print(f"  {'fold':<22}{'n':>8}{'bona':>8}{'spoof':>8}{'spk':>6}"
              f"{'srcutt':>8}  generators")
        for fold, stats in plan.summary(man).items():
            sub = plan.subset(man, fold)
            gens = sorted({i.attack_id for i in sub})
            shown = ",".join(gens[:5]) + ("..." if len(gens) > 5 else "")
            print(f"  {fold:<22}{stats['n']:>8}{stats['bonafide']:>8}"
                  f"{stats['spoof']:>8}{stats['speakers']:>6}"
                  f"{stats['source_utts']:>8}  {shown}")

        violations = verify(man, plan)
        if violations:
            failed = True
            print("  LEAKAGE VIOLATIONS:")
            for v in violations:
                print(f"    ! {v.guard}: {v.detail}")
        else:
            print("  leakage guards: ALL CLEAN")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
