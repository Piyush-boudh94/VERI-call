#!/usr/bin/env python
"""Will the whole gateway fit in a 6 GB consumer GPU?

Everything here was built on a MIG 1g.18gb slice. An RTX 4050 laptop GPU has
6 GB, and Windows/WDDM reserves a slice of that for the desktop, so the usable
budget is nearer 5 GB. This measures the real high-water mark per batch size
rather than estimating from parameter counts -- activations dominate, and a
frozen 94 M-parameter encoder at batch 32 is not 190 MB of work.

Reports the largest batch that fits a stated budget with headroom, which is
what the serving batcher should be configured with.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

torch.set_num_threads(1)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vericall.audio.chunker import Window

SR = 16_000


def mk(n):
    g = torch.Generator().manual_seed(0)
    return [Window(lane="fast", call_id=f"c{i}", t_start=0.0, t_end=1.0,
                   hop_s=0.5, samples=torch.randn(SR, generator=g) * 0.1,
                   quality=1.0) for i in range(n)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-gb", type=float, default=5.0,
                    help="usable VRAM. 6 GB card minus what the desktop holds.")
    ap.add_argument("--batches", type=int, nargs="+",
                    default=[1, 4, 8, 16, 32, 48])
    ap.add_argument("--precision", default="fp16", choices=["fp32", "fp16"])
    ap.add_argument("--out", default="docs/vram_budget.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device", file=sys.stderr)
        return 2
    dev = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"measuring on {name} ({total:.1f} GiB), budget {args.budget_gb} GiB, "
          f"{args.precision}\n")

    from vericall.branches.a2_phase import BranchA2, BranchA2Config
    from vericall.branches.a_ssl import BranchA, BranchAConfig
    from vericall.branches.b_prosody import BranchB, BranchBConfig

    base = torch.cuda.memory_allocated() / 2**30
    a = BranchA(BranchAConfig()).to(dev).eval()
    a.head.load_state_dict(torch.load("checkpoints/branch_A_seed0.pt",
                                      map_location=dev,
                                      weights_only=True)["state_dict"])
    a2 = BranchA2(BranchA2Config()).to(dev)
    a2.load_state_dict(torch.load("checkpoints/branch_A2_seed0.pt",
                                  map_location=dev,
                                  weights_only=True)["state_dict"])
    a2.eval()
    b = BranchB(BranchBConfig()).to(dev)
    b.load_state_dict(torch.load("checkpoints/branch_B.pt", map_location=dev,
                                 weights_only=True)["state_dict"])
    b.eval()
    c = None
    try:
        from vericall.branches.c_speaker import BranchC, BranchCConfig
        c = BranchC(BranchCConfig(), device=dev).to(dev).eval()
    except Exception as exc:                                  # noqa: BLE001
        print(f"  (branch C unavailable: {type(exc).__name__})")

    weights = torch.cuda.memory_allocated() / 2**30 - base
    print(f"weights resident: {weights:.3f} GiB "
          f"(A + A2 + B{' + C' if c else ''})\n")

    dtype = torch.float16 if args.precision == "fp16" else None
    rows = []
    print(f"{'batch':>6}{'peak GiB':>11}{'activations':>13}{'fits budget':>13}")
    print("-" * 44)
    for n in args.batches:
        ws = mk(n)
        x = torch.stack([w.samples for w in ws]).to(dev)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            with torch.inference_mode():
                ctx = (torch.autocast("cuda", dtype=dtype) if dtype
                       else torch.enable_grad(False) if False else None)
                if dtype:
                    with torch.autocast("cuda", dtype=dtype):
                        a.score(x); a2.score(x)
                        b.score(x, torch.zeros(n, dtype=torch.long, device=dev))
                        if c: c.embed(x)
                else:
                    a.score(x); a2.score(x)
                    b.score(x, torch.zeros(n, dtype=torch.long, device=dev))
                    if c: c.embed(x)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 2**30
            acts = peak - weights
            fits = peak < args.budget_gb * 0.85      # 15% headroom
            rows.append({"batch": n, "peak_gib": round(peak, 3),
                         "activations_gib": round(acts, 3), "fits": bool(fits)})
            print(f"{n:>6}{peak:>11.3f}{acts:>13.3f}"
                  f"{'yes' if fits else 'NO':>13}")
        except torch.cuda.OutOfMemoryError:
            print(f"{n:>6}{'OOM':>11}{'':>13}{'NO':>13}")
            rows.append({"batch": n, "peak_gib": None, "fits": False})
            torch.cuda.empty_cache()
            break

    ok = [r["batch"] for r in rows if r["fits"]]
    best = max(ok) if ok else 0
    print(f"\nlargest batch inside {args.budget_gb} GiB with 15% headroom: "
          f"{best if best else 'NONE'}")
    if best:
        wps = best / 0.10        # ~100 ms per batch, measured elsewhere
        print(f"  -> ~{wps:.0f} windows/s, i.e. ~{wps/2:.0f} concurrent calls "
              f"at a 0.5 s hop")
    out = {"device": name, "total_gib": round(total, 2),
           "budget_gib": args.budget_gb, "precision": args.precision,
           "weights_gib": round(weights, 3), "rows": rows,
           "recommended_max_batch": best}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
