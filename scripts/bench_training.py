#!/usr/bin/env python
"""Measure real training throughput, end to end.

The headline "321 GB of audio" is misleading about cost for one structural
reason: the SSL frontend is FROZEN. Only a 0.89M-parameter head trains. So the
work is dominated by a forward pass that can be computed once and cached, not by
backprop through 94M parameters every epoch.

This measures each stage separately so the projection is arithmetic on measured
numbers rather than a guess.
"""
from __future__ import annotations

import glob, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vericall.branches.a_ssl import BranchA, BranchAConfig
from vericall.data.audio_io import fixed_length, load_audio

ROOT = "/tmp/vericall-data"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def timed(fn, n, warmup=2):
    for _ in range(warmup):
        fn()
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def main() -> int:
    import pyarrow.parquet as pq

    print(f"device: {DEV}  "
          f"{torch.cuda.get_device_name(0) if DEV.type=='cuda' else ''}\n")

    # ---------- stage 1: data loading (CPU) --------------------------------
    shard = sorted(glob.glob(f"{ROOT}/asvspoof2019_la/data/train*.parquet"))[0]
    rows = pq.ParquetFile(shard).read_row_group(0, columns=["audio"]).to_pydict()["audio"]
    blobs = [r["bytes"] for r in rows[:256]]

    t0 = time.perf_counter()
    wavs = [fixed_length(load_audio(b), 1.0) for b in blobs]
    dt = time.perf_counter() - t0
    load_rate = len(blobs) / dt
    print(f"1. decode + resample + VAD-trim + crop   "
          f"{load_rate:8.0f} clips/s/core  ({dt/len(blobs)*1000:.2f} ms each)")
    print(f"   with 14 usable cores (16 quota - 2 reserved): "
          f"{load_rate*14:8.0f} clips/s")

    # ---------- stage 2: frozen frontend (GPU) -----------------------------
    torch.set_grad_enabled(False)
    model = BranchA(BranchAConfig()).to(DEV).eval()
    batch = torch.stack(wavs[:32]).to(DEV)
    dt_fe = timed(lambda: model.frontend(batch), 15)
    fe_rate = 32 / dt_fe
    print(f"\n2. frozen SSL frontend, batch 32          {fe_rate:8.0f} windows/s "
          f"({dt_fe*1000:.1f} ms/batch)")

    # ---------- stage 3: head train step (GPU) -----------------------------
    torch.set_grad_enabled(True)
    feats = model.frontend(batch).detach().clone()
    head = model.head.train()
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    y = torch.randint(0, 2, (32,), device=DEV)
    lossfn = torch.nn.CrossEntropyLoss()

    def step():
        opt.zero_grad(set_to_none=True)
        lossfn(head(feats), y).backward()
        opt.step()

    dt_tr = timed(step, 30, warmup=5)
    tr_rate = 32 / dt_tr
    print(f"3. head-only train step (cached feats)    {tr_rate:8.0f} windows/s "
          f"({dt_tr*1000:.2f} ms/batch)")
    print(f"   -> head training is {tr_rate/fe_rate:.0f}x cheaper than the frontend,")
    print(f"      which is why caching features once and reusing them wins.")

    # ---------- projection --------------------------------------------------
    print("\n" + "=" * 74)
    print("PROJECTION (measured rates, 1 random 1 s crop per clip per epoch)")
    print("=" * 74)

    sets = {
        "ASVspoof LA train": 25_380,
        "WaveFake train fold": 45_905,
    }
    core = sum(sets.values())
    print(f"\nCore anti-spoofing set: {core:,} clips")
    for name, n in sets.items():
        print(f"   {name:<24}{n:>9,}")

    cache_crops = 4       # crops cached per clip, for augmentation variety
    extract_s = core * cache_crops / fe_rate
    io_s = core * cache_crops / (load_rate * 14)
    print(f"\n  one-time feature extraction ({cache_crops} crops/clip):")
    print(f"     GPU  {extract_s/60:6.1f} min")
    print(f"     I/O  {io_s/60:6.1f} min  (overlaps with GPU in a dataloader)")
    print(f"     => ~{max(extract_s, io_s)/60:.0f} min wall clock")

    per_epoch = core / tr_rate
    print(f"\n  then head training on cached features:")
    print(f"     {per_epoch:6.1f} s/epoch  ->  30 epochs = "
          f"{per_epoch*30/60:.1f} min")
    total = max(extract_s, io_s) + per_epoch * 30
    print(f"\n  TOTAL Branch A: ~{total/60:.0f} minutes")

    naive = core / fe_rate * 30
    print(f"  (without feature caching: {naive/3600:.1f} hours for the same 30 epochs)")

    print(f"\nIndicSynth (258 GB, ~2,240 h of audio) is multilingual FINE-TUNING,")
    print(f"not core training. Capped at 50 h/language x 6 = 300 h:")
    ind_clips = 300 * 3600 / 5.0            # ~5 s per clip
    ind_extract = ind_clips / fe_rate
    print(f"   ~{ind_clips:,.0f} clips -> extraction {ind_extract/60:.0f} min, "
          f"then {ind_clips/tr_rate*10/60:.0f} min for 10 epochs")
    print(f"\nUsing ALL of IndicSynth instead:")
    all_clips = 2240 * 3600 / 5.0
    print(f"   ~{all_clips:,.0f} clips -> extraction {all_clips/fe_rate/3600:.1f} h, "
          f"then {all_clips/tr_rate*10/3600:.1f} h for 10 epochs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
