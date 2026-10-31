#!/usr/bin/env python
"""Build the training cache: decode -> trim -> crop -> augment -> frontend.

Why cache at all: the SSL frontend is frozen, so its output for a given
(clip, augmentation) never changes. Measured, the frontend runs at 784
windows/s while the head trains at 8,358 windows/s -- so recomputing it every
epoch would make training 11x slower for no benefit.

Two things are cached per crop:

  waveform  fp16, 1 s @ 16 kHz (32 KB)  -- Branches A2 and B need the signal,
                                           not SSL features
  features  fp16, 49 x 768   (75 KB)    -- Branch A's layer-5 activations

**The trade this makes, stated plainly:** augmentation is baked in at extraction
time rather than sampled fresh each epoch. With `--crops N` we get N fixed
channel variants per clip instead of unlimited random ones. That is a real
reduction in augmentation diversity, bought for an 11x speedup; N=4 across four
channel profiles covers the channels we can actually simulate.

Shards are read sequentially, never by random access: parquet row-group reads
are cheap in order and thrash badly out of order.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

# One torch thread per worker. The decode pool runs 14 threads, and torch
# defaults to using every core for each op, so without this each worker tries to
# fan out across all 16 cores: ~224 threads contending for 16, which pins the
# CPU at 1424% while delivering a fraction of the throughput. The GPU does the
# heavy lifting here; the CPU side only needs to keep it fed.
torch.set_num_threads(1)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vericall.branches.a_ssl import BranchA, BranchAConfig
from vericall.data.augment import CHANNEL_BY_NAME, CHANNELS, apply_channel
from vericall.data.manifest import Manifest
from vericall.data.rawboost import rawboost
from vericall.data.splits import SplitPlan
from vericall.data.audio_io import fixed_length, load_audio

SR = 16_000


def load_plan(path: Path) -> SplitPlan:
    raw = json.loads(path.read_text())
    return SplitPlan({k: tuple(v) for k, v in raw.items()})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--folds", nargs="+", default=["train", "dev", "calib"])
    ap.add_argument("--crops", type=int, default=4,
                    help="augmented variants cached per clip")
    ap.add_argument("--window", type=float, default=1.0)
    # 128, not 32. Extraction is throughput-only -- nothing waits on an
    # individual window -- so the latency trade that caps the serving batcher
    # at 64 does not apply here. Measured on the 2g.35gb slice: batch 32 runs
    # at 88% of peak and batch 128 at 99%, for 2.7 GiB of 32.5. The old
    # default left an eighth of the GPU unused on every extraction.
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0,
                    help="first N items, IN MANIFEST ORDER. Biased on every "
                         "corpus here -- manifests are grouped by shard or by "
                         "speaker, so the first N is a few speakers or one "
                         "split. Fine for a smoke test, wrong for a "
                         "measurement; use --sample for that.")
    ap.add_argument("--sample", type=int, default=0,
                    help="balanced random subsample of N items (N/2 per "
                         "class), seeded and reproducible. Use this, not "
                         "--limit, when the output feeds a number.")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12,
                    help="decode threads; soundfile and torch both "
                         "release the GIL, so threads scale here")
    ap.add_argument("--channels", nargs="+", default=[c.name for c in CHANNELS],
                    choices=sorted(CHANNEL_BY_NAME),
                    help="channel profiles cycled across crops. Default is all "
                         "four. Pass 'clean' alone for a corpus whose channel "
                         "effects are already REAL (RTCFake): stacking a "
                         "simulator on top of real transmission measures the "
                         "simulator, not the corpus.")
    ap.add_argument("--rawboost", default="",
                    choices=["", "1", "2", "3", "series", "parallel"],
                    help="apply RawBoost to the waveform before the channel "
                         "profile. Training folds only -- an augmentation in "
                         "the scoring path changes the distribution the "
                         "calibration was fitted against.")
    ap.add_argument("--layers", type=int, nargs="*", default=None,
                    help="extract these SSL layers instead of the config's "
                         "single layer, writing each to <fold>@L<n>. All layers "
                         "come from ONE encoder pass, so a sweep costs the same "
                         "forward as a single extraction.")
    ap.add_argument("--out", default="/tmp/vericall-cache")
    ap.add_argument("--manifests", default="data/manifests")
    ap.add_argument("--splits", default="data/splits")
    args = ap.parse_args()

    import pyarrow.parquet as pq

    man = Manifest.load(Path(args.manifests) / f"{args.dataset}.jsonl")
    plan = load_plan(Path(args.splits) / f"{args.dataset}.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # One generator for the whole run: per-crop generators seeded from the row
    # index would make the "random" augmentation a deterministic function of
    # position, which is a cue rather than a regulariser.
    _rb_rng = np.random.default_rng(1234)
    torch.set_grad_enabled(False)
    model = BranchA(BranchAConfig()).to(device).eval()

    for fold in args.folds:
        items = [i for i in man if fold in plan.fold_of(i)]
        if args.limit:
            items = items[: args.limit]
        if args.sample:
            # Balanced and seeded. Taking the head of the manifest would be a
            # speaker-biased slice on In-the-Wild (meta.csv is grouped by
            # public figure) and a split-biased one elsewhere, so a subsample
            # meant for a measurement must be drawn, not sliced.
            import random as _random

            rng = _random.Random(args.sample_seed)
            bona = [it for it in items if it.label == 0]
            spoof = [it for it in items if it.label == 1]
            k = min(args.sample // 2, len(bona), len(spoof))
            if k == 0:
                print(f"[{fold}] cannot draw a balanced sample "
                      f"({len(bona)} bona / {len(spoof)} spoof); using all")
            else:
                items = rng.sample(bona, k) + rng.sample(spoof, k)
                rng.shuffle(items)
                print(f"[{fold}] balanced sample: {k} bona + {k} spoof "
                      f"(seed {args.sample_seed})")
        if not items:
            print(f"[{fold}] empty, skipping")
            continue

        # sequential shard order, then row order inside each shard
        by_shard: dict[str, list] = {}
        for it in items:
            by_shard.setdefault(it.shard, []).append(it)
        for v in by_shard.values():
            v.sort(key=lambda i: i.row)
        # Corpora ship either parquet shards (ASVspoof, WaveFake) or plain audio
        # files (RTCFake). Dispatch on the shard suffix rather than on the
        # dataset name, so a new corpus in either shape needs no new branch.
        is_files = not any(str(k).endswith(".parquet") for k in by_shard)
        chans = [CHANNEL_BY_NAME[n] for n in args.channels]

        n_out = len(items) * args.crops
        out_dir = Path(args.out) / args.dataset / fold
        out_dir.mkdir(parents=True, exist_ok=True)
        n_frames = int(args.window * 49)          # 49 frames per second

        waves = np.lib.format.open_memmap(
            out_dir / "wave.npy", mode="w+", dtype=np.float16,
            shape=(n_out, int(args.window * SR)))
        feats = np.lib.format.open_memmap(
            out_dir / "feat.npy", mode="w+", dtype=np.float16,
            shape=(n_out, n_frames, 768))
        # One memmap per requested layer, written from the same forward pass.
        layer_feats = {}
        layer_dirs = {}
        for L in (args.layers or []):
            d = Path(args.out) / args.dataset / f"{fold}@L{L}"
            d.mkdir(parents=True, exist_ok=True)
            layer_dirs[L] = d
            layer_feats[L] = np.lib.format.open_memmap(
                d / "feat.npy", mode="w+", dtype=np.float16,
                shape=(n_out, n_frames, 768))
        labels = np.zeros(n_out, dtype=np.int64)
        codecs = np.zeros(n_out, dtype=np.int64)
        meta = []

        print(f"[{fold}] {len(items)} clips x {args.crops} crops = {n_out} rows"
              f" -> {out_dir}", flush=True)
        t0 = time.time()
        write = 0
        buf_w, buf_c, buf_meta = [], [], []

        def flush():
            nonlocal write
            if not buf_w:
                return
            x = torch.from_numpy(np.stack(buf_w)).to(device).float()
            k = len(buf_w)
            waves[write:write + k] = x.cpu().numpy().astype(np.float16)
            if layer_feats:
                got = model.frontend.extract_layers(x, args.layers)
                for L, arr in layer_feats.items():
                    arr[write:write + k] = (got[L][:, :n_frames].cpu()
                                            .numpy().astype(np.float16))
            f = (model.frontend(x)[:, :n_frames] if not layer_feats
                 else got[args.layers[0]][:, :n_frames])
            feats[write:write + k] = f.cpu().numpy().astype(np.float16)
            for j, (lab, cod, m) in enumerate(zip(
                    [b[0] for b in buf_meta], buf_c, [b[1] for b in buf_meta])):
                labels[write + j] = lab
                codecs[write + j] = cod
                meta.append(m)
            write += k
            buf_w.clear(); buf_c.clear(); buf_meta.clear()

        def prepare(payload):
            """Decode one clip into its N augmented crops.

            Runs in a worker thread: the file read, soundfile's decoder and the
            torch resampler all release the GIL, so this is the one stage that
            genuinely parallelises. Single-threaded it caps the whole pass at
            ~18 clips/s against a measured ceiling of ~810.
            """
            it, raw = payload
            if raw is None:                      # plain-WAV corpus: read here,
                raw = Path(it.shard).read_bytes()  # so I/O overlaps decode
            wav = load_audio(raw)
            out = []
            for c in range(args.crops):
                crop = fixed_length(wav, args.window, SR,
                                    random_crop=args.crops > 1)
                if args.rawboost:
                    # Before the channel profile, not after: RawBoost stands in
                    # for handset and line character, and the codec is applied
                    # to whatever the line carried.
                    crop = rawboost(crop, _rb_rng, args.rawboost)
                ch = chans[c % len(chans)]
                aug = apply_channel(crop.unsqueeze(0), ch)[0]
                out.append((
                    aug.numpy().astype(np.float32), ch.codec_id,
                    (it.label, {"utt_id": it.utt_id,
                                "attack_id": it.attack_id,
                                "speaker_id": it.speaker_id,
                                "source_utt_id": it.source_utt_id,
                                "channel": ch.name})))
            return out

        def audio_column(pf) -> str:
            """Which column holds the audio struct.

            Not every corpus calls it `audio`: EchoFake puts the bytes under
            `path`, which is also a plausible name for a filename string. So
            the column is identified by SHAPE -- a struct carrying `bytes` --
            rather than by name, and the failure when no such column exists is
            an explicit error rather than a KeyError from deep inside a loop.
            """
            for f in pf.schema_arrow:
                t = str(f.type)
                if "struct" in t and "bytes" in t:
                    return f.name
            raise KeyError(
                f"no audio column (a struct with a `bytes` field) in "
                f"{[f.name for f in pf.schema_arrow]}")

        def feed_parquet():
            """Sequential row-group reads. Random access thrashes badly."""
            for shard, rows in by_shard.items():
                pf = pq.ParquetFile(shard)
                col = audio_column(pf)
                wanted = {r.row: r for r in rows}
                offset = 0
                for rg in range(pf.num_row_groups):
                    n_rows = pf.metadata.row_group(rg).num_rows
                    local = [i for i in range(n_rows) if (offset + i) in wanted]
                    if local:
                        tbl = pf.read_row_group(
                            rg, columns=[col]).to_pydict()[col]
                        for i in local:
                            yield (wanted[offset + i], tbl[i]["bytes"])
                    offset += n_rows

        def feed_files():
            """One audio file per shard (RTCFake). Bytes are read in the pool."""
            for _shard, rows in by_shard.items():
                for it in rows:
                    yield (it, None)

        def batched(src, n):
            """Bound how much decoded audio is resident at once. pool.map()
            consumes its whole iterable eagerly, so feeding it the raw generator
            would materialise the entire fold in RAM."""
            buf = []
            for x in src:
                buf.append(x)
                if len(buf) >= n:
                    yield buf
                    buf = []
            if buf:
                yield buf

        feed = feed_files if is_files else feed_parquet
        done = 0
        pool = ThreadPoolExecutor(max_workers=args.workers)
        for group in batched(feed(), max(args.workers * 4, 32)):
            for crops in pool.map(prepare, group):
                for wav_np, cid, m in crops:
                    buf_w.append(wav_np)
                    buf_c.append(cid)
                    buf_meta.append(m)
                    if len(buf_w) >= args.batch:
                        flush()
                done += 1
                if done % 2000 == 0:
                    el = time.time() - t0
                    print(f"   {done}/{len(items)} clips  "
                          f"{done/el:.0f} clips/s  eta "
                          f"{(len(items)-done)/max(done/el,1e-9)/60:.1f} min",
                          flush=True)
        flush()
        pool.shutdown()

        # Each layer dir is a complete fold: same labels, codecs, meta and info,
        # so every downstream script can point at train@L7 with no special case.
        for L, d in layer_dirs.items():
            np.save(d / "label.npy", labels[:write])
            np.save(d / "codec.npy", codecs[:write])
            np.save(d / "wave.npy", np.asarray(waves[:write]))
            (d / "meta.jsonl").write_text("\n".join(json.dumps(m) for m in meta))
            (d / "info.json").write_text(json.dumps({
                "dataset": args.dataset, "fold": f"{fold}@L{L}",
                "rows": int(write), "clips": len(items), "crops": args.crops,
                "window_s": args.window, "n_frames": n_frames,
                "ssl_layer": L, "channels": list(args.channels),
            }, indent=2))

        np.save(out_dir / "label.npy", labels[:write])
        np.save(out_dir / "codec.npy", codecs[:write])
        (out_dir / "meta.jsonl").write_text(
            "\n".join(json.dumps(m) for m in meta))
        (out_dir / "info.json").write_text(json.dumps({
            "dataset": args.dataset, "fold": fold, "rows": int(write),
            "clips": len(items), "crops": args.crops, "window_s": args.window,
            "channels": list(args.channels),
            "n_frames": n_frames, "seconds": round(time.time() - t0, 1),
        }, indent=2))
        el = time.time() - t0
        print(f"[{fold}] wrote {write} rows in {el/60:.1f} min "
              f"({done/el:.0f} clips/s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
