#!/usr/bin/env python
"""Export real ASVspoof calls as .wav so the dashboard can stream genuine audio.

The dashboard previously streamed `Float32Array(8000)` of random noise. That is
the same failure as the mock scorer: the UI animates and means nothing. A demo
needs real audio, and the corpus lives in parquet on ephemeral pod scratch, so
this materialises a few calls as ordinary wav files under web/demo/ where the
gateway already serves them at /static/demo/.

Calls are built the same way as every other evaluation here -- consecutive clips
from one speaker with one provenance -- because ASVspoof clips are 1-4 s and a
call is not.
"""
from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vericall.data.manifest import Manifest
from end_to_end import build_calls, read_audio

SR = 16_000


def write_wav(path: Path, wav: torch.Tensor, sr: int = SR) -> None:
    pcm = (np.clip(wav.numpy(), -1.0, 1.0) * 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifests/asvspoof2019_la.jsonl")
    ap.add_argument("--splits", default="data/splits/asvspoof2019_la.json")
    ap.add_argument("--fold", default="dev")
    ap.add_argument("--clips-per-call", type=int, default=8)
    ap.add_argument("--out", default="web/demo")
    args = ap.parse_args()

    manifest = Manifest.load(args.manifest)
    plan = json.loads(Path(args.splits).read_text())
    bona, spoof = build_calls(manifest, plan, args.fold, args.clips_per_call, 2)
    cache: dict = {}
    read_audio(bona + spoof, cache)

    def audio_of(run):
        parts = [cache.get((it.shard, it.row)) for it in run]
        parts = [p for p in parts if p is not None and p.numel() > 0]
        return torch.cat(parts) if parts else None

    out = Path(args.out)
    wb, ws = audio_of(bona[0][2]), audio_of(spoof[0][2])
    if wb is None or ws is None:
        print("could not assemble calls", file=sys.stderr)
        return 1

    written = []
    write_wav(out / "genuine.wav", wb)
    written.append(("genuine.wav", wb.numel() / SR, bona[0][0], "-"))

    write_wav(out / "spoofed.wav", ws)
    written.append(("spoofed.wav", ws.numel() / SR, spoof[0][0], spoof[0][1]))

    # The mid-call swap: genuine speech spliced onto the clone. This is the
    # attack the dual-window design and change-point detector exist for, and
    # the one a single-shot classifier cannot express at all -- so it is the
    # clip worth putting in front of a panel.
    swap = torch.cat([wb, ws])
    write_wav(out / "midcall_swap.wav", swap)
    written.append(("midcall_swap.wav", swap.numel() / SR,
                    f"{bona[0][0]}->{spoof[0][0]}", f"splice @ {wb.numel()/SR:.1f}s"))

    meta = {
        "sample_rate": SR,
        "note": "Real ASVspoof 2019 LA dev audio. Not synthetic, not a mock.",
        "clips": [
            {"file": f, "seconds": round(d, 2), "speaker": s, "attack": a}
            for f, d, s, a in written
        ],
        "splice_s": round(wb.numel() / SR, 2),
    }
    (out / "manifest.json").write_text(json.dumps(meta, indent=2))

    print(f"{'file':<20}{'seconds':>9}  speaker / attack")
    for f, d, s, a in written:
        print(f"  {f:<18}{d:>9.2f}  {s} / {a}")
    print(f"\nwrote {out}/ ({sum(p.stat().st_size for p in out.glob('*.wav'))/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
