#!/usr/bin/env python
"""Build a PAIRED Hindi bona fide / spoof corpus -- the instrument this project
has been missing, and the reason the multilingual gap was called unclosable.

The standing diagnosis was: IndicSynth is spoof-only, so it can support a miss
rate at a threshold fitted elsewhere but not an EER, and no Indian-language
bona fide corpus is available to pair with it. Borrowing bona fide from an
unrelated corpus was correctly refused, because bona fide from corpus X against
spoof from corpus Y lets a model separate the classes by recording conditions
rather than by synthesis -- the leak that inflates published deepfake numbers.

That diagnosis missed a fact sitting in IndicSynth's own filenames.

    xtts_v2____1059-m-149.wav              speaker 1059, male, utterance 149
    freevc24____f____src_1087____tgt_1102  speaker 1087 converted to 1102

Those are **Kathbath speaker ids**, and all 41 source speakers also appear as
targets: IndicSynth's clones are conversions between speakers of ONE real
corpus. That corpus is Kathbath, via IndicSUPERB (Javed et al., AAAI 2023), and
it is CC0. So the bona fide class was never missing. Reuniting IndicSynth with
its own source is the opposite of cross-corpus borrowing: same speakers, same
recording sessions, same language, on both sides of the boundary.

**The trap, and why most of this script is about it.** Kathbath ships m4a
(AAC-compressed); IndicSynth ships PCM wav. Feed those in as they come and a
detector separates the classes by *codec artifacts* and reports a spectacular
number that has nothing to do with synthesis. Nothing downstream would catch
it -- it would look like the multilingual problem had been solved.

So both classes are forced through the same channel. Kathbath has already been
through AAC once, irreversibly, so the spoof side is given one matching AAC
generation. Two arms are built and both are reported:

    matched     both classes through one AAC generation   <- the honest number
    unmatched   as-shipped, AAC bona fide vs PCM spoof    <- the confound

The gap between them measures how much of the "detection" was codec. If the
unmatched arm is dramatically better, that difference was never detection.

Loudness is matched too, for the same reason and more cheaply: TTS output and
read speech sit at different levels, and level is trivially separable.

Silence is NOT handled here, because it is already handled: load_audio(trim=
True) is the single path every sample in this project takes, and
scripts/audit_silence.py measures the leading-silence shortcut per corpus. Run
it on the output -- neural TTS has systematically different pause structure
from read speech and that is exactly the kind of cue that produces an
impressive, worthless model.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vericall.data.audio_io import load_audio

SR = 16_000
TARGET_RMS = 0.05          # both classes normalised to this before writing

# Kathbath: <recording id>-<speaker id>-<gender>.m4a
KATHBATH_RE = re.compile(r"\A(\d+)-(\d+)-([mf])\Z")
# IndicSynth zero-shot TTS: <gen>____<speaker>-<gender>-<index>.wav
ITS_TTS_RE = re.compile(r"____(\d+)-([mf])-(\d+)\Z")
# IndicSynth voice conversion: <gen>____<gender>____src_<id>____tgt_<id>___<n>
ITS_VC_RE = re.compile(r"src_(\d+)____tgt_(\d+)")


def aac_roundtrip(x: np.ndarray, sr: int = SR, bitrate: int = 64_000) -> np.ndarray:
    """One AAC generation, to match what Kathbath's audio has already had.

    Real libavcodec rather than a filter approximation, for the same reason
    vericall/data/opus.py uses real libopus: AAC artifacts are specific to its
    own transform and an imitation would measure the imitation.
    """
    import av

    n = len(x)
    if n == 0:
        return x.astype(np.float32)
    buf = io.BytesIO()
    out = av.open(buf, mode="w", format="adts")        # raw AAC stream
    st = out.add_stream("aac", rate=sr, layout="mono")
    st.bit_rate = int(bitrate)
    frame = av.AudioFrame.from_ndarray(
        (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16).reshape(1, -1),
        format="s16", layout="mono")
    frame.rate = sr
    for pkt in st.encode(frame):
        out.mux(pkt)
    for pkt in st.encode(None):
        out.mux(pkt)
    out.close()

    buf.seek(0)
    inp = av.open(buf, mode="r", format="adts")
    rs = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=sr)
    chunks = [rf.to_ndarray().reshape(-1)
              for f in inp.decode(audio=0) for rf in rs.resample(f)]
    inp.close()
    if not chunks:
        return np.zeros(n, dtype=np.float32)
    y = np.concatenate(chunks).astype(np.float32) / 32768.0
    # AAC adds encoder delay and pads to a frame boundary. Trim the head by the
    # known priming amount, then match length, so the two classes are not
    # separable by an offset.
    y = y[1024:] if len(y) > n + 1024 else y
    return y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))


def equalise_silence(x: torch.Tensor, pad_ms: float = 50.0,
                     top_db: float = 35.0) -> torch.Tensor:
    """Force leading and trailing silence to the SAME fixed length on both
    classes, because trimming does not.

    Measured on this corpus with scripts/audit_silence.py: a leading-silence-
    only classifier scores 14.61% EER on the raw audio and **13.92% after
    load_audio's trim**. The cue survives because trim_silence pads 30 ms of
    original audio back onto the onset, and neural TTS often has less than
    30 ms of leading silence to give back -- so bona fide keeps 30.0 ms and
    spoof keeps 12.5 ms, and the difference is trivially separable.

    That number is a hard ceiling on the credibility of everything downstream.
    A Hindi EER above ~14% would be beaten by a stopwatch, and a Hindi EER
    below it could not be attributed to synthesis detection rather than to
    pause structure. So the cue is removed at the source: find speech onset and
    offset with a stricter threshold than the trim uses, cut to exactly those,
    then prepend and append an IDENTICAL fixed pad of digital silence.

    Both classes then carry exactly `pad_ms` of silence at each end by
    construction, and the audit must come back at chance. Building the cue out
    is better than hoping the model ignores it.

    `top_db` matches scripts/audit_silence.py's own threshold deliberately. A
    first pass cut at 45 dB left the leading cue fully removed (14.61% ->
    59.06% EER) but trailing silence still at 36.80%, because natural speech
    decay between -35 and -45 dB counts as silence to the audit and as speech
    to a stricter cut -- and read speech trails off where neural TTS stops
    abruptly. Cutting at the same threshold the cue is defined by is what makes
    it physically absent rather than merely reduced.

    `pad_ms` must exceed the 25 ms analysis frame the cue is measured
    with, or frame quantisation reintroduces it. With a 20 ms pad the
    first frame still straddles the pad/speech boundary, so a sharp onset
    reads as active while a gradual one reads as silent -- and neural TTS
    starts abruptly where read speech eases in. Measured: bona fide
    20.0 ms against spoof 13.2 ms, a 32.61% EER cue. At 50 ms the first
    two frames sit wholly inside the pad for both classes and onset
    sharpness stops mattering.
    """
    sr = SR
    frame = int(0.025 * sr)
    hop = int(0.010 * sr)
    if x.numel() < frame:
        return x
    frames = x.unfold(0, frame, hop)
    rms = frames.pow(2).mean(dim=-1).clamp(min=1e-12).sqrt()
    thresh = rms.max() * (10.0 ** (-top_db / 20.0))
    voiced = (rms > thresh).nonzero().reshape(-1)
    if voiced.numel() == 0:
        return x
    start = int(voiced[0]) * hop
    end = min(x.numel(), int(voiced[-1]) * hop + frame)
    core = x[start:end]
    if core.numel() < frame:
        return x
    pad = torch.zeros(int(pad_ms * sr / 1000.0), dtype=core.dtype)
    return torch.cat([pad, core, pad])


def normalise(x: torch.Tensor) -> torch.Tensor:
    r = float(x.pow(2).mean().sqrt())
    if r < 1e-8:
        return x
    y = x * (TARGET_RMS / r)
    peak = float(y.abs().max())
    return y / peak if peak > 1.0 else y            # never clip


def find_kathbath(root: Path, language: str = "hindi") -> dict[str, list[Path]]:
    """speaker id -> real recordings. Layout: <language>/<split>/audio/*.m4a

    Parameterised by language because the construction that produced the Hindi
    result is not Hindi-specific: IndicSynth clones Kathbath speakers in twelve
    languages and Kathbath publishes all twelve, so every one of them can be
    paired the same way. Hindi was simply first.
    """
    by_spk: dict[str, list[Path]] = defaultdict(list)
    hindi_dirs = [d for d in root.rglob("*")
                  if d.is_dir() and d.name.lower() == language.lower()]
    for hd in hindi_dirs:
        for f in hd.rglob("*"):
            if f.suffix.lower() not in (".m4a", ".wav", ".flac", ".mp3"):
                continue
            m = KATHBATH_RE.match(f.stem)
            if m:
                by_spk[m.group(2)].append(f)
    return by_spk


def indicsynth_speakers(manifest: Path) -> dict[str, list[dict]]:
    """speaker id -> spoof rows, keyed on the id IndicSynth cloned."""
    by_spk: dict[str, list[dict]] = defaultdict(list)
    with manifest.open() as fh:
        for line in fh:
            r = json.loads(line)
            stem = Path(r["utt_id"]).stem
            spk = None
            m = ITS_TTS_RE.search(stem)
            if m:
                spk = m.group(1)
            else:
                m = ITS_VC_RE.search(stem)
                if m:
                    spk = m.group(2)          # the TARGET is whose voice it is
            if spk:
                by_spk[spk].append(r)
    return by_spk


def drop_absent_shards(by_spk: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Keep only rows whose parquet shard is actually on disk.

    indicsynth_hi is fetched as 16 of 107 shards, strided for generator
    coverage, so most manifest rows point at files that were never downloaded.
    Filtering AFTER sampling wasted about 40% of every draw and quietly
    unbalanced the classes; filtering before it means the per-speaker cap means
    what it says.
    """
    have: dict[str, bool] = {}
    out: dict[str, list[dict]] = {}
    for spk, rows in by_spk.items():
        keep = []
        for r in rows:
            sh = r["shard"]
            if sh not in have:
                have[sh] = Path(sh).exists()
            if have[sh]:
                keep.append(r)
        if keep:
            out[spk] = keep
    return out


def plan_spoof_budget(its: dict[str, list[dict]], shared: list[str],
                      per_generator: int, rng) -> dict[str, list[dict]]:
    """Choose spoof rows against a GLOBAL per-generator budget.

    A per-speaker cap cannot balance generators here, and the arithmetic says
    why: 23 of the 29 shared speakers carry only FreeVC24, so they fill their
    whole quota with voice conversion and the 6 speakers who also have XTTS-v2
    contribute a fraction of it. Two attempts at a per-speaker rule produced
    396/84 and then 1820/210 -- and a head trained on the first reached 2.62%
    EER on FreeVC24 against 40.42% on XTTS-v2, the latter being the duration
    floor. The aggregate read 10.21% and looked healthy.

    Material was never the constraint: 4,145 XTTS-v2 rows are available for
    shared speakers against 4,419 FreeVC24. So the budget is set per generator
    and spread round-robin across whichever speakers have that generator,
    capped by what each actually holds. Both families then reach every fold in
    numbers large enough to report separately, which is the only way an
    aggregate over attack families means anything.
    """
    by_gen_spk: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list))
    for spk in shared:
        for r in its.get(spk, []):
            by_gen_spk[r["attack_id"]][spk].append(r)

    chosen: dict[str, list[dict]] = defaultdict(list)
    for gen in sorted(by_gen_spk):
        holders = sorted(by_gen_spk[gen], key=int)
        pools = {s: list(by_gen_spk[gen][s]) for s in holders}
        for s in holders:
            rng.shuffle(pools[s])
        taken = 0
        # Round-robin so no single prolific speaker supplies a whole generator.
        while taken < per_generator and any(pools.values()):
            for s in holders:
                if not pools[s]:
                    continue
                chosen[s].append(pools[s].pop())
                taken += 1
                if taken >= per_generator:
                    break
        print(f"  spoof budget {gen:<10} {taken:>5} rows "
              f"from {sum(1 for s in holders if s in chosen):>3} speakers")
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--language", default="hindi",
                    help="Kathbath directory name: hindi, bengali, marathi, "
                         "tamil, telugu, gujarati, ...")
    ap.add_argument("--kathbath", default="/tmp/vericall-data/kathbath_valid")
    ap.add_argument("--indicsynth-manifest",
                    default="data/manifests/indicsynth_hi.jsonl")
    ap.add_argument("--out", default="/tmp/vericall-data/hindi_paired")
    ap.add_argument("--per-generator", type=int, default=800,
                    help="spoof rows to draw PER GENERATOR, spread across the "
                         "speakers that have it. A per-speaker cap cannot "
                         "balance generators when most speakers carry only "
                         "one; see plan_spoof_budget.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", default="docs/hindi_paired_build.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    kb = find_kathbath(Path(args.kathbath), args.language)
    its = drop_absent_shards(indicsynth_speakers(Path(args.indicsynth_manifest)))
    print(f"Kathbath {args.language:<7}: {len(kb):>4} speakers, "
          f"{sum(len(v) for v in kb.values()):>6} recordings")
    print(f"IndicSynth      : {len(its):>4} speakers, "
          f"{sum(len(v) for v in its.values()):>6} clones")

    shared = sorted(set(kb) & set(its), key=int)
    print(f"\nSPEAKERS IN BOTH: {len(shared)}")
    if not shared:
        print("\nFAIL: no speaker id appears in both. The pairing assumption "
              "is wrong -- re-check the filename parsing before going on.")
        print(f"  kathbath sample ids  : {sorted(kb, key=int)[:8]}")
        print(f"  indicsynth sample ids: {sorted(its, key=int)[:8]}")
        return 2
    print(f"  {shared[:12]}{' ...' if len(shared) > 12 else ''}")

    out = Path(args.out)
    for arm in ("matched", "unmatched"):
        for cls in ("bonafide", "spoof"):
            (out / arm / cls).mkdir(parents=True, exist_ok=True)

    spoof_plan = plan_spoof_budget(its, shared, args.per_generator, rng)
    n_spoof_planned = sum(len(v) for v in spoof_plan.values())
    # Bona fide is drawn to MATCH the spoof total, spread evenly over all
    # shared speakers, so the classes stay balanced without a second knob.
    bona_cap = max(1, -(-n_spoof_planned // len(shared)))
    print(f"  bona fide cap {bona_cap} per speaker to match "
          f"{n_spoof_planned} spoof\n")

    rows = []
    gens = Counter()
    for spk in shared:
        # ---- bona fide: real Kathbath, already through AAC once ----------
        files = sorted(kb[spk])
        pick = files if len(files) <= bona_cap else [
            files[i] for i in rng.choice(len(files), bona_cap, replace=False)]
        for f in pick:
            try:
                w = load_audio(f.read_bytes())          # decode+resample+trim
            except Exception as exc:                    # noqa: BLE001
                print(f"    skip {f.name}: {type(exc).__name__}")
                continue
            if w.numel() < SR:                          # under 1 s is unusable
                continue
            w = normalise(equalise_silence(w))
            name = f"{spk}_{f.stem}.wav"
            # Bona fide is IDENTICAL in both arms: it is what it is, and the
            # arms differ only in what is done to the spoof side.
            for arm in ("matched", "unmatched"):
                _write(out / arm / "bonafide" / name, w)
            rows.append({"speaker_id": f"hi_{spk}", "label": 0,
                         "attack_id": "-", "file": name,
                         "seconds": round(w.numel() / SR, 2)})

        # spoof selection is decided globally, before this loop -- see
        # plan_spoof_budget(). Per-speaker capping cannot balance generators
        # when only 6 of 29 speakers carry XTTS-v2 at all.
        for r in spoof_plan.get(spk, []):
            raw = _read_parquet_cell(r)
            if raw is None:
                continue
            try:
                w = load_audio(raw)
            except Exception as exc:                    # noqa: BLE001
                print(f"    skip spoof: {type(exc).__name__}")
                continue
            if w.numel() < SR:
                continue
            w = normalise(equalise_silence(w))
            name = f"{spk}_{Path(r['utt_id']).stem}.wav"
            # unmatched: as shipped -- PCM against AAC bona fide. The confound.
            _write(out / "unmatched" / "spoof" / name, w)
            # matched: one AAC generation, so the codec stops being a class cue.
            wm = torch.from_numpy(aac_roundtrip(w.numpy())).float()
            _write(out / "matched" / "spoof" / name, normalise(wm))
            rows.append({"speaker_id": f"hi_{spk}", "label": 1,
                         "attack_id": r["attack_id"], "file": name,
                         "seconds": round(w.numel() / SR, 2)})
            gens[r["attack_id"]] += 1

    bona = [r for r in rows if r["label"] == 0]
    spoof = [r for r in rows if r["label"] == 1]
    print(f"\nwrote {len(bona)} bona fide + {len(spoof)} spoof "
          f"over {len(shared)} shared speakers")
    print(f"  generators: {dict(gens)}")
    print(f"  bona fide median {np.median([r['seconds'] for r in bona]):.1f}s  "
          f"spoof median {np.median([r['seconds'] for r in spoof]):.1f}s")

    rep = {"shared_speakers": len(shared), "speakers": shared,
           "n_bonafide": len(bona), "n_spoof": len(spoof),
           "generators": dict(gens), "per_generator_budget": args.per_generator,
           "target_rms": TARGET_RMS, "rows": rows,
           "kathbath_speakers": len(kb), "indicsynth_speakers": len(its),
           "note": "Paired Hindi. Bona fide = Kathbath (the corpus IndicSynth "
                   "cloned); spoof = IndicSynth clones of the SAME speakers. "
                   "The matched arm gives the spoof side one AAC generation so "
                   "the codec is not a class cue; the unmatched arm is "
                   "as-shipped and exists to measure that confound."}
    Path(args.report).write_text(json.dumps(rep, indent=2) + "\n")
    print(f"\nwrote {args.report}")
    print(f"corpus at {out}/{{matched,unmatched}}/{{bonafide,spoof}}/")
    return 0


def _write(path: Path, w: torch.Tensor) -> None:
    import soundfile as sf
    sf.write(path, w.numpy(), SR, subtype="PCM_16")


_PARQUET_CACHE: dict[str, object] = {}


def _read_parquet_cell(row: dict):
    """Pull one audio blob out of an IndicSynth parquet shard."""
    import pyarrow.parquet as pq

    shard = row["shard"]
    if not Path(shard).exists():
        return None
    tbl = _PARQUET_CACHE.get(shard)
    if tbl is None:
        if len(_PARQUET_CACHE) > 2:          # shards are ~500 MB; hold few
            _PARQUET_CACHE.clear()
        tbl = pq.read_table(shard, columns=["audio"]).column("audio")
        _PARQUET_CACHE[shard] = tbl
    cell = tbl[row["row"]].as_py()
    return cell["bytes"] if isinstance(cell, dict) else cell


if __name__ == "__main__":
    raise SystemExit(main())
