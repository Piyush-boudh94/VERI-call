#!/usr/bin/env python
"""Download the VeriCall corpora onto pod-local scratch.

Why this is a first-class script and not a one-off command:

  - The corpora total ~61 GB against 55 GB free on a 99%-full shared NFS home,
    so they cannot live there. They go on the pod's 26 TB node-local overlay.
  - That overlay is EPHEMERAL -- wiped on pod delete or reschedule -- and the
    namespace is reported to be deleted roughly twice a day. So re-fetching is a
    routine event, not an exception, and this must be idempotent and resumable.

Progress is written to a JSON state file on NFS so it survives the pod and can
be polled from the head node.

Notes on the two environment quirks this works around:
  - The pod advertises IPv6 but has no IPv6 route, so dual-stack hosts fail
    intermittently with "Network is unreachable". Resolution is pinned to IPv4.
  - The image has no curl, wget or unzip. Everything here is stdlib.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# --- IPv4 pinning: must happen before any connection is made ----------------
_orig_getaddrinfo = socket.getaddrinfo


def _ipv4_only(host, port, family=0, type=0, proto=0, flags=0):
    return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)


socket.getaddrinfo = _ipv4_only

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


@dataclass
class Source:
    name: str
    approx_gb: float
    note: str
    kind: str = "hf"                  # "hf" | "url" | "kaggle"
    repo: str = ""                    # hf repo id
    allow: tuple[str, ...] = ()       # hf allow_patterns
    url: str = ""                     # direct-url sources
    filename: str = ""
    resumable: bool = True


SOURCES: dict[str, Source] = {
    # Mirrors chosen over the canonical hosts for one reason: bandwidth.
    # Edinburgh DataShare sustains ~25 KB/s per connection (~700 KB/s with 8),
    # so the 23.6 GB deposit is a ~10 hour download to obtain 7 GB of LA. The
    # HuggingFace CDN runs at 13-23 MB/s. Both mirrors below were verified
    # against the official release before being trusted:
    #   * lossless 16-bit PCM WAV at the native rate (no lossy re-encode, which
    #     would destroy the very artifacts these branches detect)
    #   * exact official split counts
    #   * full protocol metadata retained (attack id, speaker id)
    # See docs/DATASETS.md for the verification record.
    "asvspoof2019_la": Source(
        name="asvspoof2019_la",
        kind="hf",
        repo="Bisher/ASVspoof_2019_LA",
        allow=("data/*", "*.md"),
        approx_gb=7.03,
        note="LA train/dev/eval, 16 kHz PCM. speaker_id + system_id (A01-A19) intact.",
    ),
    "wavefake": Source(
        name="wavefake",
        kind="hf",
        repo="ajaykarthick/wavefake-audio",
        allow=("data/*", "*.md"),
        approx_gb=28.20,   # tree API reported 10.07; actual is 28.20
        note="R + WF1-WF7 over LJSpeech, 22.05 kHz PCM. audio_id gives the source utterance.",
    ),
    "rtcfake": Source(
        name="rtcfake",
        kind="hf",
        repo="JunXueTech/RTCFake",
        allow=("data/*",),
        approx_gb=20.40,
        note="Speech transmitted through 7 real RTC platforms. THE test of the "
             "'survives compression' claim; eval holds out unseen platforms.",
    ),
    # Scoped to six languages ON PURPOSE. The full repo is ~845 GB across 12
    # languages; the HF tree API paginates and reported 16 GB, which is how an
    # unbounded pull got started. These six are what completed, they exceed the
    # "Hindi + 2" the plan called for, and pinning them means a re-fetch after a
    # scratch wipe reproduces this exact state instead of downloading 845 GB.
    "indicsynth": Source(
        name="indicsynth",
        kind="hf",
        repo="vdivyasharma/IndicSynth",
        allow=("Bengali/*", "Gujarati/*", "Hindi/*", "Kannada/*",
               "Malayalam/*", "Marathi/*", "*.md"),
        approx_gb=263.0,
        note="6 of 12 Indian languages (Bengali, Gujarati, Hindi, Kannada, "
             "Malayalam, Marathi). ACL 2025. CC BY-NC: research use only.",
    ),
    # A slice of Hindi, not the 52 GB language and not the 845 GB repo. The gap
    # this closes is EVALUATION on an Indian language -- every number in
    # docs/RESULTS.md is English or Mandarin, on an AICTE problem statement.
    #
    # **Shards are STRIDED across all 107, not taken from the front.** The first
    # attempt took shards 0-11 and drew 23,100 rows from exactly one generator
    # (freevc24), because the shards are ordered by generative model. That is
    # the same positional-prefix-over-a-grouped-file bug as the Branch C cohort
    # in docs/RESULTS.md section 7, which drew 3 of 20 speakers. A stride is the
    # cheap fix; it cannot guarantee coverage the way reading the metadata first
    # would, but it cannot systematically miss a group either.
    #
    # Pinned by explicit filename so a re-fetch after a scratch wipe reproduces
    # exactly this slice rather than whatever the repo holds that day.
    "indicsynth_hi": Source(
        name="indicsynth_hi",
        kind="hf",
        repo="vdivyasharma/IndicSynth",
        allow=tuple(f"Hindi/train-{i:05d}-of-00107.parquet"
                    for i in range(0, 107, 7)) + ("*.md",),
        approx_gb=8.0,
        note="16 of 107 Hindi shards, strided for generator coverage. "
             "ACL 2025. CC BY-NC: research use only. SPOOF ONLY -- the corpus "
             "publishes no bona fide, so it cannot support an EER by itself.",
    ),
    "echofake": Source(
        name="echofake",
        kind="hf",
        repo="EchoFake/EchoFake",
        allow=("*",),
        approx_gb=3.04,
        note="Physical replay attacks -- the threat vector the v3 blueprint omitted.",
    ),
    "codecfake": Source(
        name="codecfake",
        kind="hf",
        repo="ajaykarthick/codecfake-audio",
        allow=("data/*", "*.md"),
        approx_gb=17.13,   # the tree API under-reported this one too
        note="Neural-codec-synthesised speech: the newest generation of attack.",
    ),
    # No HuggingFace mirror exists, so this comes from the origin. It advertises
    # Accept-Ranges: none on HEAD but DOES honour Range on GET (verified: 206
    # with a valid Content-Range), so it parallelises after all -- an earlier
    # reading of the HEAD response wrongly wrote it off as a single slow stream.
    # Measured: the Fraunhofer origin throttles per CLIENT, not per connection,
    # so twelve parallel range requests still totalled ~150 KB/s -- a ~14 hour
    # download. The Kaggle mirror serves the same 7.6 GB at 11 MB/s and does
    # support Range. Verified against the published release after download
    # (31,779 clips: 19,963 bona fide / 11,816 spoof).
    "in_the_wild": Source(
        name="in_the_wild",
        kind="kaggle",
        url="abdallamohamed312/in-the-wild-dataset",
        filename="in_the_wild.zip",
        approx_gb=7.60,
        note="Cross-domain eval ONLY. Never trained on.",
    ),
    # The real Hindi speech behind IndicSynth's clones, and the reason the
    # multilingual gap is closable at all.
    #
    # IndicSynth is spoof-only, so on its own it can support a miss rate at a
    # threshold fitted elsewhere and NOT an EER -- docs/RESULTS.md has said so
    # since it landed. What was missed is that IndicSynth's filenames carry
    # Kathbath speaker ids (`xtts_v2____1059-m-149.wav`, and `src_NNNN____tgt_
    # NNNN` for the voice-conversion half, with all 41 sources also appearing
    # as targets). Its clones are conversions BETWEEN speakers of one real
    # corpus, and that corpus is Kathbath, via IndicSUPERB (Javed et al. 2023).
    #
    # So the bona fide class is not missing and does not need to be borrowed
    # from an unrelated corpus -- which the project rightly refused to do,
    # because bona fide from corpus X against spoof from corpus Y lets a model
    # separate them by recording conditions. Reuniting IndicSynth with its own
    # source is the opposite of that: same speakers, same sessions, same
    # language, on both sides.
    #
    # `valid` and not `train`: 3 GB against 85 GB, all twelve languages, and a
    # paired EVAL set is what is needed. CC0 licensed.
    #
    # One trap, handled in scripts/build_indic_paired.py and not here:
    # Kathbath ships m4a (AAC) and IndicSynth ships wav, so feeding them in
    # as-is would let a detector separate the classes by CODEC rather than by
    # synthesis and report a spectacular fake EER.
    # Per-language IndicSynth slices, strided for generator coverage the way
    # indicsynth_hi is. Each language pairs against the same Kathbath speakers,
    # so the construction that produced the Hindi result (docs/RESULTS.md
    # section 25) extends without change.
    #
    # GUJARATI IS A FROZEN HOLDOUT. It is fetched with the rest and must not be
    # scored until the multilingual work is finished, because the project no
    # longer has an untouched one: In-the-Wild has now been scored by six
    # checkpoints and CodecFake by five, so neither can still be called a
    # virgin holdout. Reserving a language from the start is the only way to
    # get that property back, and it costs nothing to reserve.
    **{
        f"indicsynth_{code}": Source(
            name=f"indicsynth_{code}",
            kind="hf",
            repo="vdivyasharma/IndicSynth",
            allow=tuple(f"{lang}/train-{i:05d}-of-{n:05d}.parquet"
                        for i in range(0, n, max(1, n // 10))) + ("*.md",),
            # ~0.84 GB/shard, from the 16 Hindi shards that measured
            # 13.4 GB, times the ~10 a stride of n//10 selects.
            approx_gb=8.4,
            note=(f"{lang} slice, ~10 of {n} shards. "
                  + ("FROZEN HOLDOUT -- do not score until the multilingual "
                     "work is complete." if lang == "Gujarati"
                     else "Pairs against Kathbath speakers of the same "
                          "language.")),
        )
        # Shard counts read off the repo rather than guessed:
        #   HfApi().list_repo_files("vdivyasharma/IndicSynth", repo_type="dataset")
        # All twelve IndicSynth languages are here, which is every language
        # Kathbath publishes -- so this is the whole buildable set, not a
        # chosen subset. The remaining ten of India's twenty-two scheduled
        # languages have no paired corpus in any dataset, which is a corpus
        # gap and not something more fetching can close.
        for code, lang, n in (
            ("bn", "Bengali", 54),
            ("mr", "Marathi", 144),
            ("ta", "Tamil", 171),
            ("te", "Telugu", 129),
            ("gu", "Gujarati", 72),      # frozen holdout
            ("kn", "Kannada", 91),
            ("ml", "Malayalam", 32),
            ("or", "Odia", 34),
            ("pa", "Punjabi", 134),
            ("sa", "Sanskrit", 396),
            ("ur", "Urdu", 48),
        )
    },
    "kathbath_valid": Source(
        name="kathbath_valid",
        kind="url",
        url="https://objectstore.e2enetworks.net/indic-superb/kathbath/clean/"
            "valid_audio.tar",
        filename="kathbath_valid.tar",
        approx_gb=3.0,
        note="Real Hindi (and 11 other languages) bona fide. The source "
             "corpus IndicSynth cloned; CC0.",
    ),
    "kathbath_testkn": Source(
        name="kathbath_testkn",
        kind="url",
        url="https://objectstore.e2enetworks.net/indic-superb/kathbath/clean/"
            "testkn_audio.tar",
        filename="kathbath_testkn.tar",
        approx_gb=2.0,
        note="Kathbath test, KNOWN speakers -- a second bona fide split if a "
             "speaker-disjoint eval needs more material.",
    ),
    "in_the_wild_origin": Source(
        name="in_the_wild_origin",
        kind="url",
        url="https://owncloud.fraunhofer.de/index.php/s/JZgXh0JEAF0elxa/download",
        filename="in_the_wild.zip",
        approx_gb=7.60,
        note="Authoritative origin. Slow (~150 KB/s client-throttled); fallback "
             "if the Kaggle mirror ever fails verification.",
    ),
}


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:6.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"


class State:
    """Progress ledger on NFS, so it outlives the ephemeral scratch."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data: dict = {}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                self.data = {}

    def update(self, name: str, **fields) -> None:
        entry = self.data.setdefault(name, {})
        entry.update(fields)
        entry["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2))
        tmp.replace(self.path)          # atomic; a torn state file is useless


def sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def download(src: Source, dest_dir: Path, state: State, retries: int = 6) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / src.filename
    part = target.with_suffix(target.suffix + ".part")

    if target.exists():
        print(f"[{src.name}] already present ({human(target.stat().st_size)}), skipping")
        state.update(src.name, status="complete", path=str(target),
                     bytes=target.stat().st_size)
        return target

    attempt = 0
    while attempt < retries:
        attempt += 1
        have = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": UA}
        if have and src.resumable:
            headers["Range"] = f"bytes={have}-"
        elif have and not src.resumable:
            # server ignores Range; restarting is the only correct option
            print(f"[{src.name}] server has no resume support, restarting")
            part.unlink()
            have = 0

        try:
            req = urllib.request.Request(src.url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp:
                if have and resp.status != 206:
                    print(f"[{src.name}] resume refused (status {resp.status}); restarting")
                    part.unlink(missing_ok=True)
                    have = 0
                remaining = resp.headers.get("Content-Length")
                total = (int(remaining) + have) if remaining else None
                mode = "ab" if have else "wb"
                t0, last, done = time.time(), time.time(), have

                with part.open(mode) as fh:
                    while True:
                        block = resp.read(1 << 20)
                        if not block:
                            break
                        fh.write(block)
                        done += len(block)
                        now = time.time()
                        if now - last > 20:
                            rate = (done - have) / max(now - t0, 1e-6)
                            pct = f"{done/total*100:5.1f}%" if total else "  ?  "
                            eta = ((total - done) / rate / 60) if (total and rate > 0) else -1
                            msg = (f"[{src.name}] {pct} {human(done)}"
                                   f"{'/' + human(total) if total else ''} "
                                   f"@ {human(rate)}/s eta {eta:.0f}m")
                            print(msg, flush=True)
                            state.update(src.name, status="downloading", bytes=done,
                                         total=total, rate_bps=rate)
                            last = now

            if total and part.stat().st_size < total:
                raise IOError(f"short read: {part.stat().st_size} < {total}")

            part.replace(target)
            digest = sha256(target)
            size = target.stat().st_size
            print(f"[{src.name}] DONE {human(size)}  sha256={digest[:16]}...")
            state.update(src.name, status="complete", path=str(target),
                         bytes=size, sha256=digest)
            return target

        except (urllib.error.URLError, IOError, TimeoutError, socket.timeout) as exc:
            wait = min(60, 2 ** attempt)
            print(f"[{src.name}] attempt {attempt}/{retries} failed: "
                  f"{type(exc).__name__}: {str(exc)[:90]} -- retrying in {wait}s",
                  flush=True)
            state.update(src.name, status="retrying", error=str(exc)[:200],
                         attempt=attempt)
            time.sleep(wait)

    state.update(src.name, status="failed")
    raise RuntimeError(f"[{src.name}] exhausted {retries} attempts")


def _kaggle_auth() -> dict[str, str]:
    """Basic auth from ~/.kaggle/kaggle.json."""
    import base64

    cfg = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())
    token = base64.b64encode(
        f"{cfg['username']}:{cfg['key']}".encode()
    ).decode()
    return {"Authorization": f"Basic {token}", "User-Agent": "kaggle-api"}


def fetch_kaggle(src: Source, dest_dir: Path, state: State,
                 workers: int = 8) -> Path:
    """Download a Kaggle dataset archive, in parallel where Range is honoured."""
    kaggle = Source(
        name=src.name, kind="url", approx_gb=src.approx_gb, note=src.note,
        url=f"https://www.kaggle.com/api/v1/datasets/download/{src.url}",
        filename=src.filename, resumable=True,
    )
    global _EXTRA_HEADERS
    _EXTRA_HEADERS = _kaggle_auth()
    try:
        archive = download_parallel(kaggle, dest_dir, state, workers=workers)
    finally:
        _EXTRA_HEADERS = {}
    return _unwrap_nested_zip(archive, state, src.name)


def _unwrap_nested_zip(archive: Path, state: State, name: str) -> Path:
    """Kaggle wraps the dataset in its own zip.

    Where the dataset is itself a single archive (In-the-Wild ships as one
    release zip), the result is a zip containing one entry called `download`
    which is the real archive. Unwrapping here keeps that quirk out of every
    downstream reader instead of leaving a trap for whoever opens it next.
    """
    import zipfile

    if not zipfile.is_zipfile(archive):
        return archive
    with zipfile.ZipFile(archive) as zf:
        entries = zf.infolist()
        if len(entries) != 1:
            return archive                      # a normal multi-file archive
        inner = entries[0]
        with zf.open(inner) as fh:
            if fh.read(2) != b"PK":
                return archive                  # single entry, but not a zip
        out = archive.with_name(archive.stem + "_inner.zip")
        if not out.exists():
            print(f"[{name}] unwrapping nested archive -> {out.name}", flush=True)
            with zf.open(inner) as src, out.open("wb") as dst:
                while (chunk := src.read(1 << 24)):
                    dst.write(chunk)
    with zipfile.ZipFile(out) as zf:
        n = sum(1 for x in zf.namelist() if x.lower().endswith(".wav"))
    print(f"[{name}] inner archive: {n} wav files")
    state.update(name, status="complete", path=str(out), wav_files=n,
                 bytes=out.stat().st_size)
    return out


_EXTRA_HEADERS: dict[str, str] = {}


def _probe_range(url: str) -> tuple[Optional[int], bool]:
    """Return (total_bytes, supports_range).

    Deliberately probes with a real ranged GET rather than a HEAD. Servers lie:
    the Fraunhofer host advertises `Accept-Ranges: none` on HEAD and then
    happily serves 206 with a correct Content-Range on GET. Believing the HEAD
    would cost a 17-hour single-stream download for no reason.
    """
    req = urllib.request.Request(
        url, headers={"User-Agent": UA, "Range": "bytes=0-1023", **_EXTRA_HEADERS}
    )
    with urllib.request.urlopen(req, timeout=90) as r:
        cr = r.headers.get("Content-Range")
        r.read(1024)
        if r.status == 206 and cr and "/" in cr:
            return int(cr.rsplit("/", 1)[1]), True
        size = r.headers.get("Content-Length")
        return (int(size) if size else None), False


def _chunk_worker(url: str, start: int, end: int, part: Path, progress: dict,
                  lock: "threading.Lock", retries: int = 8) -> None:
    for attempt in range(retries):
        try:
            have = part.stat().st_size if part.exists() else 0
            if have >= (end - start + 1):
                return
            req = urllib.request.Request(url, headers={
                "User-Agent": UA, "Range": f"bytes={start + have}-{end}",
                **_EXTRA_HEADERS})
            with urllib.request.urlopen(req, timeout=120) as r, \
                    part.open("ab") as fh:
                while True:
                    block = r.read(1 << 20)
                    if not block:
                        break
                    fh.write(block)
                    with lock:
                        progress["done"] += len(block)
            if part.stat().st_size >= (end - start + 1):
                return
        except Exception:
            time.sleep(min(30, 2 ** attempt))
    raise IOError(f"chunk {start}-{end} failed after {retries} attempts")


def download_parallel(src: Source, dest_dir: Path, state: State,
                      workers: int = 12) -> Path:
    """Multi-connection ranged download.

    Every origin measured throttles hard per connection but not in aggregate
    (Zenodo: 42 KB/s on one connection, 5.1 MB/s on sixteen), so connection
    count -- not bandwidth -- is the binding constraint on these hosts.
    """
    import threading

    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / src.filename
    if target.exists():
        print(f"[{src.name}] already present ({human(target.stat().st_size)})")
        state.update(src.name, status="complete", path=str(target),
                     bytes=target.stat().st_size)
        return target

    total, ranged = _probe_range(src.url)
    if not ranged or not total:
        print(f"[{src.name}] no Range support; falling back to single stream")
        return download(src, dest_dir, state)

    parts_dir = dest_dir / f"{src.filename}.parts"
    parts_dir.mkdir(exist_ok=True)
    span = -(-total // workers)
    bounds = [(i * span, min(total - 1, (i + 1) * span - 1))
              for i in range(workers) if i * span < total]

    progress = {"done": sum(p.stat().st_size for p in parts_dir.glob("*.part"))}
    lock = threading.Lock()
    print(f"[{src.name}] {human(total)} over {len(bounds)} connections", flush=True)

    threads, errors = [], []

    def run(i, lo, hi):
        try:
            _chunk_worker(src.url, lo, hi, parts_dir / f"{i:04d}.part", progress, lock)
        except Exception as exc:                       # noqa: BLE001
            errors.append(exc)

    for i, (lo, hi) in enumerate(bounds):
        t = threading.Thread(target=run, args=(i, lo, hi), daemon=True)
        t.start()
        threads.append(t)

    t0 = time.time()
    while any(t.is_alive() for t in threads):
        time.sleep(20)
        done = progress["done"]
        rate = done / max(time.time() - t0, 1e-6)
        eta = (total - done) / rate / 60 if rate > 0 else -1
        print(f"[{src.name}] {done/total*100:5.1f}% {human(done)}/{human(total)} "
              f"@ {human(rate)}/s eta {eta:.0f}m", flush=True)
        state.update(src.name, status="downloading", bytes=done, total=total)
    for t in threads:
        t.join()
    if errors:
        raise errors[0]

    with target.open("wb") as out:
        for i in range(len(bounds)):
            out.write((parts_dir / f"{i:04d}.part").read_bytes())
    size = target.stat().st_size
    if size != total:
        target.unlink()
        raise IOError(f"assembled {size} bytes, expected {total}")
    for p in parts_dir.glob("*.part"):
        p.unlink()
    parts_dir.rmdir()

    digest = sha256(target)
    print(f"[{src.name}] DONE {human(size)} sha256={digest[:16]}...")
    state.update(src.name, status="complete", path=str(target), bytes=size,
                 sha256=digest)
    return target


def fetch_hf(src: Source, dest_dir: Path, state: State, workers: int = 8,
             retries: int = 6) -> Path:
    """Pull a HuggingFace dataset repo.

    snapshot_download is resumable, verifies sha256 per file, and downloads
    files in parallel -- which matters because every host tested throttles hard
    per connection but not in aggregate.

    Wrapped in a retry loop because the pod's DNS is intermittently flaky under
    concurrent load: three repos in a row failed with EAI_NODATA ("No address
    associated with hostname") purely because another large download was in
    flight. snapshot_download resumes from whatever it already fetched, so a
    retry costs nothing but the backoff.
    """
    from huggingface_hub import snapshot_download

    target = dest_dir / src.name
    target.mkdir(parents=True, exist_ok=True)
    state.update(src.name, status="downloading", repo=src.repo, path=str(target))
    print(f"[{src.name}] snapshot_download {src.repo} -> {target} "
          f"({workers} workers)", flush=True)

    path = None
    for attempt in range(1, retries + 1):
        try:
            path = snapshot_download(
                repo_id=src.repo,
                repo_type="dataset",
                local_dir=str(target),
                allow_patterns=list(src.allow) or None,
                max_workers=workers,
            )
            break
        except Exception as exc:                        # noqa: BLE001
            msg = str(exc)
            if "401" in msg or "gated" in msg.lower() or "restricted" in msg.lower():
                state.update(src.name, status="gated", error=msg[:300])
                raise RuntimeError(
                    f"{src.repo} is a GATED repo: it needs a HuggingFace token "
                    f"AND the owner's approval. Request access on the dataset "
                    f"page, then set HF_TOKEN."
                ) from exc
            wait = min(60, 2 ** attempt)
            print(f"[{src.name}] attempt {attempt}/{retries} failed: "
                  f"{type(exc).__name__}: {msg[:100]} -- retrying in {wait}s",
                  flush=True)
            state.update(src.name, status="retrying", attempt=attempt,
                         error=msg[:200])
            time.sleep(wait)
    if path is None:
        state.update(src.name, status="failed")
        raise RuntimeError(f"[{src.name}] exhausted {retries} attempts")
    size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    n = sum(1 for f in Path(path).rglob("*") if f.is_file())
    print(f"[{src.name}] DONE {n} files, {human(size)}")
    state.update(src.name, status="complete", bytes=size, files=n, path=str(path))
    return Path(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=["asvspoof2019_la"],
                    choices=list(SOURCES) + ["all"])
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel connections; every origin tested throttles "
                         "per-connection but not in aggregate")
    ap.add_argument("--dest", default="/tmp/vericall-data",
                    help="pod-local scratch (ephemeral, 26 TB)")
    ap.add_argument("--state", default="data/fetch_state.json",
                    help="progress ledger on NFS (survives the pod)")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        total = sum(s.approx_gb for s in SOURCES.values())
        print(f"{'dataset':<18}{'size':>10}   note")
        for s in SOURCES.values():
            print(f"{s.name:<18}{s.approx_gb:>7.2f} GB   {s.note}")
        print(f"{'TOTAL':<18}{total:>7.2f} GB")
        return 0

    names = list(SOURCES) if "all" in args.datasets else args.datasets
    dest = Path(args.dest)
    state = State(Path(args.state))

    free_gb = os.statvfs(dest.parent if dest.exists() else "/").f_bavail * \
        os.statvfs("/").f_frsize / 2 ** 30
    need = sum(SOURCES[n].approx_gb for n in names)
    print(f"destination {dest}  free {free_gb:.0f} GB  need ~{need:.1f} GB")
    if free_gb < need * 1.15:
        print("REFUSING: not enough scratch headroom", file=sys.stderr)
        return 2

    failures = []
    for name in names:
        try:
            src = SOURCES[name]
            if src.kind == "hf":
                fetch_hf(src, dest, state, workers=args.workers)
            elif src.kind == "kaggle":
                fetch_kaggle(src, dest, state, workers=max(8, args.workers))
            else:
                download_parallel(src, dest, state, workers=max(8, args.workers))
        except Exception as exc:
            print(f"[{name}] FAILED: {exc}", file=sys.stderr)
            failures.append(name)

    print("\n=== summary ===")
    for name in names:
        entry = state.data.get(name, {})
        print(f"  {name:<16} {entry.get('status','?'):<12} "
              f"{human(entry.get('bytes', 0))}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
