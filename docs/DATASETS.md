# VeriCall — corpora, mirrors and the verification record

## What is on disk

All on pod scratch `/tmp/vericall-data/`. Verify any time with
`./scripts/pod.sh py 'scripts/verify_datasets.py'`.

| Corpus | Size | Items | Role | Verified |
|---|---:|---:|---|---|
| ASVspoof 2019 LA | 7.03 GB | 121,461 | primary training + in-domain eval | yes |
| WaveFake (LJSpeech) | 28.20 GB | 104,800 | held-out-vocoder generalisation | yes |
| In-the-Wild | 7.60 GB | 31,779 | cross-domain eval, never trained on | yes |
| EchoFake | 3.04 GB | — | physical replay attacks | pending |
| CodecFake | ~3.4 GB | — | neural-codec-synthesised speech | pending |
| IndicSynth | 16.01 GB | — | multilingual Indian (CC BY-NC) | queued |

### Not obtained: RTCFake

`JunXueTech/RTCFake` (20.4 GB) is a **gated repo**. Its file tree is public,
which is why it initially looked available, but every file returns HTTP 401:

```
Access to dataset JunXueTech/RTCFake is restricted.
You must have access to it and be authenticated to access it.
```

It needs a HuggingFace account, an access request approved by the dataset
owner, and `HF_TOKEN` set. **That is a human step and cannot be automated.**

This matters more than the other gaps: RTCFake is the only corpus containing
speech actually transmitted through real RTC platforms (WeChat, Zoom, QQ,
DingTalk, Lark, VooV, Telegram), so it is the only honest test of the
"survives compression" constraint. Until it lands, codec robustness is
evidenced by **simulated** G.711/Opus augmentation, which is weaker and must be
described as such rather than glossed.

Manifests and split plans live on NFS under `data/` and survive a pod wipe; the
audio does not, and `scripts/fetch_datasets.py` re-fetches it idempotently.

## Why mirrors, and why that needed verifying

The canonical hosts are unusably slow from this cluster. Measured, per
connection:

| Host | 1 conn | 8 conns | 16 conns |
|---|---:|---:|---:|
| Edinburgh DataShare | 25 KB/s | 704 KB/s | 361 KB/s |
| Zenodo | 42 KB/s | 1.4 MB/s | 5.1 MB/s |
| Fraunhofer ownCloud | 121 KB/s | ~150 KB/s (client-throttled) | ~150 KB/s |
| **Kaggle** | **11 MB/s** | — | — |
| **HuggingFace CDN** | — | — | **~45 MB/s effective** |

Two corrections worth recording, because both cost time:

**Servers lie about Range.** The Fraunhofer host advertises
`Accept-Ranges: none` on HEAD and then serves `206` with a valid
`Content-Range` on GET. Believing the HEAD wrote it off as un-resumable.
`_probe_range()` now probes with a real ranged GET.

**Per-connection throttling is not per-client throttling, and the two look
identical in a single-connection test.** Most origins here throttle per
connection, so parallel range requests help enormously (Zenodo: 42 KB/s -> 5.1
MB/s). Fraunhofer throttles per *client*: twelve connections still totalled
~150 KB/s, i.e. a ~14 hour download. Switching to the Kaggle mirror
(`abdallamohamed312/in-the-wild-dataset`) fetched the same 7.6 GB in under two
minutes, and its contents verify exactly against the published release.

Every origin throttles hard **per connection** but not in aggregate, which is
why parallel range requests help so much. A note against an earlier wrong
conclusion of mine: the pod is *not* bandwidth-starved relative to the head
node. Single-connection tests suggested a ~250× gap; with parallel connections
the pod matches or beats the head node. The throttle was per-connection all
along.

Even so, DataShare would have taken ~10 hours to deliver 23.6 GB in order to
extract 7 GB of LA. The HuggingFace mirror delivered the same corpus in
**2 minutes 40 seconds**.

## Mirror verification

A third-party mirror can be re-encoded, re-split, or stripped of metadata, any
of which silently invalidates results. Both were checked against the official
release before being used.

### `Bisher/ASVspoof_2019_LA`

| Check | Result |
|---|---|
| Audio encoding | **FLAC, lossless**, 16 kHz mono 16-bit — native ASVspoof format |
| Split sizes | train 25,380 / dev 24,844 / eval 71,237 — **exactly the published protocol** |
| Class balance (train) | 2,580 bona fide / 22,800 spoof — official |
| Class balance (eval) | 7,355 bona fide / 63,882 spoof — official |
| Attack ids retained | train/dev **A01–A06**; eval **A07–A19** — the official unseen-attack design is intact |
| Speaker ids retained | yes; 20 train, 67 eval |
| Speaker disjointness | train ∩ dev = 0, train ∩ eval = 0 |

Lossy re-encoding was the specific risk here: it would destroy exactly the
spectral artifacts Branches A and A2 detect, while leaving the dataset looking
perfectly healthy. It is lossless.

### In-the-Wild via Kaggle

| Check | Result |
|---|---|
| Clips | **31,779** — exactly the published count |
| Bona fide / spoof | **19,963 / 11,816** — exactly published |
| Speakers in `meta.csv` | 54 distinct names (the paper describes 58 public figures) |
| Audio encoding | PCM lossless, 16 kHz mono 32-bit |
| Structure | `release_in_the_wild/` with `meta.csv` and `attribution.txt` intact |

Kaggle wraps a dataset in its own archive, so this arrives as a zip containing a
single entry named `download` which is itself the real release zip.
`_unwrap_nested_zip()` handles that automatically rather than leaving it as a
trap for the next reader.

### `ajaykarthick/wavefake-audio`

| Check | Result |
|---|---|
| Audio encoding | **PCM WAV, lossless**, 22.05 kHz mono 16-bit — native LJSpeech rate |
| Items | 104,800 = 13,100 LJSpeech utterances × 8 classes |
| Classes | `R` (bona fide) + `WF1`–`WF7` (seven vocoders) |
| Source utterance id | `audio_id` (e.g. `LJ001-0001`) retained — **required** for the leakage guard |

The mirror does not publish which `WFn` index is which vocoder architecture, so
they are treated as **opaque families**. That is sufficient for a held-out-family
protocol, and we do not invent architecture names we cannot verify.

Because `R` is included, the separate LJSpeech download the original plan
called for is unnecessary.

## Splits

Generated once by `scripts/build_manifests.py`, written to `data/splits/`, never
recomputed on the fly, and verified by `tests/test_splits.py`.

### ASVspoof 2019 LA

| Fold | n | bona fide | spoof | speakers | generators |
|---|---:|---:|---:|---:|---|
| train | 25,380 | 2,580 | 22,800 | 20 | A01–A06 |
| dev | 15,962 | 1,694 | 14,268 | 13 | A01–A06 |
| calib | 8,882 | 854 | 8,028 | 7 | A01–A06 |
| eval_in_domain | 71,237 | 7,355 | 63,882 | 67 | **A07–A19** |

`calib` is carved out of the official dev set **by speaker**, not at random. Both
fit nothing that trains the model, but a shared speaker would mean the operating
point was calibrated on the very voices that drove early stopping.

### WaveFake — split on two axes at once

| Fold | n | bona fide | spoof | source utts | generators |
|---|---:|---:|---:|---:|---|
| train | 45,905 | 9,181 | 36,724 | 9,181 | R, WF1–WF4 |
| dev | 6,580 | 1,316 | 5,264 | 1,316 | R, WF1–WF4 |
| calib | 6,350 | 1,270 | 5,080 | 1,270 | R, WF1–WF4 |
| eval_seen_vocoder | 6,665 | 1,333 | 5,332 | 1,333 | R, WF1–WF4 |
| eval_unseen_vocoder | 5,332 | 1,333 | 3,999 | 1,333 | **R, WF5–WF7** |

The two eval folds hold out the **same 1,333 source utterances** and share the
**same 1,333 bona fide clips**. They differ only in which vocoder produced the
positives, so the gap between their EERs is a controlled measurement of
"generalises to a cloning tool we have never seen" — the question behind
PS 26104 — rather than two unrelated numbers.

Splitting on one axis alone would not do: by vocoder only leaks the sentence, by
utterance only means every test vocoder was trained on.

**Stated limit:** LJSpeech is a single speaker. WaveFake can measure vocoder
generalisation and nothing else. No speaker-generalisation claim may cite it,
and `tests/test_splits.py` encodes that.

## The silence shortcut — measured, not assumed

Documented for ASVspoof 2019 LA and confirmed here on 3,000 stratified samples:

| Cue | bona fide | spoof | gap |
|---|---:|---:|---:|
| leading silence | 0.925 s | 0.573 s | **+0.352 s** |
| trailing silence | 0.407 s | 0.353 s | +0.054 s |
| total duration | 3.418 s | 3.447 s | −0.029 s |

Exploitability, as EER from that cue alone with no speech content whatsoever:

| Feature | EER |
|---|---:|
| leading silence, raw audio | **36.4%** |
| trailing silence, raw audio | 49.2% |
| total duration, raw audio | 48.6% |
| **leading silence, after trimming** | **48.2%** (chance — cue removed) |

So a model trained on untrimmed audio can bank a real, transferable-looking
advantage from a stopwatch. VAD trimming is mandatory in `audio_io.load_audio`,
and `scripts/audit_silence.py` is the standing control: **Branch A must beat
36.4% EER by a wide margin or it has learned nothing worth deploying.**

Reproduce with `./scripts/pod.sh py 'scripts/audit_silence.py --n 3000'`.
