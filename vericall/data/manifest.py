"""Dataset manifests: an index over the corpora, with no audio in it.

The corpora live on ephemeral pod scratch (~17 GB) because the NFS home has
55 GB free and is shared. The manifest is the small, durable part: it lives on
NFS, survives a namespace wipe, and records exactly which row of which shard
each item is, so audio is read lazily and never copied.

It also carries the three fields every leakage guard depends on:

    speaker_id      -- speaker-disjointness
    source_utt_id   -- groups every variant of one source utterance, which is
                       what stops a WaveFake sentence appearing as bona fide in
                       train and as its own vocoded twin in test
    attack_id       -- the generator (A01-A19, WF1-WF7), for seen/unseen splits
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from functools import partial as _partial
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

BONAFIDE, SPOOF = 0, 1
REAL_TAG = "R"


@dataclass(frozen=True, slots=True)
class Item:
    dataset: str
    utt_id: str            # unique per audio file
    source_utt_id: str     # groups all variants of the same underlying sentence
    speaker_id: str
    attack_id: str         # '-' bona fide, else the generator id
    label: int             # 0 bona fide, 1 spoof
    shard: str             # parquet file
    row: int               # row index within the shard
    sample_rate: int
    orig_split: str        # the corpus's own split name

    @property
    def is_bonafide(self) -> bool:
        return self.label == BONAFIDE


@dataclass
class Manifest:
    items: list[Item] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[Item]:
        return iter(self.items)

    def filter(self, **kw) -> "Manifest":
        def ok(it: Item) -> bool:
            return all(getattr(it, k) in v if isinstance(v, (set, frozenset, list, tuple))
                       else getattr(it, k) == v for k, v in kw.items())
        return Manifest([it for it in self.items if ok(it)])

    def counts(self) -> dict[str, int]:
        return {
            "n": len(self.items),
            "bonafide": sum(1 for i in self.items if i.label == BONAFIDE),
            "spoof": sum(1 for i in self.items if i.label == SPOOF),
            "speakers": len({i.speaker_id for i in self.items}),
            "source_utts": len({i.source_utt_id for i in self.items}),
            "attacks": len({i.attack_id for i in self.items}),
        }

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        with tmp.open("w") as fh:
            for it in self.items:
                fh.write(json.dumps(asdict(it), separators=(",", ":")) + "\n")
        tmp.replace(p)          # atomic: a torn manifest is worse than none

    @classmethod
    def load(cls, path: str | Path) -> "Manifest":
        return cls([Item(**json.loads(line))
                    for line in Path(path).read_text().splitlines() if line.strip()])


# --------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------- #

def _shards(root: Path, pattern: str) -> list[Path]:
    return sorted(root.glob(pattern))


def build_asvspoof(root: str | Path) -> Manifest:
    """ASVspoof 2019 LA. Audio is FLAC inside parquet; only metadata is read here.

    The official protocol is preserved by the source and is what makes this
    corpus worth using: train/dev carry attacks A01-A06 while eval carries
    A07-A19, so the eval set is genuinely unseen-attack by construction, and the
    speaker sets are disjoint across all three splits.
    """
    import pyarrow.parquet as pq

    root = Path(root)
    items: list[Item] = []
    for split_name, glob in (("train", "data/train*.parquet"),
                             ("dev", "data/validation*.parquet"),
                             ("eval", "data/test*.parquet")):
        for shard in _shards(root, glob):
            cols = pq.read_table(
                shard, columns=["speaker_id", "audio_file_name", "system_id", "key"]
            ).to_pydict()
            for i, (spk, utt, sysid, key) in enumerate(zip(
                cols["speaker_id"], cols["audio_file_name"],
                cols["system_id"], cols["key"],
            )):
                items.append(Item(
                    dataset="asvspoof2019_la",
                    utt_id=utt,
                    # each ASVspoof utterance is its own source; there is no
                    # bona-fide/spoof pairing over a shared sentence here
                    source_utt_id=utt,
                    speaker_id=spk,
                    attack_id=sysid,
                    label=int(key),
                    shard=str(shard),
                    row=i,
                    sample_rate=16_000,
                    orig_split=split_name,
                ))
    return Manifest(items)


def build_wavefake(root: str | Path) -> Manifest:
    """WaveFake over LJSpeech: R (real) plus WF1-WF7 vocoders.

    Two things about this corpus drive the split rules:

    1. Every vocoded clip is a resynthesis of the SAME LJSpeech sentence as its
       bona fide counterpart, so `audio_id` (e.g. LJ001-0001) is the grouping
       key -- a random split would put a sentence in train as real and its
       HiFi-GAN twin in test, and the model would "detect" by recognising the
       sentence.
    2. LJSpeech is a SINGLE speaker. This corpus can measure vocoder
       generalisation and nothing else; no speaker-generalisation claim may
       cite it.

    The mirror labels vocoders WF1-WF7 without publishing which is which, so
    they are treated as opaque families. That is sufficient for a held-out-
    family protocol and we do not invent names for them.
    """
    import pyarrow.parquet as pq

    root = Path(root)
    items: list[Item] = []
    for shard in _shards(root, "data/*.parquet"):
        cols = pq.read_table(shard, columns=["audio_id", "real_or_fake"]).to_pydict()
        for i, (src, tag) in enumerate(zip(cols["audio_id"], cols["real_or_fake"])):
            items.append(Item(
                dataset="wavefake",
                utt_id=f"{src}__{tag}",
                source_utt_id=src,
                speaker_id="LJSpeech_single",      # named to make the limit loud
                attack_id=tag,
                label=BONAFIDE if tag == "R" else SPOOF,
                shard=str(shard),
                row=i,
                sample_rate=22_050,
                orig_split="all",
            ))
    return Manifest(items)


def build_rtcfake(root: str | Path) -> Manifest:
    """RTCFake: speech transmitted through real RTC platforms.

    Layout is plain WAV on disk rather than parquet, so `shard` holds the audio
    file path and `row` is 0. Readers dispatch on the `.parquet` suffix.

    Three things about this corpus drive the split rules, and one of them is a
    limitation that must not be papered over:

    1. `meta/*_offline_online_pairs.csv` pairs each utterance with ITSELF --
       the same speech offline (clean) and online (after transmission). Those
       two files are one source utterance and MUST share a fold. A split that
       separates them puts the same sentence, the same speaker and the same
       label on both sides of the evaluation boundary, differing only in codec,
       which is the leak this whole module exists to prevent.
    2. Language (en/zh) is in the path and is a real generalisation axis: a
       model trained on en only and evaluated on zh is doing cross-lingual
       transfer. RESULTS section 13 measured Branch A *inverting* there
       (79.08% EER), which is the finding this corpus is being fetched to fix.
    3. **The corpus publishes no speaker id and no generator id.** So
       `speaker_id` is the sentinel "RTCFake_unknown" and `attack_id` is "RTC"
       for every spoof. The consequence is stated rather than hidden:
       **no speaker-disjointness guarantee is available for RTCFake**, and
       "unseen generator" cannot be measured on it at all. Utterance-level
       pairing is the strongest grouping the metadata supports.
    """
    import csv

    root = Path(root)
    data = root / "data"
    items: list[Item] = []

    for split in ("train", "dev"):
        labels_path = data / f"{split}_label.txt"
        if not labels_path.exists():
            continue

        # utt -> group. Both members of an offline/online pair take the offline
        # id as their group key, so they land in the same fold by construction.
        group: dict[str, str] = {}
        pairs_path = data / "meta" / f"{split}_offline_online_pairs.csv"
        if pairs_path.exists():
            with pairs_path.open(newline="") as fh:
                for row in csv.DictReader(fh):
                    off, on = row["offline_id"], row["online_id"]
                    group[off] = off
                    group[on] = off

        for line in labels_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rel, tag = line.rsplit(None, 1)
            utt = f"{split}/{rel}"              # matches the pairs-CSV spelling
            parts = rel.split("/")
            transport = parts[0] if parts else "unknown"   # offline | online
            lang = parts[1] if len(parts) > 1 else "unknown"
            label = BONAFIDE if tag == "bonafide" else SPOOF
            items.append(Item(
                dataset="rtcfake",
                utt_id=utt,
                # unpaired files are their own group; never silently merged
                source_utt_id=group.get(utt, utt),
                speaker_id="RTCFake_unknown",   # named to make the limit loud
                attack_id="-" if label == BONAFIDE else "RTC",
                label=label,
                shard=str(data / "wav" / split / rel),
                row=0,
                sample_rate=16_000,
                orig_split=f"{split}:{lang}:{transport}",
            ))
    return Manifest(items)


def build_indicsynth(root: str | Path, dataset: str = "indicsynth_hi") -> Manifest:
    """IndicSynth (Hindi slice). **Spoof only -- there is no bona fide class.**

    Every row is synthesised: the schema carries a `Generative Model` and a
    source/target speaker pair, and the published configs have a single `train`
    split with no real-speech partition. Checked on 50,044 rows across 16
    strided shards: two generator families, 101 target speakers, zero bona fide.

    The consequence is the whole reason this docstring is long: **no EER can be
    computed from this corpus alone**, and pairing it with an arbitrary real
    Hindi corpus would be worse than reporting nothing. Bona fide from corpus X
    against spoof from corpus Y lets a detector separate them by recording
    conditions rather than by synthesis, which is exactly the leak that inflates
    published numbers. What it CAN support is a miss rate at a threshold
    calibrated elsewhere -- see scripts/eval_indic.py.

    `Target Speaker ID` is the identity being impersonated, which is the closest
    thing to a speaker label here; `source_utt_id` is the audio path, since each
    row is its own utterance with no bona fide twin to group with.
    """
    import pyarrow.parquet as pq

    root = Path(root)
    items: list[Item] = []
    for shard in _shards(root, "*/*.parquet"):
        cols = pq.read_table(shard, columns=[
            "audio", "Generative Model", "Target Speaker ID", "Gender",
        ]).to_pydict()
        lang = Path(shard).parent.name
        for i, (au, gen, tgt, gender) in enumerate(zip(
                cols["audio"], cols["Generative Model"],
                cols["Target Speaker ID"], cols["Gender"])):
            name = (au or {}).get("path") or f"{Path(shard).stem}_{i}"
            items.append(Item(
                dataset=dataset,
                utt_id=f"{lang}/{name}",
                source_utt_id=f"{lang}/{name}",
                speaker_id=f"{lang}_tgt{tgt}",
                attack_id=str(gen),
                label=SPOOF,               # the corpus publishes nothing else
                shard=str(shard),
                row=i,
                sample_rate=16_000,
                orig_split=f"train:{lang}:{gender}",
            ))
    return Manifest(items)


def build_echofake(root: str | Path) -> Manifest:
    """EchoFake: physical REPLAY attacks, the vector this project never tested.

    docs/THREAT_MODEL.md A4 lists replay as untested, and it is a different
    threat from synthesis: replaying a recording of a genuine speaker is not
    synthetic speech at all, so there is no reason a branch trained to spot
    vocoder artifacts should catch it.

    This corpus makes the distinction measurable because it separates the two
    in metadata. `replay_details` is populated (player, recorder, room,
    distance) for replayed items and null otherwise, and `synthesis_details`
    the other way round -- so `attack_id` can carry which kind of attack an
    item is, and a replay-only evaluation is a filter rather than a new corpus.

    It also publishes a real `source_speaker_id`, which RTCFake does not, so
    speaker-disjoint splitting is available here -- and it is the guard that
    matters for this corpus.

    **`source` does NOT pair a bona fide clip with its attacks.** An earlier
    version of this docstring claimed it did, by analogy with WaveFake.
    Measured instead of assumed: the train split has 19,964 rows and 19,964
    distinct `source` values, zero repeats -- each attack is built from its own
    Common Voice recording rather than from the bona fide clips in the corpus.
    So there is no utterance-level leak of the WaveFake kind here, and each
    item is correctly its own source utterance. The exposure is SPEAKER
    overlap: 16,732 speakers across 75,490 items, roughly 4.5 items each, so a
    speaker can appear on both sides of a split unless the plan prevents it.
    """
    import pyarrow.parquet as pq

    root = Path(root)
    items: list[Item] = []
    for shard in _shards(root, "data/*.parquet"):
        cols = pq.read_table(shard, columns=[
            "utt_id", "label", "source", "source_speaker_id",
            "replay_details", "synthesis_details",
        ]).to_pydict()
        split = Path(shard).stem.split("-")[0]
        for i, (utt, lab, src, spk, rep, syn) in enumerate(zip(
                cols["utt_id"], cols["label"], cols["source"],
                cols["source_speaker_id"], cols["replay_details"],
                cols["synthesis_details"])):
            label = BONAFIDE if lab == "bonafide" else SPOOF
            if label == BONAFIDE:
                attack = "-"
            elif (rep or {}).get("player"):
                # The whole point of this corpus: name replay as its own class
                # so it can be evaluated separately from synthesis.
                attack = f"replay:{rep.get('player')}"
            else:
                attack = f"tts:{(syn or {}).get('model') or 'unknown'}"
            items.append(Item(
                dataset="echofake",
                # Split-qualified, because EchoFake's utt_id is NOT unique
                # across splits: 1,986 ids collide, and EF_T_01988 is both a
                # bona fide TRAIN item and a spoof DEV item. SplitPlan is keyed
                # by utt_id, so a collision silently applies one item's fold to
                # another item with a different label -- no error, corrupted
                # split. Caught by the duplicate guard in splits.verify().
                utt_id=f"{split}/{utt}",
                # Group by the underlying recording: a bona fide clip and every
                # attack derived from it are one source utterance.
                source_utt_id=str(src or utt),
                speaker_id=str(spk or "unknown"),
                attack_id=attack,
                label=label,
                shard=str(shard),
                row=i,
                sample_rate=16_000,
                orig_split=split,
            ))
    return Manifest(items)


def build_codecfake(root: str | Path) -> Manifest:
    """CodecFake: neural-codec resynthesis over VCTK.

    The newest generation of attack, and structurally the same shape as
    WaveFake -- one source utterance rendered by several generators -- so it
    carries the same leak and the same grouping rule. `audio_id` (p225_002) is
    the source utterance and its prefix (p225) is the VCTK speaker, which makes
    this corpus multi-speaker where WaveFake is single-speaker.

    Generator ids (R, F01, F02 ...) are opaque in the mirror. Treated as such
    rather than guessed at, which is sufficient for a held-out-family protocol.
    """
    import pyarrow.parquet as pq

    root = Path(root)
    items: list[Item] = []
    for shard in _shards(root, "data/*.parquet"):
        cols = pq.read_table(
            shard, columns=["audio_id", "real_or_fake"]).to_pydict()
        for i, (src, tag) in enumerate(zip(cols["audio_id"],
                                           cols["real_or_fake"])):
            items.append(Item(
                dataset="codecfake",
                utt_id=f"{src}__{tag}",
                source_utt_id=str(src),
                # p225_002 -> p225. Falls back to the whole id rather than
                # inventing a speaker if the shape ever changes.
                speaker_id=str(src).split("_")[0] or "unknown",
                attack_id=str(tag),
                label=BONAFIDE if tag == REAL_TAG else SPOOF,
                shard=str(shard),
                row=i,
                sample_rate=16_000,
                orig_split="all",
            ))
    return Manifest(items)


def build_in_the_wild(root: str | Path,
                      dataset: str = "in_the_wild") -> Manifest:
    """In-the-Wild (Muller et al., Interspeech 2022) -- 58 English public figures.

    **Evaluation only. This corpus is never trained on**, and that is the whole
    reason it is here. Every other corpus in this project is either what the
    model was fitted on or a near relative of it, and docs/RESULTS.md sections
    13, 17, 20 and 22 are a record of cross-corpus generalisation failing in
    five separate ways. In-the-Wild is the standard instrument for that
    question: the bona fide audio is genuine public recordings -- podcasts,
    speeches, interviews -- and the spoofs are clones of the same 58 people,
    so a detector cannot separate the classes by recording provenance.

    Layout is plain WAV plus a `meta.csv` of `file,speaker,label`, so `shard`
    holds the audio path and `row` is 0, the same addressing build_rtcfake
    uses. Readers dispatch on the `.parquet` suffix.

    Three properties that decide how it may be used:

    1. **Speaker labels are real and shared across classes.** The same public
       figure appears as both bona fide and spoof, which is what makes it an
       honest test -- and also means a speaker-disjoint split is meaningless
       here. It is an eval corpus; there is nothing to be disjoint from.
    2. **There is no generator id.** The release does not say which TTS or VC
       system produced each spoof, so `attack_id` is the sentinel "ITW" and
       "unseen generator" cannot be measured on this corpus. Stated rather
       than quietly ignored.
    3. **`source_utt_id` is the file itself.** Unlike RTCFake there is no
       offline/online pairing and no bona fide twin for each spoof, so there
       is no grouping to enforce beyond the utterance.

    The published release is 31,779 clips: 19,963 bona fide and 11,816 spoof,
    ~37.9 hours. `scripts/verify_in_the_wild.py` checks a download against
    those counts, because a mirror that had been re-encoded, subsetted or
    re-labelled would invalidate every number computed from it.
    """
    import csv

    root = Path(root)
    # The Kaggle mirror nests one directory deep; the origin zip does not.
    # Find the metadata rather than assuming either shape.
    meta_path = None
    for cand in (root / "meta.csv", *sorted(root.glob("*/meta.csv"))):
        if cand.exists():
            meta_path = cand
            break
    if meta_path is None:
        return Manifest([])
    base = meta_path.parent

    items: list[Item] = []
    with meta_path.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            # Header is `file,speaker,label` in the published release; accept
            # the common casing variants rather than failing on a mirror that
            # title-cased them.
            def pick(*names: str) -> str:
                for n in names:
                    for k in (n, n.capitalize(), n.upper()):
                        if row.get(k):
                            return str(row[k]).strip()
                return ""

            fname = pick("file", "filename", "path")
            if not fname:
                continue
            speaker = pick("speaker") or "ITW_unknown"
            raw_label = pick("label", "class", "type").lower()
            # "bona-fide" in the release, with "bonafide"/"real"/"genuine"
            # seen on mirrors. Anything else is a spoof, and an unrecognised
            # value must NOT silently become bona fide.
            is_bona = raw_label.replace("_", "-") in (
                "bona-fide", "bonafide", "real", "genuine", "0")
            audio = base / fname
            items.append(Item(
                dataset=dataset,
                utt_id=f"{dataset}/{fname}",
                source_utt_id=f"{dataset}/{fname}",
                speaker_id=f"ITW_{speaker.replace(' ', '_')}",
                attack_id="-" if is_bona else "ITW",
                label=BONAFIDE if is_bona else SPOOF,
                shard=str(audio),
                row=0,
                sample_rate=16_000,
                orig_split="eval",          # the release ships one partition
            ))
    return Manifest(items)


def build_indic_paired(root: str | Path, arm: str = "matched",
                       dataset: str = "hindi_paired") -> Manifest:
    """A paired Indic corpus built by scripts/build_indic_paired.py.

    Bona fide is Kathbath -- the real corpus IndicSynth cloned -- and spoof is
    IndicSynth's clones of **the same speakers**. That is the whole point: the
    project had recorded that no Hindi EER was computable because IndicSynth
    ships no bona fide class, and that borrowing bona fide from an unrelated
    corpus would measure recording conditions rather than synthesis. Both true,
    and both beside the point once you notice IndicSynth's filenames carry
    Kathbath speaker ids. Reuniting a corpus with its own source is not
    cross-corpus borrowing.

    Layout: <root>/<arm>/{bonafide,spoof}/<speaker>_<original stem>.wav, so
    label comes from the directory and speaker from the filename prefix.

    `arm` selects the codec treatment and MUST be stated wherever a number
    from this corpus is quoted:

      matched     both classes through one AAC generation. Kathbath is AAC and
                  IndicSynth is PCM, so this is the arm where the codec is not
                  a class cue and the EER is about synthesis.
      unmatched   as-shipped. Kept deliberately, to measure how large the
                  codec confound is rather than to assert it is small.

    File-addressed like RTCFake and In-the-Wild (shard = wav path, row = 0).
    """
    root = Path(root) / arm
    items: list[Item] = []
    for label, sub in ((BONAFIDE, "bonafide"), (SPOOF, "spoof")):
        d = root / sub
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.wav")):
            spk = f.stem.split("_", 1)[0]
            # Generator is recoverable from the IndicSynth stem; bona fide has
            # none. "unknown" rather than a guess if the stem is unfamiliar.
            attack = "-"
            if label == SPOOF:
                low = f.stem.lower()
                attack = ("xtts_v2" if "xtts" in low else
                          "freevc24" if "freevc" in low else
                          "vits" if "vits" in low else "unknown")
            items.append(Item(
                dataset=dataset,
                utt_id=f"{dataset}/{arm}/{sub}/{f.name}",
                source_utt_id=f"{dataset}/{arm}/{sub}/{f.name}",
                speaker_id=f"hi_{spk}",
                attack_id=attack,
                label=label,
                shard=str(f),
                row=0,
                sample_rate=16_000,
                orig_split=f"{arm}:{sub}",
            ))
    return Manifest(items)


# Every language IndicSynth and Kathbath both publish. Hindi was measured
# first (docs/RESULTS.md section 25); the construction is identical for the
# rest, which is the point of listing them here rather than hard-coding one.
INDIC_LANGUAGES = ("hindi", "bengali", "marathi", "tamil", "telugu",
                   "gujarati", "kannada", "malayalam", "odia", "punjabi",
                   "sanskrit", "urdu")

# The two-letter codes the IndicSynth slices are fetched under. Kept beside
# the language list so the two cannot drift: a language present in one and
# absent from the other is how the spoof side of a new language silently has
# no manifest builder, which is what happened to the six added in October.
INDIC_CODES = {"hindi": "hi", "bengali": "bn", "marathi": "mr", "tamil": "ta",
               "telugu": "te", "gujarati": "gu", "kannada": "kn",
               "malayalam": "ml", "odia": "or", "punjabi": "pa",
               "sanskrit": "sa", "urdu": "ur"}

# The original name, kept so existing imports and tests do not break.
build_hindi_paired = build_indic_paired

BUILDERS = {"asvspoof2019_la": build_asvspoof, "wavefake": build_wavefake,
            "rtcfake": build_rtcfake,
            "echofake": build_echofake, "codecfake": build_codecfake,
            "in_the_wild": build_in_the_wild,
            # Every IndicSynth slice, not just Hindi. Only indicsynth_hi was
            # registered, so build_manifests.py could not produce the spoof-side
            # manifest for any other language -- and build_indic_paired.py
            # takes that manifest as its input. The paired entries below were
            # all twelve already, which made the gap look like it was not
            # there until a new language was actually attempted.
            **{f"indicsynth_{code}": _partial(build_indicsynth,
                                              dataset=f"indicsynth_{code}")
               for code in INDIC_CODES.values()},
            # Two dataset names for two codec arms, so a cached fold can never
            # be mistaken for the other arm's. Quoting a matched number while
            # serving unmatched data is the worst failure this corpus admits.
            **{
                name: _partial(build_indic_paired, arm=arm, dataset=name)
                for lang in INDIC_LANGUAGES
                for name, arm in (
                    (f"{lang}_paired", "matched"),
                    (f"{lang}_paired_unmatched", "unmatched"),
                )
            }}


def build(dataset: str, root: str | Path) -> Manifest:
    return BUILDERS[dataset](root)
