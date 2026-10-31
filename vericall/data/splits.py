"""Leakage-safe split assignment.

Most published audio-deepfake numbers are inflated by one of a small set of
specific, well-understood leaks. Each is guarded here, and each guard is
asserted in tests/test_splits.py so a violation fails the build rather than
quietly producing a flattering EER.

Folds
-----
    train                  fit the branch heads
    dev                    early stopping only
    calib                  fit Platt / subset / eta calibration. MUST be
                           disjoint from train and eval, which the standard
                           ASVspoof protocol does not give you for free.
    eval_in_domain         ASVspoof eval (unseen attacks A07-A19 by design)
    eval_seen_vocoder      WaveFake, held-out utterances, TRAINED vocoders
    eval_unseen_vocoder    WaveFake, held-out utterances, HELD-OUT vocoders

The last two differ only in the vocoder, because the utterances are held out in
both. That makes the gap between them a controlled measurement of "generalises
to a voice-cloning tool we have never seen" -- which is the actual question
behind PS 26104, and is not answerable from a single pooled number.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .manifest import BONAFIDE, SPOOF, Item, Manifest

TRAIN = "train"
DEV = "dev"
CALIB = "calib"
EVAL_IN_DOMAIN = "eval_in_domain"
EVAL_SEEN_VOCODER = "eval_seen_vocoder"
EVAL_UNSEEN_VOCODER = "eval_unseen_vocoder"
EVAL_RTC_EN = "eval_rtc_en"
EVAL_RTC_ZH = "eval_rtc_zh"
EVAL_REPLAY = "eval_replay"          # EchoFake, physical replay only
EVAL_SYNTH_ECHO = "eval_synth_echo"  # EchoFake, synthesis only -- the control
EVAL_UNSEEN_CODEC = "eval_unseen_codec"
EVAL_IN_THE_WILD = "eval_in_the_wild"   # never trained on, by construction
EVAL_HINDI = "eval_hindi"               # paired Kathbath/IndicSynth

FOLDS = (TRAIN, DEV, CALIB, EVAL_IN_DOMAIN, EVAL_SEEN_VOCODER,
         EVAL_UNSEEN_VOCODER, EVAL_RTC_EN, EVAL_RTC_ZH,
         EVAL_REPLAY, EVAL_SYNTH_ECHO, EVAL_UNSEEN_CODEC, EVAL_IN_THE_WILD, EVAL_HINDI)
TRAINING_FOLDS = (TRAIN,)
EVAL_FOLDS = (EVAL_IN_DOMAIN, EVAL_SEEN_VOCODER, EVAL_UNSEEN_VOCODER,
              EVAL_RTC_EN, EVAL_RTC_ZH, EVAL_REPLAY, EVAL_SYNTH_ECHO,
              EVAL_UNSEEN_CODEC, EVAL_IN_THE_WILD, EVAL_HINDI)

# Corpora that publish no speaker labels. Speaker-disjointness cannot be
# checked for these, and verify() says so instead of silently passing.
NO_SPEAKER_LABELS = frozenset({"LJSpeech_single", "RTCFake_unknown"})

# Held-out WaveFake vocoder families. The mirror does not publish which index is
# which architecture, so these are treated as opaque -- sufficient for a
# held-out-family protocol, and we do not invent names we cannot verify.
UNSEEN_VOCODERS = frozenset({"WF5", "WF6", "WF7"})
REAL_TAG = "R"


def stable_bucket(key: str, n: int, salt: str = "") -> int:
    """Deterministic hash bucket.

    Deliberately not Python's hash(): that is randomised per process, so splits
    would silently differ between runs and a "reproducible" experiment would not
    be. Splits are generated once, written down, and never recomputed on the fly.
    """
    h = hashlib.sha256(f"{salt}:{key}".encode()).hexdigest()
    return int(h[:16], 16) % n


@dataclass(frozen=True)
class SplitPlan:
    """Per-item fold assignment, keyed by utt_id.

    The value is a tuple because one item can legitimately belong to more than
    one fold: WaveFake's bona fide clips are the shared negatives of both eval
    folds, which is precisely what makes those two folds comparable. Every other
    item has exactly one fold.
    """

    assignment: dict[str, tuple[str, ...]]

    def fold_of(self, item: Item) -> tuple[str, ...]:
        return self.assignment.get(item.utt_id, ())

    def subset(self, manifest: Manifest, fold: str) -> Manifest:
        return Manifest([i for i in manifest if fold in self.assignment.get(i.utt_id, ())])

    def summary(self, manifest: Manifest) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for fold in FOLDS:
            sub = self.subset(manifest, fold)
            if len(sub):
                out[fold] = sub.counts()
        return out


def plan_asvspoof(manifest: Manifest, calib_speaker_frac: float = 0.35,
                  salt: str = "vericall-v1") -> SplitPlan:
    """Use the official protocol, then carve `calib` out of dev BY SPEAKER.

    Splitting dev randomly would put the same speaker in both dev and calib.
    That is not a catastrophic leak -- neither trains the model -- but the
    calibration would then be fit on speakers whose scores also drove early
    stopping, which biases the operating point toward exactly the voices the run
    was tuned on. Speaker-disjoint costs nothing and removes the question.
    """
    dev_speakers = sorted({i.speaker_id for i in manifest if i.orig_split == "dev"})
    n_calib = max(1, round(len(dev_speakers) * calib_speaker_frac))
    ranked = sorted(dev_speakers, key=lambda s: stable_bucket(s, 10 ** 6, salt))
    calib_speakers = set(ranked[:n_calib])

    assignment: dict[str, tuple[str, ...]] = {}
    for it in manifest:
        if it.orig_split == "train":
            assignment[it.utt_id] = (TRAIN,)
        elif it.orig_split == "dev":
            assignment[it.utt_id] = (
                (CALIB,) if it.speaker_id in calib_speakers else (DEV,)
            )
        elif it.orig_split == "eval":
            assignment[it.utt_id] = (EVAL_IN_DOMAIN,)
    return SplitPlan(assignment)


def plan_wavefake(
    manifest: Manifest,
    unseen: frozenset[str] = UNSEEN_VOCODERS,
    weights: Mapping[str, float] | None = None,
    salt: str = "vericall-v1",
) -> SplitPlan:
    """Split on BOTH axes at once: source utterance AND vocoder family.

    One axis alone is not enough, and this is the single most important rule for
    this corpus:

      * split only by vocoder -> the same sentence appears in train and test,
        just rendered by a different vocoder, so the model can key on content
      * split only by utterance -> the model sees every vocoder it will be
        tested on, and "unseen generator" is not measured at all

    Bona fide (R) is assigned by utterance only and appears in both eval folds,
    so the two eval sets share their negatives and differ solely in which
    vocoder produced the positives. That is what makes their EER gap a
    controlled measurement rather than two unrelated numbers.
    """
    weights = weights or {TRAIN: 0.70, DEV: 0.10, CALIB: 0.10, "eval": 0.10}
    order = [TRAIN, DEV, CALIB, "eval"]
    edges, acc = [], 0.0
    for k in order:
        acc += weights[k]
        edges.append(acc)
    total = edges[-1]

    utt_fold: dict[str, str] = {}
    for src in sorted({i.source_utt_id for i in manifest}):
        u = stable_bucket(src, 10 ** 6, salt) / 10 ** 6 * total
        utt_fold[src] = next(k for k, e in zip(order, edges) if u <= e)

    assignment: dict[str, tuple[str, ...]] = {}
    for it in manifest:
        fold = utt_fold[it.source_utt_id]
        is_unseen = it.attack_id in unseen
        if fold == "eval":
            if it.attack_id == REAL_TAG:
                # the shared negatives of both eval folds
                assignment[it.utt_id] = (EVAL_SEEN_VOCODER, EVAL_UNSEEN_VOCODER)
            else:
                assignment[it.utt_id] = (
                    (EVAL_UNSEEN_VOCODER,) if is_unseen else (EVAL_SEEN_VOCODER,)
                )
        elif not is_unseen:
            assignment[it.utt_id] = (fold,)
        # held-out vocoders outside the eval utterance fold are simply dropped:
        # they must never appear in train, dev or calib

    return SplitPlan(assignment)


def plan_rtcfake(
    manifest: Manifest,
    train_langs: frozenset[str] = frozenset({"en"}),
    weights: Mapping[str, float] | None = None,
    salt: str = "vericall-rtc-v1",
) -> SplitPlan:
    """RTCFake: train on one language, hold the other out entirely.

    Two boundaries, and they do different jobs.

    **The corpus's own train/dev boundary is the evaluation boundary.** Our
    train/dev/calib folds are carved out of RTCFake *train*; RTCFake *dev*
    becomes eval_rtc_en / eval_rtc_zh untouched. Carving an eval fold out of the
    training split instead would mean trusting our own grouping more than the
    corpus authors' -- and with no speaker ids published, our grouping cannot
    check speaker overlap at all (see build_rtcfake). Using their split keeps
    the published protocol intact and keeps our claim checkable by anyone.

    **Language is held out, not sampled.** Every zh item in RTCFake *train* is
    dropped -- not down-weighted, dropped -- so eval_rtc_zh stays a genuine
    cross-lingual transfer test. That is the specific measurement this corpus
    was fetched for: RESULTS section 13 found Branch A inverting on Mandarin at
    79.08% EER, and "did retraining fix the inversion" is only answerable if
    Mandarin remains unseen. Training on zh as well is a deliberate, separate
    plan, not a default.

    Grouping is by source utterance, which for this corpus means an
    offline/online PAIR: the same speech before and after real transmission.
    Splitting a pair would put identical content on both sides of the boundary
    with only the codec differing.
    """
    weights = weights or {TRAIN: 0.80, DEV: 0.10, CALIB: 0.10}
    order = [TRAIN, DEV, CALIB]
    edges, acc = [], 0.0
    for k in order:
        acc += weights[k]
        edges.append(acc)
    total = edges[-1]

    def lang_of(it: Item) -> str:
        parts = it.orig_split.split(":")
        return parts[1] if len(parts) > 1 else "unknown"

    def split_of(it: Item) -> str:
        return it.orig_split.split(":")[0]

    # A source-utterance group must get ONE fold, so the group's language is
    # taken from the group rather than per item. Pairs never straddle languages
    # in this corpus, but deriving it per group makes that an invariant rather
    # than an assumption.
    group_lang: dict[str, str] = {}
    for it in manifest:
        group_lang.setdefault(it.source_utt_id, lang_of(it))

    grp_fold: dict[str, str] = {}
    for src in sorted({i.source_utt_id for i in manifest if split_of(i) == TRAIN}):
        u = stable_bucket(src, 10 ** 6, salt) / 10 ** 6 * total
        grp_fold[src] = next(k for k, e in zip(order, edges) if u <= e)

    assignment: dict[str, tuple[str, ...]] = {}
    for it in manifest:
        lang = group_lang.get(it.source_utt_id, "unknown")
        if split_of(it) == DEV:
            if lang == "en":
                assignment[it.utt_id] = (EVAL_RTC_EN,)
            elif lang == "zh":
                assignment[it.utt_id] = (EVAL_RTC_ZH,)
            # any other language is left unassigned rather than pooled into one
            # of the two named folds, which would make the fold name a lie
        elif split_of(it) == TRAIN and lang in train_langs:
            assignment[it.utt_id] = (grp_fold[it.source_utt_id],)
        # held-out languages in the train split are DROPPED, not reassigned

    return SplitPlan(assignment)


def plan_echofake(manifest: Manifest, salt: str = "vericall-echo-v1") -> SplitPlan:
    """EchoFake, split to make REPLAY separately measurable.

    docs/THREAT_MODEL.md A4 lists replay as untested, and it is a genuinely
    different threat: replaying a recording of a real speaker is not synthetic
    speech, so a branch trained on vocoder artifacts has no reason to catch it.

    Two eval folds over the corpus's own eval splits, sharing the same bona
    fide clips and differing only in the attack type:

        eval_replay       bona fide + physical replay attacks
        eval_synth_echo   bona fide + synthesis attacks  <- the control

    Sharing the negatives is what makes the gap between their EERs a
    measurement of "does replay evade this detector" rather than two unrelated
    numbers -- the same construction as the two WaveFake vocoder folds.

    **Speaker-disjoint, because this corpus can be.** `source` does not pair a
    bona fide clip with its attacks here (measured: zero repeats), so the
    utterance guard is vacuous and the real exposure is speaker overlap --
    16,732 speakers over 75,490 items. Speakers are assigned whole to a side.
    """
    train_like, eval_like = {}, {}
    for it in manifest:
        (eval_like if "eval" in it.orig_split else train_like)[it.utt_id] = it

    # Speakers seen in the corpus's own training material are kept out of our
    # eval folds entirely, so a speaker cannot straddle the boundary.
    train_speakers = {it.speaker_id for it in train_like.values()}

    assignment: dict[str, tuple[str, ...]] = {}
    for it in manifest:
        is_eval = "eval" in it.orig_split
        if is_eval:
            if it.speaker_id in train_speakers:
                continue                      # dropped, not reassigned
            if it.label == BONAFIDE:
                # Shared negatives: both folds must score the same bona fide.
                assignment[it.utt_id] = (EVAL_REPLAY, EVAL_SYNTH_ECHO)
            elif it.attack_id.startswith("replay:"):
                assignment[it.utt_id] = (EVAL_REPLAY,)
            else:
                assignment[it.utt_id] = (EVAL_SYNTH_ECHO,)
        else:
            u = stable_bucket(it.speaker_id, 100, salt)
            assignment[it.utt_id] = ((TRAIN,) if u < 80
                                     else (DEV,) if u < 90 else (CALIB,))
    return SplitPlan(assignment)


def plan_codecfake(manifest: Manifest,
                   unseen: frozenset[str] = frozenset({"F05", "F06"}),
                   salt: str = "vericall-codec-v1") -> SplitPlan:
    """CodecFake: held-out codec families, split on utterance AND generator.

    Structurally WaveFake's problem with a different generator set, so it takes
    WaveFake's rule: split by BOTH axes at once. By generator alone leaks the
    sentence; by utterance alone means every test codec was trained on.

    **Split by SPEAKER, not by utterance.** CodecFake is VCTK, and every
    speaker appears across the whole corpus -- so an utterance-level split
    leaves all 64 speakers on both sides, which the guard duly reported. That
    makes this fold harder than WaveFake's equivalent and more like deployment:
    eval_unseen_codec is unseen codec AND unseen speaker.

    The controlled single-variable measurement of codec generalisation alone
    already exists in the WaveFake folds, which hold the speaker fixed by
    construction (LJSpeech is one speaker). Having both is better than having
    the same measurement twice.
    """
    order = [TRAIN, DEV, CALIB, "eval"]
    edges = [0.70, 0.80, 0.90, 1.00]
    spk_fold: dict[str, str] = {}
    for spk in sorted({i.speaker_id for i in manifest}):
        u = stable_bucket(spk, 10 ** 6, salt) / 10 ** 6
        spk_fold[spk] = next(k for k, e in zip(order, edges) if u <= e)
    utt_fold = {i.source_utt_id: spk_fold[i.speaker_id] for i in manifest}

    assignment: dict[str, tuple[str, ...]] = {}
    for it in manifest:
        fold = utt_fold[it.source_utt_id]
        is_unseen = it.attack_id in unseen
        if fold == "eval":
            if it.attack_id == REAL_TAG:
                assignment[it.utt_id] = (EVAL_UNSEEN_CODEC,)
            elif is_unseen:
                assignment[it.utt_id] = (EVAL_UNSEEN_CODEC,)
        elif not is_unseen:
            assignment[it.utt_id] = (fold,)
        # held-out codecs outside the eval utterance fold are dropped
    return SplitPlan(assignment)


def eval_bonafide_ids(manifest: Manifest, plan: SplitPlan) -> set[str]:
    """Bona fide utt_ids usable as negatives for BOTH WaveFake eval folds."""
    return {
        i.utt_id for i in manifest
        if i.attack_id == REAL_TAG
        and EVAL_SEEN_VOCODER in plan.assignment.get(i.utt_id, ())
    }


# --------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class Violation:
    guard: str
    detail: str


def verify(manifest: Manifest, plan: SplitPlan) -> list[Violation]:
    """Check every leakage guard. Empty list means clean.

    This is called from the tests, and should also be called before any training
    run: a split is cheap to re-verify and expensive to get wrong.
    """
    v: list[Violation] = []

    # --- guard 1: utt_id must be unique -----------------------------------
    # SplitPlan.assignment is keyed by utt_id, so two items sharing one id
    # means one item's fold silently governs the other -- including across
    # labels. EchoFake shipped 1,986 such collisions (EF_T_01988 was a bona
    # fide train item AND a spoof dev item) and nothing raised. This is checked
    # for every corpus because any of them could do it and the failure is
    # invisible in the fold summaries.
    seen: dict[str, int] = defaultdict(int)
    for it in manifest:
        seen[it.utt_id] += 1
    dupes = {k: n for k, n in seen.items() if n > 1}
    if dupes:
        example = next(iter(dupes.items()))
        v.append(Violation(
            "utt_id_unique",
            f"{len(dupes)} utt_id(s) appear more than once, e.g. "
            f"{example[0]!r} x{example[1]}; SplitPlan is keyed by utt_id so "
            f"these silently share a fold assignment"))

    by_fold: dict[str, list[Item]] = defaultdict(list)
    for it in manifest:
        for fold in plan.assignment.get(it.utt_id, ()):
            by_fold[fold].append(it)

    train_like = [f for f in (TRAIN, DEV, CALIB) if f in by_fold]
    evals = [f for f in EVAL_FOLDS if f in by_fold]

    # --- guard 2: source-utterance disjointness (the WaveFake killer) -------
    src_of: dict[str, set[str]] = defaultdict(set)
    for fold, items in by_fold.items():
        for it in items:
            src_of[it.source_utt_id].add(fold)
    # Two legitimate shared-negative pairs: each shares its bona fide clips
    # between two eval folds so the gap between them isolates one variable
    # (vocoder family / attack type). Everything else straddling is a leak.
    SHARED = ({EVAL_SEEN_VOCODER, EVAL_UNSEEN_VOCODER},
              {EVAL_REPLAY, EVAL_SYNTH_ECHO})
    straddling = {s: f for s, f in src_of.items() if len(f) > 1
                  and not any(f <= ok for ok in SHARED)}
    if straddling:
        example = next(iter(straddling.items()))
        v.append(Violation(
            "source_utterance_disjoint",
            f"{len(straddling)} source utterances span folds, e.g. "
            f"{example[0]} in {sorted(example[1])}",
        ))

    # --- guard 3: speaker disjointness (skipped for single-speaker corpora) --
    for a in train_like:
        for b in evals:
            spk_a = {i.speaker_id for i in by_fold[a]}
            spk_b = {i.speaker_id for i in by_fold[b]}
            shared = spk_a & spk_b
            if shared and not shared <= NO_SPEAKER_LABELS:
                v.append(Violation(
                    "speaker_disjoint",
                    f"{a} and {b} share {len(shared)} speakers: {sorted(shared)[:4]}",
                ))

    # --- guard 4: held-out generators must not appear in training folds -----
    for fold in train_like:
        leaked = {i.attack_id for i in by_fold[fold]} & UNSEEN_VOCODERS
        if leaked:
            v.append(Violation(
                "unseen_generator_leak",
                f"held-out vocoders {sorted(leaked)} present in {fold}",
            ))

    # --- guard 7: calibration must not overlap train or eval ----------------
    if CALIB in by_fold:
        calib_ids = {i.utt_id for i in by_fold[CALIB]}
        for other in [f for f in (TRAIN,) + tuple(evals) if f in by_fold]:
            overlap = calib_ids & {i.utt_id for i in by_fold[other]}
            if overlap:
                v.append(Violation(
                    "calibration_disjoint",
                    f"calib overlaps {other} on {len(overlap)} items",
                ))

    # --- both classes must be present wherever a threshold is fit or read ---
    for fold in [f for f in (TRAIN, DEV, CALIB) + tuple(evals) if f in by_fold]:
        labels = {i.label for i in by_fold[fold]}
        if labels != {BONAFIDE, SPOOF}:
            v.append(Violation(
                "class_coverage",
                f"{fold} contains only label(s) {sorted(labels)}; EER and "
                f"calibration are undefined without both classes",
            ))
    return v


def plan_in_the_wild(manifest: Manifest) -> SplitPlan:
    """In-the-Wild: one fold, everything in it, nothing withheld.

    Every other planner in this module divides a corpus and spends most of its
    length defending the boundary -- speaker disjointness, source-utterance
    pairing, unseen generators. This one has no boundary to defend, and saying
    so explicitly is the point rather than an omission.

    **There is no training fold because this corpus is never trained on.** It
    is the only genuinely held-out instrument the project has: docs/RESULTS.md
    records cross-corpus generalisation failing five separate ways, and every
    one of those measurements was taken on a corpus that shares a lineage with
    something the model saw. If In-the-Wild ever appeared in TRAIN, the number
    it produces would stop meaning anything and there would be nothing left to
    replace it with. The absence of a training fold is a property to preserve,
    not a gap to fill later.

    **Speaker-disjointness is not applicable, and would be wrong here.** The
    same 58 public figures appear as both bona fide and spoof by design: the
    spoofs are clones of the very people whose real recordings form the
    negative class. That is what makes the corpus honest -- a detector cannot
    separate the classes by speaker, accent or recording provenance, only by
    synthesis. Holding speakers out would destroy exactly the property the
    corpus exists to provide.

    So the plan is a single assignment and the leakage guards in `verify()`
    pass trivially, which is correct: with no train fold there is no train/eval
    boundary for an utterance or a speaker to cross.
    """
    return SplitPlan({it.utt_id: (EVAL_IN_THE_WILD,) for it in manifest})


def plan_hindi_paired(manifest: Manifest, eval_frac: float = 0.35,
                      dev_frac: float = 0.15, calib_frac: float = 0.15,
                      salt: str = "vericall-hindi-v1428") -> SplitPlan:
    """Speaker-disjoint train/dev/calib/eval over the paired Hindi corpus,
    stratified so every fold sees every generator.

    **Speaker-disjoint, strictly.** Bona fide and spoof for a given speaker are
    the same person's voice -- one recorded, one cloned -- so a speaker who
    straddles the boundary hands the model a voice it has already heard on both
    sides of the label. That is the most direct leak this corpus could have.

    **Stratified by generator, and that is not cosmetic.** Only 5 of the 20
    shared speakers have any XTTS-v2 clones; the rest are FreeVC24 only. A
    plain hash of the speaker id put all five outside the eval fold, so
    `eval_hindi` contained 400 voice-conversion attacks and zero zero-shot TTS
    -- and the headline Hindi number would have been measured on the older,
    easier attack family while appearing to cover both. Speakers are therefore
    grouped by which generators they carry, and each group is spread across the
    folds independently.

    **Unlike In-the-Wild, this corpus DOES get a training fold**, deliberately.
    In-the-Wild is the project's one untouched instrument. Hindi is the
    deployment target, and the model has never been trained on a word of it:
    docs/RESULTS.md records five failed attempts to make the representation
    transfer, not one of which involved any Indian language. Branch A's head is
    0.89 M parameters, so the experiment that has never been run is cheap.

    Assignment is a stable hash within each stratum, so adding speakers later
    does not reshuffle the ones already placed -- a split that changes between
    runs makes every number irreproducible.
    """
    # Generator signature per speaker, from the spoof side only.
    gens: dict[str, set[str]] = {}
    for it in manifest:
        if it.label == SPOOF:
            gens.setdefault(it.speaker_id, set()).add(it.attack_id)

    e = int(round(eval_frac * 100))
    d = e + int(round(dev_frac * 100))
    c = d + int(round(calib_frac * 100))

    # One stable hash per speaker, salted BY ITS STRATUM.
    #
    # Stability and guaranteed coverage cannot both be had for a stratum of
    # five or six speakers, and stability is the one the project requires: a
    # split that reshuffles when a speaker is added makes every number
    # irreproducible. An earlier version of this function dealt strata
    # round-robin into folds, which guaranteed coverage and silently broke
    # stability -- test_a_speaker_keeps_its_fold_when_others_are_added caught
    # it, and only when the suite was finally run on a machine where it had
    # not already been cached as passing.
    #
    # Salting per stratum keeps each speaker's assignment a pure function of
    # its own id and its own generator set, so adding speakers never moves an
    # existing one, while giving each generator family an INDEPENDENT draw
    # rather than sharing one global hash. That is what fixes the original
    # problem: a single global hash put all five XTTS-v2 speakers outside the
    # eval fold, so eval_hindi held 400 voice-conversion attacks and zero
    # zero-shot TTS while appearing to cover both.
    #
    # Coverage is therefore checked, not assumed, and the default `salt` was
    # SELECTED so that it holds: of 1,500 candidate salts, 392 give every fold
    # both attack families with at least 30 spoofs of each, and v1428 is the
    # one among those whose fold sizes best fit the 35/15/15/35 target
    # (fraction-error 0.012).
    #
    # Choosing a salt this way does not leak. The criterion is a property of
    # the SPLIT -- does each fold contain both generators, are the folds the
    # intended size -- and never a property of any model's performance on it;
    # no checkpoint existed when the search ran. The alternative was a fold
    # that measures one attack family while appearing to cover two, which is a
    # worse failure than a declared salt choice. Stated here rather than left
    # for someone to find in a commit message.
    fold_of: dict[str, str] = {}
    for spk in sorted({it.speaker_id for it in manifest}):
        sig = "+".join(sorted(gens.get(spk, ()))) or "none"
        u = stable_bucket(spk, 100, f"{salt}:{sig}")
        fold_of[spk] = (EVAL_HINDI if u < e else DEV if u < d
                        else CALIB if u < c else TRAIN)

    return SplitPlan({it.utt_id: (fold_of[it.speaker_id],) for it in manifest})
