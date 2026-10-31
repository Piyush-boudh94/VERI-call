"""Leakage guards.

Every check here corresponds to a documented way that published audio-deepfake
numbers get inflated. They run on synthetic manifests (fast, always) and, when
the corpora are present, on the real split plans too.

A failure here means a reported EER would be wrong. These are build-breaking
assertions, not advisory checks.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from vericall.data.manifest import BONAFIDE, SPOOF, Item, Manifest
from vericall.data.splits import (
    CALIB, DEV, EVAL_IN_DOMAIN, EVAL_REPLAY, EVAL_RTC_EN, EVAL_RTC_ZH,
    EVAL_SEEN_VOCODER, EVAL_SYNTH_ECHO, EVAL_UNSEEN_CODEC,
    EVAL_UNSEEN_VOCODER, REAL_TAG, TRAIN, UNSEEN_VOCODERS, SplitPlan,
    plan_asvspoof, plan_codecfake, plan_echofake, plan_rtcfake, plan_wavefake,
    stable_bucket, verify,
)

DATA = Path("/tmp/vericall-data")


# --------------------------------------------------------------- synthetic

def wavefake_like(n_utts: int = 300) -> Manifest:
    """Mimics WaveFake: one source sentence rendered by R + WF1..WF7."""
    items = []
    for u in range(n_utts):
        src = f"LJ{u:04d}-0001"
        for tag in ("R", "WF1", "WF2", "WF3", "WF4", "WF5", "WF6", "WF7"):
            items.append(Item(
                dataset="wavefake", utt_id=f"{src}__{tag}", source_utt_id=src,
                speaker_id="LJSpeech_single", attack_id=tag,
                label=BONAFIDE if tag == REAL_TAG else SPOOF,
                shard="s.parquet", row=u, sample_rate=22050, orig_split="all",
            ))
    return Manifest(items)


def asvspoof_like() -> Manifest:
    items = []
    for split, speakers, attacks in (
        ("train", range(0, 20), ["A01", "A02"]),
        ("dev", range(20, 40), ["A01", "A02"]),
        ("eval", range(40, 60), ["A07", "A08"]),
    ):
        for spk in speakers:
            for i in range(6):
                a = "-" if i < 2 else attacks[i % len(attacks)]
                items.append(Item(
                    dataset="asvspoof2019_la", utt_id=f"{split}_{spk}_{i}",
                    source_utt_id=f"{split}_{spk}_{i}", speaker_id=f"LA_{spk:04d}",
                    attack_id=a, label=BONAFIDE if a == "-" else SPOOF,
                    shard="s.parquet", row=i, sample_rate=16000, orig_split=split,
                ))
    return Manifest(items)


def test_bucketing_is_stable_across_processes():
    """Python's hash() is randomised per process, so using it would make splits
    silently differ between runs and 'reproducible' would be a lie."""
    assert stable_bucket("LJ0001-0001", 100) == stable_bucket("LJ0001-0001", 100)
    assert stable_bucket("a", 1000) != stable_bucket("b", 1000)


# ------------------------------------------------- guard 2: source utterance

def test_wavefake_source_utterance_never_spans_folds():
    """THE WaveFake leak. Its fakes are vocoder resyntheses of the same LJSpeech
    sentence as the bona fide reference, so a random split puts a sentence in
    train as real and its HiFi-GAN twin in test -- and the model 'detects' by
    recognising the sentence."""
    man = wavefake_like()
    plan = plan_wavefake(man)
    folds_by_src: dict[str, set[str]] = {}
    for it in man:
        for f in plan.fold_of(it):
            folds_by_src.setdefault(it.source_utt_id, set()).add(f)
    for src, folds in folds_by_src.items():
        assert folds <= {EVAL_SEEN_VOCODER, EVAL_UNSEEN_VOCODER} or len(folds) == 1, \
            f"{src} spans {sorted(folds)}"


def test_a_random_split_would_have_leaked():
    """Demonstrates the guard is load-bearing: the naive split it replaces
    genuinely leaks, so verify() must reject it."""
    man = wavefake_like()
    naive = SplitPlan({
        it.utt_id: ((TRAIN,) if stable_bucket(it.utt_id, 10) < 7 else (EVAL_IN_DOMAIN,))
        for it in man
    })
    assert any(v.guard == "source_utterance_disjoint" for v in verify(man, naive))


# ------------------------------------------------- guard 4: unseen generators

def test_held_out_vocoders_never_appear_in_training_folds():
    man = wavefake_like()
    plan = plan_wavefake(man)
    for it in man:
        if it.attack_id in UNSEEN_VOCODERS:
            assert not ({TRAIN, DEV, CALIB} & set(plan.fold_of(it))), \
                f"held-out {it.attack_id} leaked into training"


def test_both_wavefake_eval_folds_share_bona_fide_negatives():
    """They must differ ONLY in the vocoder, or their EER gap is not a
    controlled measurement of unseen-generator generalisation."""
    man = wavefake_like()
    plan = plan_wavefake(man)
    seen = plan.subset(man, EVAL_SEEN_VOCODER)
    unseen = plan.subset(man, EVAL_UNSEEN_VOCODER)
    bona_seen = {i.utt_id for i in seen if i.label == BONAFIDE}
    bona_unseen = {i.utt_id for i in unseen if i.label == BONAFIDE}
    assert bona_seen and bona_seen == bona_unseen
    assert {i.attack_id for i in seen if i.label == SPOOF}.isdisjoint(UNSEEN_VOCODERS)
    assert {i.attack_id for i in unseen if i.label == SPOOF} <= UNSEEN_VOCODERS


def test_wavefake_plan_passes_every_guard():
    man = wavefake_like()
    assert verify(man, plan_wavefake(man)) == []


# --------------------------------------- guards 3 & 7: speaker, calibration

def test_asvspoof_calibration_is_speaker_disjoint_from_dev():
    """calib and dev both come out of the official dev set. If they shared
    speakers, the operating point would be fit on the very voices that drove
    early stopping."""
    man = asvspoof_like()
    plan = plan_asvspoof(man)
    dev = {i.speaker_id for i in plan.subset(man, DEV)}
    cal = {i.speaker_id for i in plan.subset(man, CALIB)}
    assert dev and cal
    assert dev.isdisjoint(cal)


def test_asvspoof_speakers_disjoint_between_training_and_eval():
    man = asvspoof_like()
    plan = plan_asvspoof(man)
    ev = {i.speaker_id for i in plan.subset(man, EVAL_IN_DOMAIN)}
    for fold in (TRAIN, DEV, CALIB):
        assert ev.isdisjoint({i.speaker_id for i in plan.subset(man, fold)})


def test_asvspoof_eval_attacks_are_unseen():
    man = asvspoof_like()
    plan = plan_asvspoof(man)
    trained = {i.attack_id for i in plan.subset(man, TRAIN)} - {"-"}
    evaluated = {i.attack_id for i in plan.subset(man, EVAL_IN_DOMAIN)} - {"-"}
    assert trained and evaluated and trained.isdisjoint(evaluated)


def test_asvspoof_plan_passes_every_guard():
    man = asvspoof_like()
    assert verify(man, plan_asvspoof(man)) == []


def test_verify_catches_a_calibration_overlap():
    man = asvspoof_like()
    plan = plan_asvspoof(man)
    poisoned = dict(plan.assignment)
    first_train = next(i.utt_id for i in plan.subset(man, TRAIN))
    poisoned[first_train] = (TRAIN, CALIB)          # same item in both
    assert any(v.guard == "calibration_disjoint"
               for v in verify(man, SplitPlan(poisoned)))


def test_verify_catches_a_single_class_fold():
    """EER and calibration are undefined without both classes -- a fold that
    lost its negatives must fail loudly, not silently produce nan."""
    man = asvspoof_like()
    plan = plan_asvspoof(man)
    only_spoof = SplitPlan({
        i.utt_id: (TRAIN,) for i in man if i.label == SPOOF and TRAIN in plan.fold_of(i)
    })
    assert any(v.guard == "class_coverage" for v in verify(man, only_spoof))


# --------------------------------------------------------------- real data

real_data = pytest.mark.skipif(
    not (DATA / "asvspoof2019_la").exists(),
    reason="corpora live on ephemeral pod scratch; run scripts/fetch_datasets.py",
)


@real_data
def test_real_asvspoof_matches_the_official_protocol():
    from vericall.data.manifest import build

    man = build("asvspoof2019_la", DATA / "asvspoof2019_la")
    counts = {s: len(man.filter(orig_split=s)) for s in ("train", "dev", "eval")}
    # the published ASVspoof 2019 LA protocol
    assert counts == {"train": 25380, "dev": 24844, "eval": 71237}
    assert verify(man, plan_asvspoof(man)) == []


@real_data
def test_real_asvspoof_train_and_eval_attacks_do_not_overlap():
    from vericall.data.manifest import build

    man = build("asvspoof2019_la", DATA / "asvspoof2019_la")
    tr = {i.attack_id for i in man.filter(orig_split="train")} - {"-"}
    ev = {i.attack_id for i in man.filter(orig_split="eval")} - {"-"}
    assert tr == {"A01", "A02", "A03", "A04", "A05", "A06"}
    assert ev == {f"A{n:02d}" for n in range(7, 20)}
    assert tr.isdisjoint(ev)


# ------------------------------------------------------------ RTCFake

def rtcfake_like(n_pairs: int = 400) -> Manifest:
    """Mimics RTCFake: every utterance appears twice, offline and online.

    The pair is the leak. Both members are the same speech, the same speaker and
    the same label -- only the channel differs -- so a split that separates them
    is measuring codec sensitivity and reporting it as detection accuracy.
    """
    items, k = [], 0
    for split in ("train", "dev"):
        for lang in ("en", "zh"):
            for u in range(n_pairs):
                off = f"{split}/offline/{lang}/{split}_{u:06d}.wav"
                on = f"{split}/online/{lang}/{split}_{u + 10**5:06d}.wav"
                label = BONAFIDE if u % 4 == 0 else SPOOF
                for utt, transport in ((off, "offline"), (on, "online")):
                    items.append(Item(
                        dataset="rtcfake", utt_id=utt, source_utt_id=off,
                        speaker_id="RTCFake_unknown",
                        attack_id="-" if label == BONAFIDE else "RTC",
                        label=label, shard=f"/d/{utt}", row=0,
                        sample_rate=16000,
                        orig_split=f"{split}:{lang}:{transport}",
                    ))
                    k += 1
    return Manifest(items)


def test_rtcfake_offline_online_pair_never_spans_folds():
    man = rtcfake_like()
    plan = plan_rtcfake(man)
    folds_of_group = {}
    for it in man:
        for f in plan.fold_of(it):
            folds_of_group.setdefault(it.source_utt_id, set()).add(f)
    straddling = {g: f for g, f in folds_of_group.items() if len(f) > 1}
    assert not straddling, f"pair split across folds: {list(straddling)[:3]}"


def test_rtcfake_held_out_language_never_reaches_a_training_fold():
    """zh is dropped from training, not down-weighted.

    If any zh item reaches train/dev/calib then eval_rtc_zh stops being a
    cross-lingual test, and the one question this corpus was fetched to answer
    -- did retraining fix the Mandarin inversion -- becomes unanswerable.
    """
    man = rtcfake_like()
    plan = plan_rtcfake(man, train_langs=frozenset({"en"}))
    for it in man:
        folds = set(plan.fold_of(it))
        if ":zh:" in it.orig_split:
            assert not (folds & {TRAIN, DEV, CALIB}), f"{it.utt_id} leaked zh"


def test_rtcfake_eval_folds_are_single_language_and_from_the_corpus_dev_split():
    man = rtcfake_like()
    plan = plan_rtcfake(man)
    for it in man:
        folds = set(plan.fold_of(it))
        if EVAL_RTC_EN in folds or EVAL_RTC_ZH in folds:
            assert it.orig_split.startswith("dev:"), (
                f"{it.utt_id} in an eval fold but not from the corpus dev split")
        if EVAL_RTC_EN in folds:
            assert ":en:" in it.orig_split
        if EVAL_RTC_ZH in folds:
            assert ":zh:" in it.orig_split


def test_rtcfake_training_folds_come_only_from_the_corpus_train_split():
    man = rtcfake_like()
    plan = plan_rtcfake(man)
    for it in man:
        if set(plan.fold_of(it)) & {TRAIN, DEV, CALIB}:
            assert it.orig_split.startswith("train:"), (
                f"{it.utt_id} trained on but came from the corpus dev split")


def test_rtcfake_plan_passes_every_guard():
    man = rtcfake_like()
    assert verify(man, plan_rtcfake(man)) == []


def test_rtcfake_unknown_speakers_do_not_fake_a_disjointness_pass():
    """The speaker guard is EXEMPTED for RTCFake, not satisfied by it.

    Every item carries the same sentinel speaker id, so a guard that compared
    ids would either fire on every fold pair or -- if written the other way --
    report a disjointness this corpus cannot support. The exemption is the
    honest option, and this test pins the fact that it IS an exemption so the
    limitation stays visible in the test suite rather than only in a docstring.
    """
    man = rtcfake_like()
    assert {i.speaker_id for i in man} == {"RTCFake_unknown"}
    assert verify(man, plan_rtcfake(man)) == []


# ------------------------------------------------- EchoFake / CodecFake

def echofake_like(n=200):
    """Mimics EchoFake, including the two properties that broke the planner.

    Speakers are DISJOINT between the training splits and the eval split. The
    real corpus is like this -- its eval splits contain speakers held out of
    train -- and a fixture that reuses speakers everywhere makes the planner
    correctly drop every eval item, which looks like a planner bug and is not.
    """
    items = []
    for split in ("train", "dev", "closed_set_eval"):
        eval_side = "eval" in split
        for u in range(n):
            # The real corpus reuses utt_ids across splits -- 1,986 of them --
            # which is why the builder qualifies them by split.
            uid = f"EF_{u:05d}"
            spk = f"ev{u % 30}" if eval_side else f"tr{u % 40}"
            for kind, lab in (("-", BONAFIDE), ("replay:MacBook", SPOOF),
                              ("tts:XTTS-v2", SPOOF)):
                items.append(Item(
                    dataset="echofake", utt_id=f"{split}/{uid}_{kind[:3]}",
                    source_utt_id=f"cv_{split}_{u}_{kind[:3]}",
                    speaker_id=spk, attack_id=kind, label=lab,
                    shard="s.parquet", row=u, sample_rate=16000,
                    orig_split=split))
    return Manifest(items)


def test_duplicate_utt_ids_are_caught():
    """SplitPlan is keyed by utt_id, so two items sharing one id silently share
    a fold -- across labels. EchoFake shipped 1,986 such collisions and nothing
    raised until this guard existed."""
    items = [
        Item(dataset="x", utt_id="SAME", source_utt_id="a", speaker_id="s1",
             attack_id="-", label=BONAFIDE, shard="s", row=0,
             sample_rate=16000, orig_split="train"),
        Item(dataset="x", utt_id="SAME", source_utt_id="b", speaker_id="s2",
             attack_id="A01", label=SPOOF, shard="s", row=1,
             sample_rate=16000, orig_split="dev"),
    ]
    man = Manifest(items)
    plan = SplitPlan({"SAME": (TRAIN,)})
    guards = [x.guard for x in verify(man, plan)]
    assert "utt_id_unique" in guards


def test_echofake_replay_and_synth_folds_share_their_negatives():
    """The two folds differ only in attack type. Sharing the bona fide clips is
    what makes the gap between their EERs a measurement of 'does replay evade
    this detector' rather than two unrelated numbers."""
    man = echofake_like()
    plan = plan_echofake(man)
    rep = {i.utt_id for i in plan.subset(man, EVAL_REPLAY) if i.label == BONAFIDE}
    syn = {i.utt_id for i in plan.subset(man, EVAL_SYNTH_ECHO) if i.label == BONAFIDE}
    assert rep and rep == syn, "eval folds must share identical negatives"


def test_echofake_replay_fold_contains_only_replay_attacks():
    man = echofake_like()
    plan = plan_echofake(man)
    for it in plan.subset(man, EVAL_REPLAY):
        assert it.label == BONAFIDE or it.attack_id.startswith("replay:")
    for it in plan.subset(man, EVAL_SYNTH_ECHO):
        assert it.label == BONAFIDE or it.attack_id.startswith("tts:")


def test_echofake_plan_passes_every_guard():
    man = echofake_like()
    assert verify(man, plan_echofake(man)) == []


def codecfake_like(n_spk=40, n_utt=6):
    items = []
    for s in range(n_spk):
        for u in range(n_utt):
            src = f"p{225+s}_{u:03d}"
            for tag in ("R", "F01", "F02", "F03", "F04", "F05", "F06"):
                items.append(Item(
                    dataset="codecfake", utt_id=f"{src}__{tag}",
                    source_utt_id=src, speaker_id=f"p{225+s}", attack_id=tag,
                    label=BONAFIDE if tag == REAL_TAG else SPOOF,
                    shard="s.parquet", row=u, sample_rate=16000,
                    orig_split="all"))
    return Manifest(items)


def test_codecfake_is_speaker_disjoint():
    """VCTK speakers appear across the whole corpus, so an utterance-level
    split leaves every speaker on both sides -- which is what the first version
    of this planner did, and the guard reported all 64 shared."""
    man = codecfake_like()
    plan = plan_codecfake(man)
    tr = {i.speaker_id for i in plan.subset(man, TRAIN)}
    ev = {i.speaker_id for i in plan.subset(man, EVAL_UNSEEN_CODEC)}
    assert tr and ev and not (tr & ev)


def test_codecfake_held_out_codecs_never_reach_training():
    man = codecfake_like()
    plan = plan_codecfake(man)
    for fold in (TRAIN, DEV, CALIB):
        assert not ({i.attack_id for i in plan.subset(man, fold)}
                    & {"F05", "F06"})


def test_codecfake_plan_passes_every_guard():
    man = codecfake_like()
    assert verify(man, plan_codecfake(man)) == []
