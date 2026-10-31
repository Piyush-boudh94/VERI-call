"""The paired Hindi corpus, and the confound it exists to control.

The project recorded for weeks that no Hindi EER was computable: IndicSynth
ships no bona fide class, and borrowing bona fide from an unrelated corpus
would let a model separate the classes by recording conditions. Both true. What
was missed is that IndicSynth's filenames carry Kathbath speaker ids, so its
own source corpus supplies the bona fide side with the speakers already
matched. These tests pin the parsing that establishes that, and the codec
matching that keeps the resulting number honest.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from vericall.data.manifest import BONAFIDE, BUILDERS, SPOOF, build_hindi_paired
from vericall.data.splits import (
    CALIB, DEV, EVAL_FOLDS, EVAL_HINDI, FOLDS, TRAIN, plan_hindi_paired, verify,
)

av = pytest.importorskip("av", reason="the AAC arm needs PyAV")

_spec = importlib.util.spec_from_file_location(
    "bhp", Path(__file__).resolve().parents[1] / "scripts" / "build_indic_paired.py")
bhp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bhp)


# ------------------------------------------------- the pairing that unlocks it

@pytest.mark.parametrize("stem,spk", [
    ("xtts_v2____1059-m-149", "1059"),
    ("xtts_v2____3-f-2", "3"),
    ("vits____1191-f-7", "1191"),
])
def test_indicsynth_tts_filenames_yield_the_cloned_speaker(stem, spk):
    m = bhp.ITS_TTS_RE.search(stem)
    assert m and m.group(1) == spk


def test_voice_conversion_filenames_yield_the_TARGET_speaker():
    """src is whose words, tgt is whose VOICE. The clone impersonates the
    target, so the target is who it must be paired against."""
    m = bhp.ITS_VC_RE.search("freevc24____f____src_1087____tgt_1102___0")
    assert m and m.group(2) == "1102"


@pytest.mark.parametrize("stem,spk", [
    ("844424931537866-594-f", "594"),
    ("12345-3-m", "3"),
])
def test_kathbath_filenames_yield_the_speaker(stem, spk):
    m = bhp.KATHBATH_RE.match(stem)
    assert m and m.group(2) == spk


def test_a_kathbath_name_is_not_mistaken_for_an_indicsynth_one():
    """They are both <digits>-<...> shaped. A parser that confused them would
    silently pair the wrong speakers and the EER would be meaningless."""
    assert bhp.ITS_TTS_RE.search("844424931537866-594-f") is None
    assert bhp.KATHBATH_RE.match("xtts_v2____1059-m-149") is None


# ------------------------------------------------------- the codec confound

def test_the_aac_roundtrip_preserves_length_and_content():
    """Kathbath is AAC and IndicSynth is PCM. The spoof side gets one matching
    AAC generation so the codec stops being a class cue; if that round-trip
    mangled the audio it would be a different confound, not a fix."""
    t = np.arange(16_000 * 2) / 16_000
    x = (0.3 * np.sin(2 * np.pi * 180 * t)
         + 0.1 * np.sin(2 * np.pi * 540 * t)).astype(np.float32)
    y = bhp.aac_roundtrip(x, bitrate=64_000)
    assert len(y) == len(x)
    assert float(np.corrcoef(x, y)[0, 1]) > 0.9


def test_higher_aac_bitrate_is_closer_to_the_input():
    t = np.arange(16_000 * 2) / 16_000
    x = (0.3 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    lo = float(np.corrcoef(x, bhp.aac_roundtrip(x, bitrate=24_000))[0, 1])
    hi = float(np.corrcoef(x, bhp.aac_roundtrip(x, bitrate=128_000))[0, 1])
    assert lo < hi


def test_empty_input_does_not_raise():
    assert len(bhp.aac_roundtrip(np.zeros(0, dtype=np.float32))) == 0


def test_normalise_matches_loudness_without_clipping():
    """Level is trivially separable, and TTS output sits at a different level
    from read speech, so both classes are put at one RMS."""
    for scale in (0.001, 1.0, 20.0):
        y = bhp.normalise(torch.randn(16_000) * scale)
        assert float(y.pow(2).mean().sqrt()) == pytest.approx(
            bhp.TARGET_RMS, rel=0.05)
        assert float(y.abs().max()) <= 1.0


def test_silence_is_not_normalised_into_noise():
    z = torch.zeros(16_000)
    assert torch.allclose(bhp.normalise(z), z)


# ------------------------------------------------------- manifest and split

def _corpus(tmp_path, arm="matched", speakers=6, per=4):
    for sub in ("bonafide", "spoof"):
        d = tmp_path / arm / sub
        d.mkdir(parents=True)
        for spk in range(speakers):
            for j in range(per):
                # The recording id must vary, or the four files per speaker
                # overwrite each other and the fixture silently shrinks.
                stem = (f"{spk}_1234500{j}-{spk}-f" if sub == "bonafide"
                        else f"{spk}_xtts_v2____{spk}-f-{j}")
                sf.write(d / f"{stem}.wav", np.zeros(16_000, np.float32), 16_000)
    return tmp_path


def test_labels_come_from_the_directory_and_speaker_from_the_name(tmp_path):
    m = build_hindi_paired(_corpus(tmp_path))
    assert len(m) == 6 * 4 * 2
    assert {it.label for it in m} == {BONAFIDE, SPOOF}
    for it in m:
        assert it.speaker_id.startswith("hi_")
        assert (it.attack_id == "-") == (it.label == BONAFIDE)


def test_both_classes_carry_the_same_speakers(tmp_path):
    """The point of the corpus: one person's real voice against a clone of it.
    If the speaker sets diverged, the model could separate by voice."""
    m = build_hindi_paired(_corpus(tmp_path))
    bona = {it.speaker_id for it in m if it.label == BONAFIDE}
    spoof = {it.speaker_id for it in m if it.label == SPOOF}
    assert bona == spoof and len(bona) == 6


def test_the_arm_is_selectable_and_not_silently_defaulted(tmp_path):
    _corpus(tmp_path, arm="matched")
    _corpus(tmp_path, arm="unmatched")
    assert len(build_hindi_paired(tmp_path, arm="matched")) == 48
    assert len(build_hindi_paired(tmp_path, arm="unmatched")) == 48
    # A missing arm yields nothing rather than falling back to the other one:
    # quoting a matched number while serving unmatched data would be the worst
    # possible failure here.
    assert len(build_hindi_paired(tmp_path, arm="nonexistent")) == 0


def test_the_arm_is_recorded_in_every_utt_id(tmp_path):
    """So a cached fold can never be mistaken for the other arm's."""
    _corpus(tmp_path, arm="matched")
    for it in build_hindi_paired(tmp_path, arm="matched"):
        assert "/matched/" in it.utt_id


def test_the_split_is_speaker_disjoint(tmp_path):
    m = build_hindi_paired(_corpus(tmp_path, speakers=40))
    plan = plan_hindi_paired(m)
    seen: dict[str, set] = {}
    for it in m:
        seen.setdefault(it.speaker_id, set()).update(plan.assignment[it.utt_id])
    spanning = {k: v for k, v in seen.items() if len(v) > 1}
    assert not spanning, f"speakers on both sides of the boundary: {spanning}"


def test_a_speaker_keeps_its_fold_when_others_are_added(tmp_path):
    """Stable hashing, because a split that reshuffles between runs makes
    every number irreproducible."""
    small = build_hindi_paired(_corpus(tmp_path / "a", speakers=10))
    big = build_hindi_paired(_corpus(tmp_path / "b", speakers=30))
    fold = {}
    for m in (small, big):
        p = plan_hindi_paired(m)
        for it in m:
            spk = it.speaker_id
            f = p.assignment[it.utt_id][0]
            if spk in fold:
                assert fold[spk] == f, f"{spk} moved folds"
            fold[spk] = f


def test_it_has_a_training_fold_unlike_in_the_wild(tmp_path):
    """Deliberate asymmetry. In-the-Wild is the untouched instrument; Hindi is
    the deployment target the model has never been trained on."""
    m = build_hindi_paired(_corpus(tmp_path, speakers=40))
    folds = {f for v in plan_hindi_paired(m).assignment.values() for f in v}
    assert TRAIN in folds
    assert {DEV, CALIB, EVAL_HINDI} <= folds


def test_leakage_guards_pass(tmp_path):
    m = build_hindi_paired(_corpus(tmp_path, speakers=40))
    assert verify(m, plan_hindi_paired(m)) == []


def test_the_fold_is_declared():
    assert EVAL_HINDI in FOLDS and EVAL_HINDI in EVAL_FOLDS


def test_it_is_registered_for_every_language_and_both_arms():
    """Registered via functools.partial now, one entry per language per codec
    arm, so an identity check against the bare function no longer holds -- what
    matters is that calling the registered entry reaches the right builder with
    the right arm bound."""
    from vericall.data.manifest import INDIC_LANGUAGES

    for lang in INDIC_LANGUAGES:
        for suffix, arm in (("", "matched"), ("_unmatched", "unmatched")):
            name = f"{lang}_paired{suffix}"
            assert name in BUILDERS, name
            fn = BUILDERS[name]
            assert fn.func is build_hindi_paired, name
            assert fn.keywords["arm"] == arm, name
            assert fn.keywords["dataset"] == name, name
