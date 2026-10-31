"""In-the-Wild: the one corpus this project never trains on.

Every cross-corpus number in docs/RESULTS.md so far was measured on a corpus
that shares a lineage with something the model saw. These tests pin the
properties that keep In-the-Wild a genuinely held-out instrument, because the
way that gets lost is not a crash -- it is a planner quietly emitting a train
fold one refactor later.
"""
from __future__ import annotations

import csv

import pytest

from vericall.data.manifest import BONAFIDE, BUILDERS, SPOOF, build_in_the_wild
from vericall.data.splits import (
    EVAL_FOLDS, EVAL_IN_THE_WILD, FOLDS, TRAINING_FOLDS, plan_in_the_wild,
    verify,
)

ROWS = [
    ("0.wav", "Alec Guinness", "spoof"),
    ("1.wav", "Alec Guinness", "bona-fide"),
    ("2.wav", "Barack Obama", "bona-fide"),
    ("3.wav", "Barack Obama", "spoof"),
]


def make_corpus(tmp_path, rows=ROWS, header=("file", "speaker", "label"),
                nested=False):
    base = tmp_path / "release_in_the_wild" if nested else tmp_path
    base.mkdir(parents=True, exist_ok=True)
    with (base / "meta.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    for name, *_ in rows:
        (base / name).write_bytes(b"")
    return tmp_path


# ------------------------------------------------------------------ build

def test_labels_map_the_right_way_round(tmp_path):
    m = build_in_the_wild(make_corpus(tmp_path))
    by = {it.utt_id.split("/")[-1]: it for it in m}
    assert by["1.wav"].label == BONAFIDE
    assert by["0.wav"].label == SPOOF
    assert by["1.wav"].attack_id == "-"
    assert by["0.wav"].attack_id == "ITW"


def test_an_unrecognised_label_is_never_silently_bona_fide(tmp_path):
    """The dangerous direction. A mirror that wrote "fake" instead of "spoof"
    must not turn 11,816 attacks into the negative class and hand back a
    spectacular EER."""
    m = build_in_the_wild(make_corpus(tmp_path, [("0.wav", "X", "fake")]))
    assert list(m)[0].label == SPOOF


@pytest.mark.parametrize("word", ["bona-fide", "bonafide", "real", "genuine"])
def test_the_bona_fide_spellings_mirrors_use(tmp_path, word):
    m = build_in_the_wild(make_corpus(tmp_path, [("0.wav", "X", word)]))
    assert list(m)[0].label == BONAFIDE


def test_it_finds_the_nested_kaggle_layout(tmp_path):
    """The Kaggle mirror nests one directory deep; the origin zip does not."""
    m = build_in_the_wild(make_corpus(tmp_path, nested=True))
    assert len(m) == len(ROWS)
    assert all(it.shard.endswith(".wav") for it in m)


def test_a_missing_metadata_file_yields_nothing_rather_than_raising(tmp_path):
    assert len(build_in_the_wild(tmp_path)) == 0


def test_shard_addressing_is_one_audio_file_per_shard(tmp_path):
    """Readers dispatch on the .parquet suffix, so a file-addressed corpus
    must not look like a parquet one."""
    m = build_in_the_wild(make_corpus(tmp_path))
    for it in m:
        assert it.row == 0
        assert not it.shard.endswith(".parquet")


def test_utt_ids_are_unique(tmp_path):
    m = build_in_the_wild(make_corpus(tmp_path))
    ids = [it.utt_id for it in m]
    assert len(ids) == len(set(ids))


def test_it_is_registered(tmp_path):
    assert BUILDERS["in_the_wild"] is build_in_the_wild


# ------------------------------------------------------------------- plan

def test_everything_lands_in_one_eval_fold(tmp_path):
    m = build_in_the_wild(make_corpus(tmp_path))
    plan = plan_in_the_wild(m)
    assert {f for folds in plan.assignment.values() for f in folds} == {
        EVAL_IN_THE_WILD}
    assert len(plan.assignment) == len(m)


def test_there_is_no_training_fold_and_that_is_the_point(tmp_path):
    """If In-the-Wild ever appeared in TRAIN, the number it produces would
    stop meaning anything and there is nothing left to replace it with."""
    m = build_in_the_wild(make_corpus(tmp_path))
    plan = plan_in_the_wild(m)
    for folds in plan.assignment.values():
        for f in folds:
            assert f not in TRAINING_FOLDS


def test_speakers_deliberately_span_both_classes(tmp_path):
    """The spoofs are clones of the same public figures whose real recordings
    form the negative class. A detector therefore cannot separate the classes
    by speaker or recording provenance -- only by synthesis. Holding speakers
    out would destroy exactly the property the corpus exists to provide."""
    m = build_in_the_wild(make_corpus(tmp_path))
    bona = {it.speaker_id for it in m if it.label == BONAFIDE}
    spoof = {it.speaker_id for it in m if it.label == SPOOF}
    assert bona & spoof, "the overlap is the design, not a leak"


def test_the_leakage_guards_pass(tmp_path):
    """Trivially, and correctly: with no train fold there is no train/eval
    boundary for an utterance or a speaker to cross."""
    m = build_in_the_wild(make_corpus(tmp_path))
    assert verify(m, plan_in_the_wild(m)) == []


def test_the_fold_is_declared_as_an_eval_fold():
    assert EVAL_IN_THE_WILD in FOLDS
    assert EVAL_IN_THE_WILD in EVAL_FOLDS
    assert EVAL_IN_THE_WILD not in TRAINING_FOLDS
