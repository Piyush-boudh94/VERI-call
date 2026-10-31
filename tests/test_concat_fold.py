"""ConcatFold: several extracted caches presented as one training fold.

The risky part is _LazyConcat. It fancy-indexes across memmaps instead of
concatenating them, because the alternative pulls ~12 GB of fp16 features into
a 32 GiB shared pod. A mapping bug there would not crash -- it would silently
pair row i's features with row j's label, and training would just be worse for
no visible reason. So the tests below check the pairing, not just the shapes.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from vericall.data.cache import CachedFold, ConcatFold, _LazyConcat


def make_fold(tmp_path, name, n, offset, n_frames=4, dim=3):
    """A cache whose feature values encode their own row index.

    That is what makes a mis-mapping detectable: feat[i] == offset + i, so any
    row that ends up beside the wrong label shows up as an arithmetic mismatch
    rather than as plausible-looking noise.
    """
    root = tmp_path / name
    root.mkdir(parents=True)
    feat = np.full((n, n_frames, dim), 0, dtype=np.float16)
    wave = np.full((n, 16), 0, dtype=np.float16)
    for i in range(n):
        feat[i] = offset + i
        wave[i] = offset + i
    np.save(root / "feat.npy", feat)
    np.save(root / "wave.npy", wave)
    np.save(root / "label.npy", np.array([(offset + i) % 2 for i in range(n)]))
    np.save(root / "codec.npy", np.zeros(n, dtype=np.int64))
    (root / "meta.jsonl").write_text("\n".join(
        json.dumps({"utt_id": f"u{i}", "attack_id": "-", "speaker_id": "s",
                    "source_utt_id": f"u{i}", "channel": "clean"})
        for i in range(n)))
    (root / "info.json").write_text(json.dumps(
        {"rows": n, "window_s": 1.0, "channels": ["clean"]}))
    return CachedFold.load(root)


def test_lazy_concat_returns_the_same_rows_as_a_real_concatenation(tmp_path):
    a = make_fold(tmp_path, "a", 7, 100)
    b = make_fold(tmp_path, "b", 5, 200)
    lazy = _LazyConcat([a.feat, b.feat])
    eager = np.concatenate([np.asarray(a.feat), np.asarray(b.feat)])

    assert lazy.shape == eager.shape
    idx = np.array([0, 3, 6, 7, 8, 11])
    np.testing.assert_array_equal(lazy[idx], eager[idx])
    # every single row, including the seam
    for i in range(len(eager)):
        np.testing.assert_array_equal(lazy[np.array([i])][0], eager[i])


def test_lazy_concat_preserves_the_caller_s_index_order(tmp_path):
    """The batcher sorts indices for mmap locality, but nothing guarantees a
    caller will. Returning rows in part order while the labels stay in the
    caller's order is exactly the silent mis-pairing this class could cause."""
    a = make_fold(tmp_path, "a", 4, 100)
    b = make_fold(tmp_path, "b", 4, 200)
    lazy = _LazyConcat([a.feat, b.feat])
    idx = np.array([6, 1, 5, 0])           # interleaved and unsorted
    got = lazy[idx]
    for k, i in enumerate(idx):
        expected = 100 + i if i < 4 else 200 + (i - 4)
        assert float(got[k].flat[0]) == expected, f"row {i} landed at slot {k}"


def test_concat_fold_keeps_labels_aligned_with_features(tmp_path):
    a = make_fold(tmp_path, "a", 9, 100)
    b = make_fold(tmp_path, "b", 6, 200)
    cf = ConcatFold([a, b], ["a", "b"])

    assert len(cf) == 15
    for i in range(len(cf)):
        val = float(cf.feat[np.array([i])][0].flat[0])
        assert cf.label[i] == int(val) % 2, f"label/feature mismatch at row {i}"


def test_concat_fold_qualifies_group_ids_by_corpus(tmp_path):
    """Both corpora here use utt ids u0..u3. Unqualified, the bootstrap would
    resample two unrelated clips as one unit and report a narrower interval."""
    a = make_fold(tmp_path, "a", 4, 100)
    b = make_fold(tmp_path, "b", 4, 200)
    cf = ConcatFold([a, b], ["asvspoof2019_la", "rtcfake"])
    assert len({m["utt_id"] for m in cf.meta}) == 4       # the collision
    assert len(set(cf.groups)) == 8                        # resolved
    assert "asvspoof2019_la:u0" in set(cf.groups)
    assert "rtcfake:u0" in set(cf.groups)


def test_concat_fold_batches_cover_every_row_exactly_once(tmp_path):
    a = make_fold(tmp_path, "a", 10, 100)
    b = make_fold(tmp_path, "b", 7, 200)
    cf = ConcatFold([a, b], ["a", "b"])
    seen = []
    for x, y, c, sel in cf.batches(4, seed=3, use="feat"):
        assert len(x) == len(y) == len(sel)
        for k in range(len(sel)):
            # the pairing must survive shuffling and batching
            assert int(y[k]) == int(float(x[k].flatten()[0])) % 2
        seen.extend(sel.tolist())
    assert sorted(seen) == list(range(17))


def test_concat_fold_refuses_mismatched_row_shapes(tmp_path):
    """Two caches extracted with different windows cannot be concatenated. The
    silent alternative -- padding or truncating -- would make the model's input
    depend on which corpus a row came from."""
    a = make_fold(tmp_path, "a", 4, 100, n_frames=4)
    b = make_fold(tmp_path, "b", 4, 200, n_frames=8)
    with pytest.raises(ValueError, match="re-extract"):
        ConcatFold([a, b], ["a", "b"])


def test_concat_fold_records_which_corpus_each_row_came_from(tmp_path):
    a = make_fold(tmp_path, "a", 5, 100)
    b = make_fold(tmp_path, "b", 3, 200)
    cf = ConcatFold([a, b], ["asvspoof2019_la", "rtcfake"])
    assert list(cf.sources[:5]) == ["asvspoof2019_la"] * 5
    assert list(cf.sources[5:]) == ["rtcfake"] * 3
    assert cf.info["rows"] == [5, 3]


def test_cached_fold_accepts_the_same_arguments_as_concat_fold(tmp_path):
    """A caller must be able to pass one set of arguments to either.

    `balanced` was added to ConcatFold only, so the layer sweep -- which trains
    on a single corpus and therefore gets a CachedFold -- died with a TypeError
    AFTER the entire feature extraction had completed. The interfaces have to
    match or every caller needs an isinstance branch.
    """
    a = make_fold(tmp_path, "a", 8, 100)
    seen = []
    for x, y, c, sel in a.batches(4, use="feat", balanced=True):
        seen.extend(sel.tolist())
    assert sorted(seen) == list(range(8))
    # domain_ids too: one corpus is one domain
    assert a.domain_ids(np.arange(5)).tolist() == [0] * 5
