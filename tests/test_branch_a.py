"""Branch A structural tests. Skipped where torchaudio bundles are unavailable."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")

from vericall.branches.a_ssl import BranchA, BranchAConfig, BranchAHead

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def model():
    return BranchA(BranchAConfig()).to(DEV).eval()


def test_frontend_is_frozen_and_head_is_not(model):
    """Freezing the frontend is the largest single overfitting control here and
    is also what makes this fit a 16 GB MIG slice."""
    assert model.n_trainable > 0
    assert model.n_frozen > 50e6
    assert model.n_frozen / model.n_trainable > 50
    assert not any(p.requires_grad for p in model.frontend.parameters())
    assert all(p.requires_grad for p in model.head.parameters())


def test_frontend_stays_in_eval_even_when_train_is_called(model):
    model.frontend.train(True)
    assert not model.frontend.model.training


def test_truncation_actually_stops_at_the_configured_layer(model):
    feats, _ = model.frontend.model.extract_features(
        torch.randn(1, 16000, device=DEV), num_layers=model.frontend.layer
    )
    assert len(feats) == model.cfg.layer == 5


@pytest.mark.parametrize("seconds", [1.0, 3.0])
def test_head_is_invariant_to_window_length(model, seconds):
    """The slow lane hands it 3 s windows and the fast lane 1 s ones, so a
    fixed-width second view is required -- an earlier design keyed attention
    width to the frame count and simply would not run at two lengths."""
    x = torch.randn(2, int(seconds * 16000), device=DEV)
    with torch.inference_mode():
        s = model.score(x)
    assert s.shape == (2,)
    assert torch.isfinite(s).all()
    assert ((s >= 0) & (s <= 1)).all()


def test_scores_are_deterministic_in_eval(model):
    x = torch.randn(3, 16000, device=DEV)
    with torch.inference_mode():
        a, b = model.score(x), model.score(x)
    assert torch.allclose(a, b, atol=1e-6)


def test_head_accepts_arbitrary_frame_counts():
    head = BranchAHead(768, BranchAConfig()).to(DEV).eval()
    for frames in (25, 49, 149, 300):
        with torch.inference_mode():
            out = head(torch.randn(2, frames, 768, device=DEV))
        assert out.shape == (2, 2)
