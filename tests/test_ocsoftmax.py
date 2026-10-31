"""OC-Softmax (Zhang et al., IEEE SPL 2021).

The failure mode this guards against is a loss that trains smoothly and
optimises the wrong thing. Every test here pins a directional property -- which
class gets clustered, which way the score points -- because all of those are
silently reversible and none of them shows up as an error.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from vericall.train.ocsoftmax import OCSoftmax


def test_margins_must_be_ordered():
    """Reversed margins invert the objective into 'scatter bona fide, cluster
    spoofs'. It trains happily and yields a detector that is exactly backwards."""
    OCSoftmax(8, m_real=0.9, m_fake=0.2)
    with pytest.raises(ValueError, match="must exceed"):
        OCSoftmax(8, m_real=0.2, m_fake=0.9)
    with pytest.raises(ValueError):
        OCSoftmax(8, m_real=0.5, m_fake=0.5)


def test_score_is_higher_for_spoof():
    """Every other branch scores higher = more spoof. A flipped sign here gives
    an EER near 1-EER, which reads as a bad model rather than a reversed one."""
    torch.manual_seed(0)
    m = OCSoftmax(4)
    with torch.no_grad():
        m.center.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    aligned = torch.tensor([[1.0, 0.0, 0.0, 0.0]])        # like the centre
    opposed = torch.tensor([[-1.0, 0.0, 0.0, 0.0]])
    _, s_real = m(aligned, torch.tensor([0]))
    _, s_fake = m(opposed, torch.tensor([1]))
    assert float(s_fake) > float(s_real)


def test_loss_is_low_when_the_classes_sit_where_they_should():
    torch.manual_seed(0)
    m = OCSoftmax(4, alpha=20.0)
    with torch.no_grad():
        m.center.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    x = torch.tensor([[1.0, 0.0, 0.0, 0.0],      # bona fide, on the centre
                      [-1.0, 0.0, 0.0, 0.0]])    # spoof, far from it
    y = torch.tensor([0, 1])
    good, _ = m(x, y)
    bad, _ = m(x, torch.tensor([1, 0]))          # labels swapped
    assert float(good) < float(bad)
    assert float(good) < 0.1


def test_training_clusters_bona_fide_and_excludes_spoof():
    """The end-to-end property: after optimisation, bona fide should sit inside
    the target margin and spoof outside the exclusion margin."""
    torch.manual_seed(0)
    feat = 6
    m = OCSoftmax(feat, alpha=20.0, m_real=0.9, m_fake=0.2)
    # A learnable projection so the geometry has somewhere to move.
    proj = torch.nn.Linear(feat, feat)
    opt = torch.optim.Adam(list(m.parameters()) + list(proj.parameters()), lr=0.05)
    g = torch.Generator().manual_seed(1)

    def batch():
        y = torch.randint(0, 2, (64,), generator=g)
        x = torch.randn(64, feat, generator=g)
        x[:, 0] += (1 - y).float() * 4.0        # bona fide separable in dim 0
        return x, y

    for _ in range(400):
        x, y = batch()
        opt.zero_grad()
        loss, _ = m(proj(x), y)
        loss.backward()
        opt.step()

    x, y = batch()
    with torch.no_grad():
        cos = m.cosine(proj(x)).squeeze(1)
    assert cos[y == 0].mean() > cos[y == 1].mean() + 0.5, "classes not separated"
    assert float(cos[y == 0].mean()) > 0.5, "bona fide not clustered near centre"


def test_to_logits_agrees_with_the_score_ordering():
    """The conversion must not flip anything. A cosine consumed as a logit would
    not crash and would silently rescale every threshold in the system."""
    torch.manual_seed(0)
    m = OCSoftmax(4)
    x = torch.randn(32, 4)
    with torch.no_grad():
        _, score = m(x, torch.zeros(32, dtype=torch.long))
        p_spoof = torch.softmax(m.to_logits(x), dim=1)[:, 1]
    # score and P(spoof) must rank identically
    assert torch.equal(score.argsort(), p_spoof.argsort())


def test_logits_are_two_class_in_the_project_convention():
    m = OCSoftmax(5)
    out = m.to_logits(torch.randn(7, 5))
    assert out.shape == (7, 2)


def test_gradients_reach_the_centre():
    """The centre is the thing being learned. If it were detached the loss would
    still fall -- the projection would do all the work -- and the one-class
    geometry would be decorative."""
    m = OCSoftmax(4)
    loss, _ = m(torch.randn(8, 4), torch.randint(0, 2, (8,)))
    loss.backward()
    assert m.center.grad is not None and m.center.grad.abs().sum() > 0
