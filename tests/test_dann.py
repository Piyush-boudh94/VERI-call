"""Domain-adversarial training pieces.

These exist because a gradient reversal layer that quietly fails is
indistinguishable from one that works: training still converges, the loss still
falls, and the only symptom is that cross-corpus EER does not improve. The sign
of a gradient is not something to take on faith.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vericall.train.dann import DomainHead, dann_lambda, grad_reverse


def test_gradient_is_negated_and_scaled():
    x = torch.ones(3, 4, requires_grad=True)
    grad_reverse(x, 0.5).sum().backward()
    # d(sum)/dx would be +1 without reversal
    assert torch.allclose(x.grad, torch.full_like(x, -0.5))


def test_forward_is_an_identity():
    """The adversary must see the real representation; only the backward pass
    is unusual."""
    x = torch.randn(5, 7)
    assert torch.equal(grad_reverse(x, 0.9), x)


def test_lambda_zero_makes_the_layer_a_no_op_in_backward():
    """lambda=0 is the start of the schedule. If it leaked any gradient the
    spoof head would be fighting the adversary before it had learned anything."""
    x = torch.ones(2, 3, requires_grad=True)
    grad_reverse(x, 0.0).sum().backward()
    assert torch.allclose(x.grad, torch.zeros_like(x))


def test_lambda_ramps_from_zero_to_max():
    assert dann_lambda(0, 100) == 0.0
    mid = dann_lambda(50, 100)
    assert 0.0 < mid < 1.0
    assert dann_lambda(100, 100) == pytest.approx(1.0, abs=1e-3)
    assert dann_lambda(10, 100) < dann_lambda(60, 100)      # monotone


def test_lambda_respects_its_ceiling():
    assert dann_lambda(100, 100, max_lambda=0.3) == pytest.approx(0.3, abs=1e-3)


def test_lambda_is_safe_when_total_steps_is_zero():
    """A one-batch run would otherwise divide by zero on the first step."""
    assert dann_lambda(0, 0) == 1.0


def test_reversal_makes_the_domain_harder_to_predict_than_no_reversal():
    """The end-to-end property, as a COMPARISON rather than an absolute.

    An earlier version of this test asserted that domain loss rises during
    adversarial training. It does not reliably: with a two-layer adversary, a
    one-layer encoder and no competing objective, the adversary simply wins the
    minimax game outright (measured: domain loss fell 0.75 -> 0.01). That is a
    real property of adversarial balance, not a bug in the reversal -- and it is
    the reason DomainHead is deliberately small and lambda is ramped.

    What IS robustly true, and what actually matters, is the relative claim:
    holding everything else fixed, training the encoder through the reversal
    leaves the domain less predictable than training it without.
    """
    def train(lambd: float) -> float:
        torch.manual_seed(0)
        enc = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.GELU(),
                                  torch.nn.Linear(8, 6))
        dom = DomainHead(6, 2, hidden=16)
        opt = torch.optim.Adam(list(enc.parameters()) + list(dom.parameters()),
                               lr=0.02)
        lossfn = torch.nn.CrossEntropyLoss()
        g = torch.Generator().manual_seed(1)

        def batch():
            d = torch.randint(0, 2, (64,), generator=g)
            x = torch.randn(64, 4, generator=g)
            x[:, 0] += d.float() * 6.0      # a blatant domain marker
            return x, d

        for _ in range(400):
            x, d = batch()
            opt.zero_grad()
            lossfn(dom(enc(x), lambd), d).backward()
            opt.step()
        # Freeze and re-probe with a FRESH head: how much domain information
        # survives in the representation, independent of this head's state.
        probe = DomainHead(6, 2, hidden=16)
        popt = torch.optim.Adam(probe.parameters(), lr=0.02)
        for _ in range(200):
            x, d = batch()
            popt.zero_grad()
            with torch.no_grad():
                h = enc(x)
            lossfn(probe(h, 0.0), d).backward()
            popt.step()
        x, d = batch()
        with torch.no_grad():
            return lossfn(probe(enc(x), 0.0), d).item()

    with_rev = train(1.0)
    without = train(0.0)
    assert with_rev > without, (
        f"reversal did not reduce domain predictability "
        f"(with={with_rev:.3f} vs without={without:.3f})")


def test_domain_head_shape_and_domains():
    h = DomainHead(12, 3)
    assert h(torch.randn(9, 12), 0.7).shape == (9, 3)
