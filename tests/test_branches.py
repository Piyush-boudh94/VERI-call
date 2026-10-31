"""Branch A2 (phase) and Branch B (prosody).

Both were untested. A2 matters because it is the second weighted branch in
every profile and the one that behaves worst out of domain -- 60.36% INVERTED
on Hindi before retraining, destroyed by Opus (1.77% -> 18.07%, section 23),
at chance on In-the-Wild. B matters for the opposite reason: it is trained on
nothing and zero-weighted everywhere, and the tests here pin the structural
property that made it collapse in the first place so a future edit cannot undo
the fix silently.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from vericall.branches.a2_phase import BranchA2, BranchA2Config
from vericall.branches.b_prosody import BranchB, BranchBConfig

SR = 16_000


def speech(n=SR, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(n, dtype=torch.float32) / SR
    x = sum(torch.sin(2 * np.pi * 140 * k * t) / k for k in range(1, 9))
    return (x * 0.5 + 0.02 * torch.randn(n, generator=g)).float()


# --------------------------------------------------------------- Branch A2

@pytest.fixture(scope="module")
def a2():
    return BranchA2(BranchA2Config()).eval()


def test_a2_consumes_raw_waveform_and_returns_two_logits(a2):
    out = a2(speech().unsqueeze(0))
    assert out.shape == (1, 2)
    assert torch.isfinite(out).all()


def test_a2_accepts_a_bare_1d_waveform(a2):
    """The serving path hands it a single window; a silent shape error here
    would be a crash mid-call."""
    assert a2(speech()).shape == (1, 2)


def test_a2_is_batch_independent(a2):
    """One BranchA2 serves every concurrent call. If a row's score depended on
    its neighbours, verdicts would change with load -- the least debuggable
    failure this service can have."""
    a, b = speech(seed=1), speech(seed=2)
    alone = a2(a.unsqueeze(0))
    together = a2(torch.stack([a, b]))[:1]
    assert torch.allclose(alone, together, atol=1e-4)


def test_a2_front_end_has_no_parameters(a2):
    """The phase front end is deterministic DSP, not learned. If it ever
    acquired parameters they would sit outside the optimiser and drift."""
    n = sum(p.numel() for p in a2.features.parameters())
    assert n == 0, f"phase front end has {n} parameters"


def test_a2_is_deterministic_in_eval(a2):
    x = speech().unsqueeze(0)
    assert torch.allclose(a2(x), a2(x))


def test_a2_dropout_is_actually_off_in_eval():
    """Dropout left on at serve time makes the same call score differently on
    every chunk, which reads as instability rather than as a bug."""
    m = BranchA2(BranchA2Config()).eval()
    assert all(not mod.training for mod in m.modules())


def test_a2_handles_a_short_window_without_raising(a2):
    """quality_discount zeroes very short windows, but the branch must survive
    seeing one rather than throwing inside a live call."""
    assert torch.isfinite(a2(speech(n=SR // 4).unsqueeze(0))).all()


# ---------------------------------------------------------------- Branch B

def test_b_encoder_is_bias_free():
    """Deep-SVDD's documented requirement, and not a style choice: any constant
    path lets the encoder map every input to the centre, driving the one-class
    loss to zero while learning nothing. This project HIT that collapse --
    embedding std 0.0057, dev EER 49.8% -- so the structural guard is pinned."""
    m = BranchB(BranchBConfig())
    with_bias = [n for n, p in m.named_parameters()
                 if n.endswith(".bias") and "encoder" in n]
    assert not with_bias, f"bias terms reintroduce the collapse path: {with_bias}"


def test_b_normalisation_layers_are_not_affine():
    """Same reason: an affine BatchNorm has a learnable shift, which is a
    constant path by another name."""
    m = BranchB(BranchBConfig())
    affine = [n for n, mod in m.named_modules()
              if isinstance(mod, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))
              and mod.affine]
    assert not affine, f"affine norm layers: {affine}"


def test_b_score_is_a_scalar_per_item():
    m = BranchB(BranchBConfig()).eval()
    s = m.score(torch.stack([speech(seed=3), speech(seed=4)]))
    assert s.shape == (2,) and torch.isfinite(s).all()


def test_b_accepts_an_optional_codec_id():
    """B carries a codec embedding, and the serving path passes a channel id.
    Both call shapes must work or the slow lane breaks on channel detection."""
    m = BranchB(BranchBConfig()).eval()
    x = speech().unsqueeze(0)
    a = m.score(x)
    b = m.score(x, torch.zeros(1, dtype=torch.long))
    assert a.shape == b.shape == (1,)


def test_b_score_is_batch_independent():
    m = BranchB(BranchBConfig()).eval()
    a, b = speech(seed=5), speech(seed=6)
    alone = m.score(a.unsqueeze(0))
    together = m.score(torch.stack([a, b]))[:1]
    assert torch.allclose(alone, together, atol=1e-4)
