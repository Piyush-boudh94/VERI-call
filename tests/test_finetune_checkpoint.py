"""The fine-tune trainer must save what moved, and only what moved.

Eight fine-tuning checkpoints were committed at 363.5 MB each -- 2.9 GB -- and
almost every byte was a copy of the pretrained WavLM encoder that
`torchaudio.pipelines` downloads on its own. The frozen control arm is the
clearest case: 363.5 MB to record that nothing in the encoder changed.

`model.state_dict()` does not know which parameters were trained, so these
tests pin the two properties that make the smaller checkpoint safe: it holds
every trainable parameter, and loading it with strict=False onto a fresh model
reproduces the original scores exactly.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _helpers():
    """Load the module-level helpers without running main()."""
    path = ROOT / "scripts" / "train_finetune.py"
    # __file__ because the script resolves its own path to extend sys.path.
    ns: dict = {"__file__": str(path), "__name__": "train_finetune"}
    exec(compile(path.read_text().split("def main(")[0], str(path), "exec"), ns)
    return ns


def test_trainable_state_exists():
    assert "trainable_state" in _helpers(), (
        "train_finetune.py must not go back to saving model.state_dict()")


def test_the_trainer_calls_it():
    src = (ROOT / "scripts" / "train_finetune.py").read_text()
    assert "trainable_state(model)" in src
    assert '"state_dict": model.state_dict()' not in src, (
        "the full state dict is 363.5 MB and almost all of it is pretrained")
    assert "strict=False" in src, (
        "a partial checkpoint cannot be loaded strictly")


@pytest.mark.parametrize("unfreeze", [0, 2])
def test_round_trip_is_exact(unfreeze):
    """Perturb everything trainable, save, reload onto a fresh model, compare.

    The perturbation matters: without it a checkpoint that saved nothing at
    all would still 'round-trip', because both models would hold the same
    pretrained weights.
    """
    h = _helpers()
    from vericall.branches.a_ssl import BranchA, BranchAConfig

    torch.manual_seed(0)
    a = BranchA(BranchAConfig()).eval()
    h["unfreeze_top"](a, unfreeze)
    with torch.no_grad():
        for p in a.parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * 0.01)

    x = torch.randn(2, 16000)
    with torch.inference_mode():
        want = a.score(x).clone()

    state = h["trainable_state"](a)
    b = BranchA(BranchAConfig()).eval()
    _, unexpected = b.load_state_dict(state, strict=False)
    assert not unexpected
    with torch.inference_mode():
        got = b.score(x)
    assert torch.allclose(want, got, atol=1e-6), (
        f"unfreeze={unfreeze}: reload changed the scores, so the checkpoint "
        f"is missing a parameter that moved")


def test_frozen_checkpoint_is_far_smaller_than_the_model():
    h = _helpers()
    from vericall.branches.a_ssl import BranchA, BranchAConfig

    m = BranchA(BranchAConfig())
    h["unfreeze_top"](m, 0)
    part = sum(v.numel() * v.element_size()
               for v in h["trainable_state"](m).values())
    full = sum(v.numel() * v.element_size() for v in m.state_dict().values())
    assert part < full / 20, (
        f"frozen checkpoint is {part/2**20:.1f} MB against a {full/2**20:.1f} "
        f"MB model -- the encoder is being saved again")
