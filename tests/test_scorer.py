"""The live scoring path.

`serving/scorer.py` had no test, which is the worst of the five gaps: it is
what every real call goes through, it holds the fp16 autocast decision, and it
carries the forward/calibration split that exists so a raw forward can be
batched across calls while each call keeps its own channel mapping. A bug here
does not crash -- it produces confident numbers on the wrong scale.
"""
from __future__ import annotations

import math

import pytest
import torch

from vericall.audio.chunker import Window
from vericall.fusion.calibration import PlattCalibration
from vericall.serving.scorer import BranchCalibrations, BranchScorer


class FakeA(torch.nn.Module):
    """Returns a fixed posterior per item, so calibration is testable without
    a 94 M-parameter frontend."""

    def __init__(self, posteriors):
        super().__init__()
        self.posteriors = posteriors
        self.calls = 0

    def score(self, x):
        self.calls += 1
        n = x.shape[0]
        p = (self.posteriors * ((n + len(self.posteriors) - 1)
                                // len(self.posteriors)))[:n]
        return torch.tensor(p, dtype=torch.float32)


def win(call_id="c", t_end=1.0, lane="fast", quality=1.0, n=16_000):
    return Window(call_id=call_id, lane=lane, samples=torch.zeros(n),
                  t_start=max(0.0, t_end - 1.0), t_end=t_end,
                  quality=quality, hop_s=0.5)


def platt(a=1.0, b=0.0):
    return PlattCalibration(a=a, b=b, branch="A", channel_bucket="clean",
                            calib_id="A@clean", n_fit=10, cllr=0.1,
                            min_cllr=0.09)


# ------------------------------------------------------------ no windows

def test_no_windows_means_no_events_and_no_forward():
    a = FakeA([0.5])
    sc = BranchScorer(a, None, None, BranchCalibrations(), "cpu")
    assert sc.score_fast([], None) == []
    assert a.calls == 0, "an empty batch must not reach the GPU"


# ---------------------------------------------------------- calibration

def test_the_platt_mapping_is_actually_applied():
    """Without this, an uncalibrated posterior reaches the fusion engine as
    though it were a log-likelihood ratio, and every threshold in the system
    assumes LLRs."""
    cal = BranchCalibrations()
    cal.A = platt(a=2.0, b=-1.0)
    sc = BranchScorer(FakeA([0.75]), None, None, cal, "cpu")
    ev = [e for e in sc.score_fast([win()], None) if e.branch == "A"]
    assert len(ev) == 1
    want = 2.0 * math.log(0.75 / 0.25) - 1.0
    assert ev[0].llr == pytest.approx(want, rel=1e-4)


def test_a_per_call_calibration_overrides_the_startup_one():
    """One BranchScorer serves every concurrent call, so the channel mapping
    has to arrive as an argument. If it were scorer state, one caller's
    channel would rescale another caller's LLRs."""
    startup = BranchCalibrations(); startup.A = platt(a=1.0, b=0.0)
    percall = BranchCalibrations(); percall.A = platt(a=1.0, b=+5.0)
    sc = BranchScorer(FakeA([0.6]), None, None, startup, "cpu")
    base = [e for e in sc.score_fast([win()], None) if e.branch == "A"][0].llr
    over = [e for e in sc.score_fast([win()], percall) if e.branch == "A"][0].llr
    assert over == pytest.approx(base + 5.0, rel=1e-4)
    # and the override must not stick
    again = [e for e in sc.score_fast([win()], None) if e.branch == "A"][0].llr
    assert again == pytest.approx(base, rel=1e-4)


def test_an_unfitted_calibration_is_reported_in_calib_id():
    """The fusion engine and /health both key off this, and 'identity' is how
    a downstream reader learns the numbers are raw logits."""
    sc = BranchScorer(FakeA([0.6]), None, None, BranchCalibrations(), "cpu")
    ev = sc.score_fast([win()], None)
    assert all(e.calib_id == "identity" for e in ev)

    cal = BranchCalibrations(); cal.A = platt()
    sc2 = BranchScorer(FakeA([0.6]), None, None, cal, "cpu")
    assert all(e.calib_id == "fitted" for e in sc2.score_fast([win()], None))


# ------------------------------------------------- the batching contract

def test_forward_fast_and_events_from_raw_agree_with_score_fast():
    """The split exists so the raw forward can be batched ACROSS calls while
    each call applies its own channel mapping. If the two paths disagreed, the
    gateway would score differently depending on load -- which is the least
    debuggable failure this service could have.
    """
    cal = BranchCalibrations(); cal.A = platt(a=1.7, b=-0.4)
    sc = BranchScorer(FakeA([0.2, 0.9]), None, None, cal, "cpu")
    ws = [win(t_end=1.0), win(t_end=1.5)]

    direct = {(e.branch, e.t_end): e.llr for e in sc.score_fast(ws, None)}
    raw = sc.forward_fast(ws)
    split = {(e.branch, e.t_end): e.llr
             for e in sc.events_from_raw(ws, raw, None)}
    assert direct.keys() == split.keys()
    for k in direct:
        assert direct[k] == pytest.approx(split[k], rel=1e-5), k


def test_quality_is_carried_onto_the_event():
    """quality_discount() zeroes a window below 0.15, so losing this field
    silently re-enables scoring on audio the chunker judged unusable."""
    cal = BranchCalibrations(); cal.A = platt()
    sc = BranchScorer(FakeA([0.6]), None, None, cal, "cpu")
    ev = sc.score_fast([win(quality=0.3)], None)
    assert all(e.quality == pytest.approx(0.3) for e in ev)


def test_window_timestamps_survive_scoring():
    cal = BranchCalibrations(); cal.A = platt()
    sc = BranchScorer(FakeA([0.6, 0.6]), None, None, cal, "cpu")
    ev = [e for e in sc.score_fast([win(t_end=3.0), win(t_end=3.5)], None)
          if e.branch == "A"]
    assert sorted(e.t_end for e in ev) == [3.0, 3.5]


# ----------------------------------------------------------- precision

def test_cpu_never_uses_fp16_autocast():
    """CPU has no meaningful fp16 path and autocast there is a pessimisation,
    so the scorer forces fp32 regardless of VERICALL_PRECISION."""
    sc = BranchScorer(FakeA([0.5]), None, None, BranchCalibrations(), "cpu")
    assert sc.precision == "fp32"


def test_stats_count_what_was_actually_scored():
    cal = BranchCalibrations(); cal.A = platt()
    sc = BranchScorer(FakeA([0.6]), None, None, cal, "cpu")
    sc.score_fast([win(t_end=1.0), win(t_end=1.5)], None)
    assert sc.stats.windows >= 2
    assert sc.stats.batches >= 1
