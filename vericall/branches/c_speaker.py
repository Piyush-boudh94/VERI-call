"""Branch C -- speaker identity, and a free speaker-turn detector.

**This branch answers "is this the enrolled person?", not "is this synthetic?"**
That distinction is the single most important correction to the original
blueprint. A convincing clone of an enrolled speaker *raises* voiceprint
similarity, so a branch that fed similarity into the spoof score would report
the best attacks as the safest calls.

It is therefore kept entirely out of the spoof score and consumed at the policy
layer, where high spoof score AND high identity match is the *worst* case -- a
convincing clone of a known customer -- rather than the most reassuring one.

It also earns its keep with no enrollment at all: comparing a caller to
themselves across consecutive windows detects the "human hands the phone to a
clone" case, which the spoof branches cannot express when both segments look
equally synthetic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class BranchCConfig:
    source: str = "speechbrain/spkrec-ecapa-voxceleb"
    embed_dim: int = 192
    cohort_size: int = 200         # top-k imposters used by AS-Norm
    change_threshold: float = 0.55  # cosine distance for a speaker turn


class SpeakerEmbedder(nn.Module):
    """Frozen pretrained ECAPA-TDNN. Returns L2-normalised embeddings."""

    def __init__(self, cfg: BranchCConfig = BranchCConfig(),
                 savedir: str | None = None,
                 device: str | torch.device = "cpu"):
        super().__init__()
        from speechbrain.inference.speaker import EncoderClassifier

        self.cfg = cfg
        self.encoder = EncoderClassifier.from_hparams(
            source=cfg.source,
            savedir=savedir or ".cache/ecapa",
            run_opts={"device": str(device)},
        )
        for p in self.encoder.mods.parameters():
            p.requires_grad_(False)

    @torch.inference_mode()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if wav.dim() == 3:
            wav = wav.squeeze(1)
        emb = self.encoder.encode_batch(wav).squeeze(1)
        return F.normalize(emb, dim=-1)


class ASNorm:
    """Adaptive symmetric score normalisation against an imposter cohort.

    Raw cosine similarity carries a speaker-dependent offset, so one threshold
    across speakers is meaningless without it. AS-Norm centres each side of the
    trial on its own top-k imposter statistics.
    """

    def __init__(self, cohort: torch.Tensor, top_k: int = 200):
        self.cohort = F.normalize(cohort, dim=-1)
        self.top_k = min(top_k, self.cohort.shape[0])

    def _stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sims = x @ self.cohort.T                       # (B, cohort)
        top = sims.topk(self.top_k, dim=-1).values
        return top.mean(dim=-1), top.std(dim=-1).clamp(min=1e-6)

    def __call__(self, enroll: torch.Tensor, test: torch.Tensor) -> torch.Tensor:
        raw = (F.normalize(enroll, dim=-1) * F.normalize(test, dim=-1)).sum(dim=-1)
        me, se = self._stats(F.normalize(enroll, dim=-1))
        mt, st = self._stats(F.normalize(test, dim=-1))
        return 0.5 * ((raw - me) / se + (raw - mt) / st)


class BranchC(nn.Module):
    """Identity signal plus a free speaker-turn detector."""

    def __init__(self, cfg: BranchCConfig = BranchCConfig(),
                 savedir: str | None = None,
                 device: str | torch.device = "cpu"):
        super().__init__()
        self.cfg = cfg
        self.embedder = SpeakerEmbedder(cfg, savedir, device=device)
        self.asnorm: Optional[ASNorm] = None
        # Calibration of the ASV score to an LLR is fit on target/nontarget dev
        # trials and needs no spoof data; identity is a different question from
        # synthesis and is calibrated separately.
        self.register_buffer("calib_a", torch.tensor(1.0))
        self.register_buffer("calib_b", torch.tensor(0.0))

    def fit_cohort(self, cohort_wavs: torch.Tensor) -> None:
        self.asnorm = ASNorm(self.embedder(cohort_wavs), top_k=self.cfg.cohort_size)

    def embed(self, wav: torch.Tensor) -> torch.Tensor:
        return self.embedder(wav)

    def load_calibration(self, a: float, b: float) -> None:
        """Set the ASV score -> LLR mapping from calib/C@asv.json.

        These buffers defaulted to 1.0/0.0 and nothing ever loaded the fitted
        values (a=1.0219, b=0.4283), because their only consumer -- llr_target
        -- had no call site. An identity LLR on the wrong scale is worse than
        none: `identity_conflict` tests `identity_llr > 0`, and the intercept
        alone moves which trials land on which side of that.
        """
        self.calib_a.fill_(float(a))
        self.calib_b.fill_(float(b))

    @torch.inference_mode()
    def llr_target_from_emb(
        self, enroll: torch.Tensor, test_emb: torch.Tensor
    ) -> torch.Tensor:
        """Calibrated identity LLR from an ALREADY COMPUTED test embedding.

        Split out from `llr_target` because the serving path has the embedding
        in hand: the scorer computes it for speaker-turn detection on every
        slow-lane window. Calling llr_target there would run ECAPA a second
        time on the same audio, doubling Branch C's cost per window for a
        number already available -- and Branch C is the most expensive branch
        per second of audio in the system.
        """
        t = F.normalize(test_emb, dim=-1)
        e = F.normalize(enroll, dim=-1)
        if e.dim() == 1:
            e = e.unsqueeze(0).expand_as(t)
        raw = self.asnorm(e, t) if self.asnorm else (e * t).sum(dim=-1)
        return self.calib_a * raw + self.calib_b

    @torch.inference_mode()
    def llr_target(self, enroll: torch.Tensor, test_wav: torch.Tensor) -> torch.Tensor:
        """Calibrated identity LLR from audio. Never enters the spoof score."""
        return self.llr_target_from_emb(enroll, self.embed(test_wav))

    @torch.inference_mode()
    def speaker_changed(
        self, prev: torch.Tensor, current: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Self-similarity turn detection. Works with no enrollment at all."""
        a = F.normalize(prev, dim=-1)
        b = F.normalize(current, dim=-1)
        dist = 1.0 - (a * b).sum(dim=-1)
        return dist > self.cfg.change_threshold, dist
