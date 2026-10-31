"""Branch A -- self-supervised spectral-artifact detector.

A frozen SSL encoder truncated at layer 5, feeding a graph-attention head.

Two deliberate choices worth stating:

*Frozen frontend.* Only the head trains. That is the single largest overfitting
control available here -- it cuts trainable parameters by roughly two orders of
magnitude (0.89M trainable against 94.4M frozen) -- and it is also what lets
this fit a 16 GB / 16 SM MIG slice at a useful batch size. Fine-tuning
XLS-R-300M was never on the table on this quota.

*Truncated at layer 5.* This was originally an argument from the literature;
it is now measured. RESULTS.md §22 swept layers 3/5/7/9/11 under otherwise
identical conditions and found depth monotonically harmful -- unseen-vocoder EER
rises 29.11% (L3) -> 30.18% (L5) -> 38.71% (L11), and dev EER 1.40% -> 7.07%.
Layer 5 wins dev EER and RTCFake-en outright; layer 3 takes unseen-vocoder by
1.07 points on overlapping CIs, which is not a reason to move. Dropping layers
6-12 is therefore both cheaper and better, not a compromise.

Do not read this as a free win available elsewhere: the sweep found *no*
headroom in the layer index, and every cross-corpus cell stayed at or beyond
chance at every depth. Note also that lower layers carry the most
language-specific artefact information, so L3 *inverts* on Mandarin (56.63% EER,
CI excludes 50%) while L11 merely decays to chance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
import torch.nn as nn


@dataclass(frozen=True)
class BranchAConfig:
    bundle: Literal["wavlm_base", "wav2vec2_base"] = "wavlm_base"
    layer: int = 5                 # 1-indexed encoder layer to read out
    hidden: int = 192
    n_heads: int = 4
    dropout: float = 0.1
    sample_rate: int = 16_000


def _load_bundle(name: str):
    from torchaudio import pipelines

    return {
        "wavlm_base": pipelines.WAVLM_BASE,
        "wav2vec2_base": pipelines.WAV2VEC2_BASE,
    }[name]


class SSLFrontend(nn.Module):
    """Frozen SSL encoder, evaluated only up to `layer`."""

    def __init__(self, cfg: BranchAConfig):
        super().__init__()
        bundle = _load_bundle(cfg.bundle)
        self.model = bundle.get_model()
        self.layer = cfg.layer
        self.out_dim = 768
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True):     # never leave eval: frozen means frozen
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """(B, T) waveform -> (B, frames, 768) layer-`layer` features."""
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        feats, _ = self.model.extract_features(wav, num_layers=self.layer)
        return feats[-1]

    @torch.inference_mode()
    def extract_layers(self, wav: torch.Tensor,
                       layers: Sequence[int]) -> dict[int, torch.Tensor]:
        """Several layers from ONE forward pass.

        `extract_features(num_layers=N)` already returns every layer up to N, so
        a layer sweep costs one encoder pass rather than one per layer -- which
        is what makes sweeping affordable on a single MIG slice at all.

        Worth having because the layer choice is not a detail. The 2026
        literature on generalisable audio deepfake detection (probing-guided
        layer selection; layer selection and fusion of SSL countermeasures)
        reports it as a dominant factor in out-of-domain performance, and this
        project picked layer 5 from a general argument about lower-middle layers
        rather than from a measurement on its own corpora.
        """
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        want = sorted(set(int(l) for l in layers))
        feats, _ = self.model.extract_features(wav, num_layers=max(want))
        # extract_features returns layers 1..N in order, so layer L is index L-1
        return {l: feats[l - 1] for l in want}


class GraphAttentionLayer(nn.Module):
    """Self-attention over a set of node embeddings.

    AASIST's contribution is treating evidence as a graph and letting attention
    decide which nodes matter, rather than pooling uniformly -- spoofing
    artifacts are typically local, so uniform pooling dilutes exactly the frames
    that carry the signal.
    """

    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm(x + h)
        return self.norm2(x + self.ff(x))


class LatentGraphLayer(nn.Module):
    """A fixed set of learned nodes that cross-attend over the frame sequence.

    This is the second view. It is NOT a spectral graph: SSL features have no
    frequency axis to build one over -- their channels are learned semantic
    dimensions, so transposing to make channels the nodes would make the
    attention width equal to the frame count and therefore vary with input
    length, which does not even type-check as a fixed module.

    Learned prototype nodes give a genuinely different view (what patterns are
    present, pooled globally) from the temporal graph (where in the window
    something happened), at a fixed width independent of window length.
    """

    def __init__(self, dim: int, n_nodes: int = 16, n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.nodes = nn.Parameter(torch.randn(1, n_nodes, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.nodes.expand(x.shape[0], -1, -1)
        h, _ = self.attn(q, x, x, need_weights=False)
        z = self.norm(q + h)
        return self.norm2(z + self.ff(z))


class BranchAHead(nn.Module):
    """Two-view attention head, AASIST-inspired.

    Simplified against published AASIST, stated plainly rather than glossed: the
    heterogeneous stack-node attention is replaced by concatenated readouts of
    the two views, graph pooling is fixed rather than learned, and the second
    view is a latent-node graph rather than a spectral one (see above). What is
    kept is the part that carries the accuracy -- two complementary views with
    attention-weighted readout instead of uniform pooling, which matters because
    spoofing artifacts are local and uniform pooling dilutes exactly the frames
    that carry them.
    """

    def __init__(self, in_dim: int = 768, cfg: BranchAConfig = BranchAConfig()):
        super().__init__()
        d = cfg.hidden
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, d), nn.GELU())
        self.temporal = GraphAttentionLayer(d, cfg.n_heads, cfg.dropout)
        self.latent = LatentGraphLayer(d, 16, cfg.n_heads, cfg.dropout)
        self.pool_t = nn.Linear(d, 1)
        self.pool_l = nn.Linear(d, 1)
        self.classifier = nn.Sequential(
            nn.Dropout(cfg.dropout), nn.Linear(d * 4, d), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(d, 2),
        )

    @staticmethod
    def _readout(x: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(scores, dim=1)
        return torch.cat([(x * w).sum(1), x.max(dim=1).values], dim=-1)

    def pooled(self, feats: torch.Tensor) -> torch.Tensor:
        """(B, frames, in_dim) -> (B, hidden*4), the representation the
        classifier sees.

        Exposed so a domain adversary can attach HERE rather than to the logits.
        Attaching to the logits would only make the decision corpus-invariant,
        which is trivially satisfiable by a worse decision; attaching to the
        representation is what actually forces the features to stop encoding
        which corpus a clip came from.
        """
        x = self.proj(feats)
        t = self.temporal(x)
        l = self.latent(x)
        return torch.cat([self._readout(t, self.pool_t(t)),
                          self._readout(l, self.pool_l(l))], dim=-1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """(B, frames, in_dim) -> (B, 2) logits."""
        return self.classifier(self.pooled(feats))


class BranchA(nn.Module):
    """Frozen SSL frontend + trainable head. Emits a spoof posterior."""

    def __init__(self, cfg: BranchAConfig = BranchAConfig()):
        super().__init__()
        self.cfg = cfg
        self.frontend = SSLFrontend(cfg)
        self.head = BranchAHead(self.frontend.out_dim, cfg)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return self.head(self.frontend(wav))

    @torch.inference_mode()
    def score(self, wav: torch.Tensor) -> torch.Tensor:
        """Spoof posterior in [0, 1], one per item in the batch.

        Raw and uncalibrated by design -- the fusion engine never sees this.
        It is mapped to an LLR by the branch's calibration artifact first.
        """
        return torch.softmax(self(wav), dim=-1)[:, 1]

    @property
    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def n_frozen(self) -> int:
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)
