import torch
import torch.nn as nn
from speechbrain.inference.speaker import EncoderClassifier

class SpeakerBranch(nn.Module):
    """
    Branch C: Speaker Identity (ASV) Branch.
    Pretrained ECAPA-TDNN for speaker embedding extraction.
    Used for mid-call speaker swap detection and identity verification.
    """
    def __init__(self, embed_dim: int = 192):
        super().__init__()
        self.embed_dim = embed_dim
        # Load the pre-trained ECAPA-TDNN from speechbrain
        # Using the standard VoxCeleb trained model
        self.encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            savedir="tmpdir"
        )
        # We don't want to train this branch, freeze it
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        waveform: (batch, time)
        returns: (batch, embed_dim) L2-normalized speaker embedding
        """
        # Ensure it has no channel dim for speechbrain
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
            
        embed = self.encoder.encode_batch(waveform)
        # embed shape is (batch, 1, embed_dim), squeeze the middle dimension
        embed = embed.squeeze(1)
        
        # L2 normalize
        embed = torch.nn.functional.normalize(embed, p=2, dim=-1)
        return embed
        
    def score_similarity(self, embed1: torch.Tensor, embed2: torch.Tensor) -> torch.Tensor:
        """
        Cosine similarity between embeddings.
        """
        return torch.sum(embed1 * embed2, dim=-1)
