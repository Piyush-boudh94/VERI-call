import torch
import torch.nn as nn

class ProsodyBranch(nn.Module):
    """
    Branch B: Prosody Branch.
    Trained one-class on bona fide speech only, emitting an anomaly score.
    Relies on F0 and energy trajectories (survives compression).
    """
    def __init__(self, input_dim: int = 1): # e.g. F0 sequence
        super().__init__()
        
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=32, num_layers=2, batch_first=True)
        self.anomaly_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1) # Autoencoder reconstruction error or direct one-class score
        )

    def forward(self, f0_contour: torch.Tensor) -> torch.Tensor:
        """
        f0_contour: (batch, time, 1)
        """
        # Ensure channel dim
        if f0_contour.dim() == 2:
            f0_contour = f0_contour.unsqueeze(-1)
            
        _, (h_n, _) = self.lstm(f0_contour)
        # Use the last hidden state of the top layer
        last_hidden = h_n[-1] # (batch, 32)
        
        return self.anomaly_head(last_hidden)
