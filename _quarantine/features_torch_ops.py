import torch
import torch.nn as nn
import torchaudio.transforms as T
import math

class FeatureExtractor(nn.Module):
    """
    Batched GPU feature extraction for VeriCall.
    Implements STFT, LFCC, Group Delay (GD), Modified Relative Phase (MRP), and batched F0.
    All operations must remain on the device provided.
    """
    def __init__(self, sample_rate=16000, n_fft=512, hop_length=160, n_lfcc=60):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_lfcc = n_lfcc
        
        # LFCC transform (simulated via Mel for typical GPU compatibility, or direct LFCC if torchaudio supports it)
        try:
            self.lfcc = T.LFCC(
                sample_rate=sample_rate,
                n_lfcc=n_lfcc,
                speckwargs={"n_fft": n_fft, "hop_length": hop_length, "center": True}
            )
        except AttributeError:
            # Fallback to MFCC if LFCC is not available in the specific torchaudio version
            self.lfcc = T.MFCC(
                sample_rate=sample_rate,
                n_mfcc=n_lfcc,
                melkwargs={"n_fft": n_fft, "hop_length": hop_length, "center": True}
            )

    def extract_stft(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        waveform: (batch, time)
        returns: STFT complex tensor (batch, freq, frames)
        """
        window = torch.hann_window(self.n_fft, device=waveform.device)
        return torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
            center=True
        )
        
    def extract_group_delay(self, stft_complex: torch.Tensor) -> torch.Tensor:
        """
        Computes the group delay from the STFT phase.
        stft_complex: (batch, freq, frames)
        """
        phase = torch.angle(stft_complex)
        # Unwrapped phase difference along frequency axis
        gd = torch.diff(phase, dim=1)
        # wrap to [-pi, pi]
        gd = (gd + math.pi) % (2 * math.pi) - math.pi
        # Pad back to original size
        gd = torch.nn.functional.pad(gd, (0, 0, 0, 1))
        return gd
        
    def extract_mrp(self, stft_complex: torch.Tensor) -> torch.Tensor:
        """
        Computes Modified Relative Phase (MRP).
        """
        phase = torch.angle(stft_complex)
        # Phase difference along time axis
        mrp = torch.diff(phase, dim=2)
        mrp = (mrp + math.pi) % (2 * math.pi) - math.pi
        mrp = torch.nn.functional.pad(mrp, (1, 0))
        return mrp
        
    def extract_lfcc(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        waveform: (batch, time)
        returns: (batch, n_lfcc, frames)
        """
        self.lfcc = self.lfcc.to(waveform.device)
        return self.lfcc(waveform)

    def forward(self, waveform: torch.Tensor):
        stft = self.extract_stft(waveform)
        mag = torch.abs(stft)
        gd = self.extract_group_delay(stft)
        mrp = self.extract_mrp(stft)
        lfcc = self.extract_lfcc(waveform)
        
        return {
            "mag": mag,
            "gd": gd,
            "mrp": mrp,
            "lfcc": lfcc
        }
