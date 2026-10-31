import torch
from typing import Dict, Optional

class IncrementalF0:
    """
    Computes and caches F0 (pitch) frame by frame to avoid redundant 
    computation on overlapping windows (e.g., 3s window, 1s hop = 2s overlap).
    """
    def __init__(self, sample_rate: int = 16000, frame_length: int = 160):
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.cache: Dict[int, torch.Tensor] = {}

    def get_f0(self, waveform: torch.Tensor, start_frame_idx: int) -> torch.Tensor:
        """
        waveform: (batch, time) corresponding to a chunk
        start_frame_idx: the absolute index of the first frame in this chunk
        
        Returns the F0 contour for the provided chunk.
        """
        # If we had torchaudio and an actual batch F0 function, we would do:
        # f0 = torchaudio.functional.detect_pitch_frequency(waveform, self.sample_rate)
        # For the sake of this implementation, we will mock the F0 extraction
        
        batch_size, num_samples = waveform.shape
        num_frames = num_samples // self.frame_length
        
        f0_frames = []
        for i in range(num_frames):
            abs_frame_idx = start_frame_idx + i
            if abs_frame_idx in self.cache:
                f0_frames.append(self.cache[abs_frame_idx])
            else:
                # Mock F0 extraction for this frame
                # In practice, this would extract the pitch from the frame
                frame_f0 = torch.zeros((batch_size, 1), device=waveform.device)
                self.cache[abs_frame_idx] = frame_f0
                f0_frames.append(frame_f0)
                
        # Clean up old cache entries (keep only last 1000 frames ~ 10 seconds)
        cutoff = start_frame_idx - 1000
        keys_to_delete = [k for k in self.cache.keys() if k < cutoff]
        for k in keys_to_delete:
            del self.cache[k]
            
        return torch.cat(f0_frames, dim=1)
