import torch
from torch.utils.data import Dataset
import os
import torchaudio
from pathlib import Path
import datasets
import io

class VeriCallDataset(Dataset):
    """
    Base dataset class for ASVSpoof and RTCFake.
    Uses HuggingFace datasets to stream from local parquet files.
    """
    def __init__(self, data_dir: str, subset: str = "train", segment_length_sec: float = 3.0, sample_rate: int = 16000):
        self.data_dir = Path(data_dir)
        self.subset = subset
        self.segment_length_sec = segment_length_sec
        self.sample_rate = sample_rate
        self.target_length = int(segment_length_sec * sample_rate)
        
        # Determine the parquet file path based on the subset
        parquet_file = self.data_dir / f"{subset}-00000-of-00001.parquet"
        
        if not parquet_file.exists():
            print(f"Warning: {parquet_file} not found. Using empty dataset.")
            self.hf_dataset = []
        else:
            # Load the local parquet file
            self.hf_dataset = datasets.load_dataset("parquet", data_files={subset: str(parquet_file)}, split=subset)

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        
        # ASVSpoof datasets on HF usually have 'audio' (dict with 'array' and 'sampling_rate') 
        # and 'label' or 'label2' (bonafide vs spoof)
        label = item.get('label', 1)  # Default to spoof (1) if not found
        # Usually label 0 is bonafide, 1 is spoof for standard binary classification
        
        try:
            audio_data = item['audio']
            waveform = torch.tensor(audio_data['array'], dtype=torch.float32).unsqueeze(0)
            sr = audio_data['sampling_rate']
            
            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
                
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            # Pad or truncate to target_length
            if waveform.shape[1] > self.target_length:
                # Random crop
                start = torch.randint(0, waveform.shape[1] - self.target_length, (1,)).item()
                waveform = waveform[:, start:start+self.target_length]
            elif waveform.shape[1] < self.target_length:
                # Pad
                pad_amount = self.target_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
                
            return waveform.squeeze(0), torch.tensor(label, dtype=torch.float32)
            
        except Exception as e:
            # Return a zero tensor if loading fails
            return torch.zeros((self.target_length,)), torch.tensor(label, dtype=torch.float32)
