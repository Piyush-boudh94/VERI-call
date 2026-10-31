import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from vericall.audio.dataset import VeriCallDataset
from vericall.features.f0 import IncrementalF0
from vericall.branches.b_prosody import ProsodyBranch

def train_branch_b(data_dir: str, epochs: int = 10, batch_size: int = 32, lr: float = 1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    # Init modules
    f0_extractor = IncrementalF0()
    model = ProsodyBranch().to(device)
    
    # We train this as a one-class anomaly detector (MSE on bona fide data)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    dataset = VeriCallDataset(data_dir=data_dir, subset="train")
    if len(dataset) == 0:
        print("No metadata found, using dummy dataset for demonstration")
        # Only use bona fide labels (0) for one-class learning
        dataset.metadata = [{'filepath': f'dummy_{i}.wav', 'label': 0} for i in range(50)]
        
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for waveforms, labels in loader:
            # Filter for bona fide only
            bona_fide_mask = (labels == 0)
            if not bona_fide_mask.any():
                continue
                
            waveforms = waveforms[bona_fide_mask].to(device)
            
            optimizer.zero_grad()
            
            with torch.no_grad():
                # Get F0 contour for the full 3s waveform (simulated by batching frames)
                f0_contour = f0_extractor.get_f0(waveforms, start_frame_idx=0)
                
            # The model attempts to reconstruct or output an anomaly score close to 0
            # Since it's a stub, we will minimize its direct output towards 0
            anomaly_scores = model(f0_contour)
            targets = torch.zeros_like(anomaly_scores)
            
            loss = criterion(anomaly_scores, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(loader):.4f}")
        
    print("Training complete. Saving model...")
    torch.save(model.state_dict(), "branch_b.pth")

if __name__ == "__main__":
    train_branch_b(data_dir="/tmp/vericall-data/asvspoof2019_la/data")
