import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from vericall.audio.dataset import VeriCallDataset
from vericall.features.torch_ops import FeatureExtractor
from vericall.branches.a2_phase import PhaseBranch

def train_branch_a2(data_dir: str, epochs: int = 10, batch_size: int = 32, lr: float = 1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    # Init modules
    extractor = FeatureExtractor().to(device)
    model = PhaseBranch().to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    dataset = VeriCallDataset(data_dir=data_dir, subset="train")
    # For a real pipeline, the dataset metadata should be loaded
    if len(dataset) == 0:
        print("No metadata found, using dummy dataset for demonstration")
        dataset.metadata = [{'filepath': f'dummy_{i}.wav', 'label': i % 2} for i in range(100)]
        
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for waveforms, labels in loader:
            waveforms = waveforms.to(device)
            labels = labels.to(device).unsqueeze(1) # (batch, 1)
            
            optimizer.zero_grad()
            
            with torch.no_grad():
                features = extractor(waveforms)
                gd = features["gd"]
                mrp = features["mrp"]
                
            outputs = model(gd, mrp)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(loader):.4f}")
        
    print("Training complete. Saving model...")
    torch.save(model.state_dict(), "branch_a2.pth")

if __name__ == "__main__":
    train_branch_a2(data_dir="/tmp/vericall-data/asvspoof2019_la/data")
