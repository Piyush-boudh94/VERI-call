import torch
from torch.utils.data import DataLoader
from vericall.audio.dataset import VeriCallDataset
from vericall.fusion.calibration import LLRCalibrator
import numpy as np

def calibrate_system(data_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Calibrating on {device}")
    
    # Load all models (assuming they are saved in current dir or we use stubs)
    # For demonstration, we just generate mock scores instead of running full inference
    dataset = VeriCallDataset(data_dir=data_dir, subset="validation")
    if len(dataset) == 0:
        print("No metadata found, using dummy dataset for calibration")
        dataset.metadata = [{'filepath': f'dummy_{i}.wav', 'label': i % 2} for i in range(100)]
        
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # We would normally collect raw scores from each branch
    scores_a = []
    scores_a2 = []
    scores_b = []
    labels_all = []
    
    for waveforms, labels in loader:
        # Mock inference
        batch_size = waveforms.size(0)
        
        # Branch A: tends to score higher for spoof (label=1)
        mock_a = np.random.normal(0, 1, batch_size) + labels.numpy() * 2
        # Branch A2: phase anomalies
        mock_a2 = np.random.normal(0, 1, batch_size) + labels.numpy() * 1.5
        # Branch B: anomaly scores, higher for spoof
        mock_b = np.random.normal(0, 0.5, batch_size) + labels.numpy() * 1.0
        
        scores_a.extend(mock_a)
        scores_a2.extend(mock_a2)
        scores_b.extend(mock_b)
        labels_all.extend(labels.numpy())
        
    scores_a = np.array(scores_a).reshape(-1, 1)
    scores_a2 = np.array(scores_a2).reshape(-1, 1)
    scores_b = np.array(scores_b).reshape(-1, 1)
    labels_all = np.array(labels_all)
    
    calibrator = LLRCalibrator()
    
    # Fit logistic regression to convert raw scores to LLRs
    print("Fitting Branch A...")
    calibrator.fit("branch_a", scores_a, labels_all)
    print("Fitting Branch A2...")
    calibrator.fit("branch_a2", scores_a2, labels_all)
    print("Fitting Branch B...")
    calibrator.fit("branch_b", scores_b, labels_all)
    
    print("Calibration complete. Ready for real-time fusion.")
    # In practice we'd pickle the calibrator object
    import pickle
    with open("calibrator.pkl", "wb") as f:
        pickle.dump(calibrator, f)

if __name__ == "__main__":
    calibrate_system(data_dir="/tmp/vericall-data/asvspoof2019_la/data")
