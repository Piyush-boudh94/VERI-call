import sys, numpy as np, torch
sys.path.insert(0, ".")
from vericall.branches.a2_phase import BranchA2, BranchA2Config
from vericall.data.cache import CachedFold
from vericall.fusion.calibration import eer

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m = BranchA2(BranchA2Config()).to(dev)
blob = torch.load("checkpoints/branch_A2_seed0.pt", map_location=dev, weights_only=False)
m.load_state_dict(blob["state_dict"]); m.eval()

fold = CachedFold.load("/tmp/vericall-cache/asvspoof2019_la/dev")
scores = np.zeros(len(fold))
with torch.inference_mode():
    for x, _, _, sel in fold.batches(128, shuffle=False, use="wave", device=dev):
        scores[sel] = torch.softmax(m(x), -1)[:, 1].float().cpu().numpy()
e, _ = eer(scores, fold.label)
rec = blob["dev_eer"]
print(f"recorded dev EER   {rec*100:.2f}%")
print(f"reproduced dev EER {e*100:.2f}%")
print(f"delta              {abs(e-rec)*100:.3f} pp")
print("VERDICT:", "EXACT MATCH - reconstruction correct"
      if abs(e - rec) < 0.002 else "MISMATCH - activations differ")
