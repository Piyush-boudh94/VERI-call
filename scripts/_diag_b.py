import sys, numpy as np, torch
sys.path.insert(0, ".")
from vericall.branches.b_prosody import BranchB, BranchBConfig
from vericall.data.cache import CachedFold

dev = torch.device("cuda")
m = BranchB(BranchBConfig()).to(dev)
blob = torch.load("checkpoints/branch_B.pt", map_location=dev, weights_only=False)
m.load_state_dict(blob["state_dict"]); m.eval()

fold = CachedFold.load("/tmp/vericall-cache/asvspoof2019_la/dev")
idx = np.arange(0, len(fold), 20)
embs, labs = [], []
with torch.inference_mode():
    for i in range(0, len(idx), 128):
        sel = idx[i:i+128]
        x = torch.from_numpy(np.asarray(fold.wave[sel], dtype=np.float32)).to(dev)
        c = torch.from_numpy(fold.codec[sel]).to(dev).long()
        embs.append(m.embed(x, c).float().cpu()); labs.append(fold.label[sel])
z = torch.cat(embs).numpy(); y = np.concatenate(labs)
centre = m.center.float().cpu().numpy()

d = ((z - centre) ** 2).sum(1)
print(f"embedding dim {z.shape[1]}, n={len(z)}")
print(f"  per-dim std across samples : {z.std(0).mean():.6f}  (collapse -> ~0)")
print(f"  ||z|| mean                 : {np.linalg.norm(z,axis=1).mean():.4f}")
print(f"  ||centre||                 : {np.linalg.norm(centre):.4f}")
print(f"  dist^2  bona fide          : {d[y==0].mean():.5f}")
print(f"  dist^2  spoof              : {d[y==1].mean():.5f}")
sep = (d[y==1].mean() - d[y==0].mean()) / (d.std() + 1e-9)
print(f"  separation (effect size)   : {sep:+.4f}")
print()
print("VERDICT:", "COLLAPSE - encoder maps everything to the centre"
      if z.std(0).mean() < 0.05 else "not collapsed; failure is elsewhere")
