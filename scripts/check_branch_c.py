"""Branch C sanity check: does the embedder actually separate speakers?"""
import math, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import torch, torch.nn.functional as F
from vericall.branches.c_speaker import BranchC

dev = "cuda" if torch.cuda.is_available() else "cpu"
m = BranchC(device=dev)
sr = 16000
t = torch.arange(3 * sr, device=dev) / sr

def voice(f0, phase=0.0, jitter=0.0):
    x = sum(0.7 ** k * torch.sin(2 * math.pi * f0 * (k + 1) * t + phase) for k in range(6))
    return x + jitter * torch.randn_like(x)

a1, a2 = voice(120.0), voice(121.0, 0.4, 0.01)      # same-ish speaker
b1 = voice(215.0, 1.1)                               # clearly different
E = m.embed(torch.stack([a1, a2, b1]))
print("embeddings", tuple(E.shape), "| L2 norm", round(float(E.norm(dim=-1).mean()), 4))
same = float(F.cosine_similarity(E[0:1], E[1:2]))
diff = float(F.cosine_similarity(E[0:1], E[2:3]))
print(f"  cos(same-ish)  = {same:+.3f}")
print(f"  cos(different) = {diff:+.3f}")
print(f"  margin         = {same - diff:+.3f}")
for label, pair in (("a1->a2 (same)", (0, 1)), ("a1->b1 (change)", (0, 2))):
    ch, d = m.speaker_changed(E[pair[0]:pair[0]+1], E[pair[1]:pair[1]+1])
    print(f"  turn {label:<16} detected={bool(ch)!s:<5} dist={float(d):.3f}")
ok = same > diff
print("RESULT:", "PASS" if ok else "FAIL")
