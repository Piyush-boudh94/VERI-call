"""F0 accuracy + incremental-cache check. Run inside the pod."""
import math, sys, time
import torch

sys.path.insert(0, ".")
from vericall.features.prosody import (
    F0Config, F0Tracker, IncrementalF0, frame_energy, prosody_statistics,
)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tr = F0Tracker().to(dev)
sr = 16000
t = torch.arange(2 * sr, device=dev) / sr

truths = [110.0, 150.0, 220.0, 300.0]
x = torch.stack([
    sum(0.7 ** k * torch.sin(2 * math.pi * f * (k + 1) * t) for k in range(5))
    for f in truths
])

with torch.inference_mode():
    f0, voiced = tr(x)

print("--- F0 accuracy on synthetic harmonic tones ---")
ok = True
for f, row in zip(truths, f0):
    est = row[row > 0]
    med = float(est.median()) if est.numel() else float("nan")
    vr = float((row > 0).float().mean()) * 100
    err = abs(med - f) / f * 100
    ok &= err < 2.0 and vr > 80
    print(f"  true {f:6.1f} Hz -> est {med:7.2f} Hz  ({err:4.1f}% err)  voiced {vr:5.1f}%")

print("--- unvoiced input must be rejected ---")
noise = torch.randn(2, 2 * sr, device=dev) * 0.1
with torch.inference_mode():
    f0n, _ = tr(noise)
frac = float((f0n > 0).float().mean()) * 100
print(f"  white noise: {frac:5.1f}% frames marked voiced (want low)")

print("--- throughput ---")
with torch.inference_mode():
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        tr(x)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 10
print(f"  F0 4x2s: {dt*1000:.2f} ms -> {8/dt:.0f}x realtime")

print("--- incremental cache (3 s window, 1 s hop) ---")
inc = IncrementalF0(tr, window_s=3.0, hop_s=1.0)
for _ in range(6):
    with torch.inference_mode():
        f0w, vw = inc.push(torch.randn(1, sr, device=dev) * 0.05)
naive = 6 * inc._frames_per_window
print(f"  frames computed incrementally: {inc.frames_recomputed}")
print(f"  frames a naive recompute would cost: {naive}")
print(f"  saving: {(1 - inc.frames_recomputed / naive) * 100:.0f}%")

e = frame_energy(x)
st = prosody_statistics(f0, voiced, e[:, : f0.shape[1]])
print(f"--- prosody stats {tuple(st.shape)} finite={bool(torch.isfinite(st).all())} ---")
print("RESULT:", "PASS" if ok else "FAIL")
