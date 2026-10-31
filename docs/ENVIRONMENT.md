# VeriCall — measured environment

Everything here was verified by running it, not read from documentation. Where a
prior document disagreed, the measurement wins and the discrepancy is noted.

## Topology

| | Head node (`bmu-headnode`) | GPU pod (on `bmu-worker`) |
|---|---|---|
| CPU / RAM | 40 threads, 125 GiB | 16 vCPU / 32 GiB (quota) of 224 / 2 TiB |
| GPU | **none at all** | 1× `mig-1g.18gb` (H200 slice, ~16 GB VRAM, 16 SMs) |
| Python | 3.12.12 | 3.11.9 |
| torch | 2.13.0+cu130, `cuda.is_available() == False` | **2.4.0 + torchaudio 2.4.0, CUDA live** |
| ffmpeg | installed but **broken** (`liblapack.so.3` missing, unfixable without root) | works (conda build 4.3) |
| Purpose here | fusion engine, calibration, tests, docs | all GPU work |

Slurm exists but is non-functional (`resolve_ctls_from_dns_srv` failure).
**Kubernetes is the scheduler.**

## Quota — the binding constraint

```
limits.cpu                       16    / 16      (100% used)
limits.memory                    32Gi  / 32Gi    (100% used)
requests.nvidia.com/gpu          0     / 0       (full GPUs FORBIDDEN)
requests.nvidia.com/mig-1g.18gb  1     / 1       (100% used)
requests.nvidia.com/mig-2g.35gb  0     / 0       (FORBIDDEN)
requests.nvidia.com/mig-3g.71gb  0     / 0       (FORBIDDEN)
```

The whole quota is held by the `wmamba-dev` Deployment (a separate thesis
project). **VeriCall runs as a co-tenant inside that pod** and does not create,
modify or restart it. The slice was idle (`sleep infinity`, no training) when
this was set up.

Consequence: if WMamba training starts, serving contends for the same 16 cores
and demo p99 latency will suffer. Documented fallback is a manifest split —
`dev-pod` to 8 CPU / 16 Gi / no GPU, plus `vericall` at 8 CPU / 16 Gi / 1 MIG —
which fits the quota exactly. That change is **not** made without the owner's
explicit go-ahead.

## Storage — the surprise

| Path | Size | Free | Nature |
|---|---|---|---|
| NFS home `/home/...` | 5.3 T | **55 G (99% full)** | shared across ~20 users, persistent |
| Pod container fs `/` (`/dev/md2`) | 28 T | **26 T** | node-local, **ephemeral** |
| `/dev/md1` (in pod) | 1.8 T | 1.7 T | node-local, ephemeral |

There is **no StorageClass and no PersistentVolume** in the cluster and
`kubectl get pvc` is empty — so the 26 TB **cannot be made persistent**. It is
scratch, wiped on pod delete or reschedule.

Hence the split: raw corpora on pod scratch, and only the VAD-trimmed working
set, manifests, calibration artifacts and checkpoints on NFS. A prior document
recorded 1.3 T free five weeks earlier; it is 55 G now, so the shared filesystem
is filling fast and could reach zero mid-project.

## Python environment

`.venv` is built **inside the pod** with `--system-site-packages` so it inherits
the image's matched **torch 2.4.0 / torchaudio 2.4.0** pair. This is deliberate
on two counts: reinstalling either would break the ABI pairing, and a standalone
venv with torch would cost ~3 GB on a filesystem with 55 GB free. As built it is
**269 MB**.

Note `~/research/.venv` pairs torch 2.10 with torchaudio 2.4 — a broken pairing
that fails on any C++-backed torchaudio op. VeriCall does not reuse it.

**No compiler and no root in the pod**, so every dependency must resolve to a
prebuilt cp311 manylinux wheel. This rules out `webrtcvad` and `pyworld`, which
is why VAD is silero (pure torch) and F0 is the torch-native YIN in
`vericall/features/prosody.py`.

## Codec support (pod ffmpeg)

| Codec | Decode | Encode |
|---|---|---|
| Opus | yes | **yes** |
| G.711 µ-law / A-law | yes | **yes** |
| G.722, GSM | yes | no |
| **AMR-NB / AMR-WB** | yes | **NO** |

AMR encode would need `--enable-libopencore-amrnb`, i.e. a rebuilt ffmpeg we
cannot install. Codec augmentation therefore covers Opus + G.711 honestly, and
the AMR gap is documented rather than hidden — mitigated by RTCFake, which
carries *real* platform transmission rather than simulated degradation.

## Gotchas that cost real time

**1. NFS bytecode staleness — the expensive one.** Editing a `.py` on the head
node and running it in the pod can silently execute the *old* code. The pod is a
different NFS client and reuses a `__pycache__/*.pyc` it believes is current;
observed with a `.pyc` **7 minutes older** than its `.py` and still used. This
produced two rounds of "the fix didn't work" on a fix that was correct.
`scripts/pod.sh` now exports `PYTHONDONTWRITEBYTECODE=1` for every pod
invocation. If results ever look impossibly stale, clear caches first:

```bash
find . -name __pycache__ -type d -prune -exec rm -rf {} +
```

**2. The head node's scratch is invisible to the pod.** `/tmp` is node-local.
Anything the pod must read goes in the project directory (NFS).

**3. No audio hardware anywhere.** No `/dev/snd`, no ALSA, no PulseAudio, no
PortAudio. `sounddevice`/`pyaudio` will install and find zero devices.
"Real-time" therefore means audio streamed in over WebSocket from a browser —
which is the intended deployment anyway.

**4. `/dev/shm` defaults to 64 MB in Kubernetes** and deadlocks PyTorch
DataLoader workers. The existing manifest already overrides it to 8 GiB.

**5. Namespace deletion.** Prior project notes record the namespace being wiped
externally roughly twice a day. Any load-test ramp longer than ~40 minutes will
hit one, so the serving stack reuses the existing `supervisor.sh` reconcile
pattern.

## Verified working

```
torch 2.4.0  cuda True  NVIDIA H200 MIG 1g.18gb   VRAM 16384 MiB
batched STFT+LFCC, batch 64 x 1 s @ 16 kHz : 0.54 ms  -> ~119k windows/s
batched phase features (MGDCC + MRP), batch 64 : 2.51 ms -> ~25k windows/s
batched YIN F0, 4 x 2 s : 0.54 ms -> ~14,000x realtime
```

Feature extraction is emphatically **not** the bottleneck; the SSL frontend will
be. Capacity numbers in the plan remain derived until Branch A is measured.
