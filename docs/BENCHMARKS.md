# VeriCall — measured performance

All figures measured on the deployed quota: **1× `mig-1g.18gb`** (H200 slice,
16 SMs, 16 GB) inside the shared pod. Nothing here is extrapolated from
published RTFs; reproduce with:

```bash
./scripts/pod.sh py '-m vericall.bench.rtf --json docs/rtf_mig.json'
```

## Why the usual quoted numbers are wrong

Two figures circulate for systems like this and neither survives contact with it:

- **"AASIST RTF 0.0155."** That is the graph-attention *head alone* on a full
  GPU. Measured here, the head is **0.6–1.6 ms** and the frozen SSL frontend is
  **4–80 ms** — the frontend is 25–50× the head. Quoting the head's RTF for an
  SSL-based system understates cost by more than an order of magnitude.
- **"1.2M events/sec, 85 ms."** From transaction-fraud streaming, where an event
  is a ~500-byte JSON record rather than 16,000 audio samples through a
  95M-parameter encoder. Four orders of magnitude off in work-per-event.

## Branch A (frozen WavLM-base truncated at layer 5 + attention head)

| Batch | p50 | p99 | windows/s | Concurrent calls (GPU-side) | VRAM |
|---:|---:|---:|---:|---:|---:|
| 1 | 35.0 ms | 60.2 ms | 29 | 14 | — |
| 8 | 36.8 ms | 85.1 ms | 217 | 109 | — |
| 16 | 23.9 ms | 62.0 ms | 669 | 335 | — |
| **32** | **41.7 ms** | **54.7 ms** | **767** | **384** | 2.1 GB |
| 64 | 80.1 ms | 80.2 ms | 800 | 400 | 3.5 GB |

Fast lane needs 2 windows/s/call, hence the calls column.

**Operating point: batch 32.** Throughput saturates around 800 win/s, so batch
64 buys ~4% more throughput for ~2× the latency. Batch 32 sits at 96% of peak
throughput with the best p99 in the table.

## Latency stability is the real argument for batching

Repeated timing of the frontend alone:

| Batch | min | p50 | max | spread |
|---:|---:|---:|---:|---:|
| 1 | 4.1 ms | 35.6 ms | 60.0 ms | **14×** |
| 8 | 13.2 ms | 18.3 ms | 86.8 ms | 6.6× |
| 32 | 40.7 ms | 40.8 ms | 43.7 ms | **1.07×** |
| 64 | 78.4 ms | 78.4 ms | 78.6 ms | 1.00× |

A sparsely-fed MIG slice idles between kernels and clocks down, so small batches
are not merely slower on average — they are *wildly* variable, which is what a
p99 SLO actually cares about. Large batches keep the slice busy and latency
becomes deterministic.

This is a second, independent reason for global tick alignment: it was adopted
so a cross-call batcher could fill batches at all (unsynchronised streams give
~2 arrivals per 20 ms window, not 32), and it turns out to also convert a 14×
latency spread into a 1.07× one.

## Feature extraction is not the bottleneck

| Stage | batch 32 p50 | windows/s |
|---|---:|---:|
| LFCC (STFT + filterbank + DCT + deltas) | 0.30 ms | 108,000 |
| Phase features (MGDCC + MRP) | 1.31 ms | 24,500 |
| YIN F0 (torch-native) | — | ~14,000× realtime |
| **Branch A (SSL + head)** | **41.7 ms** | **767** |

Every feature extractor is 30–140× cheaper than the SSL frontend. Putting them
on the GPU costs almost nothing and keeps the pipeline resident on the
accelerator; the SSL encoder is the only thing worth optimising.

## Headline

> **~400 concurrent calls GPU-side** on a single `mig-1g.18gb` slice, at batch
> 32/64, p99 55–80 ms per batch, using 2.1–3.5 GB of 16 GB VRAM.
> Cost: **150 GPU-ms per call-minute.**

**This is the GPU ceiling, not the system capacity.** End-to-end throughput is
CPU-bound — ingest, VAD, ring buffers, the slow lane and WebSocket handling all
run on the same 16 vCPU the pod is limited to. The number that goes on a slide
is the one the load test produces (Phase 5), not this one. VRAM is emphatically
not the constraint: 3.5 GB of 16 GB at batch 64.

---

# Training cost

Measured on the same MIG slice: `./scripts/pod.sh py 'scripts/bench_training.py'`

| Stage | Rate | Note |
|---|---:|---|
| decode + resample + VAD-trim + crop | 58 clips/s/core (810 on 14 cores) | CPU |
| frozen SSL frontend | 784 windows/s | GPU, batch 32 |
| head-only train step on cached features | **8,358 windows/s** | GPU, 11× cheaper |

**The frozen frontend is what makes this cheap.** Only a 0.89M-parameter head
trains, so layer-5 features can be extracted **once** and reused for every epoch
and every seed. That converts a repeated 94M-parameter forward pass into a
one-time cost.

## Core anti-spoofing training (ASVspoof LA + WaveFake, 71,285 clips)

| Step | Time |
|---|---:|
| one-time feature extraction (4 crops/clip) | ~6 min |
| head training, 30 epochs | 4.3 min |
| **total** | **~10 min** |
| *the same run without feature caching* | *~50 min* |

Feature cache is ~21 GB in fp16 — trivial against 25 TB of scratch.

## Adding multilingual fine-tuning

| Configuration | Extraction | 10 epochs |
|---|---:|---:|
| IndicSynth capped at 50 h/language × 6 | 5 min | 4 min |
| IndicSynth in full (~2,240 h) | 36 min | 30 min |

## What this estimate excludes

Multiple seeds (extraction is shared, so each extra seed is only ~4 min), full
evaluation sweeps (~115,000 clips ≈ 2.5 min each), and calibration fitting
(CPU, seconds). None changes the order of magnitude.

**The one thing that would**: unfreezing the SSL frontend. Fine-tuning 94M
parameters removes the caching advantage and adds a backward pass through the
encoder — roughly 3–4× the per-step cost with no reuse across epochs or seeds,
turning a 10-minute run into hours. The frozen-frontend decision was made for
generalisation (it is the largest single overfitting control available here);
the training speed is a consequence, not the reason.
