# VeriCall

Real-time voice-cloning and impersonation detection for live calls.
**SIH 2026 · PS 26104** — *AI-Powered Real-Time Detection and Prevention of Voice
Cloning Impersonation Attacks* (AICTE, Software).

## What it does

Audio from a live call is chunked on two cadences. Four detectors score it —
SSL spectral artifacts, phase/group-delay, prosody, and speaker identity — at
different rates and on different scales. Their scores are calibrated to a common
log-likelihood-ratio scale and accumulated sequentially into one continuously
updated risk score, with change-point detection so a mid-call speaker swap is
caught rather than averaged away. Crossing a cost-weighted threshold triggers a
tiered response: silent flag → step-up verification → agent alert.

## Design commitments

- **Two outputs, not one.** Voiceprint similarity is an *identity* signal and is
  deliberately excluded from the spoof score — a good clone of an enrolled
  person makes similarity go **up**, so adding it positively would reward the
  attack. It is consumed at the policy layer, where *spoof-high ∧ identity-high*
  is the highest-consequence case, not the safest.
- **Calibrated LLRs, not a gating MLP.** A fixed-width network cannot consume
  2 Hz + 1 Hz + sometimes-absent streams. Accumulating calibrated LLRs over
  *time* makes multi-rate fusion fall out for free, and a missing branch becomes
  a dropped term rather than a different model.
- **Decisions act on a lower confidence bound**, so "reduced confidence" is a
  behaviour, not a label. The UI can never show confident green on a starved call.
- **Honest numbers.** Cross-dataset EER is the headline, per-call not per-window,
  with bootstrap intervals over calls. In-domain numbers are reported alongside,
  not instead.

## Status

| Phase | State |
|---|---|
| 0 · Fusion + risk engine | **done** — 35 tests |
| 1 · Pod environment | **done** — torch 2.4.0 + torchaudio 2.4.0, CUDA on MIG |
| 2 · GPU feature pipeline | **done** — all extractors GPU-resident |
| 3 · Chunker + Branch A + measured RTF | **done** — ~400 calls GPU-side, `docs/BENCHMARKS.md` |
| 4a · Branches A2 / B / C | **done** — all four branches run on the MIG slice |
| 4b · Corpora + leakage-safe splits | **done** — 35 GB verified, guards clean, see `docs/DATASETS.md` |
| 4c · Training + calibration fitting | **done** — 3 seeds, calibration loss 0.082 -> 0.011 |
| 4d · End-to-end on real audio | **done** — 40/40 spoof alerted, 0/40 bona fide, `docs/RESULTS.md` |
| 9 · Production readiness (FM-0..FM-8) | **done** — 9 new conflicts, `docs/PRODUCTION_READINESS.md` |
| 5 · Serving, concurrency, load test | pending |
| 6 · Dashboard + demo | pending |
| 7 · Whitepaper + bibliography | pending |

## Layout

```
docs/     CONFLICTS_RESOLVED.md  23 defects found in the v3 blueprint, with fixes
          ENVIRONMENT.md         measured cluster facts and the gotchas
vericall/ fusion/    LLR accumulation, calibration, hysteresis, risk policy
          features/  spectral (LFCC), phase (MGDCC + MRP), prosody (YIN F0)
          audio/ branches/ risk/ prevent/ serving/ bench/
scripts/  pod.sh     run anything inside the GPU pod
tests/
```

## Running

The fusion engine and calibration are pure Python and run anywhere:

```bash
python3 -m pytest tests/test_fusion.py tests/test_calibration.py -q
```

Feature extraction needs torch; run the full suite in the GPU pod:

```bash
./scripts/pod.sh py '-m pytest tests/ -q -p no:cacheprovider'   # 90 tests
./scripts/pod.sh py 'scripts/check_prosody.py'                  # F0 accuracy report
./scripts/pod.sh py '-m vericall.bench.rtf'                     # measured RTF / capacity
./scripts/pod.sh py 'scripts/fetch_datasets.py --datasets all'  # corpora -> pod scratch
./scripts/pod.sh py 'scripts/build_manifests.py'                # splits + leakage verification
./scripts/pod.sh py 'scripts/audit_silence.py --n 3000'         # the silence-shortcut control
```

VeriCall is a **co-tenant** of the `wmamba-dev` pod, which holds the entire
namespace quota. It never creates, modifies or restarts that Deployment.
See `docs/ENVIRONMENT.md`.

### Vercel

Vercel is configured as a static dashboard deployment in `vercel.json`. The
GPU-backed FastAPI gateway and its WebSocket endpoints must run on a
long-running Python host; serverless Vercel Functions cannot package or serve
the detector runtime.
