# VeriCall — plan from here

*Written 2026-09-05. Supersedes the original build plan, which is complete
through Phase 4. Target: internal SIH hackathon, ~2 weeks out.*

## Where the project actually is

| | State |
|---|---|
| Fusion + risk engine | done, 103 tests |
| GPU feature pipeline | done — LFCC, MGDCC+MRP, torch YIN F0 |
| Branches A / A2 / B / C | built; **only A is trained** |
| Datasets + leakage-safe splits | splits and manifests survive on NFS |
| Training + calibration | done — 3 seeds, calibration loss 0.082 → 0.011 |
| End-to-end on real audio | done — 40/40 spoof, 0/40 bona fide |
| Section 9 production readiness | done — 9 new conflicts, C11–C19 |
| **Serving + load test** | **code exists, never run** |
| **Dashboard / demo** | **`web/index.html` exists, never driven** |
| **Whitepaper + bibliography** | **not started** |

**The honest headline number is 30.29% EER cross-corpus**, not the 2.48%
in-domain figure. Everything below is sequenced around either improving that or
presenting it credibly.

## Two operational facts that shape the sequence

1. **The pod gets recreated and wipes scratch.** It happened once already and
   cost 375 GB (330 GB corpora + 45 GB feature cache). The NFS-resident split
   held — checkpoints, calibration, splits and manifests all survived. Re-fetch
   is fast (ASVspoof 7 GB in ~2 min via the HF mirror), but it must be one
   command, not a rediscovery each time.
2. **A second Claude session works this same folder.** It built the serving
   layer, dashboard and data pipeline while this session built fusion, features
   and Branch A. Both sets coexist and all 103 tests pass, but I have already
   seen it overwrite a file of mine. Shared written rules are worth more than
   coordination attempts.

Both are addressed by the two artifacts in "Do first" below.

---

## Do first (smoothness, ~30 min)

**A. `scripts/rehydrate.sh`** — one command that restores a wiped pod to a
working state: re-fetch the corpora the current task needs, rebuild the feature
cache, verify checkpoints and calibration are present. Idempotent, resumable,
safe to run when nothing is missing.

**B. `CLAUDE.md`** — project rules any session picks up automatically: the
storage split (raw on scratch, derived on NFS), never touch the `wmamba-dev`
Deployment, `PYTHONDONTWRITEBYTECODE` for the NFS bytecode trap, which numbers
are the honest ones to quote, and the leakage guards that must never be relaxed.
With two sessions on one folder this is the highest-leverage thing available.

---

## Critical path to the demo

### 1 — Serving, measured (highest priority)
`vericall/serving/app.py` and `vericall/bench/load.py` exist and read well but
have never been run. Until they have, the "handles many concurrent users"
requirement is unevidenced.

- Smoke test: one simulated call through the WebSocket path end to end
- Then the ramp: N ∈ {1, 2, 5, 10, 20, 40, 80}, wall-clock-paced 20 ms frames,
  80% bona fide / 10% spoof-from-t=0 / 10% mid-call swap
- **Pass criterion, declared before running:** largest N where p99 chunk latency
  ≤ 250 ms **and** drops < 0.1% **and** CPU < 80% **and** `degrade_level` stayed 0
- **The gate most teams skip:** EER at N=1 must equal EER at N=N_max. Scaling by
  quietly getting worse is trivially easy

Expected: GPU ceiling is ~400 calls (measured), so the binding constraint will be
the 16 vCPU. A defensible number in the tens is the realistic outcome, and it is
worth far more than a borrowed one.

### 2 — Demo rehearsal
Drive `web/index.html` against the live service: gauge with a numeric score,
per-branch sparklines, alert card, timeline marker on the change point.

Three moments to stage, in order of impact:
1. **Mid-call swap** — genuine voice, then the clone, gauge climbing, alert at
   ~3.7 s. This is the moment that sells it, and a single-shot classifier cannot
   express it at all
2. **Compression survival** — same clone through G.711, spectral-only baseline
   misses it, fused model catches it
3. **Kill the worker mid-call** — confidence dips, recovers, no lost call

### 3 — Cross-domain improvement (the weakest number)
Re-fetch RTCFake (20 GB) and an IndicSynth subset, retrain Branch A with domain
diversity, re-evaluate. Target: move 30.29% meaningfully. Even a partial
improvement is a strong slide, and the honest framing survives either way.

Also: train A2, B and C. Only A is trained today, so the four-branch fusion has
never run on real audio with all branches live — and the single-branch path is
what surfaced conflict E24.

### 4 — Section 9 implementation (what is not yet code)
- **Stage 2** augmentation, branch-specific: RawBoost for A/A2/B/C, RIR for
  A/A2/C, **no time-stretch for A2** (C16)
- **Stage 2b** re-measure η after augmentation — reverb raises within-call
  correlation and will push η below 0.080 (C17)
- **Stage 5** per-group stratified reporting; the fitting stays blocked on
  speaker diversity (C13), and saying so is the correct output
- **Stage 8** a Proteus-style chain search at depth ≥ 3 as a release gate

### 5 — Deliverables
- `docs/VERICALL_V4.md` — the corrected blueprint, every claim cited
- `docs/RESEARCH.md` — annotated bibliography; every source in it verified to
  exist and to say what is claimed, after v3's fabricated citation
- Slide content mapped to the Build with Bharat template in `~/papers/`

---

## Sequence

| Day | Work |
|---|---|
| 1 | rehydrate + CLAUDE.md; serving smoke test |
| 2 | load-test ramp → real concurrency number |
| 3–4 | dashboard driven live; demo rehearsal |
| 5–6 | RTCFake + IndicSynth retrain; train A2/B/C |
| 7 | augmentation, re-measure η, stratified reporting |
| 8–9 | whitepaper + annotated bibliography |
| 10 | red-team pass; slide content; buffer |

## Decisions I need

1. **Demo mode** — browser-mic live, or pre-recorded clips played through a real
   WebRTC path? The staging notes favoured pre-recorded, which I agree with:
   there is no audio hardware on this cluster, and a live clone on conference
   wifi is a needless risk.
2. **Do we spend a day on cross-domain retraining, or accept 30% and present it
   well?** Both are defensible. Presenting it well is genuinely differentiating —
   most teams will show an in-domain number and not know the difference.
3. **The other session** — should I keep coordinating around it, or do you want
   to consolidate to one session now that the pieces have merged?

## Risks

- **Scratch loss recurs** — mitigated by rehydrate.sh, not eliminated
- **NFS at 52 GB free and falling**, shared with ~20 users; a checkpoint sprawl
  would hurt. Keep derived artifacts small
- **Only Branch A is trained** — the four-branch claims in the architecture are
  currently unevidenced end to end
- **Serving is unmeasured** — the largest unknown on the critical path, which is
  why it is first
