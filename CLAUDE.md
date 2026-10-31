# VeriCall — project record and working rules

Real-time voice-cloning and impersonation detection for live calls.
**SIH 2026 · PS 26104** — *AI-Powered Real-Time Detection and Prevention of
Voice Cloning Impersonation Attacks* (AICTE, Software).

**More than one agent session works this folder.** Read this file first. It is
the single source of truth for what exists, what is measured, and what must not
be changed casually.

---

## 1. What the system is

A live call is chunked on two cadences. Four branches score it — SSL spectral
artifacts (A), phase/group-delay (A2), prosody (B), speaker identity (C) — at
different rates and on different scales. Their scores are calibrated to a common
log-likelihood-ratio scale in nats and accumulated sequentially into one
continuously-updated risk score, with CUSUM change detection so a mid-call
speaker swap is caught rather than averaged away. Crossing a cost-weighted
threshold triggers a tiered response: silent flag → step-up verification →
agent alert.

**Branch C is an identity signal, never a spoof score.** A convincing clone of
an enrolled person *raises* voiceprint similarity, so adding it to the spoof
score would make the best attacks look safest. It is consumed at the policy
layer, where *spoof-high ∧ identity-high* is the worst case, not the best.

---

## 2. Status

| Component | State |
|---|---|
| Fusion + risk engine | **done** — 25 tests |
| Calibration + metrics | **done** — 16 tests |
| n-of-m aggregation (Section 9) | **done** — 13 tests |
| GPU feature pipeline | **done** — LFCC, MGDCC+MRP, torch YIN F0 |
| Dual-rate chunker | **done** — 15 tests |
| Branch A (SSL + attention head) | **trained**, 3 seeds, 2.48%/11.01%/30.29% EER |
| Branch A2 (phase) | **trained**, dev EER 6.31% |
| Branch B (one-class prosody) | **trained**, but ceiling-limited: ~39% supervised bound on this corpus (§6) |
| Branch C (ECAPA + AS-Norm) | **fitted**, 20-speaker cohort, identity EER 7.98% (§7) |
| Per-subset calibration table | **fitted** — A+A2 reaches 1.46% EER vs 2.52% for A alone (§8) |
| Serving gateway | **production-hardened** — **N=64** all SLOs over real WebSockets, chunk p99 22 ms; auth, admission + fair-share caps, bounded queue, drain, integrity manifest (§9.12) |
| Dashboard | **real** — streams bundled ASVspoof clips, verified on-stage path (§12) |
| Whitepaper + bibliography | **done** (`docs/VERICALL_V4.md`, `docs/RESEARCH.md`) |
| Corpora integrated | **8 built, 7 measured.** Trained on: ASVspoof, WaveFake, RTCFake, hindi_paired (Kathbath+IndicSynth). Eval only: EchoFake (§21), In-the-Wild (§24). **CodecFake has a manifest, a split and an `eval_unseen_codec` fold and has never been measured** — the only fold in EVAL_FOLDS with no result. Branch B is trained on nothing and zero-weighted everywhere |
| Replay (EchoFake) | **measured** — evades by 1.2–1.5x, best 33.68% (§21). Needs a different mechanism |
| Cross-corpus fixes | **all three tested, all failed** (§20). Recorded, not hidden |
| Inference precision | **fp16**, 2.00x at batch 32, decisions unchanged |
| Call history (SQLite + API + UI) | **done** — no audio/features/transcript, retention callable |
| Live microphone demo | **done** — resampler verified exactly against the clip path |
| Local deployment (6 GB GPU) | **done** — `run_local.py`, `docs/LOCAL_DEPLOYMENT.md`; 0.96 GiB at batch 32 |
| Browser-to-browser calling | **done** — `/call`, signalling on `/rtc/{room}`, each peer scores the decoded remote track. Media is SRTP peer-to-peer: the server never sees it |
| SSL layer selection | **measured, closed** (§22) — depth monotonically hurts, layer 5 was already right, no headroom in the layer index |
| Opus / WebRTC channel | **measured and fitted** (§23) — A survives (1.18→1.33% EER), **A2 destroyed** (1.77→18.07%, it reads phase). opus bucket fitted, A2 weight cut 0.87→0.42, channel *declared* because it cannot be detected |
| **Hindi (the deployment target)** | **WORKS** (§25) — 7.54% EER [6.65–8.55] on a paired Kathbath/IndicSynth corpus; 1.92% on voice conversion, 9.64% on zero-shot TTS. Deployed ASVspoof head scored 42.29% on the same fold. Codec confound ruled out (matched 7.54% vs unmatched 7.81%) |
| Hindi over telephony | **measured and fitted** (§25) — clean 9.24% / G.711 13.93% / noisy 23.21%, slope spread 1.25×. Channel-trained head beats clean-only by 4.7 points on channel-mixed audio (13.04% vs 17.72%) |
| Hindi, live | **0.00% call-level** over 21 real-time streamed calls, 25-nat margin. In the export gate |
| GPU | **in use** — `deploy/gpu-pod.yaml`, H200 MIG 1g.18gb. Was idle the whole project: the head node has no GPU and the co-tenant pod it borrowed had been deleted. 5 → 309 clips/s |
| Strict mode | `VERICALL_STRICT=1` refuses to start degraded. There were 19 ways to lose a capability and still answer status "ok" |
| Portability | `scripts/make_bundle.sh` — one 165 MB file, complete history, `git clone <file>` restores it. There is still no git remote |
| In-the-Wild | **integrated and measured** (§24) — 22.48% EER, the best OOD result the project has, and **all three generalisation fixes are worse on it** (6th confirmation of §20). A2 is at chance (50.65%) |
| Per-tenant calibration | **implemented** — `scripts/fit_tenant_calibration.py`, refuses to write a profile when min_Cllr says there is no signal |
| TLS / two-device demo | **done** — `run_local.py --tls`; the pages now use their own origin instead of a hardcoded `ws://…:8001` |
| identity_conflict | **live** (was dead code) — enrollment API, C calibration actually loaded, `?claims=` on the call |
| Cold export | **verified** — `scripts/verify_export.sh` clones, boots, streams and asserts. Found 4 defects no amount of reading would have |

357 tests pass on the head node, 2 skipped (measured 2026-09-22: `python3 -m pytest tests/ -q`, 15 s). The two skips are `tests/test_splits.py` real-corpus protocol checks, which need ASVspoof on pod scratch -- so the *real-data* leakage guards do not run in the default head-node suite, only their synthetic fixtures do.

---

## 3. Measured results — the honest numbers

**Detection (Branch A, 3 seeds, CIs bootstrapped over utterances):**

| Evaluation | EER |
|---|---:|
| dev — in-domain, seen attacks | 2.48% ± 0.09 |
| ASVspoof eval — in-domain, **unseen attacks** | 11.01% ± 0.42 |
| WaveFake — seen vocoders | 31.43% ± 0.65 |
| WaveFake — **unseen vocoders** | **30.29% ± 0.33** |

**Cross-corpus, Branch A by training corpus** (`docs/eval_matrix.json`, one code
path from every checkpoint to every fold):

| trained on | ASVspoof eval | WaveFake unseen | RTCFake en | RTCFake zh |
|---|---:|---:|---:|---:|
| ASVspoof | **11.35%** | **30.08%** | 45.57% | 55.34% INVERTED |
| RTCFake-en | 36.54% | 49.36% | **22.93%** | **33.34%** |
| both | **9.12%** | 49.89% | 25.03% | 37.21% |

**No configuration is good on more than one corpus.** Retraining removed the
Mandarin inversion completely (paired protocol: 79.08% -> 9.51%, on a language
never trained on) and proved section 13's diagnosis right -- it was a data
problem, not an architectural one -- but it relocated rather than solved the
generalisation failure. Mixing destroys WaveFake outright (30.08% -> 49.89%)
even with ASVspoof at 64% of the training rows, so "just add more data" is not
the strategy.

**Hindi** (IndicSynth, spoof-only so no EER exists): at a 5% false-positive
budget the ASVspoof head catches **5.3%** of attacks (chance), the RTCFake head
**13.9%**. This is the biggest remaining gap.

**End-to-end on real streamed audio** (40 calls each, no tuning): 40/40 spoof
alerted, **0/40** bona fide, mid-call swap caught **2.3 s after the splice**,
bona fide calls at **-50.23 nats**.

**Runtime channel detection: 98.8% per call** (clean 100%). Closes the section 11
gap. Per-channel calibration now covers clean/g711/noisy_line for A and A2;
Branch A2's calibration loss falls **94%** and detection captures 97.4% of what
a perfect oracle would give. Fitting the fusion weights -- hand-set and never
fitted until now -- moved A2 from 0.45 to **0.87** and made fusion beat its own
best single branch (2.79% vs 3.62%).

**Calibration:** loss 0.082 → 0.011. Per-channel fitting is justified, not
assumed: G.711 more than doubles EER (1.52% → 3.30%) *and* needs a different
slope (0.510 vs 0.411).

**Throughput:** **64 concurrent calls** meeting every SLO over real WebSockets
(chunk p99 22.2 ms against a 250 ms budget, first score 1.01 s, 64/64 complete).
Server-side, excluding transport, the ceiling is **N=160 at 0.99x real time**.
The previous figure was N=4.

The single biggest cause was **torch sizing its intra-op pool from the host core
count -- 112 threads on 16 cores**. Every offline script here already set it to
1; the serving path never did. Invisible at one request, the whole problem under
concurrency. One line took N=50 from 1.74x real time / p99 1102 ms to 0.97x /
p99 56.7 ms.

**η, per branch** — measured on 800 calls, each branch scored on the lane it
actually runs on: **A = 0.207** (4.8x inflation), **A2 = 0.928** (near
independent), **B = 0.565**. A's autocorrelation decays only from +0.38 to +0.23
across eight lags, which is call-level consistency rather than window overlap;
A2's phase residual decorrelates within one hop; B's profile reproduces its own
3 s / 1 s overlap fractions. Now in `FusionConfig`.

**An earlier η = 0.080 / 12.5x is RETRACTED.** It was measured while the chunker
emitted byte-identical windows for a multi-tick push, so its autocorrelation was
float non-determinism (median score std 1.8e-15). See docs/RESULTS.md section 15.

### Rules for quoting numbers
- **Two capacity numbers, never one.** `scripts/capacity_probe.py` is
  server-side and excludes transport; `vericall.bench.load` is end-to-end and
  includes it. The load generator shares the pod's 16 cores with the server, so
  at high N it starves what it is measuring -- a caveat at N=4 and the answer
  at N=50.
- **Quote the cross-corpus number, not 2.48% (in-domain).** The gap is the
  finding, not an embarrassment. Which cross-corpus number depends on the
  checkpoint -- use the matrix above, and say what it was trained on.
- **Never quote a trivial-baseline figure from a different corpus than the
  model's.** ASVspoof's silence artifact is worth 36.4% EER; RTCFake has none
  (best cue 41.43% zh, 43.96% en). Re-measure per corpus with
  `scripts/audit_silence.py --dataset <name>`.
- Report **paired FPR/FNR**, never blended accuracy or AUC.
- Bootstrap CIs resample **utterances**, never rows.
- Per-call is the headline; per-window is diagnostic.
- **Weights and calibration belong to a deployment profile, not to the code.**
  `calib/` and `calib_rtc/` each carry their own; never quote a number produced
  by mixing one profile's heads with another's mappings.
- Never compute an aggregate error rate binomially — measure it (see §4, C11).

---

## 4. Principal findings

**The vocoder gap is ~0; the domain gap is 12×.** Holding out HiFi-GAN,
FB-MelGAN and TTS costs nothing (30.29% vs 31.43%). Changing corpus costs
everything. Effort spent anticipating new cloning tools is close to wasted;
corpus and channel robustness is where the accuracy lives. Generator diversity
must therefore be specified by **synthesis paradigm**, not family.

**24 defects found in the v3/v4 blueprints**, each pinned by a test —
`docs/CONFLICTS_RESOLVED.md`. The ones that mattered were invisible on paper:
- **E4** Branch C inverted (above).
- **E20** the overlap fix cancelled algebraically, so a 0.5 s hop banked twice
  the evidence of a 1.0 s hop from identical audio.
- **E22** ESS deflation applied to weights is a mathematical no-op — a weighted
  mean *and its variance* are both scale-invariant in the weights.
- **E24** failed escalation gates suppressed alerts entirely: 0% of spoofed
  calls alerted despite perfect separation. Only end-to-end audio caught it.
- **C11** the n-of-m binomial error model is **9.3×–17.8× optimistic** on real
  correlated scores (measured, `docs/aggregation.json`).

**Silence shortcut:** ASVspoof LA scores 36.4% EER from leading-silence duration
alone. VAD-trim before windowing, always.

---

## 5. Environment — non-obvious and costly to rediscover

- **No GPU on the head node.** GPU work runs via `./scripts/pod.sh py '<args>'`.
- **We are a co-tenant of the `wmamba-dev` Deployment**, which holds the entire
  namespace quota (16 CPU / 32 GiB / 1× mig-1g.18gb) and belongs to another
  project. **Never create, modify, restart or delete it.**
- **Never recreate the venv from the head node.** It is a pod-side venv built
  with `--system-site-packages` on conda Python 3.11. A head-node `python -m
  venv` rewrites `pyvenv.cfg` to Python 3.12 with system site-packages
  disabled, and torch disappears. This has already happened once (§7).
- **NFS bytecode staleness.** Editing a `.py` here and running it in the pod can
  execute the *old* code — observed with a `.pyc` seven minutes older than its
  source. `pod.sh` sets `PYTHONDONTWRITEBYTECODE=1`. If a fix "doesn't work",
  suspect this first: `find . -name __pycache__ -type d -prune -exec rm -rf {} +`
- **The head node's `/tmp` is invisible to the pod.**
- **`ffmpeg` on the head node is broken** (missing `liblapack.so.3`, no root).
- **No compiler, no root in the pod** — prebuilt cp311 wheels only. Rules out
  `webrtcvad` and `pyworld`; VAD is energy-based, F0 is torch-native YIN.
- **AMR-NB/WB are decode-only.** Codec augmentation covers Opus and G.711.

### Storage — the split is load-bearing

| Location | Holds | Lifetime |
|---|---|---|
| Pod `/tmp/vericall-{data,cache}` | corpora, feature cache | **ephemeral** |
| NFS (this directory) | code, checkpoints, calib, splits, manifests, docs | persistent |

The pod has been recreated once and wiped 375 GB. Everything derived survived
because of this split. **Never stage corpora on NFS** — 99% full, ~20 users.
After a wipe: `./scripts/rehydrate.sh`.

---

## 6. Rules that must not be relaxed

- **Leakage guards are assertions.** `tests/test_splits.py` enforces them.
  WaveFake fakes are vocoder resyntheses of the same LJSpeech sentences, so a
  source utterance and all its variants share one fold. LJSpeech is one speaker
  — **no speaker-generalisation claim may cite WaveFake.**
- **Cross-domain eval sets are touched once, at the end.**
- **VAD-trim before windowing.**
- **Recalibrate after any weight change.** Platt constants describe one model.
- **Never ship mock scores.** A dashboard fed random numbers looks alive and
  means nothing; one judge question ends the demo. If a component is not ready,
  the UI must say so.

---

## 7. Progress log

- **Phase 0** fusion engine + calibration, head-node testable, no GPU needed.
- **Phase 1** pod venv on matched torch 2.4.0 / torchaudio 2.4.0.
- **Phase 2** GPU features. F0 exact (0.0% error at 110–300 Hz); MRP verified
  *exactly* shift-invariant where raw phase differs by 3.4 rad.
- **Phase 3** chunker + Branch A; RTF measured. Batch 32 chosen: 96% of peak
  throughput with the best p99. Small batches vary 14×, batch 32 varies 1.07×.
- **Phase 4** corpora, leakage-safe splits (121,461 utterances, exact official
  protocol), training, calibration, end-to-end on real audio.
- **Section 9** production readiness: FM-0..FM-8, C11–C19, fix ordering.
- **2026-09-05 incident** — source loss, environment corruption, mock scores in
  the demo path. Recovered. Full account in `docs/INCIDENT_2026-09-05.md`.
  Key lesson: **a checkpoint that loads cleanly is not evidence the
  architecture is right** — the A2 reconstruction loaded with zero missing keys
  and scored at chance, because activations carry no parameters and so are
  invisible in a `state_dict`. Only reproducing the measured metric proves it.

---

## 8. What is left

Detection is built, trained, measured and demoable. The gaps below are ordered
by distance from the problem statement, not by effort.

### A. Prevention -- BUILT (this entry was stale, corrected Sep 15)
`vericall/prevent/` holds `policy.py` (6 actions, 7 transaction contexts keyed
on prior / miss cost / **reversibility**), `challenge.py` (challenge-response
liveness with an asymmetric risk adjustment: pass -1.5 nats, timeout +4.0) and
`audit.py` (PreventionRecord with CNAP / CERT-In / IT Rules 2026 fields and a
FORBIDDEN_KEYS check). Wired into `serving/app.py`. The compliance fields are
**fields to populate, not integrations** -- whether a call is reportable is a
legal determination the deploying institution makes.

### B. Multilingual -- measured Sep 15, and this is now the BIGGEST gap
No longer absent, but the measurement is bad news rather than good.

* **Mandarin** (RTCFake zh, never trained on): Branch A retrained on RTCFake-en
  scores **33.41% EER**, against a best-trivial-cue baseline of 41.43% measured
  on the same corpus. Informative, far from usable. The ASVspoof-trained
  baseline was 79.08% -- *inverted*, i.e. worse than guessing.
* **Hindi** (IndicSynth, 50,044 rows, 2 generators, 101 target speakers): the
  corpus publishes **spoof only**, so no EER exists for it. At a 5%
  false-positive budget set on real bona fide, the ASVspoof head catches
  **5.3%** of Hindi attacks (chance) and the RTCFake head **13.9%**. Neither is
  usable.

Unblocking a real Hindi EER needs an Indian-language **bona fide** corpus paired
to IndicSynth's own source recordings. Borrowing bona fide from an unrelated
corpus would let a detector separate the two by recording conditions rather than
by synthesis -- so it is refused, not caveated.

### C. Requested deliverables -- whitepaper and bibliography DONE
`docs/VERICALL_V4.md` (11 sections, ~640 lines) and `docs/RESEARCH.md`
(annotated bibliography, 14 sources) are written and committed. Every source in
the bibliography was verified against origin during the project -- v3 contained
a fabricated one ("Pindrop's OCCAS pattern", replaced with Pindrop Pulse), so
the bibliography also records where a source does NOT support the use it was
put to (SafeEar's shuffle layer, the patent's binomial model).

`docs/DEMO_SCRIPT.md` is written and **every command in it was executed**, not
recalled -- launch, the health check, the pre-flight gate, and the mis-paired
calibration case. Two commands were wrong on the first draft (the pod has no
`curl`; `$PWD` does not resolve to the project there) and were corrected by
running them.

Still 0 bytes: `THREAT_MODEL.md`, `EVALUATION.md`. Never explicitly requested;
the whitepaper covers the threat surface in section 1 and the methodology in
section 8.

### D. Measurement gaps carried forward
- **eta is now measured for all three** scoring branches. Only A's was real
  (0.207, 0.928, 0.565 on 800 calls) -- and the three differ by 4.5x between
  them, so a single global eta would have been wrong for two of them.
- ~~**Replay (E13) never tested**; EchoFake never fetched.~~ **Done** --
  EchoFake is integrated, `eval_replay` / `eval_synth_echo` share their bona
  fide, and replay evades by 1.2-1.5x (best 33.68%, §21).
- **RTCFake is gated**, so real-transmission testing is blocked. Section 10's
  paired simulator was the substitute and answered the narrower question.
- ~~**In-the-Wild** was fetched once, lost to a pod wipe, never re-evaluated.~~
  **Done** -- integrated and measured, 22.48% EER (§24). Cross-corpus
  generalisation no longer rests on WaveFake alone.
- ~~**N=4 is a floor**~~ -- superseded: **N=64** end to end over real
  WebSockets, N=160 server-side (§9.12, and the status table above).

*(This subsection had gone stale against section 2 of this same file; corrected
in the 2026-09-22 audit.)*

### E. Section 9 production-readiness: partly built, and the split matters
Corrected in the 2026-09-22 audit -- this entry used to say none of it was
implemented, which stopped being true on Sep 16.

- **Built and tested:** RawBoost (`vericall/data/rawboost.py`, exposed as
  `extract_features.py --rawboost`), domain-adversarial training
  (`vericall/train/dann.py`), OC-Softmax (`vericall/train/ocsoftmax.py`),
  per-group fairness *reporting* (`scripts/eval_fairness.py`,
  `docs/fairness.json`).
- **Built but NOT in the shipped profile:** none of the above produced the
  served checkpoints. `calib/provenance.json` records the deployment as
  `branch_A_seed0.pt` / `branch_A2_seed0.pt` -- the plain ASVspoof baseline --
  because §20 and §24 measured every one of these interventions as neutral or
  worse out of domain. That is a deliberate choice; it is not a gap, but it
  must not be reported as "we ship adversarial training".
- **Still not built:** continual learning with a replay buffer (FM-6), the
  Proteus-style red-team release gate (FM-3), and per-group threshold
  *fitting* (still blocked on speaker diversity, C13).

## 9. Conventions

- Tests: `./scripts/pod.sh py '-m pytest tests/ -q -p no:cacheprovider'`
- Long jobs: `setsid nohup … > logs/x.log 2>&1 &`
- Findings go in `docs/CONFLICTS_RESOLVED.md` with a test that pins them
- Standing audits go in `docs/AUDIT_<date>.md`. The latest is
  **`docs/AUDIT_2026-09-22.md`** — outside-eye review; its three live
  findings are the untrimmed live-ingest path (F2), the unwired
  `DomainGuard` (F3), and In-the-Wild having become a selection set (F4)
- Comments explain *why*, especially where the obvious approach is wrong
- `_quarantine/` holds superseded duplicates; nothing there is imported
