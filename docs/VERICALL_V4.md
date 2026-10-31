# VeriCall: streaming detection and prevention of voice-cloning impersonation

**Technical whitepaper — v4**
Smart India Hackathon 2026 · PS 26104 (AICTE, Software)
Measured on ASVspoof 2019 LA, WaveFake, and RTCFake. September 2026.

---

## Abstract

VeriCall detects synthetic-speech impersonation **during** a live call and acts
on it before money moves. Four detection branches score overlapping audio
windows; each is Platt-calibrated to a log-likelihood ratio in nats; a fusion
layer accumulates those LLRs over time with per-branch measured effective-sample-size
corrections and time discounting, so evidence strengthens with call duration
instead of being re-decided every window. A prevention layer converts the
accumulated LLR into an action chosen by transaction context and cost.

The system reaches **2.48% EER in-domain** and holds **0/40 false positives with
40/40 detections** end-to-end on real audio, catching a mid-call speaker swap
**2.3 s after the splice** — the attack a single-shot classifier cannot express
at all. It serves **4 concurrent calls** on one 18 GB MIG slice within every
declared latency SLO.

**It does not generalise across corpora, and that is the most important result
here.** On RTCFake — real speech transmitted through live RTC platforms — the
ASVspoof-trained Branch A *inverts*: 79.08% EER on Mandarin, systematically
anti-correlated rather than merely uninformative. Retraining on RTCFake removes
the inversion completely (Mandarin 79.08% → 9.51%, on a language never trained
on) and proves the diagnosis was a data problem rather than an architectural one
— but the retrained model is then at chance on ASVspoof and WaveFake, and
training on both corpora destroys WaveFake outright. **No configuration we
tested is good on more than one corpus at a time.** On Hindi, at a 5%
false-positive budget, the best model catches 13.9% of attacks. Retraining the
branches and refitting the calibration does make the fusion layer earn its place
again (§7.3) — in the language it was refitted on.

Sections 7 and 10 report this at the same prominence as the headline numbers. A
voice-fraud system whose published accuracy does not survive a change of channel,
corpus or language is a specific and common failure, and the measurement that
reveals it is worth more than the number that hides it. Section 8.5 and §3.2
apply the same standard inward: two of this project's own metrics were wrong —
including a headline η — and both are retracted here rather than quietly
restated.

---

## 1. The problem, stated precisely

Voice cloning has crossed the threshold where three seconds of a target's speech
— from a voicemail greeting, a conference recording, a social-media clip —
produces real-time synthesis convincing to a listener under telephone bandwidth.
The documented losses are large and the attack requires no technical skill.

Most published detection work answers a different question than the one a bank
actually faces. It asks: *given this recording, is it synthetic?* The
operational question is: *given that this call is in progress and the caller is
about to authorise a transfer, should we act, and how?* Four differences follow
from that, and they drive the entire architecture:

1. **Evidence arrives over time.** A decision at 0.5 s and a decision at 20 s
   should not be the same decision, and the system should say *why* its
   confidence changed.
2. **The attacker can switch mid-call.** A genuine caller may be
   social-engineered off the line, or a call may begin real and continue
   synthetic. A model that classifies a whole utterance cannot represent this
   event; it can only average it away.
3. **The channel is not the lab.** G.711 at 8 kHz, packet loss, noise
   suppression, echo cancellation. Every one of these removes exactly the
   fine spectral and phase detail detectors rely on.
4. **Detection alone changes nothing.** A score that does not become an action
   is a dashboard, not a control.

**Scope.** VeriCall targets synthetic-speech impersonation on live voice calls.
It is not a caller-ID spoofing defence (CNAP is), not a replay defence (untested
— see §9), and not a general deepfake detector.

---

## 2. Architecture

```
  audio ──▶ VAD + dual-rate chunker (global tick alignment)
              │
              ├─ fast lane 1.0 s window / 0.5 s hop ─▶ Branch A   (SSL spectral)
              │                                    └─▶ Branch A2  (phase/recon)
              └─ slow lane 3.0 s window / 1.0 s hop ─▶ Branch B   (prosody)
                                                   └─▶ Branch C   (identity)
              │
              ▼
        per-branch Platt calibration  (per channel bucket)
              │
              ▼
        fusion: discounted LLR accumulation in nats
                + η deflation + per-subset recalibration
                + CUSUM change-point + speaker-turn detection
                + domain guard
              │
              ▼
        tiered risk view ──▶ prevention policy ──▶ action + audit record
```

### 2.1 Branches

| | What it sees | Independent of | Trainable params |
|---|---|---|---:|
| **A** | Frozen WavLM-base truncated at layer 5 → attention-weighted graph readout | — | 0.89 M (94.4 M frozen) |
| **A2** | Phase / reconstruction residual | A's spectral view | small |
| **B** | Prosody: F0 trajectory, jitter, pause structure, voiced ratio; one-class Deep-SVDD on bona fide only | having seen the attack | small |
| **C** | ECAPA-TDNN speaker embedding + AS-Norm | the spoof question entirely | 0 (frozen) |

Branch A's head is **AASIST-inspired, not AASIST**. It keeps the two-view
attention-weighted readout — spoofing artifacts are local, and uniform pooling
dilutes precisely the frames that carry them — but replaces heterogeneous
stack-node attention with concatenated readouts over a latent-node graph. The
substitution is forced: SSL features have no frequency axis to build a spectral
graph over.

**Branch C never produces a spoof score.** It answers "is this the enrolled
person?", which is a different question, and a cloned voice can pass speaker
verification by construction — that is what cloning *is*. Its outputs feed
identity conflict and speaker-turn detection only. This is enforced structurally:
C is excluded from the fusion subset, and a regression test asserts the set of
subsets observed end-to-end is `{A, A+A2, A+A2+B}` and never contains C.

**Branch B carries zero weight, on measurement.** §6 explains why, and §4.4
explains the bug that made zero-weighting harder than setting a number to zero.

### 2.2 Why LLR accumulation, and not a fusion network

The obvious design is a small MLP over branch scores. It does not work here, for
a structural reason rather than an accuracy one: **the branches do not all fire
at the same rate, and some are absent.** The fast lane emits at 2 Hz, the slow
lane at 1 Hz, Branch B may be unavailable on a short call, Branch C requires
enrollment. A fixed-width network needs a fixed-width input vector, so every
absent branch must be imputed — and imputing a spoof score is inventing evidence.

Calibrated LLRs compose by addition, so an absent branch contributes zero by
definition rather than by imputation. The cost is that each branch must be
genuinely calibrated, which is a real engineering burden (§3) and the reason
Cllr, not EER, is the metric that governs the fusion layer.

The accumulator, per branch *k*:

```
T_eff  ←  min(T_cap,  T_eff·e^(−Δt/τ)  +  η_k · hop_s)
S_k    ←  S_k·e^(−Δt/τ)  +  η_k · hop_s · (llr_k / window_s)
S      =  Σ_k w_k S_k   ,  then recalibrated by the subset table for K
```

Three details that were each a bug before they were a design:

- **`window_s` normalises, `hop_s` credits.** An earlier version used one
  duration for both. The algebra then cancels — `S = (N·hop)·(llr/hop) = N·llr`
  — so overlapping windows received no correction at all despite the code
  appearing to apply one.
- **η deflates the accumulated *time*, not the weights.** Deflating weights is a
  mathematical no-op: a weighted mean and its variance are both scale-invariant
  in the weights.
- **A stale branch stops contributing; it is not held at its last value.**
  Hold-last-value silently converts "we have not looked in four seconds" into
  "we looked and it was fine."

### 2.3 Time constants and tiers

`τ_fast = 1.5 s`, `τ_slow = 4.0 s`, `T_cap = 10 s`. The cap matters: without it a
long genuine call accumulates unbounded confidence in its own genuineness, and
a mid-call swap then has to overcome minutes of banked evidence before it can
move the score. Capping credited audio at 10 s bounds how much prior evidence a
swap must overcome; measured time-to-detection after a splice is **2.3 s**
(§4). The cap's effect on *long* calls specifically is an argument from the
accumulator's algebra, not a separate measurement — the test calls are tens of
seconds, not minutes.

Tier transitions use hysteresis: an enter/exit gap, a consecutive-crossing
requirement, a dwell time, and one-step-only de-escalation. Escalation is fast,
de-escalation is slow, and the system cannot oscillate at a boundary.

---

## 3. Calibration is the load-bearing component

Every threshold in this system assumes branch outputs are LLRs in nats. If they
are not, the fusion arithmetic is meaningless even when the ranking is perfect.
So the governing metric is **Cllr**, and the reported quantity is the
*calibration loss* `Cllr − min-Cllr` — the part of the cost attributable to bad
probabilities rather than bad ranking.

| | n | EER | Cllr | min-Cllr | calibration loss |
|---|---:|---:|---:|---:|---:|
| uncalibrated | 17,764 | 2.34% | 0.174 | 0.092 | **0.082** |
| per-channel fitted | 17,764 | — | 0.103 | 0.092 | **0.011** |

Calibration is fitted **per channel bucket**, because G.711 more than doubles
Branch A's EER (1.52% → 3.30%) *and* requires a different slope (0.510 vs
0.411). A single pooled mapping is mis-calibrated on both. Branch A2's slope
spread is wider still — 1.67× — which is one of two independent lines of
evidence that A2 is the channel-sensitive component (§6).

Calibration is fitted on a dedicated `calib` fold, disjoint from train and dev.
It is never fitted on anything a reported number is measured on.

### 3.1 Per-subset recalibration

Summing calibrated LLRs is only exactly right if the branches are independent.
They are not — A and A2 see the same audio. So a per-subset table `(α_K, s_K)`
is fitted for each branch-availability subset K, absorbing the residual
correlation for that combination:

| Subset | α | scale | EER |
|---|---:|---:|---:|
| A | −0.007 | 0.991 | 2.52% |
| **A+A2** | 0.166 | 0.960 | **1.46%** |
| A+A2+B | 0.289 | 0.957 | 1.46% |
| A2 | 0.000 | 2.222 | 8.08% |

This table settles two questions that were previously argued rather than
measured. **A2 earns its place**: 2.52% → 1.46%, a 42% relative reduction from
one branch, which is direct evidence that phase is not redundant with SSL
spectral features. **B contributes nothing once A is present**: 1.46% → 1.46%,
unchanged to two decimals.

### 3.2 Effective sample size, and a measurement that was itself wrong

Overlapping windows are not independent observations. The design assumed roughly
√2 inflation from window overlap. Measured per branch over 800 streamed calls,
each branch scored on the lane it actually runs on:

| Branch | Lane | η | inflation | lag-1 … lag-8 |
|---|---|---:|---:|---|
| A | fast 1.0 s / 0.5 s | **0.207** | 4.8× | +0.38 +0.28 +0.27 +0.27 +0.26 +0.25 +0.23 +0.23 |
| A2 | fast 1.0 s / 0.5 s | **0.928** | 1.1× | +0.03 −0.04 −0.03 −0.03 −0.03 −0.03 −0.03 −0.02 |
| B | slow 3.0 s / 1.0 s | **0.565** | 1.8× | +0.34 +0.03 −0.15 −0.10 −0.10 −0.08 −0.07 −0.07 |

Branch A's correlation does not decay to zero across eight lags. That is **not**
a window-overlap signature — a 1 s window at a 0.5 s hop shares no samples at
all with its neighbour twice removed. It is **call-level consistency**: one
speaker, one channel, one generator makes an entire call close to a single
observation. About 4.8 of A's windows count as one independent look.

**The three branches do not move together**, which a single global η would have
hidden. A2 is nearly independent — phase residual is a local property that
decorrelates within one hop — so crediting it at the old 0.6 default discarded
over a third of its evidence. Branch B's profile reproduces its own geometry:
at a 3 s window and 1 s hop, adjacent windows share 2/3 of their audio, lag-2
shares 1/3, lag-3 shares none, and the measured +0.34 / +0.03 / −0.15 tracks
that. Numbers reproducing the overlap fractions they should is the best evidence
available that the measurement reads signal.

**An earlier version of this section reported η_A = 0.080 and 12.5× inflation.
That was wrong, and the reason is worth more than the number.** `push()` on the
dual-rate chunker appended its whole input to the ring buffer and then cut every
due window from the *end* of that buffer, so a push spanning several ticks
emitted windows whose timestamps advanced while their audio did not. η was
measured by pushing whole 14 s calls at once: 40/40 sequences had byte-identical
windows, the branch scores differed only by floating-point non-determinism
(median std 1.8e-15), and the reported autocorrelation was of that noise.

Only the batch measurement was affected — the gateway and the end-to-end runner
push well under one hop — which is precisely why it survived. There was one
chunker test and it never inspected window contents. The fix is in the chunker
rather than its callers, and the strongest of the five new tests asserts that a
single bulk push produces byte-identical output to many small pushes.

Adopting the corrected values changes no end-to-end decision: still 0/40 false
positives and 40/40 detections on the four-branch run.

Caveats stated rather than buried: the estimate truncates at lag 8 while Branch
A's is still positive, so 0.207 is an **upper bound** on A's independence; and
the synthetic calls are concatenated clips, so splice points add noise a real
call would not have.

### 3.3 Choosing the bucket at runtime

Per-channel calibration is worthless if nothing selects the channel. Three
signal-processing features — energy above 4 kHz, zero-variance 20 ms frames
(packet loss, which mu-law turns into a *constant* frame rather than a silent
one), and p5-over-median frame RMS — with thresholds fitted on labelled audio
and checked on held-out clips. **98.8% per call**; clean calls 100%.

What this buys, measured on held-out data with each condition given its own
subset calibration:

| | Branch A calib loss | Branch A2 calib loss | fused Cllr | fused EER |
|---|---:|---:|---:|---:|
| one bucket for every call | 0.020 | 0.185 | 0.121 | 2.85% |
| **detected** | **0.013** | **0.011** | **0.111** | **2.79%** |
| oracle (true channel) | 0.013 | 0.012 | 0.110 | 2.79% |

Branch A2's calibration loss falls **94%**, and detection captures **97.4%** of
what a perfect oracle would deliver — so the detector is not the limiting factor
and more accuracy there would buy almost nothing.

**Fixing that exposed an error it had been hiding.** The fusion weights
(A = 1.0, A2 = 0.45) were set by hand and never fitted. That was harmless only
because two errors cancelled: a weight that under-stated A2 against a mapping
that over-stated it. Fitted properly — logistic regression of the label on the
branch LLRs, whose coefficients *are* the weights — A2's weight nearly doubles to
0.87, fused EER goes 3.04% → 2.79%, calibration loss halves, and fusion finally
beats its own best single branch (2.79% vs 3.62% for Branch A alone).

Weights therefore live in the calibration directory rather than in the config:
they are only valid against the calibrations they were fitted for, which is the
head/calibration pairing rule (§9) one level up.

### 3.4 Where decision theory stops being useful

The cost-optimal threshold is `θ* = log[C_fa(1−π) / (C_miss·π)]`. Worked through
realistic banking priors and costs, θ* lands **below the reliable-calibration
floor in every high-value context** — that is, below the ±0.5–1 nat band where a
fitted Platt mapping means anything. Thresholds are therefore clamped to
[1.0, 8.0] nats and the clamp is documented as such.

The honest reading: **the real constraint on alerting is analyst capacity, not
decision theory.** Presenting a fake-precise θ* would be worse than saying so.

---

## 4. Streaming behaviour and the mid-call swap

Real ASVspoof audio streamed in 0.5 s increments through the full four-branch
pipeline, fitted calibration, and the real fusion engine. 40 calls per scenario,
no tuning:

| Scenario | alerted | median time-to-alert | median S_lo | change pts/call |
|---|---:|---:|---:|---:|
| bona fide | **0/40** | — | −50.23 | 0.00 |
| spoof from t=0 | **40/40** | 2.5 s | +16.92 | 0.00 |
| mid-call swap | **40/40** | **2.3 s after splice** | +16.95 | 1.43 |

The change-point column is the interesting one. CUSUM fires on 1.43 calls on
average in the swap scenario, 0.00 in the spoof-from-t0 scenario, and 0.00 on
genuine calls. That asymmetry is exactly right and is a correctness check, not a
performance number: **a call that was synthetic from the first window contains
no change to detect.** A change-point detector that fired there would be
responding to noise.

These are 40-call samples on one corpus. The honest statement is "0/40", not
"0% FPR" — the upper 95% bound on a 0/40 observation is about 7%.

### 4.1 A bug that passed its own test

An earlier version suppressed alerts entirely when a quality gate failed,
producing 0% TPR despite perfect score separation. The unit test asserted
`tier <= 1`, which passes on tier 0 — the bug itself. Fixed by clamping to the
highest gate-satisfying tier rather than suppressing, and the test rewritten to
assert the tier the evidence justifies.

---

## 5. Prevention: turning a score into an action

Detection without action is a dashboard. The prevention layer maps
(risk in nats, confidence, transaction context) to one of six actions:
`ALLOW`, `SILENT_FLAG`, `CHALLENGE`, `STEP_UP`, `HOLD`, `FRAUD_DESK`.

The context matters more than the score does. Seven contexts are defined, each
carrying a prior, a miss cost, and — critically — **reversibility**. A statement
request and a new-payee addition at the same risk level warrant different
responses because one can be undone and the other cannot. Irreversible,
high-value contexts escalate at lower risk than the raw threshold implies.

**Challenge-response liveness.** An in-band challenge uses `secrets` for
unpredictability and scores the response by timing: a human ceiling of 2.5 s and
a suspicious floor of 0.25 s. The risk adjustment is deliberately **asymmetric**
— passing subtracts 1.5 nats, timing out adds 4.0. A clone that fails a
challenge is strong evidence; a clone that passes one is weak evidence of
innocence, because passing is exactly what a well-built real-time system would
do. Symmetric adjustment would let an attacker buy down their score.

**Audit.** Every decision writes a `PreventionRecord` with compliance fields for
CNAP mismatch, CERT-In incident handling, and the IT Amendment Rules 2026
36-hour takedown clock. These are **fields to populate, not integrations** —
whether a given call is reportable is a legal determination the deploying
institution makes, and the record stores the clock start rather than claiming
compliance. A `FORBIDDEN_KEYS` check enforces that no raw audio or biometric
template enters the log.

---

## 6. Robustness, branch by branch

Paired design: identical utterances, identical speakers, only the channel
changes. Dev fold, 5,321 windows, CIs bootstrapped over utterances.

| Channel | Branch A | Branch A2 | Branch B |
|---|---:|---:|---:|
| clean | **1.39%** | 2.86% | 43.36% |
| G.711 µ-law | 2.83% (2.04×) | 10.79% (3.77×) | 40.71% (**0.94×**) |
| noisy line | 5.84% (4.21×) | 30.15% (**10.54×**) | 43.40% (1.00×) |

**The v3 blueprint's mechanism was right and its conclusion was wrong.** It
claimed prosody is the main defence against compression because prosody survives
codecs. Prosody *does* survive — Branch B moves 0.94× and 1.00×, flat within
noise, while spectral detail degrades 2–4× and phase 4–10×. The physics is
correct. But Branch A is better than Branch B at **every severity tested**,
5.84% vs 43.40% even on the harshest channel. A's EER never crosses B's.

Robustness of a near-useless signal does not make it useful. The blueprint made
a *relative* argument and stated an *absolute* conclusion.

**New finding: A2 is the channel-fragile branch.** Unflagged by any prior
analysis. A2 earns its place on clean audio (§3.1) but degrades 10.54× on a
noisy line, 2.86% → 30.15%. That drove the per-channel A2 calibration in §3 —
two independent lines of evidence (EER degradation, slope spread) for the same
conclusion.

### 6.1 Branch B, and why it is still in the codebase at zero weight

Branch B was tested in every condition available and reached chance in all of
them:

- **~39% supervised ceiling** on clean ASVspoof prosody features. A one-class
  failure is ambiguous — optimisation problem, or no signal? Supervised learning
  on the same features is strictly easier, so it bounds what any one-class model
  could reach. Logistic regression: 39.39%. Gradient boosting: 41.04%. That is a
  *ceiling*, about 11 pp from chance.
- **Deep-SVDD collapsed**: embedding std 0.0057, loss 0.023 (which looks like
  success), dev EER 49.8% (which is chance). Loss alone cannot distinguish
  collapse from success — it looks *better*. The prescribed bias-free fix
  improved spread only to 0.0116, still collapsed. A `COLLAPSE_STD = 0.05` guard
  now aborts the run rather than shipping a confident constant.
- **No fusion contribution** (§3.1), **no compression advantage in absolute
  terms** (§6), **42–53% EER on real RTC transmission** including cross-lingual
  (§7).

The individual feature EERs are physically sensible — synthetic speech has lower
pitch variability and shorter pauses — which is evidence the features measure
the right thing, weakly. The code stays because the front end is sound and a
future corpus may vindicate it. The weight is zero because nothing has measured
it contributing.

**A three-branch system that knows why it is three branches is a stronger
position than a four-branch one carrying a dead limb.**

---

## 7. Cross-domain collapse, and what retraining did and did not fix

RTCFake pairs each utterance with itself — the same speech offline (clean) and
online (after passing through a real RTC platform). Same speaker, same content,
same label, only the channel differs. It also splits en/zh, and since every
model here only ever saw English, Mandarin is genuine cross-lingual
generalisation. 1,200 paired utterances per language.

| Lang | Branch | offline | online | degradation |
|---|---|---:|---:|---:|
| EN | A | 50.21% | 52.57% | 1.05× |
| EN | **A2** | **19.28%** | 39.99% | 2.07× |
| EN | B | 42.44% | 45.08% | 1.06× |
| ZH | A | **79.08%** | 76.60% | 0.97× |
| ZH | **A2** | **28.72%** | 50.71% | 1.77× |
| ZH | B | 48.23% | 53.05% | 1.10× |

### Finding 1 — Branch A does not fail, it inverts

79.08% EER is not "uninformative". Above 50% means systematically
**anti-correlated**: flipped, it would score 20.92%. Branch A learned something
real and corpus-specific that points the *wrong way* under domain shift, ranking
RTCFake spoofs as more genuine than RTCFake bona fide.

Offered as hypothesis, not conclusion: trained on ASVspoof LA, Branch A learned
characteristics of *that corpus's bona fide recordings*. Mandarin bona fide does
not share them, while RTCFake's spoofs — from different generators — may resemble
ASVspoof bona fide more closely. This is the shortcut-learning failure mode
(§8.1) surfacing as inversion rather than as noise.

### Finding 2 — the branch hierarchy reverses

On ASVspoof, A is strong (2.52%) and A2 weaker (8.08%). On RTCFake it reverses
completely: **A2 is the only branch carrying transferable signal** while A is at
or below chance. This sits awkwardly beside §6, where A2 was the *most*
channel-fragile branch. The reconciliation: **A2 is fragile to channel but
robust to corpus; A is the reverse.** Those are different axes, and the
architecture had only ever measured one of them.

### Finding 3 — the fitted fusion actively hurts

| | fused | best single branch | |
|---|---:|---:|---|
| EN offline | 39.87% | **19.28%** | 2.1× worse |
| ZH offline | **72.70%** | **28.72%** | 2.5× worse |

Fusion is worse than its own best branch in **all four conditions**. The reason
is mechanical: the weights were fitted where A is strongest, so they put the most
weight on the branch that is inverted here. **Fusion faithfully amplifies the
worst signal available.**

This is a more serious limitation than any single branch's EER, because it means
the fusion layer — otherwise this system's strongest engineering — cannot be
fitted once and deployed.

### 7.1 Retrained on RTCFake: the inversion is gone

The diagnosis above was that this is a training-data problem, not an
architectural one. That was tested. Branch A was retrained on **RTCFake English
only** — Mandarin dropped from training entirely, so it stays a genuine
cross-lingual test — with no architectural change whatsoever: same frozen WavLM
layer 5, same attention head, same 0.89 M trainable parameters.

Same paired protocol, same crops, same utterances, only the checkpoint differs:

| Branch A trained on | EN offline | EN online | ZH offline | ZH online |
|---|---:|---:|---:|---:|
| ASVspoof (above) | 50.21% | 52.57% | **79.08%** | **76.60%** |
| **RTCFake-en** | **14.23%** | **16.25%** | **9.51%** | **13.57%** |
| ASVspoof + RTCFake-en | 17.34% | 18.59% | 11.67% | 18.17% |

**Mandarin goes from 79.08% to 9.51%, on a language never trained on.** Every
cell is now far below 50%: the inversion is not reduced, it is gone. And it is
not a trivial cue — the best duration-only baseline on RTCFake Mandarin is
41.43%, measured on that same corpus, which carries no silence artifact at all.

### 7.2 But the problem moved rather than dissolved

| Branch A trained on | ASVspoof eval | WaveFake unseen | RTCFake en | RTCFake zh |
|---|---:|---:|---:|---:|
| ASVspoof | **11.35%** | **30.08%** | 45.57% | 55.34% ⚠ |
| RTCFake-en | 36.54% | 49.36% | **22.93%** | **33.34%** |
| ASVspoof + RTCFake-en | **9.12%** | 49.89% | 25.03% | 37.21% |

The RTCFake model is at **chance on ASVspoof and WaveFake**. It did not learn a
general synthetic-speech detector; it learned an RTCFake-specific one, exactly
as the ASVspoof model had learned an ASVspoof-specific one. **We swapped one
corpus-specific detector for another.** What transfers is across *language within
a corpus*, not across *corpus* — and only the second is what deployment needs.

**Mixing trades, it does not combine.** The mixed model posts the best in-domain
number in the project (9.12%, beating the ASVspoof-only baseline) and is worse
than the RTCFake-only model on both RTCFake folds. It also destroys WaveFake
outright — 30.08% → 49.89%, chance — even though ASVspoof rows are 64% of the
mixed training set. That is not proportional dilution, it is destruction: with
RTC-transmitted audio present, channel becomes an available discriminator, and
WaveFake is clean studio resynthesis with no channel to read. A naive
concatenation of corpora lets the model take the cue that is cheapest across the
mixture rather than the one that generalises.

**So "just add more data" is not the generalisation strategy.** Balanced
sampling, domain-adversarial training, or per-domain heads are the next things to
try, and none is tested here.

### 7.3 Fusion: the diagnosis, and acting on it

With the retrained Branch A but the **stale** ASVspoof-fitted weights and subset
table, fusion helped in one of four conditions and hurt in three — worst on ZH
offline, 17.73% fused against 9.51% for Branch A alone. The diagnosis was that
the weights encode a branch hierarchy retraining changed.

That diagnosis implies a fix, so it was tested. Branch A2 retrained on RTCFake
(Mandarin still held out), Platt mappings refitted, subset table refitted on the
RTCFake `calib` fold. Measured on **held-out** eval folds:

| fold | A | A2 | A+A2 fused | vs best single |
|---|---:|---:|---:|---|
| eval_rtc_en | 22.93% | 20.55% | **16.07%** [15.0–17.6] | **helps, 1.28×** |
| eval_rtc_zh | 33.34% | 29.52% | 29.89% [28.9–31.0] | tie (1.01×) |

**Fusion now helps decisively in-language and is indistinguishable from its best
branch cross-language.** The Mandarin fused confidence interval contains the best
single branch's 29.52%, so that cell is a tie rather than a loss.

Two things worth reading off this. Retraining A2 barely changed its RTCFake
numbers — the ASVspoof-trained A2 was already at 19.28% / 28.72% — which
confirms §7.2's reading that A2 is the corpus-robust branch. A2 was never the
problem; everything downstream of it was.

And the residual is **the same failure one level up**. English is where the
calibration was fitted; Mandarin is not. The branches transfer to Mandarin, but
the combination rule gains nothing there, because the per-subset (α, scale) is
itself a fitted object fitted on one language. Every fitted component of this
system — branch heads, Platt mappings, subset tables, weights — carries the
domain it was fitted on, and there is no layer at which that stops being true.

The practical consequence, stated as an implication rather than a built feature:
calibration is cheap in a way retraining is not. The whole subset table is a
couple of kB of JSON and refits in seconds from cached branch outputs with no
GPU. A deployment procedure that refits calibration on a sample of the tenant's
own channel and language is the obvious design, and it is not implemented.

Branch C was not retrained; it produces no spoof score, so it does not enter
this table.

### 7.4 The domain guard

The serious part of §7 was not a bad number. It was that fusion reported a
**confident** result while scoring worse than its own best branch, and nothing
detected it. A production system that fails this way fails silently.

`domain_guard.py` watches the two signatures visible in that data:
**score drift** (branch means far from the distribution the Platt parameters
were fitted on — the mapping is extrapolating) and **branch disagreement** (A and
A2 were fitted as correlated, so sustained opposite signs means one has
inverted). Validated on real scores, not on the synthetic signatures it was
designed from:

| Data | A mean LLR | A2 mean LLR | Guard |
|---|---:|---:|---|
| ASVspoof dev (in-domain) | −5.10 | −8.23 | **quiet** |
| RTCFake en | −4.58 | **+1.24** | **suspect** — disagreement |
| RTCFake zh | **+3.08** | +0.20 | **suspect** — drift, 6.6 sd |

The detectors are complementary and each caught a case the other missed. On
English, A's mean stayed inside its fitted distribution so drift did not fire,
but the sign split against A2 did. On Mandarin, A inverted hard enough — +3.08
nats on *bona fide* audio, confidently calling genuine Mandarin speech synthetic
— that drift caught it outright.

When tripped it latches (later agreement does not restore trust in evidence
already banked), forces confidence to LOW, halves the confidence score, and
surfaces the reason on the `RiskView`. It is opt-in: a state with no reference
distribution behaves exactly as before, because a guard with a fabricated
reference fires for reasons unrelated to the data.

**What it does not do:** detect domain shift in general. It detects the specific
failure that actually occurred — which is worth more than a general claim that
does not fire.

---

## 8. Methodology

### 8.1 The shortcut that invalidates naive results

ASVspoof 2019 LA bona fide clips have systematically longer leading and trailing
silence than spoofed ones. Müller et al. (2021) showed a model trained on
leading-silence duration *alone* reaches 85% accuracy / 15.1% EER with no speech
content whatsoever.

Consequences enforced in code: VAD trimming before windowing is mandatory, and a
**duration-only control baseline** runs on our own split — measured at **36.4%
EER**. Branch A reaches 2.48% in-domain, so it clearly has not learned a
stopwatch. The sobering corollary: at ~30% cross-corpus, the margin over that
trivial baseline is thin.

### 8.2 Leakage control

Fold assignment is by **source utterance**, not by row. In WaveFake a source
utterance and all its vocoder resyntheses must share a fold, because they are the
same sentence — a split that separates them is a memorisation test wearing an
evaluation's clothes. This is enforced by `tests/test_splits.py`, not by
discipline.

Bootstrap confidence intervals resample **utterances, not rows**. Resampling
rows treats several crops of one clip as independent and reports an interval
several times too narrow.

Calibration is fitted on a `calib` fold disjoint from both train and dev.
Branch C's AS-Norm cohort is drawn from bona fide **only** — identity calibration
must never see synthetic speech, whose speaker label is ambiguous by
construction.

### 8.3 Overfitting is visible in flight, and early stopping selects for it

Over the same epochs, dev EER improves 2.97% → 2.36% while WaveFake EER *worsens*
27.7% → 30.1%. The methodological consequence is sharp: **early stopping on dev
EER selects the most domain-overfit checkpoint.** Stopping on dev EER rather than
dev loss is right as far as it goes, but dev is in-domain and cannot see this. A
cross-domain selection signal is the correct fix and is not yet implemented.

### 8.4 The vocoder gap is not the problem

Holding out entire vocoder families (HiFi-GAN, FB-MelGAN, TTS) costs **nothing**
— unseen 30.29% is marginally *better* than seen 31.43%. What destroys the
detector is changing corpus: 2.48% → ~30%, a 12× degradation.

This inverts the assumption the original design was built on. Effort spent
anticipating new vocoders is close to wasted; effort spent on channel and corpus
robustness is where the accuracy is. It also means **the number to quote is ~30%,
not 2.48%.**

### 8.5 Two metric bugs worth recording

**EER with ties.** A constant score returned 1.0 rather than 0.5, and in one
documented case the implementation reported 95.67% EER for a feature whose class
means differed by 0.8 ms. Fixed with tie collapsing and interpolation. A metric
that reports a dramatic number on a non-signal will be believed.

**A verdict that passed while measuring nothing.** An early `eval_rtcfake.py`
concluded "A crosses B, so the compression claim holds." Technically true,
substantively wrong: **A crossed B by collapsing**, not by B being good. The
verdict now requires B to be usable in absolute terms (<40% EER) before crediting
it. Same class of error as declaring prosody discriminative at 39.4% EER because
the bar was set at 40%.

---

## 9. Serving, and what the system actually costs

Criterion declared **before** the run: all calls complete, `chunk_inference`
p99 ≤ 250 ms, `time_to_first_score` p99 ≤ 1.6 s, server keeps up with real time.
One `mig-1g.18gb` slice, co-tenanted with another project.

| N | wall (12 s audio) | first_score p50 | chunk p99 | verdict |
|---:|---:|---:|---:|---|
| 1 | 13.0 s | 1.11 s | 104.5 ms | pass |
| **4** | **13.1 s** | **1.36 s** | **103.9 ms** | **pass** |
| 8 | 18.1 s | 1.36 s | 101.4 ms | fails real time |
| 16 | 18.2 s | 2.71 s | 101.6 ms | fails |

**N = 4** is the largest configuration meeting every SLO simultaneously.

**The bottleneck is not the GPU.** `chunk_inference` p99 is flat at ~102–105 ms
from N=1 to N=16 — it does not degrade with load at all. What degrades is wall
clock. The constraint is CPU-side: feature extraction, VAD, ring buffers,
WebSocket handling. The ~400-call figure in `BENCHMARKS.md` is a **GPU ceiling,
not system capacity**, and that caveat is now confirmed rather than merely
stated.

**Honest confound:** the load generator runs on the same pod, competing for the
same 16 cores at N=16. True capacity with the generator off-box is higher than 4
by an unmeasured margin. **N=4 is a floor, not a ceiling.**

**Cold start was a real defect.** The first sweep showed p99 **830 ms at N=1** —
worse than at N=16 — because with only 22 windows, one un-warmed CUDA forward
dominates the percentile. That would have landed on the first call of a live
demo. The gateway now warms each branch at startup and excludes the warmup from
the histogram; p99 is a stable ~104 ms at every N.

**Cross-call batching is implemented and disabled by default.** A/B measurement:
direct execution p99 101 ms; batching under an async lock on the event loop 780
ms; concurrent flushes contending for one MIG slice 973 ms. Batching a
synchronous GPU call from an event loop serialises the whole server. It stays in
the tree behind `VERICALL_BATCHING=0` because on a full GPU with a proper
executor the calculus changes.

---

## 10. Known gaps

Stated as gaps, not as future work.

| Gap | Status |
|---|---|
| **Cross-domain transfer** | Retraining removed the inversion (Mandarin 79.08% → 9.51%, §7.1) but relocated the problem: the RTCFake model is at chance on ASVspoof and WaveFake, and mixing the two corpora destroys WaveFake outright (§7.2). No configuration tested is good on more than one corpus. |
| **Per-deployment calibration** | Refitting branch calibration and the subset table on a corpus makes fusion help again on that corpus (§7.3), and it is cheap — kB of JSON, seconds, no GPU. A deployment procedure that refits on a sample of the tenant's own channel and language is the obvious design and is **not implemented**. Cross-language, the refitted rule is a tie rather than a gain: the calibration carries its own domain too. |
| **Multilingual coverage** | **The biggest remaining gap.** Measured on Hindi (IndicSynth, 50,044 rows, 2 generators, 101 target speakers): at a 5% false-positive budget the ASVspoof head catches **5.3%** of Hindi attacks — chance — and the RTCFake head **13.9%**. Neither is usable. IndicSynth publishes **spoof only**, so no EER exists for it; pairing it with an unrelated real Hindi corpus would let a detector separate the two by recording conditions rather than by synthesis, so that number is refused rather than caveated. Unblocking a real Hindi EER needs bona fide from IndicSynth's own source recordings. |
| **Channel detection is simulator-tuned** | Implemented and measured at **98.8% per call**, which closes the previous "not implemented" gap (§3.3). But it detects the channel profiles *we simulate*: real telephony adds adaptive noise suppression and comfort noise, and a real narrowband codec resamples to 8 kHz rather than low-passing at 16 kHz. The noisy/clean split must be re-fitted against real transmitted audio. RTCFake is the instrument and has not been used for it. |
| **Replay attacks** | Untested. EchoFake never obtained. |
| **η truncation** | The lag window stops at 8 while Branch A's autocorrelation is still positive, so η_A = 0.207 is an upper bound on its independence. All three branches are now measured (§3.2). |
| **RawBoost augmentation** | Specified, not implemented. |
| **Proteus adversarial gate** | Specified, not run. |
| **In-the-Wild evaluation** | Fetched once, lost to a pod wipe; cross-corpus claims rest on WaveFake and RTCFake instead. |
| **Cross-domain early stopping** | §8.3 identifies the fix; not implemented. |

---

## 11. What this project is actually offering

Two things, and it is worth being clear which is which.

**A working streaming detection and prevention system.** Calibrated LLR
accumulation with a measured η correction, per-channel and per-subset
calibration, change-point detection that catches a mid-call swap 2.3 s after the
splice, a prevention layer that acts on transaction context and reversibility
rather than on a score alone, and a serving path measured against
declared-in-advance SLOs. All of it reproducible from committed code on this
cluster.

**A measurement discipline that catches its own failures.** The most valuable
results in this document are the ones that contradict the design: the vocoder
gap that did not matter, the prosody branch that had to be zero-weighted, the
phase branch that turned out channel-fragile and corpus-robust, the fusion that
amplified an inverted branch, the fabricated citation in the source blueprint,
and the tests that passed while measuring nothing.

Two of those failures were in our own metrics, and they are retracted here
rather than restated:

- **η = 0.080** was an autocorrelation of floating-point noise, because the
  chunker emitted byte-identical windows for a push spanning several ticks. One
  chunker test existed and it never looked at what audio a window contained
  (§3.2).
- An **EER of 95.67%** was once reported for a feature whose class means
  differed by 0.8 ms, because ties were walked as if they were distinct
  thresholds (§8.5). The same saturation problem reappeared later in a different
  disguise, as quantile thresholds that all landed at exactly 1.0.

Both share a shape worth naming: **a number that looked like a result while
measuring nothing.** That is the failure mode this project is built to catch,
and it is why §7 reports the cross-corpus collapse beside the 2.48%.

A voice-fraud system that reports 2.48% and collapses on the first real channel
change is worse than useless, because it is *trusted*. The domain guard exists
because we found that failure in our own system and could not detect it from the
inside. Everything in §7 and §10 is here for the same reason.

---

*Reproduce every number in this document:*
```bash
# corpora -> manifests -> leakage-verified splits -> feature cache
./scripts/pod.sh py 'scripts/fetch_datasets.py --datasets asvspoof2019_la wavefake rtcfake'
./scripts/pod.sh py 'scripts/build_manifests.py --datasets asvspoof2019_la wavefake rtcfake'
./scripts/extract_asvspoof.sh && ./scripts/extract_rtcfake.sh

# training: baseline, RTCFake-only, and mixed
./scripts/pod.sh py 'scripts/train.py --branch A --dataset asvspoof2019_la \
    --eval-on eval_in_domain wavefake:eval_seen_vocoder wavefake:eval_unseen_vocoder'
./scripts/pod.sh py 'scripts/train.py --branch A --dataset rtcfake --tag _rtc \
    --eval-on rtcfake:eval_rtc_en rtcfake:eval_rtc_zh'
./scripts/stage_mixed_train.sh          # mixed run + the comparison matrix

# calibration and the measurements that govern the fusion layer
./scripts/pod.sh py 'scripts/fit_calibration.py --eta-calls 400'
./scripts/pod.sh py 'scripts/fit_subsets.py'
./scripts/pod.sh py 'scripts/measure_eta_branches.py --calls 800'

# evaluation
./scripts/pod.sh py 'scripts/end_to_end_full.py --calls 40'
./scripts/pod.sh py 'scripts/codec_robustness.py'
./scripts/stage_paired_rerun.sh         # section 13 protocol, every checkpoint
./scripts/pod.sh py 'scripts/eval_indic.py --ckpt checkpoints/branch_A_rtc_seed0.pt'
./scripts/pod.sh py 'scripts/audit_silence.py --dataset rtcfake --orig-split-prefix dev:zh'
./scripts/pod.sh py 'scripts/validate_domain_guard.py'
./scripts/sweep_concurrency.sh
```

*Full measurements: `docs/RESULTS.md`. Sources: `docs/RESEARCH.md`.*
