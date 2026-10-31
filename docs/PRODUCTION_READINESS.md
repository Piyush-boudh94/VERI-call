# Section 9 — Production Readiness & Known Failure Modes

*VeriCall v4. Documents the gap between benchmark accuracy and deployed
accuracy, the failure modes that cause it, the fix for each, and the order those
fixes must be applied in.*

**Status of the source spec.** Unlike the v3 blueprint, the Section 9 draft's
citations check out. RawBoost (Tak et al., ICASSP 2022, arXiv 2111.04433), PACT
(Ndichu et al., arXiv 2605.22324), Proteus (Müller, Bukkapatnam & Ahmed,
Resemble AI, arXiv 2606.29544) and US 10,764,043 were each verified to exist and
to say what the draft claims. That is a real improvement.

Nine further conflicts were found anyway, seven of them by measurement against
this system rather than by reading. They are C11–C19 in §9.10, and two are
severe enough to change the design: the n-of-m aggregation scheme **cannot** use
the binomial error model on VeriCall's scores, and the draft omits the failure
mode we measured as dominant.

---

## 9.0 Why this section exists

Every number in Sections 4–8 is a benchmark number. Benchmark accuracy and
deployed accuracy are different quantities, and the gap is not small. The best
system in the ACM MM 2026 AT-ADD challenge reached 90.71% Macro-F1 under
realistic channel variation, and the challenge summary still listed
generalization to unseen generators and robustness to realistic distortions as
open problems.

None of the failure modes below is unique to VeriCall. All affect every
production voice-fraud detector currently deployed. What separates a production
system from a prototype is whether they are measured and mitigated, or unmeasured
and assumed away.

---

## 9.1 The failure modes

| # | Failure mode | Severity | Status here |
|---|---|---|---|
| **FM-0** | **Cross-corpus generalisation collapse** | **Critical** | **Measured: 2.48% → 30.29% EER.** Absent from the source draft — see C14 |
| FM-1 | Base-rate collapse / alert fatigue | Major | Partly pre-solved by Layer 4; see C12, C19 |
| FM-2 | Replay & channel-effect masking | Major | Augmentation designed, not yet measured |
| FM-3 | Adversarial evasion (deliberate) | Major | Red-team loop specified as a release gate |
| FM-4 | Watermark shortcut learning | Major | Prevented architecturally (§9.5) |
| FM-5 | Accent / language / gender disparity | Major | Mechanism built; **fitting blocked on data** — see C13 |
| FM-6 | Training-distribution obsolescence | Major | Replay buffer + recalibration (C15) |
| FM-7 | Short / low-quality audio degradation | Minor | Implemented: `quality_discount()` |
| FM-8 | Aggregate-metric masking | Minor | Already enforced — we report paired FPR/FNR |

---

## 9.2 FM-0 — Cross-corpus generalisation collapse **[new, and dominant]**

The source draft lists eight failure modes and omits the one that costs the most
on this system. Measured here (`docs/RESULTS.md`, 3 seeds):

| Evaluation | EER |
|---|---:|
| dev — in-domain, seen attacks | 2.48% |
| ASVspoof eval — in-domain, **unseen attacks** | 11.01% |
| WaveFake — **seen** vocoders | 31.43% |
| WaveFake — **unseen** vocoders | 30.29% |

Two things follow, and the second contradicts the draft.

**Changing corpus costs 12×. Changing vocoder costs nothing.** Holding out
HiFi-GAN, FB-MelGAN and TTS entirely moved EER from 31.43% to 30.29% — no worse.
The collapse is not caused by unseen generators; it is caused by unseen *domain*
(speakers, channel, recording chain, language distribution).

**This partially contradicts §9.6 of the draft**, which cites a 68.5 pp
generator-recall spread (31.2% → 99.7%) to argue that "which generator the
attacker used matters roughly ten times more than who the speaker is." Both can
be true, and the reconciliation is the useful part: that study spanned distinct
synthesis *paradigms*, whereas our held-out families (HiFi-GAN, FB-MelGAN, TTS)
are all GAN-vocoders over the same LJSpeech corpus. **The axis that matters is
paradigm, not family.** The generator-diversity mandate in §9.7 is therefore
correct but must be specified as paradigm coverage — GAN-vocoder, flow-matching /
diffusion, neural-codec, autoregressive — which is exactly what it says. Adding
a ninth GAN vocoder buys nothing measurable.

**Consequence for reporting:** the number to quote is ~30%, not 2.48%. Anyone
citing in-domain EER as system performance is reporting the wrong quantity.

**Fix:** domain-diverse training (RTCFake and IndicSynth are already fetched for
this), plus the mandatory recalibration in C15. There is no known fix that
closes this gap; it is managed, not solved.

---

## 9.3 FM-1 — Base-rate collapse and alert fatigue

### The problem
At low prevalence, headline accuracy stops predicting operational usability.
Even a 0.1% false-positive rate over a million-event day yields a thousand false
alerts.

### What the draft proposes, and why it needs changing

The draft's fix (a) is n-of-m aggregation: group chunk scores in fives, alert on
3 of 5, compute the combined rate from the binomial distribution over Bernoulli
trials, citing US 10,764,043 (0.0058 FP against degraded audio, 0.00089 clean).

**That scheme is sound where it comes from and unsound here.** In the patent,
each one-second digest is compared against a *cryptographically bound reference*
digest — the trials really are independent. VeriCall has no reference. Each chunk
score is a model output on audio that overlaps its neighbour by 50% and shares a
speaker, channel and generator with every other chunk in the call. We measured
that correlation: **lag-1 autocorrelation +0.824, still +0.630 at lag 8**, giving
η = 0.080. Five consecutive chunks are worth **0.4** independent observations,
not 5.

> **RETRACTED, 2026-09-22 audit.** η = 0.080 was an autocorrelation of
> floating-point noise: the chunker was emitting byte-identical windows for a
> multi-tick push when it was measured (median score std 1.8e-15). See
> `docs/RESULTS.md` §15 and `docs/VERICALL_V4.md` §3.2. The measured values are
> **η_A = 0.207, η_A2 = 0.928, η_B = 0.565** over 800 calls, and they are what
> `FusionConfig` carries. **The conclusion of this section is unchanged and in
> fact understated** — the binomial model is 9.3×–17.8× optimistic on real
> correlated scores (`docs/aggregation.json`) — but the specific figures 0.080
> and "0.4 independent observations" must not be quoted.

**Measured** (`scripts/measure_aggregation.py`, 120 bona fide calls, 3,586
chunks, results in `docs/aggregation.json`):

| threshold | chunk FP | binomial predicts | empirical | optimism |
|---:|---:|---:|---:|---:|
| 0.0 nats | 3.15% | 0.0298% | **0.3220%** | **10.8×** |
| 1.0 nats | 2.20% | 0.0103% | **0.0966%** | **9.3×** |
| 2.0 nats | 1.78% | 0.0055% | **0.0966%** | **17.5×** |
| 3.0 nats | 1.23% | 0.0018% | **0.0322%** | **17.8×** |

For the proposed 3-of-5 the binomial model is **9.3×–17.8× optimistic**; across
all eight informative operating points, median 7.2×. (Stricter votes — 4-of-5,
5-of-5 — never fired in 3,586 chunks, which is absence of data, not agreement.)

A design that sized its alerting budget from the binomial figure would be
under-provisioned for false positives by roughly an order of magnitude.

**Also: VeriCall never alerted per chunk.** The draft's framing — "VeriCall
scores at roughly one chunk per second per call, which is a far higher event
rate" — misreads the architecture (C19). Layer 4 already performs sequential LLR
accumulation with dual-window smoothing, consecutive-crossing, per-tier dwell and
hysteresis, and alerts **per call**. Measured per-call FPR is 0/40. Alert-fatigue
exposure scales with calls/day, not chunks/day.

### Resolution
1. **Keep LLR accumulation as primary.** A binary n-of-m vote discards the LLR
   *magnitude*, which is the basis of both the cost-weighted threshold and the
   S_lo confidence bound. Replacing evidence accumulation with a vote would throw
   away the calibration Stage 5 exists to produce (C12).
2. **n-of-m is retained only as a secondary confirmation gate** on tier
   escalation, implemented in `vericall/risk/aggregation.py`.
3. **Its error rate is measured empirically, never computed binomially.**
   `effective_m(m, eta)` reports the honest window size.
4. **PACT** is adopted as specified, offline against the Redis decision store
   (C7). The draft's guardrail is correct and worth restating: a frozen
   threshold-only baseline pushed FP lower but collapsed recall by 55 pp.
   Optimise the Pareto point, not the FP rate.

---

## 9.4 FM-2 — Replay and channel-effect masking

Physical channel effects mask the synthesis artifacts Branches A and A2 depend
on; ReplayDF measured a leading detector going from 4.7% to 18.2% under replay.

**RawBoost** is the right primary tool — telephony-designed, needs no external
noise or impulse-response corpora, model-agnostic, +27% relative on ASVspoof
2021 LA. All three processes together gave the best published result.
**Reverberation** covers the replay-specific half (FRAM-RIR, rooms ~3×3×2.5 m to
10×6×4 m, RT60 0.2–1.0 s, p = 0.7).

**Stated limit:** RIR-augmented retraining only *partially* compensates. Replay
EER is reported as a separate line item, never folded into the headline.

**Two conflicts the draft misses** — see C16 (time-stretch also breaks Branch
A2, not just Branch B) and C17 (reverb depresses η further, so η must be
re-measured after Stage 2).

---

## 9.5 FM-4 — Watermark shortcut learning

If synthetic training samples are disproportionately watermarked and genuine ones
are not, the model learns "watermarked = fake" — so unwatermarked fakes evade and
genuine watermarked recordings are flagged.

**Architectural separation.** The watermark check is a Layer-0 pre-filter that
never touches training data:

```
audio ─┬─► Layer 0: watermark probe ─► positive ─► flag, logged separately
       │                                  (never fed to the trainer)
       └─► Layers 1-6: behavioural stack, trained watermark-blind
```

Enforced in the data pipeline: no sample carries watermark status as feature or
label; watermarked and unwatermarked synthetics are balanced within the synthetic
class; Layer-0 outcomes never become training labels.

**Scope limit, stated plainly:** coverage is narrow. No platform other than
ElevenLabs can currently detect ElevenLabs SynthID, their own classifier does not
reliably handle their newest model, and independent testing found it
misclassifies audio from other engines. Layer 0 is a bonus signal on a slice of
traffic, never the primary defence.

---

## 9.6 FM-5 — Accent, language and gender disparity

Audits find real, systematic disparity: FairSSD found most of six detectors
biased by gender, age and accent, with higher FPR for male speakers.
Critically, **more data is not the fix** — the bias stems from acoustic
representation differences and gender leakage in learned features, not from
imbalanced training data. Per-group threshold calibration reduced unfairness by
54–75% at no accuracy cost.

**Conflict C13 — the mechanism is buildable here; the fitting is not.**
ASVspoof 2019 LA has ~20 dev speakers, one language. {language × gender} needs
per-cell calibration data, and regional-accent subgroups need far more. Fitting
thresholds on ~10 speakers per cell would produce noise dressed as fairness.

**Resolution:** implement the mechanism (it reuses the per-subset calibration
machinery already in `vericall/fusion/calibration.py`), report stratified
EER/FPR/FNR wherever n permits, and state that per-group *threshold fitting* is
blocked pending IndicSynth-scale speaker diversity. Shipping under-fit fairness
thresholds is worse than shipping none and saying so.

**Reporting requirement:** stratified always. Aggregate numbers hide real gaps —
one study found gender gaps under 0.7 pp EER but a 7.5 pp regional gap for
AASIST within one country.

---

## 9.7 FM-6 — Training-distribution obsolescence

Detection and generation co-evolve; a frozen model degrades monotonically. Naive
retraining fails — sequential fine-tuning causes catastrophic forgetting.

**Replay buffer + knowledge distillation** is the right choice here specifically
because VeriCall's privacy design already permits storing irreversible feature
embeddings (the disclosed Branch C exception), so the buffer holds embeddings
rather than raw audio — no new privacy surface (C6).

- **Cadence:** quarterly minimum, event-triggered on major TTS releases
- **Generator diversity:** each cycle adds ≥2 generators from **paradigms** not
  yet represented (see FM-0 — new families within a paradigm buy nothing)
- **Regression gate:** evaluate all prior attack families; degradation beyond
  threshold blocks release
- **Mandatory recalibration (C15):** any weight update invalidates the Platt
  constants. Stage 5 must re-run after Stage 4, every time.

---

## 9.8 Minor failure modes

**FM-7 — short / low-quality audio.** Reliability degrades with both duration
and quality, and the 1 s fast-lane window is the most exposed configuration.
Implemented as `quality_discount()` in `vericall/risk/aggregation.py`: chunks
below a quality/duration floor contribute proportionally less rather than at full
weight. This composes with the existing quality gate, which already zeroes
silence and hold music.

**FM-8 — aggregate-metric masking.** Production benchmarking of eight systems
showed detectors with 1.0% and 2.4% FPR missing 23% and 31% of fakes, while
others flagged ~50% of genuine audio. Never report a blended accuracy or AUC.
VeriCall already reports paired FPR/FNR (0/40 and 40/40 in `docs/RESULTS.md`)
with bootstrap CIs over utterances.

---

## 9.9 Fix ordering

```
STAGE 1  DATA         corpus + paradigm diversity + watermark balancing
   │                  guard: watermark status is never a feature or label
   ▼
STAGE 2  AUGMENTATION  branch-specific (C1, C16)
   │     A  : RawBoost(3) + RIR + pitch/time-stretch
   │     A2 : RawBoost(3) + RIR only -- NO time-stretch (C16)
   │     B  : RawBoost channel-noise ONLY (timing-preserving)
   │     C  : RawBoost channel-noise + mild RIR
   ▼
STAGE 2b RE-MEASURE eta  (C17: reverb raises within-call correlation)
   ▼
STAGE 3  ADVERSARIAL TRAINING
   │     shifts error across demographic groups -> fairness measured AFTER
   ▼
STAGE 4  CONTINUAL LEARNING   replay buffer (embeddings) + distillation
   ▼
STAGE 5  CALIBRATION   per channel bucket; per {language x gender} where n
   │                   permits. MANDATORY after any Stage-4 update (C15)
   ▼
STAGE 6  AGGREGATION   n-of-m as secondary gate on the LLR accumulator,
   │                   rate measured empirically (C11, C12); FM-7 discount
   ▼
STAGE 7  ALERTING POLICY   sector prevalence profiles; PACT offline (C7),
   │                       writing back to the Stage-5 table (C18)
   ▼
STAGE 8  RED TEAM (release gate)   Proteus-style chain search, depth >= 3,
                                   quality-gated -> feeds back to STAGE 2
```

**Orderings that must never invert:**
- **Calibration before aggregation.** Aggregating uncalibrated scores bakes group
  disparity into the vote, where threshold adjustment can no longer reach it.
- **Adversarial training before fairness measurement.** Thresholds calibrated on
  a pre-adversarial model are wrong for the shipped model.
- **Recalibration after every weight update (C15).** Otherwise Stage 5's output
  describes a model that no longer exists.

---

## 9.10 Conflict matrix

C1–C10 are from the source draft. **C11–C19 are new**, and seven were found by
measurement against this system rather than by reading.

| ID | Interaction | Status | Resolution |
|---|---|---|---|
| C1 | Time/pitch augmentation **vs** Branch B (prosody) | ⚠️ Real | Correct as drafted. B receives channel-noise only. |
| C2 | RIR **vs** Branch A2 (phase) | ⚠️ Validate | Ablate; do not carry RT60/p over blind. |
| C3 | Per-group thresholds **vs** n-of-m | ✅ | Order-dependent only: calibrate, then vote. |
| C4 | Continual learning **vs** LoRA adapters | ✅ | Adapters are a natural CL unit. |
| C5 | Adversarial training **vs** fairness | ⚠️ Sequencing | Fairness measured after Stage 3. |
| C6 | Replay buffer **vs** minimal retention | ✅ | Embeddings, not audio; existing exception. |
| C7 | PACT **vs** stateless workers | ⚠️ Placement | Offline against Redis, never in the inference path. |
| C8 | Layer-0 watermark **vs** training data | ✅ | Enforced at the pipeline level. |
| C9 | Quality discount **vs** dual-rate chunking | ✅ | Applied at aggregation, which already reconciles cadences. |
| C10 | Aggregation smoothing **vs** rising-trend detector | ⚠️ Latent | Vote *within* the short window, not across both. |
| **C11** | **n-of-m binomial model vs measured correlation** | ⚠️ **Major** | The patent's Bernoulli independence holds because each digest is compared to a cryptographic reference. Ours are model outputs on correlated audio (lag-1 ρ=+0.824; η measured per branch at 0.207 / 0.928 / 0.565 — the η=0.080 originally quoted here is retracted, see §9.3). **Measure the rate empirically; never compute it binomially.** |
| **C12** | **n-of-m vs the existing LLR accumulator** | ⚠️ **Major** | A binary vote discards LLR magnitude — the basis of the cost-weighted threshold and S_lo. It is strictly weaker than what Layer 4 already does. **Keep accumulation primary; n-of-m becomes a secondary confirmation gate only.** |
| **C13** | **Per-group thresholds vs available speaker diversity** | ⚠️ **Major** | ~20 dev speakers, one language. Cannot fit {language × gender} cells. **Build the mechanism, report stratified metrics, do not ship under-fit thresholds.** |
| **C14** | **Failure-mode list vs measured reality** | ⚠️ **Major** | The dominant failure mode (cross-corpus collapse, 12×) is absent from the draft. **Added as FM-0.** |
| **C15** | **Continual learning vs fixed calibration** | ⚠️ **Major** | Stage 4 changes weights; the Platt constants (a=0.510, b=−0.507, fit on ASVspoof clean) then describe a model that no longer exists. **Recalibration is mandatory after every weight update, and per deployment domain.** |
| **C16** | **Time-stretch vs Branch A2** | ⚠️ Moderate | C1 protects Branch B only. Phase-vocoder time-stretch destroys phase coherence — precisely A2's signal, whose shift-invariance we verified exactly. **Exclude time-stretch from A2 as well.** |
| **C17** | **RIR augmentation vs measured η** | ⚠️ Moderate | Reverb is a call-constant channel effect, so it *raises* within-call correlation and lowers η further, widening every confidence bound. **Re-measure η after Stage 2 (new Stage 2b).** |
| **C18** | **PACT vs the frozen Stage-5 table** | ⚠️ Minor | PACT adapts thresholds that Stage 5 fixed and Stage 6 builds on — circular. **PACT writes a new versioned Stage-5 table; Stage 6 re-derives from it.** |
| **C19** | **Alert-exposure premise vs the architecture** | ⚠️ Minor | The draft computes exposure from chunks/second. VeriCall alerts per *call*, gated by hysteresis, dwell and consecutive-crossing; measured per-call FPR 0/40. **Exposure is calls/day.** |

---

## 9.11 Panel-facing summary

> The architecture is provider-agnostic and benchmark-validated. Deployed
> accuracy depends additionally on four things: telephony-and-replay augmentation
> during training, base-rate-aware per-call alerting (which this design already
> does through sequential evidence accumulation rather than per-chunk voting),
> per-group threshold calibration for equitable performance across Indian
> languages and accents, and a continual-learning cycle with a defined cadence
> and paradigm-diversity mandate. The residual limits are stated rather than
> assumed away: replay mitigation is partial, cross-corpus generalisation
> degrades roughly twelve-fold and is managed rather than solved, and per-group
> fairness thresholds are not yet fittable on the speaker diversity available.
> These limits are shared by every state-of-the-art detector in the field,
> including the 2026 grand-challenge leaderboard systems — the difference is that
> here they are measured.

---

## 9.12 Serving and platform readiness

§9.0–9.11 are about the model. This section is about the thing that runs it,
which was audited separately and turned out to be the more urgent of the two:
a model that generalises perfectly is worth nothing behind a gateway that
serializes every caller or that anyone can connect to.

Every item was found by reading the code against the failure lists in the OWASP
WebSocket Security cheat sheet and the standard FastAPI/PyTorch serving
guidance, then confirmed by measurement.

### The capacity defect that dominated everything else

**`torch` sized its intra-op thread pool from the host core count — 112 threads
on a 16-core pod.** Every offline script in this project already set it to 1.
The serving path never did.

For one request at a time this is invisible, which is why it survived. Under
concurrency it is the whole problem: N in-flight forwards each try to claim the
entire machine, and the resulting context switching inflated a 9 ms forward into
**296 ms**. Measured with `scripts/capacity_probe.py`, one line moved N=50 from
1.74× real time and p99 1102 ms to **0.97× real time and p99 56.7 ms**.

| | before | after |
|---|---|---|
| N=50 server-side | 1.74× real time, p99 1102 ms | **0.97×, p99 56.7 ms** |
| server-side ceiling | — | **N=160 at 0.99× real time** |

### Three more defects on the same path

**The GPU forward ran synchronously on the asyncio event loop.** One forward
blocked every other call's socket reads for its duration, so N calls did not
share the server — they queued behind each other, invisibly, because each
individual forward still looked fast. The audit write did the same with blocking
file I/O, on the code path that runs at the most interesting moment of a call.

**Cross-call batching and per-call channel calibration were mutually
exclusive**, so batching was disabled — and the A/B that justified disabling it
had compared two broken configurations (batcher holding the loop versus direct
path holding the loop). Splitting the batched *forward* from the per-call
*Platt mapping* dissolves the conflict: batch what is identical for every call,
apply what is not.

**Measurement was confounded by the instrument.** The load generator shares the
pod's 16 cores with the server it measures. At N=4 that was a caveat; at N=50 it
was the answer. `capacity_probe.py` measures the pipeline without a network
client in the way, and the gap between it and `bench.load` is the transport cost
— two numbers worth having separately rather than one that hides which is which.

### Measured result

Over real WebSockets, 12 s of paced audio per call:

| N | wall clock | first score p50 | chunk p99 | completed |
|---:|---:|---:|---:|---:|
| 16 | 13.3 s | 1.02 s | 22.7 ms | 16/16 |
| **50** | **13.8 s** | **1.01 s** | **25.3 ms** | **50/50** |
| 64 | 14.0 s | 1.01 s | 22.2 ms | 64/64 |

Against the previous ceiling of **N = 4**, and with a 10× margin on the 250 ms
chunk budget at 64.

### Security surface

The gateway had none: no authentication, no cap on concurrent calls, no frame
size limit, no idle timeout, and a client-supplied `call_id` used directly as a
dictionary key — so a second connection with the same id silently *replaced* the
first in the session table.

| Control | Status |
|---|---|
| Shared-secret auth | `hmac.compare_digest` — `==` returns on the first differing byte and leaks the matching prefix through timing |
| Admission cap | Semaphore, refuse with close code 1013, distinct code per refusal reason |
| Duplicate `call_id` | Refused, not substituted |
| `call_id` charset/length | Anchored `\A…\Z` — Python's `$` also matches before a **trailing** newline, so `"call\nid"` was rejected while `"callid\n"` got through |
| Frame size | 64 KiB; frame refused, call kept — dropping the call would be its own denial of service |
| Idle timeout | 30 s; the documented WebSocket session leak is a half-open socket nobody collects |
| Inference queue | Bounded, sheds and reports; an unbounded queue turns overload into an OOM kill that outlives the overload |
| Audit log | Size-based rotation, bounded generations, `fsync` — a full disk is a total outage, not a logging one |
| Checkpoint deserialisation | `weights_only=True`; `False` executes pickle, so write access to `checkpoints/` was write access to the process |
| Artifact integrity | SHA-256 manifest verified at load, `VERICALL_REQUIRE_SEALED=1` to enforce |
| Information disclosure | `/health` returned absolute filesystem paths; basenames answer the operator's question without the reconnaissance |

All verified against a running server, each refused with its own distinct error.

### Operability

`/metrics/prom` exposes Prometheus text — `/metrics` is JSON because the load
generator reads it, and nothing in a monitoring stack does. **Queue depth is
exported alongside p99 because it is the leading indicator: it rises before
latency does.** SIGTERM stops admission and drains within a timeout set below
the orchestrator's grace period, so a rolling restart no longer kills calls
mid-sentence — a failure that appears in no dashboard, because the fraud
decision is simply never made.

### Still open

- ~~**Per-client rate limiting.**~~ **Closed.** `vericall/serving/limits.py`
  carries a per-client cap as a fraction of the global admission cap
  (`Limits.max_per_client`), so one client can no longer occupy every slot;
  `Admission.acquire(client=...)` enforces it and `tests/test_limits.py` pins
  it. Verified in the 2026-09-22 audit.
- **`wss://` termination** is the ingress's job and must not be skipped.
- **Single process, no replication.** A crash loses every in-flight call and
  there is no session handoff.
- **Manifest signing.** The integrity check is a tripwire, not a signature:
  anyone who can rewrite a checkpoint can rewrite the manifest beside it.
- **DoS by crafted audio** — no measurement of whether specific inputs make the
  pipeline pathologically slow.
