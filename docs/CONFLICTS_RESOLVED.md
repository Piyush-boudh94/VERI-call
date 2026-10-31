# VeriCall — conflicts, errors and issues found in the v3 blueprint

Every item is a defect found in the v3 architecture document (or, for E20–E23,
in the v3 design as it was being implemented), with the correction applied in
v4. Items marked **[code]** are enforced by a passing test.

Twenty-three items. Four of them (E20–E23) were invisible on paper and only
appeared once the fusion engine was executable — which is why Phase 0 built the
fusion layer *before* any model.

---

## A. Factual and citation errors

### E1 — "Pindrop's OCCAS pattern" does not exist
The v3 preamble grounds the design partly in "Pindrop's OCCAS pattern". There is
no such product, pattern or paper; a targeted search for `"OCCAS" Pindrop`
returns nothing related. Pindrop's actual technology is **Pindrop Pulse**, built
on *liveness detection* and a per-utterance representation they call a
"Fakeprint" — reported at ~99% detection on known engines, >90% on unseen, at
<1% FPR from about 2 s of netspeech.

**Fix:** cite Pindrop Pulse / liveness detection correctly. The architectural
idea it was standing in for (score early, on short netspeech) is sound and is
retained — only the attribution was wrong. A judge who googles the original term
finds nothing, which is a needless credibility loss.

### E2 — branch-count contradiction
§4 opens "This is a **3-branch** model with late fusion", then defines four
branches (A, A2, B, C) and refers to fusing "a **4th** score".
**Fix:** it is a 4-branch model, stated once and consistently.

### E3 — the phase-branch headline number is indefensible
v3 cites "EER 1.88% → 0.002% from adding phase information". That result is on
**ASVspoof 2015**, a saturated and retired benchmark whose numbers do not
transfer to telephony conditions. (The stated "two orders of magnitude" is also
wrong — 1.88 → 0.002 is roughly three.)
**Fix:** cite it only as directional evidence that phase carries information
complementary to magnitude. Never quote 0.002% as an achievable operating point.

---

## B. Architecture bugs

### E4 — Branch C is conceptually inverted **[code]**
v3 scores voiceprint *similarity* to an enrolled speaker and adds it as a fourth
spoof score. But a **good clone of an enrolled person raises similarity**. As
specified, the better the attack, the safer the system reports it. Branch C
detects "wrong person", not "cloned person".

**Fix:** Branch C is an **identity** signal, kept entirely out of the spoof
score, and consumed at the policy layer instead — where
*(spoof high ∧ identity-match high ∧ enrolled)* is the **highest**-consequence
cell: a convincing clone of a known customer. Additive fusion gets that cell
exactly backwards. Separately, *(spoof low ∧ identity-match low)* expresses a
human impostor, which a spoof detector alone cannot represent at all.
*Enforced by `test_branch_c_never_moves_the_spoof_score`.*

Bonus: C's embedder is useful with **no** enrollment — comparing the caller to
themselves across consecutive windows is a free speaker-turn detector.

### E5 — the v3 fix for Conflict A/B silently discards the privacy property
v3 resolves the shuffle-vs-prosody conflict by moving all four branches inside
the trust boundary and demoting the SafeEar bottleneck to archival transport.
That works for accuracy, but Mode B is then a server ingesting **raw audio** —
precisely what the design philosophy's third constraint forbids — while the
pitch still says "privacy-preserving".

**Fix, two parts.** (a) State the guarantee honestly: Mode B's privacy is
*operational* (no persistence, no ASR, no transcript ever produced,
enclave-scoped, DPDP-consented), not structural. (b) Offer **Mode B-strict**,
which is a better fix than v3's: SafeEar's semantic/acoustic split is *not* what
destroys prosody — its **shuffle layer** is (verified against the paper: random
permutation within a 1 s / 50-frame window). Transporting acoustic tokens
*without* the shuffle preserves prosodic trajectories while still preventing
content recovery. v3 discarded the entire bottleneck when only one layer was the
problem.

### E6 — Layer 4 was specified as a gating MLP that cannot exist
v3 says a gating MLP fuses four scores and that "Layer 4 is designed to accept
different rates". A fixed-width MLP structurally cannot consume 2 Hz + 1 Hz +
sometimes-absent streams; it would need ~8 models (one per availability subset)
or imputation — and imputing inputs to a *detector* means inventing evidence.
Layer 4 was also never actually written in the document.

**Fix:** calibrated **LLR accumulation** — linear logistic-regression fusion
composed with an exponentially-discounted sequential accumulator over LLR
*rates*. Multi-rate fusion then falls out for free, because the accumulator
integrates over time rather than over events. A missing branch is a dropped
term, restored by two scalars per subset fit on cached dev outputs.

### E7 — score-scale mismatch feeding fusion **[code]**
A/A2 emit supervised posteriors, B a one-class anomaly score with no negative
distribution behind it, C a cosine similarity. Fusing them unconverted is a bug;
the weights would silently absorb an order-of-magnitude scale mismatch.
**Fix:** per-branch calibration to a common LLR scale before fusion, plus a hard
clip on B (κ = 2.0 nats) so an uncalibrated one-class branch can never dominate.
*Enforced by `test_one_class_llr_is_clipped_and_monotone` and
`test_one_class_branch_b_cannot_dominate`.*

### E8 — the Mode A edge model is not edge-deployable
"Truncated Wav2Vec2 at layer 7, ~101M params, light enough for browser
inference" is ~400 MB at fp32, executed twice a second on a phone mid-call.
**Fix:** use that backbone as a **teacher** and deploy an int8-quantised
distilled student. A published reference point for the target: XLS-R+AASIST
real-time detection at **0.62 MB ONNX / 2.84% EER**.

---

## C. Quantitative and evaluation issues

### E9 — throughput figures borrowed from the wrong domain
"1.2M events/sec, 85 ms" comes from transaction-fraud streaming, where an event
is a ~500-byte JSON record — not 16,000 audio samples through a ~95M-parameter
model. That is four orders of magnitude off in work-per-event.
The literature RTF anchors are a second trap: **AASIST's RTF 0.0155 is the
graph-attention head alone on a full GPU**; with an SSL frontend the real cost is
roughly 20× that.
**Fix:** derive capacity from measured RTF on our own hardware, and report cost
as **core-seconds per call-minute**, which extrapolates honestly.

### E10 — two incompatible latency claims presented as one
"<2 s first score" and "250 ms p99.9" are different quantities.
**Fix:** four SLOs defined and measured separately — `chunk_inference_latency`,
`time_to_first_score`, `time_to_alert`, `ui_publish_latency`. `time_to_alert` is
bounded by *evidence accumulation*, not compute, so shaving 50 ms off inference
does essentially nothing to it. Optimising the wrong one is the classic mistake.

### E11 — no base-rate analysis
At a realistic prevalence of ~1 in 10,000 calls, a 1% per-call FPR makes ~99% of
alerts false, and the system is rejected regardless of its EER.
**Fix:** cost-weighted operating point θ* = log[C_fa(1−π)/(C_miss·π)] with
per-tenant, per-context priors, plus a tiered response so a false positive costs
a silent flag rather than a hung-up customer. Working the arithmetic through
realistic banking costs shows θ* lands **below the reliable-calibration floor in
every high-value context**, so thresholds are clamped to [1.0, 8.0] nats and the
real constraint on alerting is agent capacity, not decision theory. Saying that
out loud is a stronger position than presenting a fake-precise θ*.

### E12 — the evaluation plan would produce meaningless numbers
**Fix:** mandatory cross-dataset protocol, per-pair rather than pooled EER, and
honest reporting of the in-domain → in-the-wild degradation (published figures:
error rates rising 200–1000%; strong models near ~21% EER in the wild against
~1–2% in domain).

### E13 — replay attacks absent from the threat model
A phone held to a microphone playing a clone is the realistic low-cost attack;
detectors average ~48% EER on replay.
**Fix:** replay added to the threat model and to channel augmentation.

---

## D. Scope gap

### E14 — "Detection **and Prevention**" — prevention was entirely missing
PS 26104 asks for both; v3 is 100% detection.
**Fix:** a prevention layer — challenge-response liveness, out-of-band step-up
verification before sensitive actions, call-flow intervention, and India-specific
compliance hooks (TRAI **CNAP**, the CERT-In deepfake advisory, IT Amendment
Rules 2026 synthetic-media labelling and 36-hour takedown).

### E15 — infrastructure assumed that does not exist on the target cluster
**Fix:** a demo tier that genuinely runs on the available quota, and a separately
documented scale-out design. Kafka is scoped to the downstream audit log only,
where small JSON records and durability requirements make its published
throughput numbers actually applicable.

---

## E. Dataset errors

### E18 — DEEP-VOICE cannot support the claims made from it
Listed in v3 as a grounding source. It is **62 minutes, 8 speakers, RVC-only** —
trivially overfittable, and a judge who checks will find it is under an hour of
audio.
**Fix:** retained only as a smoke-test fixture. No accuracy number is quoted from
it.

### E19 — InDeepFake is a video dataset
v3 cites it for audio fine-tuning. It is a corpus of high-resolution **facial
videos** (189 subjects plus 200 celebrity YouTube videos), gated to academic
institutions via IEEE DataPort.
**Fix:** removed from the audio plan. **IndieFake** is the genuinely audio,
genuinely Indian substitute to evaluate.

### Silence shortcut (not in v3 at all)
ASVspoof 2019 LA has systematically longer leading/trailing silence on bona fide
clips. A model trained on **leading-silence duration alone — no speech content —
reaches 85% accuracy / 15.1% EER.** Training naively yields an excellent
in-domain number from a model that learned a stopwatch.
**Fix:** mandatory VAD trimming before windowing, plus a duration-only control
classifier that must be clearly beaten.

---

## F. Found during implementation — invisible on paper

### E20 — hop-crediting cancels out, so overlap is still double-counted **[code]**
The v3-era correction for overlapping windows was "credit each event for its
non-overlapping hop duration", i.e. `rate = llr / hop`, with credited time
advancing by `hop`. Those cancel exactly:

```
S = T_eff · mean_rate = (N·hop) · (llr/hop) = N·llr
```

which is just the raw sum of every window's LLR — **no overlap correction at
all**. A 0.5 s hop banks exactly twice the evidence of a 1.0 s hop from the same
audio.

**Fix:** the event carries **two** durations. The LLR is normalised by the
**window** it was computed over (evidence density, nats per second of *analysed*
audio) while credited time advances only by the **hop** (non-overlapping audio):

```
rate = llr / window_s        weight = quality · hop_s        T_eff += quality · hop_s
```

Now `S = N·hop·llr/window`, which is invariant to sampling density.
*Enforced by `test_hop_not_window_determines_credited_evidence`, which fails with
40.0 vs 20.0 under the old scheme.*

### E21 — the change detector re-fires on a sustained spoof **[code]**
CUSUM was specified as firing whenever the accumulator crosses its threshold,
and a change point discards pre-change evidence and resets `T_eff`. On a call
synthetic from t=0 there is no *change* — the first evidence is the baseline —
but the CUSUM crossed anyway, roughly every 9 s, wiping accumulated evidence
each time and littering the UI timeline with speaker changes that never
happened.

A first attempt to gate on "was the pre-change rate low?" failed too, because
`declare_change_point` wipes the accumulators, so the baseline check saw a fresh
zero and immediately re-armed.

**Fix:** a persistent long-horizon `regime` accumulator (τ = 20 s) that is
deliberately **not** reset by a change point, plus a refractory period and
suppression once tier ≥ 1 ("it changed" stops being news after you have already
alerted).
*Enforced by `test_spoof_from_t_zero_declares_no_change_points` and
`test_regime_survives_a_change_point`.*

### E22 — the ESS deflation factor was a mathematical no-op **[code]**
`eta` was specified as a multiplier on the accumulator weights. But a weighted
mean **and its variance** are both scale-invariant in the weights:

```
mean = Σwᵢrᵢ / Σwᵢ                 var = Σwᵢ²vᵢ / (Σwᵢ)²
```

Scaling every `wᵢ` by η leaves both unchanged. The parameter did nothing.

**Fix:** effective-sample-size deflation belongs on the **variance**
(`rate_var /= η`), not the weight. Fewer effective independent samples must mean
a *wider* bound — which is the entire point.
*Enforced by `test_eta_deflation_reduces_banked_confidence`.*

### E23 — the escalation cooldown made early alerts impossible **[code]**
The 10 s cooldown between escalations was measured against `tier_since`, which
initialises to 0.0. Every call therefore spent its first 10 seconds unable to
alert — against a median `time_to_alert` target of 3 s.
**Fix:** cooldown gates on `last_escalation_t`, which is `None` until the first
escalation. Measured `time_to_alert` is now 2.0 s for a spoof from t=0 and 1.0 s
after a mid-call swap.
*Enforced by `test_time_to_alert_after_a_mid_call_swap_is_within_budget`.*

### E24 — failed escalation gates suppressed the alert entirely **[code]**
Found by end-to-end testing on real audio, invisible to the unit tests.

The tier gates require >= 2 live branches before an agent-visible alert. With
only Branch A deployed, a call scoring deep in tier-3 territory (S_lo = +60
nats) selected target tier 3, failed the two-branch gate, and escalated to
**nothing at all** — instead of the tier-1 step-up challenge the evidence
plainly supported. Measured effect: **0% of spoofed calls alerted** despite
perfect score separation.

The unit test that should have caught it asserted `tier <= 1`, which passes on
tier 0 — and tier 0 *was* the bug.

**Fix:** clamp the target to the highest tier whose gates are satisfied rather
than aborting. A single branch now reaches step-up but not an agent alert, which
is the intended policy. After the fix: 100% of spoofed calls alert, 0/40 bona
fide. *Enforced by `test_single_branch_reaches_step_up_but_not_an_agent_alert`
and `test_gates_clamp_the_tier_instead_of_suppressing_the_alert`.*

Degrading to silence is precisely the failure mode graceful degradation exists
to prevent, and it took real audio to expose it.

---

## G. Section 9 (production readiness) — conflicts C11–C19

Nine further conflicts, seven found by measurement rather than reading. Full
detail and the fix-ordering constraints are in `docs/PRODUCTION_READINESS.md`.

The two that change the design:

**C11 — the n-of-m binomial error model does not hold on VeriCall's scores.**
The 3-of-5 scheme is borrowed from cryptographic call authentication, where each
one-second digest is compared against a *cryptographically bound reference* and
the trials genuinely are independent Bernoulli draws. VeriCall has no reference:
each chunk score is a model output on audio that overlaps its neighbour by 50%
and shares a speaker, channel and generator with the whole call. Measured on 120
bona fide calls / 3,586 chunks, the binomial prediction is **9.3×–17.8×
optimistic** for 3-of-5 (median 7.2× across eight informative operating points).
Sizing an alerting budget from the binomial figure would under-provision false
positives by about an order of magnitude. *Fix: measure the rate empirically;
`effective_m(5, 0.080) = 0.4` reports the honest window size.*

**C14 — the dominant failure mode was missing from the list.** The source draft
enumerates eight failure modes, none of which is cross-corpus generalisation
collapse — the one we measured at 12× (2.48% → 30.29% EER). Added as FM-0.

Also new: **C12** (a binary vote discards the LLR magnitude the cost-weighted
threshold depends on, so n-of-m is strictly weaker than the existing accumulator
and is demoted to a secondary gate), **C13** (per-group fairness thresholds are
not fittable on ~20 speakers — build the mechanism, report stratified metrics,
ship no under-fit thresholds), **C15** (weight updates invalidate the Platt
constants; recalibration must be mandatory after every continual-learning
cycle), **C16** (time-stretch breaks Branch A2's phase coherence, not just
Branch B's timing), **C17** (reverb augmentation raises within-call correlation
and lowers η further, so η must be re-measured after augmentation), **C18**
(PACT adapts thresholds that a later stage treats as frozen — needs versioned
write-back), and **C19** (alert exposure is calls/day, not chunks/day; VeriCall
already alerts per call).
