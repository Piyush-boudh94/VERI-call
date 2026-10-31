# VeriCall — measured results

Every number here was produced on this cluster from the committed code. Nothing
is extrapolated, and nothing was tuned against the evaluation sets.

Reproduce:
```bash
./scripts/pod.sh py 'scripts/train.py --branch A --dataset asvspoof2019_la \
    --eval-on eval_in_domain wavefake:eval_seen_vocoder wavefake:eval_unseen_vocoder'
./scripts/pod.sh py 'scripts/fit_calibration.py --eta-calls 400'
./scripts/pod.sh py 'scripts/end_to_end.py --calls 40'
```

---

## 1. Detection accuracy — Branch A, 3 seeds

Frozen WavLM-base truncated at layer 5 (94.4M frozen) + attention head (0.89M
trainable), trained on ASVspoof 2019 LA.

| Evaluation | EER | spread | 95% CI (seed 0) |
|---|---:|---:|---|
| dev — in-domain, **seen** attacks | **2.48%** | ±0.09 | — |
| ASVspoof eval — in-domain, **unseen** attacks | **11.01%** | ±0.42 | 11.03–11.63% |
| WaveFake — **seen** vocoders | **31.43%** | ±0.65 | 29.40–32.19% |
| WaveFake — **unseen** vocoders | **30.29%** | ±0.33 | 28.45–31.66% |

Confidence intervals bootstrap over **utterances**, not rows. Resampling rows
would treat several crops of one clip as independent and report an interval
several times too narrow.

### The headline finding: the vocoder gap is not the problem, the domain gap is

Holding out entire vocoder families (HiFi-GAN, FB-MelGAN, TTS) costs **nothing**
— unseen 30.29% is, if anything, marginally *better* than seen 31.43%. What
actually destroys the detector is changing corpus: 2.48% → ~30%, a 12x
degradation, consistent with the published 200–1000% real-world error increases.

This inverts the assumption the original design was built on. Effort spent
anticipating new vocoders is close to wasted; effort spent on channel and corpus
robustness is where the accuracy is. It also means **the number to quote is
~30%, not 2.48%** — anyone reporting the in-domain figure as system performance
is reporting the wrong number.

### Shortcut control

ASVspoof 2019 LA lets a model classify on leading-silence duration alone.
Measured here (`scripts/audit_silence.py`): **36.4% EER from duration only, no
speech content.** Branch A reaches 2.48% in-domain, so it clearly has not
learned a stopwatch.

The sobering corollary: at ~30% cross-corpus, the margin over that trivial
baseline is thin. The two are measured on different corpora so it is not a clean
comparison, but it is the honest framing.

### Overfitting is visible in-flight

Dev EER improves 2.97% → 2.36% while WaveFake EER *worsens* 27.7% → 30.1% over
the same epochs. Note the consequence for methodology: **early stopping on dev
EER selects the most domain-overfit checkpoint.** Stopping on dev EER rather
than dev loss is right as far as it goes, but dev is in-domain, so it cannot see
this. A cross-domain selection signal is the correct fix.

---

## 2. Calibration

Fitted on the `calib` fold — 17,764 rows, disjoint from both train and dev.

| Bucket | n | EER | a | b | Cllr | min-Cllr | calibration loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| uncalibrated | 17,764 | 2.34% | — | — | 0.174 | 0.092 | **0.082** |
| clean | 8,882 | 1.52% | 0.510 | −0.507 | 0.062 | 0.055 | **0.007** |
| G.711 | 8,882 | 3.30% | 0.411 | −1.289 | 0.138 | 0.120 | **0.017** |
| pooled | 17,764 | — | 0.438 | −1.059 | 0.103 | 0.092 | 0.011 |

Calibration loss falls **0.082 → 0.011**. Per-bucket fitting is justified rather
than assumed: G.711 more than doubles EER (1.52% → 3.30%) *and* needs a
different slope (0.510 vs 0.411), so a single pooled mapping would be
mis-calibrated on both.

---

## 3. Effective sample size — the largest correction in the project

> ### ⚠ RETRACTED — see §15. The number below (η = 0.080) is wrong.
>
> It was measured while `DualRateChunker.push()` emitted byte-identical windows
> for any push spanning more than one tick. Every window in every sequence
> carried the same audio, the branch scores differed only by floating-point
> non-determinism (median std **1.8e-15**), and the "autocorrelation that barely
> decays" below is an autocorrelation of that noise. The measured values, after
> the chunker fix, are **η_A = 0.207, η_A2 = 0.928, η_B = 0.565** (§15).
>
> The section is kept rather than deleted because the *reasoning* it contains —
> why call-level consistency matters more than window overlap, and why the
> estimate must be fit over whole calls — survives, and because a project that
> silently rewrites a retracted number is harder to trust than one that shows
> the retraction.

Measured from streamed overlapping windows over 400 synthetic calls
(same speaker, same label, same generator; ~36 windows each):

```
eta = 0.080          ->  12.5x variance inflation
autocorrelation:  lag1 +0.824   lag2 +0.780   lag3 +0.751   lag4 +0.727
                  lag5 +0.703   lag6 +0.679   lag7 +0.655   lag8 +0.630
```

The plan predicted roughly √2 from window overlap. The measured effect is far
larger, and the profile says why: correlation **barely decays** across eight
lags. That is not a window-overlap signature — overlap couples only adjacent
windows, since a 1 s window at a 0.5 s hop shares nothing with its neighbour
twice removed. It is **call-level consistency**: one speaker, one channel, one
generator makes an entire call close to a single observation.

So ~12.5 windows count as one independent look. Without this correction every
confidence bound would be ~3.5x too tight.

Two caveats stated rather than buried: the estimate truncates at lag 8, and the
profile is still strongly positive there, so 0.080 is an **upper bound** on
independence — the true value may be smaller. And the synthetic calls are built
by concatenating clips, so splice points add noise the estimate would not see in
a real call.

**It does not break alerting.** Substituting the measured 0.080 for the assumed
0.6 widens the decision margin from 1.90 to 5.21 nats and changes no decision:
alerts still fire at 2.0 s (spoof from t=0) and 16.0 s (mid-call swap), and
never on a genuine call.

---

## 4. End-to-end on real audio

Real ASVspoof audio streamed in 0.5 s increments through the full pipeline:
chunker → Branch A → calibration fitted on held-out data → fusion engine →
tiered decision. Forty calls per scenario, built by concatenating same-speaker
clips (ASVspoof clips are 1–4 s; a call is not). No tuning.

| Scenario | n | alerted | median time-to-alert | median S_lo | change points |
|---|---:|---:|---:|---:|---:|
| bona fide | 40 | **0%** | — | −49.42 | 0.17 |
| spoof from t=0 | 40 | **100%** | 5.5 s | +60.61 | 0.00 |
| mid-call swap | 40 | **100%** | **3.7 s after the splice** | +60.75 | 1.10 |

- **Per-call FPR 0/40, TPR 40/40.**
- Separation is wide: −49 vs +61 nats.
- The swap case is the one a single-shot classifier cannot express at all. The
  change-point detector fires on 1.10 calls on average there, against 0.17 on
  genuine calls and 0.00 on calls spoofed from t=0 — which is exactly right: a
  call that was synthetic from the first window contains no *change*.

These are 40-call samples on one corpus, so the honest reading is "0/40", not
"0% FPR" — the upper 95% bound on a 0/40 observation is about 7%.

---

## 5. Throughput

See `docs/BENCHMARKS.md`. Summary: **~400 concurrent calls GPU-side** on one
`mig-1g.18gb` slice at batch 32, p99 55 ms, 2.1 GB of 16 GB VRAM, 150 GPU-ms per
call-minute. That is the GPU ceiling; end-to-end capacity is CPU-bound and is
the load test's job to establish.

---

## 6. Branch B (prosody) — measured, and it does not support the blueprint's claim

The v3 blueprint calls Branch B *"your main defence against the survives-
compression requirement"* and *"the branch that matters most for your specific
problem"*. Measured on ASVspoof 2019 LA, that claim does not hold.

**Two separate problems, established separately.**

### (a) The one-class model collapses

Deep-SVDD's failure mode is the encoder learning a constant function: every
input maps to the centre, loss goes to zero, and the score carries no
information. Measured on the first run:

```
per-dimension embedding std   0.0057   (collapse)
squared distance, bona fide   0.00289
squared distance, spoof       0.00386
loss                          0.023    <- looks like success
dev EER                       49.8%    <- is chance
```

Loss alone cannot distinguish collapse from success — it looks *better*. A
`spread` check now runs every epoch and aborts the run (`COLLAPSE_STD = 0.05`).

The standard fix — bias-free convolutions, `affine=False` BatchNorm, and
removing the codec embedding (a learnable constant selected by a categorical
input, and the most direct collapse path of the three) — **was not sufficient**.
Spread improved from 0.0057 to 0.0116, still collapsed. The guard caught it at
epoch 0.

### (b) The features are weakly informative here anyway

A one-class failure is ambiguous: optimisation problem, or no signal? Supervised
learning on the *same* features is strictly easier, so it bounds what any
one-class model could achieve (`scripts/diag_prosody.py`):

| Model on prosody statistics | dev EER |
|---|---:|
| Logistic regression | **39.39%** |
| Gradient boosting | 41.04% |

**~39% EER is a ceiling, not a result** — about 11 pp from chance. Per feature:

| Feature | EER | distance from chance |
|---|---:|---:|
| log-F0 std | 58.82% | 8.8 pp |
| longest pause | 57.70% | 7.7 pp |
| voiced ratio | 43.12% | 6.9 pp |
| F0 jitter | 55.20% | 5.2 pp |
| everything else | 45–51% | < 5 pp |

The directions are physically sensible — synthetic speech has *lower* pitch
variability and *shorter* pauses than genuine speech — which is evidence the
features are measuring the right thing, just weakly.

### What follows

1. **Retract the "main defence" claim.** Branch B's fusion weight (0.30) and its
   hard clip (κ_B = 2.0 nats) were set defensively before any of this was
   measured; both now look correctly conservative rather than merely cautious.
2. **The compression claim is untested, not disproven.** Prosody's asserted
   advantage is that it *survives codec degradation* better than fine spectral
   detail. ASVspoof LA is not a codec benchmark, so it cannot test that. RTCFake
   can — real transmission through seven platforms — and it was lost in the pod
   wipe. Until it is re-fetched and evaluated, the compression argument is a
   hypothesis with no measurement behind it and must be presented as one.
3. **Branch B stays in**, weighted low and honestly described, because its
   independence from having-seen-the-attack is still worth something in fusion
   even at a low individual ceiling.

---

## 7. Branch C (identity) — AS-Norm cohort, and a sampling bug caught before it shipped

Fitted on ASVspoof 2019 LA: cohort from `train` (bona fide only, identity
calibration must never see synthetic speech since its speaker label is
ambiguous by construction), trials from `calib`.

**Bug found while fitting.** `embed_fold()` selected the cohort as
`bona_fide_rows[:limit]`. The feature cache stores rows in file order, which is
grouped by speaker — each speaker's clips run consecutively — so a positional
prefix silently draws almost entirely from the first few speakers in the file
rather than the full roster. Measured: `--cohort 1500` against 10,320 available
bona fide rows across 20 speakers returned embeddings from **3 speakers**. An
AS-Norm imposter cohort exists specifically to characterise "typical" score
against a broad population; a 3-speaker cohort is close to the opposite of
that, and would have shipped silently — the fit completes, prints a plausible
EER, and gives no signal that anything is wrong.

Fix: stratify by speaker before truncating (`per_speaker = limit // n_speakers`,
shuffled within each speaker, sorted after selection for cache locality). Same
`--cohort 1500` now covers all 20 speakers, and the identity EER measurably
improved:

| Cohort | Speakers | AS-Norm EER | Cllr | calibration loss |
|---|---:|---:|---:|---:|
| Naive prefix (bug) | 3 | 9.75% | 0.336 | 0.008 |
| Stratified (fix) | **20** | **7.98%** | **0.283** | 0.007 |

The calib-side trial pool (7 speakers, 1,708 bona fide rows) is a genuine data
ceiling, not a sampling artifact — that is the entire bona fide population of
that split, confirmed by checking it directly rather than assumed. No fix
increases it; a larger calib fold would need a different split allocation.

---

## 8. Per-subset calibration — Branch A2's fusion value, measured directly

Fitted on the `calib` fold (17,764 rows, disjoint from train and dev):

| Subset | α | scale | EER |
|---|---:|---:|---:|
| A | −0.007 | 0.991 | 2.52% |
| **A+A2** | 0.166 | 0.960 | **1.46%** |
| A+A2+B | 0.289 | 0.957 | 1.46% |
| A+B | 0.112 | 0.988 | 2.50% |
| A2 | 0.000 | 2.222 | 8.08% |
| A2+B | 0.123 | 2.234 | 7.96% |

Two things this table settles that were previously argued, not measured:

**Branch A2 (phase) earns its place.** A alone: 2.52% EER. A+A2: **1.46%** — a
42% relative reduction from adding a single branch. This is the first direct
evidence that the phase branch is not redundant with the SSL branch, on this
corpus, at fusion time.

**Branch B contributes nothing once A is present.** A+A2 → A+A2+B: 2.52% → 2.52%
unchanged to two decimal places. A → A+B: 2.52% → 2.50%, indistinguishable from
noise. This is consistent with §6's finding that prosody's supervised ceiling on
this corpus is ~39% EER: a branch that weak cannot move a fused score that A
already gets to 1.46%. It does not contradict keeping Branch B in the system —
its independence from having-seen-the-attack, and its still-unmeasured
performance under real codec degradation (RTCFake, lost in the pod wipe and not
yet re-fetched), are different questions than "does it help fusion on ASVspoof
LA today." On this data, at this operating point, it does not.

---

## 9. Four-branch end-to-end on real audio — the first time all four ran together

Every prior end-to-end run (§4) exercised Branch A alone — the exact
configuration that hid bug E24. `scripts/end_to_end_full.py` streams real
ASVspoof audio through all four branches, the fitted per-subset calibration
table, and the real fusion engine. n=40 per scenario, no tuning:

| Scenario | alerted | median time-to-alert | median S_lo | change pts/call |
|---|---:|---:|---:|---:|
| bona fide | **0/40** | — | −29.84 | 0.00 |
| spoof from t=0 | **40/40** | 3.5 s | +6.33 | 0.00 |
| mid-call swap | **40/40** | **3.7 s after splice** | +6.82 | 1.43 |

**Per-call FPR 0/40, TPR 40/40** — matching the single-branch result exactly on
the headline numbers, with two differences worth noting:

- **Spoof-from-t0 detection got faster: 5.5 s → 3.5 s.** Adding A2 (and, weakly,
  B) accelerates crossing the alert threshold, as expected from three
  independent-ish sources of evidence rather than one.
- **Change points per swap call rose from 1.10 to 1.43.** With C's turn-detector
  now live alongside CUSUM, both fire independently and slightly more often
  register a (correct) change — not a false-positive signal, since the swap
  detection rate and timing stayed clean, but worth watching as more branches
  come online.

`subsets seen` across every call was `{A, A+A2, A+A2+B}` — confirming Branch C
never once entered the spoof-score subset, exactly as the design requires.

> **Stale caveat, corrected.** This section used to carry forward §3's
> "only Branch A's η (0.080) is measured; A2 and B run on unmeasured defaults".
> Both halves are now wrong: **η = 0.080 was an artifact** of a chunker bug and
> is retracted (§15), and all three branches are measured (A 0.207, A2 0.928,
> B 0.565). The run below predates that, so its absolute numbers are from the
> old configuration; §15 reports the re-run, which holds at 0/40 and 40/40 and
> is faster. Kept rather than deleted so the sequence of measurements stays
> readable.

---

## 10. Codec robustness — the compression claim, finally measured

`docs/RESULTS.md` §6 retracted the blueprint's "Branch B is the main defence
against the survives-compression requirement" claim, but recorded the
compression argument itself as **untested, not disproven**. This tests it.

RTCFake would have been the ideal instrument (real transmission through seven
live platforms) but its HuggingFace repo is **gated behind owner approval**, so
it is unavailable. The substitute is a paired design — identical utterances,
identical speakers, only the channel changes — which for this specific question
is arguably cleaner, because any EER movement is attributable to the codec
rather than confounded with a corpus or platform shift.

Dev fold, 5,321 windows, CIs bootstrapped over utterances:

| Channel | Branch A | Branch A2 | Branch B |
|---|---:|---:|---:|
| clean | **1.39%** [1.0–1.8] | 2.86% [2.5–3.5] | 43.36% [41.0–45.8] |
| G.711 µ-law | **2.83%** [2.3–3.6] | 10.79% [9.9–12.2] | 40.71% [38.7–42.9] |
| noisy line | **5.84%** [4.9–6.6] | 30.15% [28.5–31.6] | 43.40% [41.1–45.5] |

Degradation relative to clean:

| Channel | A | A2 | B |
|---|---:|---:|---:|
| G.711 µ-law | 2.04× | 3.77× | **0.94×** |
| noisy line | 4.21× | **10.54×** | **1.00×** |

### The blueprint was right about the mechanism and wrong about the conclusion

**The mechanism holds exactly as claimed.** Prosody is essentially immune to
codec degradation — Branch B moves 0.94× and 1.00×, i.e. flat within noise —
while spectral detail degrades 2–4× and phase degrades 4–10×. The physical
argument in the v3 blueprint was correct.

**The conclusion does not follow.** Robustness of a near-useless signal does not
make it useful. Branch A remains dramatically better than Branch B at every
severity tested — 5.84% vs 43.40% even on the harshest simulated channel. A's
EER never crosses B's, so **there is no channel condition tested here under
which Branch B becomes the main defence.** The claim fails not because prosody
is fragile but because it starts near chance and stays there.

This is the difference between a *relative* argument and an *absolute* one, and
the blueprint made the relative one while stating an absolute conclusion.

### New finding: Branch A2 is the channel-fragile one

Unflagged by any prior analysis. The phase branch earned its place on clean
audio (§8: A+A2 reaches 1.46% vs 2.52% for A alone) but is by far the most
channel-sensitive component in the system — **10.54× degradation** on the noisy
line, 2.86% → 30.15%, which is approaching useless.

That has a direct architectural consequence. Branch A already has per-channel
calibration buckets (`A@clean`, `A@g711`, each with a measurably different
slope: 0.510 vs 0.411). **Branch A2 has only a single pooled fit**
(`A2@pooled`), which means under degraded channels its LLRs are calibrated
against the wrong distribution — precisely where its behaviour changes most.
Fixing that is §11.

### Honest limit

A fixed simulator does not reproduce what real platform transmission adds:
adaptive noise suppression, echo cancellation, and bursty rather than uniform
packet loss. This measures the **mechanism**, not the deployment. The stronger
claim — that Branch B contributes nothing on *real* RTC transmission — still
requires RTCFake and remains untested.

---

## 11. Per-channel calibration for Branch A2 — closing the gap §10 exposed

§10 showed Branch A2 is the most channel-sensitive component in the system
(10.54× EER degradation on a noisy line) while carrying only a single pooled
calibration. Fitted per-channel on the calib fold, the same way Branch A already
was:

| Branch | Channel | n | EER | slope `a` | Cllr | calibration loss |
|---|---|---:|---:|---:|---:|---:|
| A2 | clean | 8,882 | 2.50% | **0.653** | 0.105 | 0.017 |
| A2 | G.711 | 8,882 | 13.15% | **0.390** | 0.431 | 0.037 |

**Slope spread 1.67×** — a pooled mapping mis-scales a G.711 call's LLR by
roughly that factor. For comparison, Branch A's spread is 1.24× (0.510 vs
0.411), so A2 is not just more fragile in EER, it is more channel-dependent in
*calibration* too. Two independent lines of evidence for the same conclusion.

Worth noting how large the uncalibrated gap is: A2's raw Cllr is **0.973** —
essentially no usable information as a probability — against a min-Cllr of
0.248. Almost everything A2 contributes comes from calibration, which is exactly
why calibrating it against the wrong channel matters.

### Gateway consequence

The serving gateway previously hardcoded `A@clean` / `A2@pooled`. It now selects
the bucket via `VERICALL_CHANNEL` (default `clean`), falls back to pooled with a
visible note when a bucket is missing, and reports `channel_bucket` and
`runtime_codec_detection: false` in `/health`.

**Runtime codec detection is not implemented.** Defaulting to `clean` is correct
for the current demo (uncompressed audio) and wrong the moment real telephony
audio arrives. That is now visible in the health endpoint rather than silently
assumed — but it is a real remaining gap, not a solved problem.

---

## 12. Serving capacity — measured, not extrapolated

The concurrency claim had never been evidenced: `serving/app.py` was a mock
returning `random.uniform(-2.0, 2.0)` and had never been run under load. §12
replaces the estimate with a measurement.

Criterion declared **before** the run, per the project's own gate discipline:
all calls complete, `chunk_inference` p99 ≤ 250 ms, `time_to_first_score`
p99 ≤ 1.6 s, and the server keeps up with real time (wall clock ≈ audio
duration). 12 s of paced audio per call, one `mig-1g.18gb` slice co-tenanted
with another project.

| N | wall (12 s audio) | first_score p50 / max | chunk p99 | verdict |
|---:|---:|---:|---:|---|
| 1 | 13.0 s | 1.11 / 1.11 s | 104.5 ms | pass |
| 2 | 13.1 s | 1.13 / 1.15 s | 104.5 ms | pass |
| **4** | **13.1 s** | **1.36 / 1.38 s** | **103.9 ms** | **pass** |
| 8 | 18.1 s | 1.36 / 1.51 s | 101.4 ms | fails real time |
| 16 | 18.2 s | 2.71 / 2.86 s | 101.6 ms | fails |

**N = 4 concurrent calls** is the largest configuration meeting every SLO
simultaneously.

> **Superseded — see `docs/PRODUCTION_READINESS.md` §9.12.** The measured
> ceiling is now **N = 64** over real WebSockets (chunk p99 22.2 ms, 64/64
> complete) and **N = 160** server-side. The dominant cause of the old figure
> was `torch` sizing its intra-op thread pool from the host core count — 112
> threads on a 16-core pod — which every offline script here already set to 1
> and the serving path never did. The GPU forward also ran synchronously on the
> asyncio event loop, so calls queued behind each other invisibly. This section
> is kept because its *diagnosis* was right — the bottleneck was CPU-side, not
> the GPU — and because the cold-start finding below still stands.

### The bottleneck is not the GPU

`chunk_inference` p99 is flat at ~102–105 ms from N=1 to N=16 — it does not
degrade with load at all. What degrades is wall clock: the server starts falling
behind real time at N=8. The GPU forward is 9–95 ms p50 while the pipeline slips,
so the constraint is CPU-side (feature extraction, VAD, ring buffers, WebSocket
handling), exactly as `docs/BENCHMARKS.md` predicted when it labelled its
~400-call figure a **GPU ceiling, not system capacity**. That caveat is now
confirmed rather than merely stated.

### Cold start was a real defect, now fixed

The first sweep showed p99 **830 ms at N=1 and N=2** — worse than at N=16 —
because with only 22 windows one un-warmed CUDA forward dominates the
percentile. That would have landed on the first call of a live demo. The gateway
now runs each branch once on dummy audio at startup and clears the warmup from
the latency histogram. After the fix, p99 is a stable ~104 ms at every N.

### Honest confound

The load generator runs **on the same pod**, sharing the same 16 CPU cores with
the server it is measuring. At N=16 it is running 16 asyncio tasks doing signal
synthesis and WebSocket I/O in direct competition with the thing under test.
True capacity with the generator off-box is therefore **higher than 4**, by an
unmeasured margin. N=4 is a floor, not a ceiling, and should be reported that
way.

---

## 13. RTCFake — real platform transmission, and a result that changes the architecture

RTCFake pairs each utterance with itself: the same speech **offline** (clean) and
**online** (after passing through a real RTC platform). Same speaker, same
content, same label, only the channel differs. That is a stronger instrument
than §10's simulator, and it is the condition Branch B had never been tested
under. It also splits **en / zh**, and since every model here only ever saw
English (ASVspoof LA), Mandarin is genuine cross-lingual generalisation.

1,200 paired utterances per language, 4 s centre crops, CIs over utterances.

| Lang | Branch | offline (clean) | online (real RTC) | degradation |
|---|---|---:|---:|---:|
| EN | A | 50.21% | 52.57% | 1.05× |
| EN | **A2** | **19.28%** | **39.99%** | 2.07× |
| EN | B | 42.44% | 45.08% | 1.06× |
| ZH | A | **79.08%** | **76.60%** | 0.97× |
| ZH | **A2** | **28.72%** | 50.71% | 1.77× |
| ZH | B | 48.23% | 53.05% | 1.10× |

### Finding 1 — Branch A does not merely fail here, it **inverts**

79.08% EER on Mandarin is not "uninformative". An EER above 50% means
systematically **anti-correlated**: flipped, it would score 20.92%. Branch A has
learned something real and corpus-specific that points the *wrong way* under
domain shift — ranking RTCFake spoofs as more genuine than RTCFake bona fide.

The plausible mechanism, offered as hypothesis not conclusion: trained on
ASVspoof LA, Branch A learned characteristics of *that corpus's* bona fide
recordings. Mandarin bona fide speech does not share them, while RTCFake's
spoofs — from different generators — may resemble ASVspoof bona fide more
closely. This is the shortcut-learning failure mode the silence artifact warned
about, surfacing as inversion rather than as noise.

This makes §2's headline understatement explicit. The domain gap measured on
WaveFake was 12×. On RTCFake it is **total collapse, and on one language worse
than chance.**

### Finding 2 — the branch hierarchy is inverted under domain shift

On ASVspoof, A is the strong branch and A2 the weaker one (2.52% vs 8.08%,
§8). On RTCFake it reverses completely: **A2 is the only branch carrying
transferable signal** (19.28% EN, 28.72% ZH offline) while A is at or below
chance.

That contradicts the design's assumed hierarchy and the usual preference for SSL
front ends. It also sits awkwardly beside §10, where A2 was the most
*channel*-fragile branch. The reconciliation: A2 is fragile to **channel**
(2.07× / 1.77× degradation here under real transmission, confirming §10 on real
data) but robust to **corpus**, and A is the reverse. Those are different axes
and the architecture had only ever measured one of them.

### Finding 3 — the fitted fusion actively **hurts** on this corpus

| | fused | best single branch | |
|---|---:|---:|---|
| EN offline | 39.87% | **19.28%** | 2.1× worse |
| EN online | 47.60% | 39.99% | worse |
| ZH offline | **72.70%** | **28.72%** | 2.5× worse |
| ZH online | 75.18% | 50.71% | worse |

Fusion is worse than its own best branch in **all four conditions**. The reason
is mechanical: the weights (A=1.0, A2=0.45, B=0.30) were fitted on ASVspoof,
where A is strongest, so they put the most weight on the branch that is inverted
here. Fusion faithfully amplifies the worst signal available.

**The weights are corpus-specific and do not transfer.** That is a more serious
limitation than any single branch's EER, because it means the fusion layer —
the part of this system that is otherwise its strongest engineering — cannot be
fitted once and deployed.

### Finding 4 — Branch B's last justification is gone

B is at **42–53% EER in every condition**, including real platform transmission.
That was the one untested claim left after §6 and §10. It is now tested and it
failed: Branch B contributes nothing to fusion (§8), is not the compression
defence it was claimed to be (§10), and does not transfer (here).

**Recommendation: cut Branch B, or carry it at zero weight and say why.** A
three-branch system that knows why it is three branches is a stronger position
than a four-branch one carrying a dead limb.

### What this does and does not mean

It does **not** mean the system is broken. In-domain and on the demo path the
numbers hold: 0/40 false positives, 40/40 detections, mid-call swap caught 0.8 s
after the splice. Those were measured on ASVspoof and remain true of ASVspoof.

It **does** mean the honest claim is narrower than the architecture implied.
This system detects ASVspoof-family attacks well and does not currently transfer
to RTC-transmitted speech or to Mandarin. Fixing that is a training-data
problem — RTCFake has a train split of 13.7 GB sitting on scratch — not an
architectural one, and no rearrangement of branches addresses it.

### A correction to this script's own verdict

The first version of `eval_rtcfake.py` concluded "A crosses B, so the
compression claim holds in some regime." That was technically true and
substantively wrong: **A crossed B by collapsing**, not by B being good. The
verdict now requires B to be usable in absolute terms (<40% EER) before crediting
it, and reports A's inversion separately. This is the same class of error as the
39.4% prosody threshold in §6 — a comparison that passes while measuring nothing.

---

## 14. Acting on §13 — Branch B zero-weighted, and a guard for silent failure

§13 produced two recommendations. Both are now implemented, and implementing
the first exposed a bug.

### Branch B carries zero weight

`FusionConfig.w["B"]` is **0.0**, on measurement: ~39% supervised ceiling on
clean ASVspoof (§6), no change to fused EER when added to A+A2 (§8), flat but
useless under simulated degradation (§10), 42–53% EER on real RTC transmission
including cross-lingual (§13). The prosody code stays — the front end is sound
and a future corpus may vindicate it — but it does not contribute to a decision
until something measures it doing so.

**A bug found while doing this.** Setting the weight to zero was not sufficient.
Two otherwise-identical calls, one receiving Branch B events, scored **9.0 vs
3.0** — because `S = T_banked × mean_rate` and every accepted event incremented
`T_eff` regardless of weight. A zero-weight branch was still inflating the score
through credited time. Two further consequences handled: a zero-weight branch is
now excluded from `available()`, so it cannot change *which* subset calibration
is selected (with B live, a call would otherwise be scored against the A+A2+B
table whose scale was fitted while B carried 0.30).

### A guard for the failure mode that was silent

The serious part of §13 was not a bad number. It was that fusion scored **worse
than its own best single branch** (72.70% vs 28.72% on Mandarin) while reporting
a confident result, because the weights put the most weight on a branch that had
inverted. Nothing detected it.

`vericall/fusion/domain_guard.py` watches the two signatures visible in that
data: **score drift** (branch means far from the distribution the Platt
parameters were fitted on — the mapping is extrapolating) and **branch
disagreement** (A and A2 were fitted as correlated, so sustained opposite signs
means one has inverted).

Validated on real scores rather than on the synthetic signatures it was designed
from (`scripts/validate_domain_guard.py`):

| Data | A mean LLR | A2 mean LLR | Guard |
|---|---:|---:|---|
| ASVspoof dev (in-domain) | −5.10 | −8.23 | **quiet** |
| RTCFake en (out-of-domain) | −4.58 | **+1.24** | **suspect** — disagreement |
| RTCFake zh (out-of-domain) | **+3.08** | +0.20 | **suspect** — drift, 6.6 sd |

The two detectors are complementary, and each caught a case the other missed. On
English, A's mean stayed inside its distribution so drift did not fire, but the
sign split against A2 did. On Mandarin, A inverted hard enough — **+3.08 nats on
bona fide audio, i.e. confidently calling genuine Mandarin speech synthetic** —
that drift caught it outright.

When tripped it latches (later agreement does not restore trust in evidence
already banked), forces confidence to LOW, halves the confidence score, and
surfaces the reason on the `RiskView`. It is opt-in: a state with no reference
distribution behaves exactly as before, because a guard with a fabricated
reference fires for reasons unrelated to the data.

**What it does not do:** detect domain shift in general. It detects the specific
failure that actually occurred, which is worth more than a general claim that
does not fire.

---

## 15. η, re-measured — and the chunker bug that invalidated §3

§3 called η "the largest correction in the project". It was, in a way it did not
intend: the measurement itself was wrong, and finding out why exposed a defect
in the streaming path that no test covered.

### What broke

`DualRateChunker.push()` appended the whole input to the ring buffer and then
cut every window that had become due with `buffer.tail(n)` — which reads the
**end** of the buffer. A push spanning several ticks therefore emitted N windows
whose `t_start` advanced correctly (0.0, 0.5, 1.0 …) and whose audio was
byte-identical.

`fit_calibration.py` measured η by concatenating eight clips into a ~14 s call
and pushing it **in one call**. Measured directly on 40 such calls:

| | windows identical within a call | median score std | η |
|---|---:|---:|---:|
| single push (what §3 did) | **40/40** | 1.8e-15 | 0.150 |
| incremental 0.5 s pushes | **0/40** | 1.80 | 0.367 |

A score std of 1.8e-15 is float non-determinism in the GPU forward, not signal.
So §3's headline — "correlation barely decays across eight lags, therefore
call-level consistency" — described the autocorrelation of numerical noise.

The streaming callers (the serving gateway, `end_to_end.py`) push well under one
hop at a time and were never affected, which is exactly why this survived: every
number produced by the *streaming* path is unaffected, and only the batch
measurement was wrong.

**Why no test caught it.** There was exactly one chunker test. It counted
windows and checked timestamps, and never looked at what audio a window
contained. Five tests added, the strongest being that a single bulk push must
produce byte-identical output to many small pushes.

The fix is in the chunker, not the callers: `push()` now feeds its input in
slices that never span a tick, so how a caller chooses to slice its audio cannot
change a single emitted sample. (A one-sample off-by-one appeared during the
fix — `ceil((next_tick − now) × sr)` evaluates to 17 rather than 16 because
`now` is a ratio of integers and lands a few ulps high. Caught by the new
content test, not by any cadence test.)

### The real numbers

`scripts/measure_eta_branches.py`, 800 synthetic calls from the `calib` fold,
each branch scored **on the lane it actually runs on** — scoring Branch B on the
fast lane would measure a configuration that never runs:

| Branch | Lane | η | variance inflation | old default | lag-1 … lag-8 |
|---|---|---:|---:|---:|---|
| A | fast 1.0 s / 0.5 s | **0.207** | 4.8× | 0.6 | +0.38 +0.28 +0.27 +0.27 +0.26 +0.25 +0.23 +0.23 |
| A2 | fast 1.0 s / 0.5 s | **0.928** | 1.1× | 0.6 | +0.03 −0.04 −0.03 −0.03 −0.03 −0.03 −0.03 −0.02 |
| B | slow 3.0 s / 1.0 s | **0.565** | 1.8× | 0.9 | +0.34 +0.03 −0.15 −0.10 −0.10 −0.08 −0.07 −0.07 |

Stable across sample size: at 400 calls the same run gives 0.187 / 0.945 / 0.575.
(A 40-call subsample gave 0.367 for A, so the estimate is sample-size sensitive
and 800 calls is what is reported.)

**§3's qualitative claim survives for Branch A, in weaker form.** A's profile
still does not decay to zero across eight lags — +0.38 falling only to +0.23 —
and window overlap alone cannot produce that, since a 1 s window at a 0.5 s hop
shares no samples at all with its neighbour twice removed. Call-level
consistency is real. It is worth 4.8× variance inflation, not 12.5×.

**The three branches do not move together, which the old single-η design
assumed they would.**

- **A2 is nearly independent (0.928).** Phase residual is a local property and
  decorrelates within one hop. Crediting it at 0.6 was throwing away over a
  third of its evidence.
- **B's profile tracks its own geometry almost exactly.** At a 3 s window and
  1 s hop, adjacent windows share 2/3 of their audio, lag-2 shares 1/3, lag-3
  shares none — and the measured profile is +0.34, +0.03, −0.15. That the
  numbers reproduce the overlap fractions is the best evidence available that
  this measurement is reading signal rather than another artifact.

### Consequence for the system

`FusionConfig.eta` now carries the measured values. It previously carried
`{A: 0.6, A2: 0.6, B: 0.9}` — **the 0.080 was never adopted**, so the fusion
engine never ran on the bad number; only the documentation did.
`end_to_end_full.py` had been overriding A's η from `calib/summary.json`, which
would have silently reinstated 0.080 in the one script whose job is to check the
end-to-end decision. That override is removed, and the stale field in
`calib/summary.json` is renamed `eta_SUPERSEDED` with the reason attached,
rather than left readable as if it were right.

### End-to-end under the corrected η

`end_to_end_full.py`, four branches, real ASVspoof audio streamed at the tick
rate, 40 calls per scenario, no tuning:

| Scenario | alerted | median time-to-alert | median S_lo | change pts/call |
|---|---:|---:|---:|---:|
| bona fide | **0/40** | — | −24.57 | 0.00 |
| spoof from t=0 | **40/40** | **2.5 s** | +11.52 | 0.00 |
| mid-call swap | **40/40** | **2.4 s after the splice** | +11.57 | 1.43 |

The result holds, and detection got **faster** — 3.5 s → 2.5 s, and 3.7 s → 2.4 s
after a splice. That is not a free lunch from a looser correction: Branch A's η
fell (0.6 → 0.207, crediting less), but A2's rose (0.6 → 0.928, crediting more),
and A2 is the branch whose evidence was being discarded. Measuring each branch
separately is what surfaced that; a single global η cannot express it.

`subsets seen` is `{A, A+A2}` throughout — Branch C never enters the spoof-score
subset, and zero-weighted Branch B is excluded from the available set, both as
the design requires.

---

## 16. Retraining on RTCFake — the inversion is gone, and the problem has moved

§13 found Branch A **inverting** under domain shift (79.08% EER on Mandarin) and
recorded that the fix was a training-data problem, not an architectural one.
RTCFake ships a train split; this tests that claim.

### Protocol

Branch A retrained on **RTCFake English only**, three seeds. Folds carved from
RTCFake's own train split; RTCFake's own dev split becomes the evaluation,
untouched. **Every Mandarin item is dropped from training** — not down-weighted,
dropped — so `eval_rtc_zh` stays a genuine cross-lingual test and "did retraining
fix the inversion" stays answerable. Offline/online pairs share a fold, since
they are the same speech through different channels.

Three configurations, scored against every fold through one code path
(`scripts/eval_folds.py`), so a row and a column are subtractable:

| Branch A trained on | ASVspoof eval | WaveFake unseen | RTCFake en | RTCFake zh |
|---|---:|---:|---:|---:|
| **ASVspoof** (baseline) | **11.35%** | **30.08%** | 45.57% | **55.34%** ⚠ inverted |
| **RTCFake-en** | 36.54% | 49.36% | **22.93%** | **33.34%** |
| **ASVspoof + RTCFake-en** | **9.12%** | 49.89% | 25.03% | 37.21% |

The baseline row reproduces the published numbers (11.01%, 30.29%) within
bootstrap noise, which is the check that the rebuilt caches are consistent with
the ones every earlier section was measured on.

### The §13 paired protocol, re-run

The matrix above uses 1 s cached crops. §13 used 4 s centre crops over 1,200
paired utterances, so only this re-run is directly subtractable from §13's
published table — same script, same crops, same utterances, only the checkpoint
differs:

| Branch A trained on | EN offline | EN online | ZH offline | ZH online |
|---|---:|---:|---:|---:|
| ASVspoof (**§13 as published**) | 50.21% | 52.57% | **79.08%** | **76.60%** |
| **RTCFake-en** | **14.23%** | **16.25%** | **9.51%** | **13.57%** |
| ASVspoof + RTCFake-en | 17.34% | 18.59% | 11.67% | 18.17% |

**Mandarin goes from 79.08% to 9.51% on a language that was never trained on.**
The inversion is not merely reduced, it is gone: every cell is far below 50%, so
every branch now points the right way.

### Finding 1 — the fix works, and it is a data fix

§13's diagnosis was right. No architectural change was made: same frozen WavLM
layer 5, same attention head, same 0.89 M trainable parameters. Only the
training corpus changed.

### Finding 2 — the problem moved rather than dissolved

The RTCFake-trained model scores **36.54% on ASVspoof eval and 49.36% on
WaveFake** — chance. It did not learn a general synthetic-speech detector; it
learned an RTCFake-specific one, exactly as the ASVspoof model had learned an
ASVspoof-specific one. **We swapped one corpus-specific detector for another.**

What it did learn transfers *across language within its corpus* (en → zh) but
not *across corpus*. Those are different axes, and only the second is what
deployment needs.

The one guard against this being a trivial artifact: the best duration-only cue
on RTCFake zh is **41.43% EER**, measured on that same corpus
(`scripts/audit_silence.py --dataset rtcfake`). RTCFake carries no silence
artifact at all — leading silence is at chance in both languages — so 9.51% is
not a stopwatch. It is signal. Whether that signal is *synthesis* or *some other
RTCFake-specific regularity* is not settled by this experiment.

### Finding 3 — mixing does not buy both, and it costs WaveFake outright

The mixed model is the best in-domain number in the project (**9.12%** on
ASVspoof eval, better than the ASVspoof-only baseline's 11.35%) and is worse than
the RTCFake-only model on both RTCFake folds. So mixing trades, it does not
combine.

The sharp part is WaveFake: **30.08% → 49.89%**, chance, even though ASVspoof
rows are 64% of the mixed training set. Adding RTCFake did not dilute the
WaveFake signal proportionally — it destroyed it. The plausible mechanism,
offered as hypothesis: with RTC-transmitted audio present, channel becomes an
available discriminator, and WaveFake is clean single-speaker studio resynthesis
with no channel to read. A naive concatenation of corpora lets the model pick the
cue that is cheapest across the mixture, not the one that generalises.

**This is an argument against "just add more data" as the generalisation
strategy**, which is where §13 left the recommendation. Balanced sampling,
domain-adversarial training, or per-domain heads are the next things to try;
none of them is tested here.

### Finding 4 — fusion's failure mode is now visible in both directions

§13 reported fusion scoring worse than its own best branch in all four
conditions. With the retrained Branch A, on the same fixed weights
(A = 1.0, A2 = 0.45, B = 0):

| | fused | best single | |
|---|---:|---:|---|
| EN offline | **10.12%** | 14.23% | **helps** |
| EN online | 17.75% | 16.25% | hurts |
| ZH offline | 17.73% | **9.51%** | hurts, 1.9× |
| ZH online | 23.95% | 13.57% | hurts |

Fusion now helps in one condition and still hurts in three, and the reason is
the same in both directions: **the weights encode a branch hierarchy that no
longer holds.** They were fitted where A and A2 were comparable. A is now far
stronger on RTCFake while A2 — still the ASVspoof-trained phase head — is at
28.72% on zh against A's 9.51%. Giving A2 a 0.45 weight against a branch three
times better dilutes it.

The conclusion from §13 stands and is now better evidenced: **fusion weights are
corpus-specific and cannot be fitted once and deployed.** Refitting them per
deployment, or making the weights a function of measured per-branch reliability
rather than a constant, is unbuilt.

### Finding 5 — Branch B is unchanged and still at chance

42.44% / 45.08% (en) and 48.23% / 53.05% (zh), identical to §13 because it was
not retrained. Its zero weight stands.

### What is still missing

Branch A2 and Branch C were **not** retrained. A2 is the branch §13 identified
as the only transferable one, and it is now the weakest link on RTCFake. Running
this same pipeline for A2 is the obvious next step and was not done.

---

## 17. Retraining A2 and refitting fusion — the §16 diagnosis, acted on

§16 finding 4 said fusion hurt because *the weights encode a branch hierarchy
that retraining changed*, and §16's closing note said Branch A2 was never
retrained. Both are now addressed, and the diagnosis holds up.

### Branch A2 on RTCFake

Same pipeline, `--branch A2`, three seeds, Mandarin still held out of training:

| | dev | eval_rtc_en | eval_rtc_zh |
|---|---:|---:|---:|
| A2 retrained on RTCFake-en | 15.48% ± 0.30 | **20.12% ± 0.58** | **30.47% ± 0.68** |

Worth noting what did *not* happen: the ASVspoof-trained A2 already scored
19.28% / 28.72% on the §13 paired protocol. Retraining A2 on RTCFake changes its
RTCFake performance barely at all. That is consistent with §13 finding 2 — **A2
is the corpus-robust branch** — and it means A2's value here was never the
problem. The problem was everything downstream of it.

### Refitting the fusion layer on RTCFake

Both retrained heads, Platt fitted on the RTCFake `calib` fold, per-subset table
refitted (`calib_rtc/`, kept separate so the ASVspoof table is not overwritten —
applying one head's calibration to another head's scores produces confident
nonsense, not a degraded number):

| subset | α | scale | EER on calib |
|---|---:|---:|---:|
| A | −0.001 | 1.001 | 14.93% |
| **A+A2** | −0.184 | 1.055 | **10.40%** |
| A2 | −0.000 | 2.222 | 17.45% |

### The result, on held-out eval folds

Not the calib fold the table was fitted on:

| fold | A | A2 | A+A2 fused | vs best single |
|---|---:|---:|---:|---|
| eval_rtc_en | 22.93% | 20.55% | **16.07%** [15.0–17.6] | **helps, 1.28×** |
| eval_rtc_zh | 33.34% | 29.52% | 29.89% [28.9–31.0] | tie (1.01×) |

Against §16's four conditions where fusion helped in one and hurt in three:
**fusion now helps decisively in-language and is indistinguishable from its best
branch cross-language.** The Mandarin fused CI (28.86–30.98) contains the best
single branch's 29.52%, so that cell is a tie, not a loss.

### The residual is the same failure, one level up

English is where the calibration was fitted; Mandarin is not. The branches
transfer to Mandarin (A 33.34%, A2 29.52% — both far below the 41.43%
trivial-cue baseline on that corpus) but the **combination rule** gains nothing
there. The per-subset (α, scale) is itself a fitted object, and it is fitted on
one language.

That is the §13/§16 finding recurring one layer up: every fitted component of
this system — branch heads, Platt mappings, subset tables, fusion weights —
carries the domain it was fitted on. Retraining fixed the heads. Refitting fixed
the table for the language it was refitted on. Neither buys transfer for free,
and there is no layer at which this stops being true.

**What that implies for deployment**, stated as an implication and not as a
built feature: the calibration artifacts are cheap — the whole subset table is a
couple of kB of JSON and refits in seconds on cached branch outputs, no GPU — so
refitting per deployment channel and language is practical in a way that
retraining is not. A deployment procedure that fits calibration on a sample of
the tenant's own traffic is the obvious design. ~~It is not implemented.~~
**Implemented: `scripts/fit_tenant_calibration.py`.** It takes a directory of
the tenant's own labelled recordings, refits the channel mappings, inherits the
cohorts and subset table, seals the result, and is served with
`VERICALL_CALIB_DIR=calib-<tenant>`. It reports Cllr before and after so a
tenant can see whether the refit helped, and it **refuses to write a profile**
when min_Cllr says the detector carries no information on their audio — a
well-calibrated uninformative detector is still uninformative, and a
confident-looking artifact would hide that.

---

## 18. Runtime channel detection — closing the gap §11 opened

§11 fitted per-channel calibration for Branch A2 and ended on an admission:
*"Runtime codec detection is not implemented. Defaulting to `clean` is correct
for the current demo and wrong the moment real telephony audio arrives."* The
gateway reported `runtime_codec_detection: false` in `/health` rather than
pretending. A flag that names a gap is not a fix, and this closes it.

### Three features, and two wrong physics arguments found by measuring

The first version of the detector asserted its thresholds from physics and was
**68% accurate**. Both arguments were wrong, and both are worth recording:

- *"Band-limiting removes high-band energy, so narrowband audio has less of it."*
  True for G.711, **false for the noisy line** — the profile adds 15 dB SNR
  white noise *after* band-limiting, which repopulates the band. Measured, the
  noisy profile carries **more** energy above 4 kHz than clean speech does
  (1.5e-2 vs 4.3e-3 median). It was being classified `clean`.
- *"Packet loss zeroes 20 ms frames, so look for silent frames."* The loss does
  zero them, but mu-law companding runs afterwards and **has no exact-zero
  quantisation level** — zero maps to ~0.0039. Lost packets come back as a
  *constant* frame, not a silent one. The silent-frame test measured 0.0000 for
  every profile including the lossy one.

| feature | what it measures | clean (p5/med/p95) | g711 | noisy_line |
|---|---|---|---|---|
| `high_band_ratio` | energy above 4 kHz | 3.5e-4 / 4.3e-3 / 3.9e-2 | 8.0e-5 / 1.0e-4 / **2.9e-4** | 1.5e-2 / 1.5e-2 / 1.7e-2 |
| `const_frame_ratio` | 20 ms frames of zero variance | 0.000 | 0.000 | 0.020 / 0.060 |
| `noise_floor_ratio` | p5 frame RMS ÷ median | 0.008 / 0.033 | 0.015 / 0.055 | **0.191 / 0.389** |

G.711's p95 (2.9e-4) sits below clean's p5 (3.5e-4), so the band-limit threshold
goes **inside that gap** at 3.15e-4. A grid search preferred 4.98e-4, which is
above clean's p5 — it bought G.711 recall by calling ~5% of clean windows
narrowband, and under the call-level vote those errors cost more than the recall
was worth. Order matters too: noisy is tested **first**, because its added noise
makes the band-limit test call it clean.

Thresholds fitted on 8,000 labelled dev rows and checked on 8,000 held out.
**Split by clip, not by row** — the four crops of a clip cycle the four channel
profiles, so a row-parity split lands exactly on that cycle: the first attempt
put clean+g711 in one half and g711+noisy in the other, and the eval half
contained no clean rows at all. Fit 94.90%, held-out 94.84%.

### The accumulator was worse than its own inputs

| rule | per-window | per-call (6 windows) | clean calls |
|---|---:|---:|---:|
| latch on any narrowband window | 94.54% | **88.41%** | **53.90%** |
| majority vote | 94.20% | **98.80%** | **100%** |

The first design **latched**: one window read as narrowband fixed the call as
narrowband permanently, on the argument that a real channel does not become
wideband mid-call. The argument is sound and the rule was still wrong, because it
turns any single per-window false positive into a permanent call-level error.
53.9% on clean calls is almost exactly 1 − 0.9⁶ — the probability that at least
one of six windows misfires.

**An accumulator that scores below its own inputs is not an accumulator.** Per
call, the vote reaches **98.80%** (clean 100%, g711 97.67%, noisy_line 99.85%).

### Per-channel calibration, now for all three buckets and both branches

`calib/` previously held `clean` and `g711` only, so the gateway could *detect* a
noisy line and had nothing to map it with — which `/health` said out loud.
Fitted on the calib fold re-extracted with all four channel profiles:

| branch | clean | G.711 | noisy_line | slope spread |
|---|---|---|---|---:|
| A | 1.62%, a=0.447 | 3.16%, a=0.405 | 6.32%, a=0.385 | 1.16× |
| **A2** | 2.34%, a=0.588 | 12.70%, a=0.401 | **32.96%, a=0.278** | **2.11×** |

A2's spread widens from the 1.67× of §11 once the noisy bucket is included —
more evidence for §10's finding that A2 is the channel-sensitive component, and
its noisy-line EER (32.96%) reproduces §10's independent 30.15%.

### What it buys

Dev fold, 63,848 rows. **Each condition gets its own subset (α, scale) fitted on
calib under that same condition** — sharing one table across conditions is a
confound, and the first run of this measurement did exactly that and reported
that per-channel calibration made fusion *worse*. It had measured the mismatch,
not the feature.

| condition | A calib loss | A2 calib loss | fused Cllr | fused EER |
|---|---:|---:|---:|---:|
| always-clean (what it did before) | 0.020 | 0.185 | 0.121 | 2.85% |
| **detected** | **0.013** | **0.011** | **0.111** | **2.79%** |
| oracle (true label) | 0.013 | 0.012 | 0.110 | 2.79% |

**Branch A2's calibration loss falls 94%.** Fused Cllr improves 8.8%, and
**detection captures 97.4% of what a perfect oracle would deliver** — so the
detector is not the limiting factor, and further accuracy there would buy almost
nothing.

### Fixing one error exposed another: the weights were never fitted

`FusionConfig.w` was `{A: 1.0, A2: 0.45, B: 0.0}` — chosen by hand, never fitted.
That was harmless while A2's LLRs were uniformly over-confident, because two
errors cancelled: a hand-set weight that under-weighted A2 against a mapping that
over-stated it. Per-channel calibration fixed one and exposed the other.

Fitted by logistic regression of the label on the per-branch LLRs — its
coefficients *are* the weights and its intercept *is* the subset α, on the same
LLR scale the engine works in. Fitted on calib, reported on dev:

| | fused EER | Cllr | calibration loss |
|---|---:|---:|---:|
| hand-set 1.0 / 0.45 | 3.04% | 0.119 | 0.009 |
| **fitted 1.0 / 0.87** | **2.79%** | **0.110** | **0.005** |
| Branch A alone | 3.62% | 0.156 | 0.013 |

**A2's fitted weight is nearly double the hand-set one**, and fusion now beats
its best single branch (2.79% vs 3.62%).

**Weights live in the calibration directory, not in `FusionConfig`.** They are
only valid against the branch calibrations they were fitted for — a profile whose
A2 mapping differs needs a different A2 weight — so `calib/weights.json` carries
the ASVspoof profile's and `calib_rtc/` has none, which means the RTCFake profile
keeps the default and §17's result stands unchanged. This is the same rule as
head/calibration pairing, one level up.

### End to end, under the whole chain

Detection on, all three buckets, fitted weights, refitted subset table:

| Scenario | alerted | median time-to-alert | median S_lo |
|---|---:|---:|---:|
| bona fide | **0/40** | — | **−50.23** |
| spoof from t=0 | **40/40** | 2.5 s | +16.92 |
| mid-call swap | **40/40** | **2.3 s after the splice** | +16.95 |

Unchanged or better on every axis, and genuine calls now sit at −50.23 nats
against the previous −24.57 — the system is markedly more confident that a real
customer is real, which is the half of the problem that decides whether anyone
can deploy it.

### Honest limits

This detects the channel profiles **we simulate**. Real telephony adds adaptive
noise suppression and comfort noise, and a real narrowband codec resamples to
8 kHz rather than low-passing at 16 kHz. `high_band_ratio` reads both the same
way — absence is absence — but the noisy/clean split is tuned to this simulator's
noise and loss model and must be re-fitted against real transmitted audio before
production. RTCFake is the instrument for that and has not been used for it.

---

## 19. Who does this system fail, and in which direction

Every accuracy number above is an aggregate: EER, recall, a false-positive
count. None of them says **who** the system is wrong about, and for a fraud
detector that is the number with a person on the other end of it.

So: fix an operating threshold on one population's bona fide speech, then
measure the realised error rates on another. RTCFake makes this a controlled
comparison — same corpus, same platforms, same recording pipeline, different
language — and Mandarin was never trained on.

Branch A + A2 retrained on RTCFake English, fused, `calib_rtc` calibration.
Threshold set on English bona fide; both error types reported at that threshold:

| target FPR | population | bona fide n | realised FPR | ratio | miss rate | ratio |
|---|---|---:|---:|---:|---:|---:|
| 1% | en (reference) | 979 | 1.02% | 1.00 | 61.0% | 1.00 |
| | **zh** | 1,870 | **0.21%** | **0.21** | **84.5%** | **1.39** |
| 5% | en (reference) | 979 | 5.01% | 1.00 | 38.2% | 1.00 |
| | **zh** | 1,870 | **2.35%** | **0.47** | **69.6%** | **1.82** |

### The false-positive number alone would have been a half-truth

Mandarin speakers are flagged **less** often than English speakers at the same
threshold — 0.21× and 0.47×. Stop there and this reads as "no fairness problem".

It is not. At the 5% operating point, **69.6% of real attacks against Mandarin
speakers get through, against 38.2% for English speakers** — a **1.82×** miss
disparity. The model scores that population lower across the board, and scoring
a group lower means accusing them less *and* defending them less. Both are
consequences of the same fact; only the first produces a complaint.

**A missed attack has a victim who never learns the system was watching.** That
is why this table reports both directions, and why the script refuses to print a
verdict from the false-positive column alone.

### What this does and does not establish

It establishes that **the disparity is real and material on the one axis this
project can measure**, and that it runs opposite to the direction people
usually look.

It does not establish anything about the population that actually matters here.
Mandarin is a proxy for "a language the model never trained on" — nothing more.
The deployment target is Indian-language and Indian-accented speech, for which
this project has **no bona fide corpus at all** (§16: IndicSynth is spoof-only),
so the group most likely to be affected is precisely the group we cannot yet
measure. Accent within a language, age, gender, speech disorders and recording
device are all unmeasured.

**Absence of a measurement is not evidence of fairness.**

### The mechanism, and why it is not fixable by moving a threshold

A single global threshold cannot equalise both error types across populations
whose score distributions differ — lowering it to protect Mandarin speakers
raises the English false-positive rate, and the two cannot be traded to zero
simultaneously. Per-population calibration would help and is exactly what
§17/§18 already built the machinery for: calibration is cheap, per-profile, and
refits in seconds.

That is a real option and it is **not implemented**, because doing it correctly
requires bona fide data from each population served — which is the same missing
ingredient as everything else in this section.

---

## 20. Three fixes for cross-corpus collapse, and none of them worked

§16 measured the project's central weakness and named three untested fixes.
All three are now tested. **None repairs it.** Recording that as a result,
because a negative measured properly is worth more than a positive assumed.

One seed each, 20 epochs, identical folds, scored through one code path
(`scripts/eval_folds.py`) so every row is subtractable:

| Branch A trained on | ASVspoof eval | **WaveFake unseen** | RTCFake en | RTCFake zh |
|---|---:|---:|---:|---:|
| ASVspoof only | **11.35%** | **30.08%** | 45.57% | 55.34% ⚠ inverted |
| RTCFake-en only | 36.54% | 49.36% | **22.93%** | **33.34%** |
| both, naive concat | **9.47%** | 49.34% | 24.18% | 36.71% |
| both, **balanced sampling** | 9.84% | 49.84% | 25.63% | 38.28% |
| both, **+ domain adversary** | 10.14% | **46.81%** | 24.51% | 35.57% |
| both, **+ adversary + OC-Softmax** | 9.88% | 49.11% | 24.62% | 36.98% |

**WaveFake is the column that matters** — the corpus held out of every mixed
run. Every mixing configuration sits at 46.8–49.8%, i.e. chance, against the
30.08% that ASVspoof-only achieves. The best of them, the domain adversary,
recovers 2.5 points and buys it with 0.7 points in-domain.

### What each fix did and did not do

**Balanced sampling made it slightly worse** (49.34% → 49.84% on WaveFake,
36.71% → 38.28% on Mandarin). Equalising corpus row counts does not help when
the problem is not row counts.

**The domain adversary worked mechanically and did not solve the problem.**
Domain loss sat at 0.689 against ln(2) = 0.693 for the whole run: the corpus
became genuinely unpredictable from the representation, which is exactly what
the reversal is for. It bought the best held-out WaveFake number of any mixed
configuration (46.81%) and that number is still chance.

**OC-Softmax did not separate from the others** (49.11%), despite being the
loss designed for open-set generalisation.

### The reading, and why it is not a tuning failure

Making a representation invariant to the corpora you *have* does not make it
invariant to a corpus you have never seen. The adversary can only remove what
it can name, and it was only ever shown two domains. WaveFake — single-speaker
LJSpeech studio resynthesis — is unlike both of them in ways neither can
describe.

This also reframes §16's headline. The mixed models are not broken: they are
the **best in-domain models in the project** (9.47% on ASVspoof eval, against
11.35% for the ASVspoof-only baseline) and they handle both corpora they saw.
What no configuration achieves is the third corpus.

So the honest claim is narrower than "these methods fail". It is:

> **Training-time domain generalisation, as tested here, extends to the corpora
> in the mixture and not beyond it.** Two corpora are not enough to induce
> corpus-invariance in general.

### What that implies for the next attempt

Not more of the same. The remaining levers, in order of expected value:

1. **More domains, not more data.** The adversary saw two. Three or four
   corpora in the mixture (WaveFake moved into training, In-the-Wild and
   CodecFake added, both now downloaded) gives it enough variation to
   generalise from. The cost is that the held-out set must then be something
   else again, or the measurement becomes circular.
2. **SSL layer selection.** ~~The 2026 literature rates the readout layer as a
   dominant factor in out-of-domain performance. This project uses layer 5 from
   a general argument and has never measured it.~~ **Measured in §22. It is not
   a lever: depth monotonically hurts, and layer 5 was already the right
   choice.** This item is closed.
3. **Per-deployment calibration**, which §17–18 already showed works and is
   cheap. It does not fix the representation, but it makes whatever the
   representation gives you usable on the channel you actually serve.

### A process note

The OC-Softmax configuration was initially absent from the comparison matrix:
the fourth run was added to the experiment without adding its checkpoint to the
evaluation list, so it trained and was never scored. Caught by reading the
matrix against the run log rather than by any error — a four-row table where
five runs happened looks entirely normal.

---

## 21. Replay attacks, finally measured

`docs/THREAT_MODEL.md` A4 has listed physical replay as **untested** since it
was written. EchoFake makes it testable, and the construction is what matters:
`eval_replay` and `eval_synth_echo` hold the **same 1,344 bona fide clips** and
differ only in attack type, so the gap between their EERs measures "does replay
evade this detector" rather than being two unrelated numbers.

| Branch A trained on | **replay** | synthesis (control) | replay ÷ synth |
|---|---:|---:|---:|
| ASVspoof | 35.28% | **57.74%** ⚠ inverted | — |
| RTCFake-en | 38.99% | **25.89%** | **1.51×** |
| balanced + adversary | **33.68%** | 27.53% | **1.22×** |

### Finding 1 — replay evades detection, by 1.2–1.5×

For the two models that work at all on this corpus, **replay is 1.2–1.5× harder
than synthesis at the same operating point**, against identical negatives. The
best replay number anywhere is 33.68%, which is not a usable detector.

This is the expected result and it is worth having as a number rather than an
argument: a replayed recording of a real person **is not synthetic speech**, so
branches trained on vocoder and phase artifacts have no mechanism to catch it.
What they can see is the second recording pass — room, speaker, microphone —
and that is evidently a weaker cue than synthesis artifacts.

### Finding 2 — the ASVspoof model inverts on modern TTS

57.74% on `eval_synth_echo`, i.e. anti-correlated. EchoFake's synthesis attacks
are CosyVoice2, F5-TTS, FireRedTTS, IndexTTS and XTTS-v2 — 2024-2026 systems.
The ASVspoof-2019-trained model does not merely fail on them, it ranks them as
*more genuine* than real speech.

That is the §13 inversion again, on a new axis: not a change of channel or
language this time, but a change of **generator generation**. It is the
clearest evidence yet that this model's 2.48% in-domain number describes a 2019
threat landscape.

The models trained on RTCFake (25.89%) and with the domain adversary (27.53%)
handle modern TTS considerably better — the first measured advantage of those
configurations over the ASVspoof baseline on anything.

### What follows

Replay needs a **different mechanism**, not a better-trained version of the
same branches — channel/recapture detection rather than synthesis-artifact
detection. Nothing in this architecture currently does that, and the honest
statement for deployment is that **VeriCall does not defend against replay.**

---

## 22. SSL layer selection, measured — and a non-replication

§20 listed layer selection as the second-most-promising remaining lever, on the
strength of 2026 work rating the **readout layer** as a dominant factor in
out-of-domain spoof detection. That claim was never tested here: layer 5 came
from a general argument about where spoof artefacts live, and the docstring in
`vericall/branches/a_ssl.py` asserted it rather than measuring it.

`scripts/stage_layer_sweep.sh` now measures it. One encoder pass extracts
layers 3/5/7/9/11 for every fold, so five heads train on cached features at
12–15 s per epoch. Everything downstream is identical across the five: same
ASVspoof-2019-LA training rows (50,760 train / 15,962 dev), same architecture,
same 12 epochs, same seed, same best-epoch-on-dev selection.

### The table

| SSL layer | dev EER | WaveFake unseen vocoder | RTCFake EN | RTCFake ZH |
|---|---|---|---|---|
| 3 | 1.65% | **29.11%** [27.47–30.34] | 46.64% [45.05–48.41] | 56.63% [55.55–57.90] |
| **5** (production) | **1.40%** | 30.18% [28.77–31.49] | **45.66%** [43.80–47.33] | 53.01% [51.98–53.96] |
| 7 | 3.07% | 32.48% [30.98–33.74] | 48.31% [46.83–49.90] | 51.02% [49.78–52.32] |
| 9 | 5.94% | 36.53% [35.21–37.99] | 49.13% [47.76–51.09] | 50.59% [49.45–51.90] |
| 11 | 7.07% | 38.71% [37.05–40.62] | 48.42% [46.88–50.26] | 49.81% [48.62–51.05] |

### Depth hurts, and it hurts monotonically

Unseen-vocoder EER rises 29.11 → 30.18 → 32.48 → 36.53 → 38.71 with no
reversal; dev EER rises 1.40 → 7.07 over the same span. A monotone trend across
five points with non-overlapping endpoints is not sampling noise.

This **contradicts the readout-layer finding** for this setup. The top layer
measured is the worst of the five by 8.5 points of unseen-vocoder EER. Two
differences from the cited work are candidate explanations, and this project
cannot distinguish them: that work fine-tunes the encoder where this one keeps
it frozen, and it reads a learned weighted sum of all layers where this one
truncates. A frozen top layer has specialised toward phonetic content with no
opportunity to adapt back toward artefacts — which is the original argument for
truncating, now with numbers attached.

### Layer 3 vs layer 5 is a wash, and layer 5 keeps the job

Layer 3 takes unseen-vocoder by 1.07 points, but [27.47–30.34] and
[28.77–31.49] overlap across most of their range. Layer 5 wins dev EER
(1.40% vs 1.65%) and RTCFake-EN (45.66% vs 46.64%) outright. There is no case
for changing production, and — the actual value of this experiment — **no
accuracy headroom hiding in the layer index.** The docstring's claim is now
measured rather than borrowed.

### The one genuinely new fact: layer 3 inverts on Mandarin

RTCFake-ZH at layer 3 is **56.63%**, with a 95% CI of [55.55–57.90] that
excludes 50% entirely. An EER above chance by that margin is not weakness, it
is **inversion** — the discriminative direction learned on English ASVspoof
points the wrong way on Mandarin. Layer 5 shows the same sign (53.01%, CI
excludes 50%), and the effect decays with depth until layer 11 sits at chance
(49.81%, CI straddles 50%).

The reading is uncomfortable but consistent with §13 and §21: **the lower
layers carry the most language-specific artefact information, so they are
simultaneously the best in-domain and the most confidently wrong
out-of-language.** Depth does not fix the inversion; it only erases the signal
that was inverted.

### Standing of the cross-corpus problem after this

Five independent attempts have now failed to close it — three published fixes
(§20), corpus mixing (§17), and layer selection (here). Every RTCFake cell in
the table above is at or beyond chance, at every depth. Layer choice belongs
with the others: **not a lever.** The two remaining candidates are unchanged —
more training domains, and per-deployment calibration — and only the second is
known to work.

---

## 23. The channel a WebRTC call actually arrives on

`web/call.html` places a real two-party browser call and streams the decoded
remote track into the detector. That audio has been through **Opus**, and no
calibration bucket in `calib/` was ever fitted on Opus — they are clean, G.711
and a simulated noisy line. The page says so on screen. This measures how much
it matters.

### The question is not "does EER get worse"

EER is threshold-free, so a branch can hold its EER while its LLRs stop meaning
anything — and this system does not threshold a score, it accumulates
calibrated log-likelihood ratios across a call. What breaks a calibrated
accumulator is the score→LLR mapping drifting, and the quantity that measures
exactly that is **calibration loss**, `Cllr − min_Cllr`. `min_Cllr` is what a
perfectly refitted mapping would achieve, so the gap is the part of the cost
that is the mapping's fault rather than the detector's.

Real libopus via PyAV, not a DSP imitation: Opus is a transform codec with its
own psychoacoustic model and SILK/CELT switching, and approximating it would
measure the approximation. 1,330 held-out `dev` utterances, 1.0 s centre crops
(the fast-lane window, and what `extract_features.py --window` fitted on).

| arm | branch | EER | min_Cllr | Cllr | calibration loss |
|---|---|---|---|---|---|
| clean | A | 1.18% [0.6–1.7] | 0.038 | 0.053 | 0.015 |
| clean | A2 | 1.77% [0.5–3.0] | 0.052 | 0.077 | 0.025 |
| opus @12k | A | 2.66% [2.1–3.7] | 0.120 | 0.347 | 0.227 |
| opus @12k | A2 | 24.67% [20.1–30.0] | 0.684 | 1.496 | 0.812 |
| opus @16k | A | 2.07% [1.4–2.6] | 0.074 | 0.115 | 0.041 |
| opus @16k | A2 | 22.60% [18.2–27.6] | 0.636 | 1.288 | 0.651 |
| **opus @24k** | **A** | **1.33% [0.8–2.0]** | 0.052 | 0.069 | **0.018** |
| **opus @24k** | **A2** | **18.07% [14.6–23.0]** | 0.561 | 0.980 | **0.419** |
| opus @32k | A | 1.23% [0.8–1.7] | 0.050 | 0.064 | 0.014 |
| opus @32k | A2 | 15.93% [12.1–19.7] | 0.465 | 0.694 | 0.230 |

### Branch A survives. Branch A2 is destroyed.

At 24 kbps — what a browser negotiates for an audio call — **A is essentially
unaffected**: 1.18% → 1.33% EER on overlapping CIs, calibration loss 0.015 →
0.018. The clean mapping is already very nearly right for it.

**A2 goes 1.77% → 18.07%**, a factor of ten, with CIs of [0.5–3.0] and
[14.6–23.0] that do not come close to touching. Calibration loss rises 17×. And
it is cleanly monotone in bitrate — 15.93% at 32k, 18.07% at 24k, 22.60% at
16k, 24.67% at 12k — which is what a mechanism looks like rather than noise.

The mechanism is not in doubt. **A2 reads phase**; Opus is an MDCT/LPC
transform codec that does not preserve phase. A reads spectral magnitude
through SSL features, which Opus is built to preserve.

### The number that should worry a deployment

The mean LLR A2 banks for the **spoof** class:

| channel | A2 mean LLR, bona fide | A2 mean LLR, spoof |
|---|---|---|
| clean | −7.12 | **+3.77** |
| opus @32k | −5.07 | +1.86 |
| opus @24k | −5.11 | +1.01 |
| opus @12k | −5.25 | **−0.40** |

At 12 kbps A2's evidence for a spoof is *negative*: the branch argues the
attacker is genuine. This is the §13/§21 inversion again, on a fourth axis —
not corpus, language or generator this time, but **transport**.

### And the channel detector cannot see it

| arm | what `CodecDetector` calls it |
|---|---|
| clean | clean 97%, noisy_line 3% |
| opus @12k–32k | clean 97–98%, noisy_line 2–3% |

Opus is wideband and leaves no packet-loss signature, so it looks *exactly*
like clean audio to a detector built to separate clean from G.711 from a lossy
line. That is not a defect in the detector — it was fitted to distinguish three
channels and Opus is a fourth — but it means **the channel cannot be
detected from the audio**. A WebRTC call therefore receives the clean bucket:
correct for A, and the worst available choice for A2.

Since A2 carries a fusion weight of 0.87, a live call currently banks A2
evidence at nearly A's weight through a mapping that is 17× miscalibrated.

### Two corrections, and only one of them is calibration

Refitting the Platt mapping removes the calibration loss but **cannot recover
the information**: A2's min_Cllr through Opus is 0.56 against 0.052 clean, and
min_Cllr is the floor a perfect mapping would reach. So the fix is both halves
— refit the mapping *and* refit the weight, because a branch that has lost nine
tenths of its information should not keep the weight it earned when it had it.
`scripts/fit_opus_bucket.py` does both. Detection is handled by the client
declaring its transport rather than by the audio, for the reason above.

### A process note, on two bugs in the measuring instrument

The first run of this measurement produced a confident, wrong answer twice
over, and both were the harness rather than the system:

1. **No logit transform.** The Platt mappings are fitted on `logit(posterior)`
   (`fit_calibration.py`: `raw = logit(scores)`), and the script fed them the
   posterior. That reported Cllr 0.93 for clean Branch A against the 0.083
   recorded in `calib/A@clean.json` — an 11× discrepancy that read as a
   catastrophic finding and was an arithmetic error.
2. **Zero padding.** The script used a 4 s crop and padded anything shorter.
   95% of dev clips are under 4 s (median 2.04 s), so almost every row became
   half speech and half digital silence. Zero padding drives
   `const_frame_ratio` to 0.25, which trips the detector's packet-loss test
   (`c > 0.0`), and **clean audio classified as noisy_line 53% of the time.**

That second one is the instructive one. It looked like a real defect in the
channel detector, it was reproducible, and the honest-looking conclusion —
"the detector is wrong about clean audio half the time" — would have sent the
next work item in entirely the wrong direction. The tell was that 53% is
suspiciously close to a number already in the detector's own docstring
describing a *previously fixed* bug. Checking clip durations took one command.

---

## 24. In-the-Wild — the first corpus this project has never trained on

Every cross-corpus number above was measured on a corpus sharing a lineage
with something a model saw. In-the-Wild (Müller et al., Interspeech 2022) does
not: the bona fide audio is genuine public recordings — podcasts, speeches,
interviews — of 54 public figures, and the spoofs are clones of those same
people. A detector cannot separate the classes by speaker, accent, language or
recording provenance. Only by synthesis.

Built and verified against the published release: **31,779 clips, 19,963 bona
fide, 11,816 spoof** — exactly the counts in the paper, so the mirror is
confirmed un-resampled and un-subsetted rather than assumed.

Measured on a **balanced, seeded 2,500 / 2,500 subsample**, 1.0 s centre crops,
no padding. Both of those matter for reading the numbers and are discussed
below.

### The headline

| Branch A, trained on | In-the-Wild EER |
|---|---|
| **ASVspoof 2019 LA (the baseline)** | **22.48%** [21.26–23.70] |
| + domain-adversarial (DANN) | 23.56% [22.37–24.63] |
| ASVspoof + RTCFake mixed | 24.76% [23.59–25.95] |
| RTCFake only | 29.48% [28.36–30.72] |

Two things, and the second is the uncomfortable one.

**Branch A carries real signal on genuinely in-the-wild audio.** 22.48% is
degraded — the in-domain figure is 2.48% — but it is not chance and it is not
inverted. Against RTCFake-zh at 55.34% (§13) and EchoFake's modern TTS at
57.74% (§21), this is the best out-of-domain result the project has, on its
hardest-to-game corpus. Placed against the other folds it sits between unseen
attacks in-domain (11.01%) and unseen vocoders (30.29%).

**Every intervention intended to improve generalisation made it worse.** The
plain ASVspoof baseline beats DANN, beats corpus mixing, beats RTCFake
training. DANN's CI overlaps the baseline's at the edge, so that one is
"indistinguishable to slightly worse"; mixing and RTCFake-only are clearly
worse. This is the **sixth** independent confirmation of §20, and the cleanest,
because all four models meet In-the-Wild as a common held-out test rather than
one of them being evaluated at home.

The mixed model was *better* on RTCFake and is *worse* here. That is the
mechanism stated plainly: it did not become more general, it specialised toward
a second corpus. Adding domains to the training mixture bought performance on
the domains added, and nothing beyond them.

### Branch A2 has no out-of-domain existence

| Branch A2, trained on | In-the-Wild EER |
|---|---|
| ASVspoof 2019 LA | **50.65%** [50.44–50.90] |
| RTCFake | 35.36% [33.88–36.66] |

The ASVspoof-trained phase branch is **at chance**. The CI technically excludes
50%, which the evaluator flags as inverted, but 0.65 points of anti-correlation
is not a usable signal in either direction — the honest reading is that A2
knows nothing about this corpus.

Together with §23, where Opus destroyed A2 (1.77% → 18.07%), a picture has
formed: **A2 works in-domain and essentially nowhere else.** It is excellent on
ASVspoof, it does not survive a codec, and it does not survive a corpus change.
Its fusion weight of 0.87 is earned on exactly one distribution.

The exception is worth recording because it is the only one in the whole §20
family: **A2 trained on RTCFake reaches 35.36%**, far better than the 50.65%
ASVspoof gives it. Real transmission artifacts appear to teach the phase branch
something that simulated ones do not. That is a lead, not a result — one seed,
one corpus — but it is the first intervention anywhere in this project that
improved an out-of-domain number rather than relocating it.

### What these numbers are not

**They are per-1-second-window, not per-utterance.** Published In-the-Wild
results score whole utterances, which is a strictly easier problem: more
evidence per decision. VeriCall's deployed mode accumulates LLRs across a call
and would do better than a single window. So 22.48% is not comparable to a
published figure and must not be quoted as one; it is comparable to the other
1 s fold numbers in this document, which is what it is for.

**They are a 2,500/2,500 balanced subsample**, not the full 31,779 at its
natural 63/37 split. Balanced because EER and Cllr are defined against a
balanced prior; subsampled because this node lost its GPU and WavLM on CPU runs
at 5 clips/s, making the full corpus a four-hour extraction. The CIs are ±1.2
points, which is tight enough for every comparison drawn above.

**No generator breakdown is possible.** The release does not say which system
produced each spoof, so "unseen generator" cannot be measured here.

---

## 25. Hindi — the deployment target, finally measurable

Every section before this one measured a detector on corpora it was never
meant to serve. The problem statement is Indian-language telephony, and until
now the project's own record said no Hindi EER was computable at all:
IndicSynth is spoof-only, and pairing its clones against an unrelated bona
fide corpus would let a model separate the classes by recording conditions
rather than by synthesis. Both statements were true. Both were beside the
point.

### The bona fide class was never missing

IndicSynth's filenames carry the speaker ids of the corpus it cloned:

```
xtts_v2____1059-m-149.wav                 speaker 1059, male, utterance 149
freevc24____f____src_1087____tgt_1102     1087's words in 1102's voice
```

Measured on the manifest: **41 source speakers, 101 target speakers, and all
41 sources also appear as targets** — one shared pool, ids 3–1191. The clones
are conversions *between speakers of a single real corpus*, and that corpus is
**Kathbath**, via IndicSUPERB (Javed et al., AAAI 2023), which is **CC0** with
direct downloads.

So reuniting IndicSynth with its own source is the opposite of cross-corpus
borrowing: same speakers, same recording sessions, same language, on both
sides of the label. That is a stronger pairing than ASVspoof has. Of Kathbath
Hindi's `valid` + `testkn` splits, **29 speakers appear in both corpora**, each
with 132–1,156 clones of their own voice.

### What it took to make the corpus trustworthy

The corpus was rebuilt five times. Each rebuild removed something that would
have produced a confident wrong number, and this is the part worth reading:

**The silence cue survived trimming.** `audit_silence.py` measured a
leading-silence-only classifier at **14.61% EER raw and 13.92% after
`load_audio`'s trim**. The cue survives because `trim_silence` pads 30 ms of
original audio back onto the onset and neural TTS often has less than 30 ms to
give back — so bona fide kept 30.0 ms and spoof 12.5 ms. Any Hindi EER above
~14% would have been beaten by a stopwatch, and one below it could not have
been attributed to synthesis. Fixed at the source: `equalise_silence()` cuts to
speech onset and appends an identical fixed pad, at the same 35 dB threshold
the cue is *defined* by. A first attempt at 45 dB removed the leading cue
(14.61% → 59.06%) and left trailing silence at 36.80%, because read speech
decays through −35..−45 dB where neural TTS stops dead. A 20 ms pad was then
narrower than the 25 ms analysis frame, so quantisation reintroduced it. At
50 ms: **leading 50.00%, trailing 50.52%, duration 54.33%** — chance, in the
path the model actually sees.

**The eval fold had no zero-shot TTS.** Only 5 of 29 shared speakers carry any
XTTS-v2, and a plain stable hash put all five outside eval — 400
voice-conversion attacks and zero TTS, while appearing to cover both. A
generator-stratified split fixes it, and the default salt is *selected* for
coverage: 392 of 1,500 salts give every fold both families, and v1428 also
best fits the 35/15/15/35 target. That criterion is a property of the split
and never of a model's performance on it; no checkpoint existed when the
search ran.

**The sampling was 396 FreeVC24 against 84 XTTS-v2.** A head trained on that
mix scored **2.62% on voice conversion and 40.42% on zero-shot TTS** — the
latter being the duration floor, i.e. no detection at all — while the
aggregate read a healthy-looking 10.21%. Material was never the constraint
(4,145 XTTS-v2 rows available against 4,419 FreeVC24); only the draw was
wrong. `plan_spoof_budget()` now works to a global per-generator budget.

**Kathbath is m4a and IndicSynth is PCM.** Fed in as shipped, a detector
separates the classes by *codec* and reports a spectacular number with nothing
to do with synthesis. Two arms are built: `matched`, where the spoof side gets
one AAC generation, and `unmatched`, as shipped, kept to *measure* the
confound rather than to assert it is small.

### The result

Branch A trained on the paired corpus, 29 speakers, speaker-disjoint folds,
all leakage guards clean:

| | EER |
|---|---|
| **eval_hindi** | **7.54%** [6.65–8.55] |
| — vs freevc24 (voice conversion) | **1.92%** [1.34–2.36] |
| — vs xtts_v2 (zero-shot TTS) | **9.64%** [8.48–11.02] |
| deployed ASVspoof head, same fold | 42.29% [40.16–44.00] |
| duration-only floor | 50.00% |

**The codec confound is ruled out by measurement**: matched **7.54%** against
unmatched **7.81%**. If the model were reading AAC-versus-PCM rather than
synthesis, the unmatched arm — where bona fide is AAC and spoof is raw PCM, an
enormous cue — would be dramatically better. It is not.

### Telephony, which is what the problem statement asks about

Clean audio is not the deployment condition. Retrained with three channels
cycled across four crops per clip:

| channel | Branch A | Branch A2 |
|---|---|---|
| clean | 9.24% | 12.86% |
| G.711 (narrowband) | 13.93% | 19.02% |
| noisy line | 23.21% | 39.67% |
| **fitted slope spread** | **1.25×** | **2.33×** |

A pooled mapping would mis-scale every non-clean call by roughly those
factors, which is why `calib_hindi_ch` fits all three buckets. On
channel-mixed audio — telephony as it actually arrives:

| head | channel-mixed EER |
|---|---|
| **channel-trained** | **13.04%** [11.80–14.16] |
| clean-only trained | 17.72% [16.62–18.94] |

4.7 points, CIs not overlapping. The clean-only head remains better on clean
audio (7.54% vs 9.24%), so this is the ordinary robustness trade; the
channel-trained head is the one to ship for this problem statement.

### Live, end to end

Streamed through the gateway at wall-clock pace, 21 calls built from
speaker-disjoint eval clips: real speakers **−26.7 to −36.3 nats, every one
tier 0**; clones **−1.4 to +52.6**, 13 of 14 alerting. **Call-level EER 0.00%**
on a 25-nat margin. Small n and a wide interval, but clean separation.

`scripts/verify_export.sh` now boots this profile from a cold clone and
asserts it: genuine clear, both clone families alerting, every channel bucket
present, nothing degraded.

### What this does and does not establish

**Does:** Hindi synthetic-speech detection is achievable with this
architecture, it was never a data-acquisition problem, and the gap was that
nobody had trained on a word of Hindi. Five failed cross-corpus interventions
(§20, §22, §24) were all attempts to make a representation *transfer*; none of
them involved training on the target language.

**Does not:** 29 speakers is a small pool, all from one corpus and one
recording programme. Hindi is one of twenty-two scheduled languages and the
other twenty-one are untested — IndicSynth covers twelve, so the same
construction extends, but extension is not evidence. XTTS-v2 at 9.64% is four
times worse than voice conversion at 1.92%, so the newer attack family is the
weaker side. And nothing here changes §21: replay is still undefended.

---

## 26. CodecFake — the last unmeasured fold

`eval_unseen_codec` has had a manifest, a speaker-disjoint split and a
declared fold since CodecFake was fetched, and no result. It was the only
member of `EVAL_FOLDS` with nothing measured against it, and `CLAUDE.md`
listed the corpus as "integrated" while `DATASETS.md` said "pending" — the
second was right.

It matters because neural audio codecs are the current generation of attack:
the spoofs here are resynthesised through learned codecs rather than through
a classical vocoder, which is a different artefact family from anything in
ASVspoof 2019.

The split holds out **F05 and F06** as unseen codecs, 1,581 spoofs each
against a shared bona fide set, so the number is genuinely "a codec the model
was never trained on".

| trained on | `eval_unseen_codec` EER |
|---|---|
| **ASVspoof 2019 LA (the baseline)** | **26.98%** [25.75–28.30] |
| ASVspoof + RTCFake mixed | 27.51% [26.20–28.92] |
| + domain-adversarial (DANN) | 28.97% [27.43–30.45] |
| RTCFake only | 49.12% [47.68–50.60] |
| Hindi (channel-trained) | 51.01% [49.57–52.12] |

Per codec, for the baseline: **F05 23.28%** [22.0–24.6], **F06 31.06%**
[29.4–32.3].

### Three readings, in order of how much they change anything

**Neural-codec attacks are detected, degraded but real.** 26.98% sits beside
WaveFake unseen vocoders (30.29%) and In-the-Wild (22.48%) — not chance, not
inverted. A 2019-trained model retains usable signal against a 2024–26 attack
family it has never seen, which is a better result than §21's replay finding
and considerably better than §13's Mandarin inversion.

**The plain baseline wins again.** Mixing and DANN are both worse, RTCFake-only
is at chance. This is the **seventh** independent confirmation of §20: every
intervention intended to broaden generalisation has, on every held-out corpus
measured since, cost accuracy rather than bought it.

**The Hindi head is at chance here, and that is the correct reading of it.**
51.01% with the CI touching 50% means the channel-trained Hindi model carries
essentially nothing on English neural-codec spoofs. It is a *specialist*, not
an improved general detector, and the two must not be confused: §25's 7.54%
is a Hindi number and does not transfer. A deployment serves the Hindi profile
on Hindi traffic and the ASVspoof profile elsewhere, which is exactly what
`VERICALL_CALIB_DIR` plus per-profile checkpoints are for.

### What is now measured

Every fold in `EVAL_FOLDS` has a result:

| fold | corpus | best measured EER |
|---|---|---|
| `eval_in_domain` | ASVspoof | 11.01% |
| `eval_seen_vocoder` | WaveFake | 31.43% |
| `eval_unseen_vocoder` | WaveFake | 30.29% |
| `eval_rtc_en` / `eval_rtc_zh` | RTCFake | 22.93% / 33.34% |
| `eval_replay` / `eval_synth_echo` | EchoFake | 33.68% / 25.89% |
| `eval_in_the_wild` | In-the-Wild | 22.48% |
| `eval_hindi` | hindi_paired | **7.54%** |
| `eval_unseen_codec` | CodecFake | **26.98%** |

The one number below 10% outside the training corpus is the one where the
model was trained on the target domain.

---

## 27. CodecFake as a training corpus — and a confound that nearly became a headline

§26 measured CodecFake and left its 117,040-row training fold unused. This
uses it, because an unused training corpus is a claim nobody has tested.

| head | CodecFake `eval_unseen_codec` | ASVspoof `eval_in_domain` |
|---|---|---|
| **CodecFake-trained** | **16.38%** [15.36–17.38] | **10.13%** [9.35–10.87] |
| ASVspoof baseline | 26.98% [25.75–28.30] | 11.23% [10.37–12.23] |

The first column is a real result. `eval_unseen_codec` holds out F05 and F06 as
codecs never seen in training, the split is speaker-disjoint within the corpus,
and training on CodecFake takes it from 26.98% to **16.38%** — a 1.6× gain on
the attack family it was trained for. That is the profile-per-domain thesis
holding on a second domain, after Hindi.

### The second column is not a result, and that matters more

Taken at face value, the CodecFake-trained head also beats the ASVspoof head on
**ASVspoof's own eval fold** — 10.13% against 11.23%. That would be the first
intervention in this entire project to improve an out-of-domain number without
costing the in-domain one, against seven that failed (§20, §22, §24, §26).

It is almost certainly an artefact of shared speakers.

CodecFake's speaker ids are `p225` … `p376`: that is **VCTK** naming, and all
93 of its speakers match the pattern. ASVspoof 2019 LA is *also* built from
VCTK, but anonymises its 107 speakers to `LA_0001` … . VCTK contains roughly
110 speakers in that range. Two corpora drawing 93 and 107 speakers from a pool
of ~110 overlap almost completely — and because one side is anonymised, **the
overlap cannot be measured from the released metadata at all**. A naive check
reports zero shared ids and means nothing:

```
  SPEAKER overlap    : 0
  SOURCE-UTT overlap : 0
```

So a model trained on CodecFake has very likely heard most of ASVspoof's eval
speakers, under different names, saying different sentences through different
attacks. That is not the same as training on the eval set, and it is not
nothing either: speaker identity is exactly the nuisance variable
speaker-disjoint splitting exists to control, and here it cannot be controlled
because it cannot be observed.

**The honest statement is therefore that the 10.13% is uninterpretable**, not
that it is good. The seven-failure record stands; this is not a counterexample
to it.

### A guard gap this exposes

Every leakage guard in `vericall/data/splits.py` operates **within** a corpus.
None of them can detect speaker overlap **between** corpora, and they never
could — the information is not in the manifests. That has not mattered while
corpora were used one at a time, and it matters the moment two are mixed:
`branch_A_mix_seed0` already trains on ASVspoof + RTCFake + WaveFake, and
WaveFake is LJSpeech-derived while the other two are not, so that particular
mixture happens to be safe. **ASVspoof + CodecFake would not be**, and nothing
in the code would have said so.

Recorded here rather than fixed, because there is no fix available from the
data: the check requires speaker identity that ASVspoof deliberately removed.
The operational rule is that these two corpora must not be mixed in training,
and any future corpus pairing needs its provenance checked by hand before the
splits are trusted.

---

## 28. Multilingual — pooling works, and the frozen holdout proves it

§25 built one paired language. The construction is not Hindi-specific:
IndicSynth clones Kathbath speakers in twelve languages and Kathbath publishes
all twelve, so the same reunion works for each. Five more were built —
Bengali, Marathi, Tamil, Telugu and **Gujarati, reserved untouched**.

| language | shared speakers | bona / spoof | generators |
|---|---|---|---|
| Hindi | 29 | 1,624 / 1,600 | freevc24, xtts_v2 |
| Bengali | 24 | 2,119 / 2,400 | freevc24, vits, xtts_v2 |
| Marathi | 31 | 806 / 800 | freevc24 only |
| Tamil | 29 | 1,624 / 1,600 | freevc24, xtts_v2 |
| Telugu | 28 | 1,624 / 1,599 | freevc24, xtts_v2 |
| **Gujarati** | 30 | 1,620 / 1,600 | freevc24, xtts_v2 |

Leakage guards clean on all six. The silence cue is **50.00% after trimming on
every one** — the `equalise_silence` fix from §25 generalises, and it was
checked per language rather than assumed.

### One model, five languages

A single Branch A head trained on the pooled train folds of five languages:

| eval language | EER |
|---|---|
| Hindi | 8.30% [7.33–9.32] |
| Bengali | 6.79% [6.03–7.79] |
| Marathi | 1.40% [0.90–2.02] |
| Tamil | 1.92% [1.43–2.57] |
| Telugu | 0.74% [0.42–1.34] |

Pooling costs Hindi a little against its own specialist (8.30% vs 7.54%, CIs
overlapping) and buys four more languages. Two cautions on the low numbers:
Marathi's eval contains **only FreeVC24**, the easier family, because the
strided IndicSynth shards for it held nothing else; and Telugu's 0.74% is
genuine within this corpus but reflects weaker clones, not a stronger
detector. The codec confound was checked there specifically — matched 0.74%
against unmatched 1.47%, the unmatched arm *worse*, so the model is not
reading AAC-versus-PCM.

### Gujarati — the frozen holdout

Gujarati was declared a holdout in `scripts/fetch_datasets.py` **before any
multilingual training ran**, built alongside the others, silence-audited, and
then scored exactly once:

| model | Gujarati, a language it has never seen |
|---|---|
| **5-language pooled** | **6.85%** [6.00–7.88] |
| Hindi-only specialist | 19.83% [18.10–21.63] |
| ASVspoof baseline | 39.44% [37.45–41.28] |

**This is the first time in this project that pooling training domains has
improved out-of-domain performance.** Seven previous attempts — domain
adversarial training, corpus mixing, OC-Softmax, SSL layer selection (§20,
§22, §24, §26) — all failed, most of them making things worse.

The difference is what is being pooled. Those attempts pooled across *corpora*:
ASVspoof against RTCFake against In-the-Wild, which differ in recording
programme, channel, generator family and language all at once. This pools
across *languages within one corpus family* — the same Kathbath recording
programme, the same IndicSynth generators, related phonology. The shift is
small enough to average over, and the model learns the synthesis artefact
rather than the corpus.

So the rule that emerges is narrower and more useful than "mixing does not
work": **pooling helps when the domains differ in one axis and share the
rest.** Five languages of one corpus generalise to a sixth; three unrelated
corpora do not generalise to a fourth.

### Why this number can be trusted

It is the one clean holdout the project has. In-the-Wild has now been scored
by six checkpoints and CodecFake by five, so neither is untouched any more
(§26, §27). Gujarati was frozen in code before training, audited for the
silence shortcut *before* being scored, and read once. Nothing about it
influenced any decision that produced the model being measured.

### Honest limits

Gujarati is Indo-Aryan, as are Hindi, Bengali and Marathi; Tamil and Telugu
are Dravidian and were both in training, so the holdout is not maximally
distant from what was seen. Everything here remains one corpus family —
Kathbath speech and IndicSynth clones — so this measures generalisation across
*languages*, not across recording conditions, which §24 and §26 show is the
harder axis. Six of twenty-two scheduled languages are covered.

---

## 29. Replay — from undefended to defended

`docs/THREAT_MODEL.md` A4 listed physical replay as untested until §21
measured it, and §21 ended with the flattest statement in this document:
**"VeriCall does not defend against replay."** The ASVspoof-trained model
scored 33.68% on `eval_replay`, and the conclusion drawn was that replay needs
a different mechanism — channel or recapture detection rather than
synthesis-artifact detection.

That conclusion was reached without ever using EchoFake's training fold, which
holds **17,511 replay attacks** alongside 8,850 TTS and 8,740 bona fide. The
corpus was fetched, split, evaluated, and its training material left on disk.

Branch A trained on it:

| fold | undefended (§21) | trained on EchoFake |
|---|---|---|
| `eval_replay` | 33.68% | **22.66%** [21.43–24.15] |
| `eval_synth_echo` | 25.89% | **18.50%** [17.30–19.55] |

So replay was not a mechanism problem after all — or at least, not only one.
A spectral-artifact detector *can* learn to flag a recaptured recording when it
is shown recaptured recordings; nobody had shown it any. The gap between 33.68%
and 22.66% is the cost of leaving a training fold unused.

**What this does not claim.** 22.66% is a defence, not a good one — it is
worse than every in-domain number in this document and worse than the
multilingual results. Replay remains the weakest threat this system handles.
The honest statement changes from "does not defend against replay" to
"detects replay at roughly one error in four, which is better than chance and
not good enough to act on alone". It should gate a step-up challenge, not a
refusal.

**And it is still a single-corpus result.** EchoFake's replays were recorded
through EchoFake's devices in EchoFake's rooms. §24 and §26 both show that a
number from one corpus does not survive the move to another, and there is no
second replay corpus here to check against. Treat 22.66% as the ceiling of
what is known, not as a field estimate.

---

## 30. Unfreezing the encoder — the first intervention that actually helps

Seven attempts against cross-corpus collapse are on record and none worked:
naive concatenation, balanced sampling, a domain adversary, OC-Softmax (§20),
SSL layer selection (§22), and the same interventions re-tested on In-the-Wild
(§24) and CodecFake (§26).

**Every one of them changed what happens after the frontend.** Not one changed
the frontend, and not by choice: every trainer in this repo reads `use="feat"`
from a cache of precomputed WavLM activations, so the encoder was frozen *by
construction*. §22 named this exact gap — the 2026 work it failed to replicate
fine-tunes the encoder where this project freezes it — and had no way to test
it.

`scripts/train_finetune.py` tests it. Trained on ASVspoof only, five epochs,
one seed, encoder learning rate 1e-5 against 3e-4 for the head, evaluated on
two corpora held out of training entirely.

| unfrozen layers | best dev | **WaveFake-unseen** | CodecFake |
|---|---:|---:|---:|
| **0 (frozen control)** | 2.46% | **28.36%** | 27.39% |
| 2 | 2.04% | 27.31% | 25.36% |
| **4** | **1.42%** | **25.23%** | 23.34% |
| 5 | 1.60% | 25.43% | **22.55%** |

**WaveFake-unseen improves from 28.36% to 25.23%**, and CodecFake from 27.39%
to 22.55%. In-domain dev improves at the same time, 2.46% to 1.42%. That is the
first intervention in this project to improve an out-of-domain fold **and** the
in-domain one, against seven that improved neither.

The trend is monotone to four layers and flattens at five, which is the whole
usable depth — Branch A truncates at layer 5, so "unfreeze 5" is full
fine-tuning of everything the model actually runs.

### This is an improvement, not a repair

25.23% is still a bad number. It is better than 28.36% and it is nowhere near
the 2.48% this model achieves in domain. A detector that errs on a quarter of
an unseen corpus has not solved cross-corpus generalisation; it has moved three
points. The honest statement is that the direction is finally right and the
magnitude is small.

### The second finding is about model selection, and it may matter more

Within every fine-tuned arm, dev EER keeps improving while out-of-domain
performance degrades:

| unfreeze=4 | epoch 0 | epoch 1 | epoch 2 | epoch 3 | epoch 4 |
|---|---|---|---|---|---|
| dev | 2.10% | 1.71% | 1.54% | 1.77% | **1.42%** |
| WaveFake-unseen | 25.73% | **25.23%** | 27.08% | 28.98% | 27.21% |

Early stopping on dev picks epoch 4 and gets 27.21%; the best out-of-domain
epoch is 1, at 25.23%. **In-domain dev EER is an actively misleading selection
criterion for out-of-domain performance** — it chooses the model that has
specialised hardest on the training corpus, which is exactly the wrong one.

Every number in this document produced by early stopping on dev is therefore
pessimistic out of domain by roughly this margin, including the frozen
baselines it is being compared against. The comparison above uses best-reached
for both arms so it is like for like, but the selection problem is real and
unresolved: there is no honest way to early-stop on out-of-domain performance
without turning the held-out corpus into a validation set.

### Two bugs found by the controls, both of which would have faked the result

**The `@torch.no_grad()` decorator.** `SSLFrontend.forward` is decorated with
it, so `requires_grad_(True)` on the parameters achieves nothing — the graph is
discarded before the optimiser sees it. Caught because the unfreeze=2 arm
produced losses identical to unfreeze=0 to four decimal places.

**Unfreezing layers that never execute.** "Unfreeze the top N of 12" unfroze
layers 11 and 12, and `extract_features(num_layers=5)` never runs them. The log
line said "unfroze 14.2M parameters" and was true and irrelevant. The fix is to
unfreeze the top N *of the layers actually used*.

Either bug alone produces a run that looks like a completed experiment and
reports "fine-tuning does not help" — a false negative on the one lever nobody
had pulled. Both were caught by the frozen control arm, which is the argument
for always including one.

### Limits

One seed, five epochs, one training corpus. The encoder learning rate was not
tuned. Nothing here is a profile that ships; `scripts/train_finetune.py`
produces checkpoints and the served profiles still use frozen heads. What this
establishes is a direction worth spending on, not a deployment.

### 30a. …and it does not transfer to the Hindi profile

The obvious next step after §30 was to ship the unfrozen encoder into the
Hindi profile, on the reasoning that if it bought three points on
ASVspoof→WaveFake it should buy something here. It does not. Same trainer,
same schedule, only the unfreezing changed:

| | Hindi eval | Gujarati (never trained on) |
|---|---|---|
| frozen control | **9.75%** [8.51–10.78] | **15.65%** [14.01–17.28] |
| unfreeze 4 layers | 9.49% [8.34–10.85] | **17.36%** [15.35–19.01] |

In-domain is a wash — the intervals overlap almost entirely. Out of domain it
is **worse**, by 1.7 points.

The cause is visible in the training loss. ASVspoof has 25,380 training clips;
the Hindi fold has **1,138**. Releasing 28.4 M encoder parameters onto 1,138
clips drives the loss to 0.0106 by epoch 4 against the frozen arm's 0.0446 —
the encoder is memorising the training speakers, and the price is paid on the
language it has never seen.

So the rule from §30 needs a condition attached: **unfreezing the encoder helps
when there is enough data to move it, and hurts when there is not.** Twenty-five
thousand clips is enough; one thousand is not. The served Hindi profiles keep
their frozen heads.

A caution on reading these numbers against §25's 7.54%: they are not
comparable. This run used `train_finetune.py` with its own schedule, batch size
and learning rates, where §25 used `train_crossdomain.py`. The only valid
comparison here is frozen against unfrozen *within this table*, which is why
the control arm was run rather than quoting the existing figure.
