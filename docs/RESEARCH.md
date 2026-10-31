# VeriCall — annotated bibliography

**Every entry here was checked against source during this project.** That is not
a formality. The v3 blueprint cited "Pindrop's OCCAS pattern" as a grounding
reference; a targeted search returns nothing by that name, and the real
technology is Pindrop Pulse. A fabricated citation is worse than a missing one,
because it survives review by looking like diligence — so each entry below
records what it actually justifies in this system, and where a source does *not*
support the use it was put to.

Annotations say what the work is used **for here**, not what it is about in
general.

---

## 1. Corpora

**ASVspoof 2019** — Edinburgh DataShare, handle `10283/3336` (LA subset, 7.1 GB).
The primary training and in-domain evaluation corpus. Chosen because its
protocol is the field's reference point, so our EER is comparable to published
work rather than to itself. Train attacks A01–A06, eval attacks A07–A19 are
disjoint by design, which gives an unseen-attack axis for free.
*Limits, measured:* see Müller et al. 2021 below — it carries a silence artifact
severe enough to invalidate naive results.

**WaveFake** — Frank & Schönherr, NeurIPS Datasets & Benchmarks 2021.
Zenodo record `5642694`, 104,885 clips, ~175 h. Used for the **held-out vocoder
family** axis that ASVspoof does not provide: train on MelGAN / PWG / WaveGlow,
test on HiFi-GAN / FB-MelGAN / TTS.
*Limit we enforce in code:* LJSpeech is one speaker and JSUT is one speaker, so
WaveFake supports **no speaker-generalisation claim whatsoever**. Its fakes are
also vocoder resyntheses of the same sentences as the bona fide references, so
a source utterance and all its variants must share a fold — enforced in
`tests/test_splits.py`, not left to discipline.

**In-the-Wild** — Müller et al., Fraunhofer AISEC. 37.9 h, 58 English public
figures. Intended as the never-trained-on cross-domain set. *Honest status: it
was fetched once and lost to a pod wipe, and the cross-corpus result reported in
`RESULTS.md` rests on WaveFake and RTCFake instead.*

**RTCFake** — Xue et al., ACL 2026, arXiv `2604.23742`; HuggingFace
`JunXueTech/RTCFake` (gated, ~600 h, 20.4 GB fetched). **The most important
corpus in this project's conclusions.** Speech transmitted through real RTC
platforms, with metadata pairing each utterance offline (clean) against itself
online (transmitted). That pairing is what makes channel effects separable from
corpus shift — same speaker, same content, same label, only the channel differs.
Its en/zh split also gave a free cross-lingual test. `RESULTS.md §13` is built on
it.

**IndicSynth** — ACL 2025 (Outstanding Paper), HuggingFace
`vdivyasharma/IndicSynth`, CC BY-NC. 4,000 h across 12 low-resource Indian
languages. *Status: wired into the fetcher, never fetched or used.* Every number
in this project is English or Mandarin, which is a real gap for an AICTE problem
statement and is recorded as one rather than glossed.

**EchoFake** — replay-aware corpus (IEEE). Cited for the threat model only;
never obtained, so **replay attacks remain untested here**.

---

## 2. Model architecture

**WavLM** — Chen et al., *IEEE JSTSP* 2022, DOI `10.1109/JSTSP.2022.3188113`,
arXiv `2110.13900`. The frozen SSL front end for Branch A, truncated at layer 5.
Freezing it is the single largest overfitting control available on a 16 GB MIG
slice: 0.89 M trainable parameters against 94.4 M frozen, a 105:1 ratio.
*What this project adds:* §13 measured WavLM-layer-5 features **inverting** under
corpus shift (79% EER on Mandarin RTCFake). That is not a criticism of WavLM; it
is evidence that a frozen general-purpose front end plus a corpus-specific head
learns corpus-specific artifacts.
§22 then swept layers 3/5/7/9/11 under otherwise identical conditions and found
**depth monotonically harmful** (unseen-vocoder EER 29.11% at layer 3 rising to
38.71% at layer 11), with layer 3 *inverting* on Mandarin at 56.63% (CI excludes
50%). Lower layers are thus both the most discriminative in-domain and the most
confidently wrong out-of-language.

**SSL layer selection for OOD generalisation** — the 2026 line of work rating
the **readout layer** as a dominant factor in out-of-domain spoof detection.
*This project does not replicate it*, and the disagreement is stated rather than
smoothed over: §22 finds the top layer measured to be the worst of five by 8.5
points. Two design differences are plausible causes and this project cannot
separate them — the cited work **fine-tunes** the encoder and reads a **learned
weighted sum** of all layers, whereas this one keeps the encoder frozen and
**truncates**. A frozen top layer has specialised toward phonetic content with
no chance to adapt back toward artefacts. The practical consequence is recorded
honestly: the sweep produced **no accuracy gain**, only the retirement of a
hypothesis and a confirmation that layer 5 was already correct.

**AASIST** — Jung et al., *ICASSP 2022*, pp. 2405–2409, DOI
`10.1109/ICASSP43922.2022.9747766`, arXiv `2110.01200`; reference implementation
`clovaai/aasist`. Branch A's head is **AASIST-inspired, not AASIST**: it keeps
the two-view attention-weighted readout — which is what carries the accuracy,
because spoofing artifacts are local and uniform pooling dilutes exactly the
frames that carry them — but replaces heterogeneous stack-node attention with
concatenated readouts, and uses a latent-node graph rather than a spectral one.
That substitution is forced, not stylistic: SSL features have no frequency axis
to build a spectral graph over.

**ECAPA-TDNN** — Desplanques, Thienpondt & Demuynck, *Interspeech 2020*,
pp. 3830–3834, DOI `10.21437/Interspeech.2020-2650`. Branch C's speaker
embedder, via SpeechBrain `spkrec-ecapa-voxceleb`. Used **only** as an identity
signal and for speaker-turn detection, never as a spoof score — see §4 below.

**Deep-SVDD** — Ruff et al., *ICML 2018*, PMLR 80:4393–4402. The one-class
objective behind Branch B. Its documented requirement that the network be
**bias-free** is the reason `ProsodyEncoder` has `bias=False` throughout and
`affine=False` BatchNorm: any constant path lets the encoder map every input to
the centre, driving loss to zero while learning nothing.
*What this project adds:* we hit that collapse (embedding std 0.0057, dev EER
49.8%), and recorded that the prescribed fix had been applied and **was not
sufficient** — spread only improving to 0.0116.

**Correction, 2026-09-23.** The fix had never been applied. `ProsodyEncoder`
carried default `bias=True` Conv1d/Linear and `affine=True` BatchNorm until a
test asserted otherwise, so this paragraph described a guard that was not in
the code — most likely lost in the September reconstruction that also
overwrote `a2_phase.py`. It went unnoticed because Branch B is zero-weighted
in every profile, so no served number depended on it.

With the guard genuinely in place the conclusion survives, but it is now
earned rather than assumed. Collapse is learning-rate sensitive, over six runs
on ASVspoof:

| lr | seed 0 | seed 1 |
|---|---|---|
| 1e-4 | collapse, std 0.0499 | **survives**, spread 0.0534 |
| 3e-4 | collapse, std 0.0315 | collapse, std 0.0338 |
| 1e-3 | collapse, std 0.0157 | collapse, std 0.0172 |

The one configuration that trains reaches **46.06% dev EER** — chance. So
bias-freedom was necessary and nowhere near sufficient, and the remaining
cause is the one already identified: prosody carries little signal on this
corpus (~39% supervised ceiling). `checkpoints/branch_B.pt` is that model, and
Branch B stays zero-weighted in every profile.

---

## 3. Calibration and evaluation methodology

**Cllr / min-Cllr** — application-independent LLR cost, standard in NIST SRE
evaluation. Used because EER measures *ranking* and is blind to
**calibration**, which is exactly what the fusion layer depends on: every
threshold assumes the branch outputs are LLRs. The gap `Cllr − min-Cllr` is
reported as calibration loss (0.082 → 0.011 after fitting).

**t-DCF / cost-weighted detection cost** — the ASVspoof line of work on
application-weighted evaluation. Underpins the cost-optimal threshold
`θ* = log[C_fa(1−π) / (C_miss·π)]` used by the prevention policy.
*Finding worth stating:* worked through realistic banking costs, θ* lands
**below the reliable-calibration floor in every high-value context**, so
thresholds are clamped and the real constraint on alerting is agent capacity,
not decision theory.

**"Speech is Silver, Silence is Golden"** — Müller, Dieckmann, Czempin, Canals,
Böttinger & Williams, ASVspoof 2021 Workshop (Interspeech satellite), arXiv
`2106.12914`. **The single most load-bearing methodological citation here.**
Bona fide ASVspoof clips have systematically longer leading and trailing
silence than spoofed ones, and a model trained on leading-silence duration
*alone* reaches 85% accuracy / 15.1% EER with no speech content at all.
This is why VAD trimming before windowing is mandatory in our pipeline and why
we run a duration-only control baseline (measured at 36.4% EER on our split)
that every real model must clearly beat.

---

## 4. Robustness, and where sources were misapplied

**RawBoost** — Tak, Kamble, Patino, Todisco & Evans, *ICASSP 2022*, arXiv
`2111.04433`; code `TakHemlata/RawBoost-antispoofing`. Telephony-designed
augmentation needing no external noise or impulse-response corpora; +27%
relative on ASVspoof 2021 LA. *Status here: specified in
`PRODUCTION_READINESS.md`, **not implemented**.*

**SafeEar** — Li, Li, Zheng, Yan, Ji & Xu, *ACM CCS 2024*, arXiv `2409.09272`.
Content-privacy-preserving detection via a neural-codec decoupling that keeps
acoustic and discards semantic information; EER down to 2.02%, WER above 93.93%
against content recovery.
*Correction this project made:* the v3 design claimed SafeEar's bottleneck
destroys prosody and therefore moved all branches outside it. Reading the method,
it is the **shuffle layer** (random permutation within a 1 s / 50-frame window)
that destroys temporal trajectories, not the semantic/acoustic split. Transport
the acoustic tokens without shuffle and prosody survives while content does not.
v3 discarded the whole mechanism when one layer was the problem.

**Proteus** — Müller, Bukkapatnam & Ahmed (Resemble AI), arXiv `2606.29544`.
Automated adversarial robustness testing by breadth-first search over chains of
everyday audio transformations. Cited for the release-gate design; **not run**.
Its structural finding — that depth-3 chains reveal interaction effects invisible
to single-step evaluation — is the reason the gate is specified as chain search
rather than per-perturbation testing.

**PACT** — Ndichu, Ban, Ozawa, Takahashi & Inoue, arXiv `2605.22324`. Triggered
active learning for low-prevalence SOC alert streams; 43% / 21% reduction in
false-positive burden using 3.8× / 5.2× fewer analyst queries. Cited for the
alerting-policy design. Its *guardrail* matters as much as its result: a frozen
threshold-only baseline pushed FP lower but collapsed recall by 55 points.

**US Patent 10,764,043** — "Identity and content authentication for phone
calls". Source of the n-of-m aggregation pattern (3-of-5 one-second digests).
*Where it does NOT transfer, measured:* its binomial error model is valid there
because each digest is compared against a **cryptographically bound reference**,
so the trials really are independent. VeriCall has no reference — our chunks are
model outputs on overlapping, same-speaker audio with lag-1 autocorrelation
+0.824. Measured on real bona fide streams, the binomial prediction is
**9.3×–17.8× optimistic** for 3-of-5. Sizing an alerting budget from it would
under-provision false positives by an order of magnitude.

**Pindrop Pulse** — commercial liveness detection ("Fakeprint"), reported at
~99% on known engines and >90% on unseen at <1% FPR from ~2 s of netspeech.
**Replaces the v3 blueprint's citation of "Pindrop's OCCAS pattern", which does
not exist.** The architectural idea it stood in for — score early on short
netspeech — is sound and retained; only the attribution was wrong.

---

## 5. India-specific regulatory surface

Recorded as **fields to populate, not integrations**, because whether a given
call is reportable is a legal determination the deploying institution makes.

- **TRAI CNAP** (Calling Name Presentation) — network-verified, KYC-bound caller
  name. Valuable precisely because it is bound to identity records rather than to
  the audio, so a CNAP mismatch alongside a high spoof score is materially
  stronger evidence than either alone.
- **CERT-In deepfake advisory (Nov 2024)** — incident-handling expectations.
- **IT Amendment Rules 2026** — synthetic-media labelling and a 36-hour takedown
  window for intermediaries. The audit record stores the clock start rather than
  claiming compliance.

---

## 6. Sources consulted and deliberately not used

- **ASVspoof 5** — ~500 GB. Ruled out on disk and time, not on merit.
- **DEEP-VOICE** — 62 minutes, 8 speakers, RVC only. Listed as a grounding
  source in v3; too small to support any claim, and trivially overfittable.
  Retained only as a smoke-test fixture, with no accuracy quoted from it.
- **InDeepFake** — cited in v3 for audio fine-tuning, but it is a **video**
  corpus (facial videos, 189 subjects), gated to academic institutions.
  Removed from the audio plan.
