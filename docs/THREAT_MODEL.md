# VeriCall — threat model and harm analysis

A fraud detector is a machine for making accusations. This document is about
what happens when it is wrong, who it is wrong about, and who can make it wrong
on purpose.

Everything here is classified as **measured**, **argued**, or **unknown**.
"Unknown" is the honest majority for a system this young, and an unknown listed
is worth more than a control asserted.

---

## Part 1 — Harm to people

The failure everyone plans for is the missed attack. The failure that actually
recurs in deployed fraud systems is the false accusation, because it is frequent,
it lands on people who did nothing, and it is invisible in an aggregate accuracy
number.

### H1. A legitimate customer is refused — **the primary harm**

A real person calls their bank and is told their voice is synthetic. What
follows is a held payment, a frozen account, an interrogation, and the burden of
proving they are themselves. For someone paying rent or a medical bill that day,
"resolved within 48 hours" is not resolution.

**What bounds it.** Per-call false positives measured at **0/40** on streamed
ASVspoof bona fide audio. An honest reading of 0/40 is an upper 95% bound of
about **7%**, not 0%. At 50 concurrent calls, 7% is a great many people.

**What makes it worse than the number suggests:** the tier policy escalates to
`fraud_desk` on accumulated evidence, and a long genuine call accumulates more
evidence than a short one. Nothing has measured whether false-positive rate rises
with call duration; the accumulator's design implies it can.

**Mitigations in place:** `T_cap` bounds banked evidence; hysteresis and a 10 s
escalation cooldown prevent a single spike from reaching the fraud desk;
`SILENT_FLAG` exists precisely so low-confidence suspicion does not touch the
customer.
**Not in place:** any appeal path, any human-review requirement before an
irreversible action, any measurement of how long a wrongly-flagged customer
stays flagged.

### H2. The harm is not evenly distributed — **the one to check first**

A system trained on English and Mandarin, deployed in India, will meet
Indian-accented English and twelve Indian languages it has never heard. If its
false-positive rate is higher on those speakers, it does not merely perform
worse — it **systematically accuses one group more than another for the same
behaviour.** That is discriminatory harm whether or not anyone intended it, and
aggregate EER cannot see it.

**Measured, and it runs the opposite way to the one people look.** At a
threshold set on English bona fide speech, Mandarin speakers are flagged **less**
often (0.47x the false-positive rate at the 5% operating point) and protected
**far less**: **69.6% of real attacks against them get through, against 38.2%
for English speakers** — a 1.82x miss disparity (§19).

Reporting only the false-positive column would have read as "no problem". A
group the model scores lower is accused less *and* defended less, and only the
first produces a complaint — **a missed attack has a victim who never learns the
system was watching.**
**Unknown, and material:** accent within a language, age, gender, speech
disorders, and recording device. This project has no Indian-language bona fide
corpus at all (IndicSynth is spoof-only — §16), so the population that matters
most for PS 26104 is the one we cannot yet measure.
**Required before deployment:** a bona fide corpus for the deployment's actual
population, and a per-group false-positive report as a release gate rather than
a post-hoc audit.

### H3. Automation bias — the score becomes a verdict

The system emits calibrated log-likelihood ratios in nats and a tier. An agent
under time pressure reads "tier 3: fraud desk" as a fact about the caller. The
uncertainty the system carefully computes is discarded by the human at the last
step.

**Argued, not measured.** The dashboard shows a risk trace and a confidence
band, and `RiskView` carries `calibration_suspect` with reasons. Whether an
agent under load reads any of it is a human-factors question nobody here has
tested.
**What would help:** phrase the surface as evidence rather than verdict, require
an explicit reason code for irreversible actions, and log agent overrides so
that disagreement between human and model is measurable.

### H4. Voice is biometric data

Audio transits the service and features are computed from it. Under India's
**DPDP Act 2023**, voice used to identify a person is personal data with
consent, purpose-limitation and breach-notification duties attached.

**In place:** the audit log enforces `FORBIDDEN_KEYS` — no audio, PCM,
waveform, samples, transcript or text can enter a prevention record, checked on
every write. Records carry decisions and scores, not speech.
**Not in place:** retention limits on audio in flight, encryption at rest for
the audit log, consent capture, a documented data-processing agreement, and
anything resembling a deletion path. `SafeEar`'s decoupling (see
`docs/RESEARCH.md`) is the architecturally right answer and is not implemented.

---

## Part 2 — The adversary

### A1. Evasion by channel manipulation — **the cheapest attack, and it works**

An attacker who knows the detector degrades on channel artifacts simply adds
them. Measured degradation with no adversarial effort at all: Branch A **2.0×**
on G.711 and **4.2×** on a noisy line; Branch A2 **10.5×**. An attacker does not
need to defeat the model — they need to make the call sound like a bad line,
which is free and indistinguishable from a genuinely bad line.

**Partially mitigated:** runtime channel detection (98.8% per call) now selects
the matching calibration, so degraded audio is scored against the right
distribution rather than the clean one.
**Unmitigated:** the underlying discriminability still falls. Detection picks
the right ruler; it does not restore the signal.

### A2. Cross-corpus collapse as an attack surface

§16 measured a model trained on one corpus scoring at chance on another, and the
original ASVspoof-trained Branch A **inverting** on Mandarin (79.08% EER —
worse than guessing). An attacker whose generator resembles the training corpus'
*bona fide* class rather than its spoof class gets a system that argues on their
behalf.

**Mitigated for detection:** the domain guard (§14) watches for score drift and
branch disagreement, latches on trip, forces confidence LOW and surfaces the
reason. Validated on real out-of-domain data.
**Unknown:** whether an attacker can craft audio that stays inside the guard's
reference distribution while inverting the branches. Nobody has tried.

### A3. Adversarial perturbation

Small, deliberately-optimised perturbations are the standard attack on any audio
classifier. **Completely untested here.** `Proteus` (chained everyday
transformations, `docs/RESEARCH.md`) is specified as a release gate and has never
been run. Treat robustness to a motivated adversary as **unestablished**.

### A4. Replay

Physical replay of recorded genuine speech. **Untested.** EchoFake exists, is
3 GB, and has never been fetched. A replay attack is not synthetic speech, so
there is no reason to expect branches trained on synthesis to catch it.

### A5. Attacks on the service rather than the model

| Vector | Status |
|---|---|
| Unauthenticated access | **Fixed** — constant-time shared-secret check (`hmac.compare_digest`; `==` leaks the matching prefix through timing) |
| Session hijack via guessed `call_id` | **Fixed** — a duplicate id is refused, not silently substituted |
| Log injection via `call_id` | **Fixed** — charset+length, anchored `\A..\Z` (Python's `$` also matches before a trailing newline, and a trailing `\n` was getting through) |
| Resource exhaustion by connection count | **Fixed** — admission cap, refuse with 1013 |
| Memory exhaustion by frame size | **Fixed** — 64 KiB cap, frame refused, call kept |
| Session leak via half-open sockets | **Fixed** — idle timeout, release in `finally` |
| Unbounded inference queue → OOM | **Fixed** — bounded queue that sheds and reports |
| Disk exhaustion via audit log | **Fixed** — size-based rotation with bounded generations |
| **Arbitrary code execution via checkpoint** | **Fixed** — `weights_only=True`. `torch.load` with `weights_only=False` executes pickle, so anyone who could write to `checkpoints/` owned the serving process |
| Checkpoint integrity | **Unknown** — nothing verifies a checkpoint is the one that was evaluated. A hash manifest checked at load is the obvious control and is not built |
| Transport encryption | **Deployer's** — the gateway speaks `ws://`; `wss://` termination is the ingress's job and must not be skipped |
| Rate limiting per client | **Not built** — the admission cap bounds total calls, not calls per caller, so one client can occupy every slot |
| DoS by expensive audio | **Unknown** — no measurement of whether crafted audio makes the pipeline pathologically slow |

---

## Part 3 — Operational failure

### O1. Silent mis-calibration — the failure that looks like success

A head served against a calibration fitted for a different head produces
**confident nonsense**, not a degraded score, and nothing downstream can tell
from the numbers.

**Mitigated:** `fit_subsets.py` records which heads it fitted for;
the gateway compares and reports `calibration_pairing` in `/health`. Verified by
deliberately mis-pairing. Weights live in the calibration directory for the same
reason.
**Residual:** the check compares filenames, not content. Renaming a checkpoint
defeats it.

### O2. Falling behind real time

A live-call detector that processes 1 s of audio in 1.2 s has not "slowed down"
— it has stopped being a live-call detector, and the gap grows for the whole
call.

**Measured:** ≤ 1.0× real time server-side through N=160; 50 and 64 concurrent
calls pass every SLO over real WebSockets.
**Mitigated:** the bounded queue sheds rather than queues, and every shed window
is recorded on the fusion state so the UI cannot show confidence built on
evidence that was dropped.
**The leading indicator is queue depth**, exported at `/metrics/prom`; it rises
before latency does.

### O3. Restart drops calls mid-sentence

**Mitigated** — SIGTERM stops admission and drains within a timeout set below
the orchestrator's grace period.
**Residual:** single process, no replication. Any crash loses every in-flight
call, and there is no session handoff.

### O4. Model and data drift

The generators change monthly. A model fitted in September is measuring a threat
that has moved by December.

**Partially mitigated:** the domain guard detects the specific drift that
occurred. Per-deployment calibration refitting is cheap and supported.
**Not built:** scheduled re-evaluation, an alert on distribution shift, or any
gate that stops a stale model serving.

---

## Part 4 — What would have to be true to deploy this

Not a roadmap — an entry condition. Each is currently **not met**.

1. **A bona fide corpus for the actual deployment population**, and a per-group
   false-positive report as a release gate (H2).
2. **A human in the loop before any irreversible action**, plus an appeal path
   with a measured resolution time (H1).
3. **Adversarial robustness established**, not assumed — at minimum the Proteus
   chain gate (A3) and a replay evaluation (A4).
4. **Checkpoint integrity** verified at load against a signed manifest (A5).
5. **Data protection**: retention limits, encryption at rest, consent capture,
   deletion path (H4).
6. **Per-client rate limiting** and `wss://` termination (A5).
7. **Drift monitoring** with a gate that can stop a stale model (O4).

The honest summary: this is a **well-measured research prototype with a
production-shaped serving layer**. The serving layer is now hardened and load-
tested. The model's behaviour on the population it would actually meet is the
gap, and no amount of engineering closes it — only data and evaluation do.
