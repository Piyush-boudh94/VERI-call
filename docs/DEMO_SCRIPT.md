# VeriCall — demo runbook

Written for the person standing in front of the judges. Every command here was
run on this cluster; every number was produced by the run, not remembered.

**Total time on stage: about 4 minutes.** Setup is 2 minutes and must be done
*before* you are called up — the gateway takes 60–90 s to load and warm four
branches, and that silence is the worst thing that can happen to you.

---

## 0. Pre-flight — do this 15 minutes early, not on stage

```bash
cd "/home/dgx-s-bmu-cse-240577/SIH final"

# 1. Is the pod alive? Its name changes on every reschedule; never hardcode one.
./scripts/pod.sh name

# 2. Start the gateway. Takes 60-90 s: four branches load and each runs once on
#    dummy audio so the first real call does not eat a cold CUDA forward.
#    pod.sh already cds into the project, so the path below is relative to it.
#    setsid + </dev/null is what detaches it from the kubectl exec session --
#    without that it dies when the exec returns.
./scripts/pod.sh exec 'rm -f logs/serve.log && setsid nohup scripts/serve.sh \
    > logs/serve.log 2>&1 < /dev/null &'

# 3. Wait for the line that means it is actually listening.
./scripts/pod.sh exec 'grep -m1 "Uvicorn running" logs/serve.log'

# 4. THE GATE. Do not go on stage until this prints DEMO PATH READY.
./scripts/pod.sh py 'scripts/verify_demo.py'
```

Expected — this is the real output, from this cluster:

```
demo clips from web/demo  (splice at 21.18s)

genuine.wav          risk   -6.63 ->  -94.98  (max -6.63)  tier=0  alert=none
spoofed.wav          risk   +5.55 ->  +79.70  (max +80.90)  tier=3  alert=3.0s
midcall_swap.wav     risk   -6.63 ->  +79.64  (max +80.40)  tier=3  alert=23.0s

  [PASS] genuine.wav          stays low, no alert
  [PASS] spoofed.wav          climbs and alerts
  [PASS] midcall_swap.wav     alerts after the splice

DEMO PATH READY
```

Then put the dashboard on the projector:

```bash
kubectl port-forward -n dgx-s-bmu-cse-240577-restricted \
    "$(./scripts/pod.sh name)" 8001:8001
# open http://localhost:8001
```

**Check `/health` says what you think it says** — it is the honest-config
endpoint and a judge may well ask you to open it. From your laptop once the
port-forward is up, `http://localhost:8001/health` in a browser tab is enough.
From inside the pod there is **no `curl`**, so use Python:

```bash
./scripts/pod.sh py '-c "import json,urllib.request; print(json.dumps(
  json.load(urllib.request.urlopen(\"http://127.0.0.1:8001/health\")), indent=1))"'
```

Look at seven fields: `calibrated: true`, `calibration_pairing: "ok"`,
`notes: []`, `eta_measured: ["A","A2","B"]`, `fusion_weights_fitted: true`,
`calibration_buckets` containing clean/g711/noisy_line, and
`branches_loaded: ["A","A2","B","C"]` — if C is missing, speaker-turn detection
and identity conflict are off and the note will say why. If `notes` is non-empty,
read what it says before you start — it is telling you something about this run
that you would otherwise claim falsely.

---

## 1. The demo, in three clips (~3 min)

The clips are **real ASVspoof 2019 LA dev audio**, not synthetic, not a mock —
`web/demo/manifest.json` records the speaker (LA_0069) and the attack id (A01).
They are committed to git, so they survive the pod wipes that have taken this
project's scratch disk seven times.

### Clip 1 — genuine call. *Say: "nothing happens, and that is the point."*

Risk goes **−6.63 → −94.98 nats** and stays there. Tier 0. No alert.

The line worth saying out loud: *"a fraud system that cannot stay quiet on a
real customer is unusable, whatever its detection rate."* Then leave it running
a few seconds longer than feels comfortable — the flat line is the evidence.

### Clip 2 — cloned voice from the first second. *Alerts at 3.0 s.*

Risk climbs **+5.55 → +79.70**, hits tier 3, alert fires at **3.0 seconds**.

Point at the *accumulation*, not the final number: each window contributes a
calibrated log-likelihood ratio in nats, and they add. That is why confidence
grows with call duration instead of being re-decided every window.

### Clip 3 — the one that matters. *Mid-call speaker swap.*

Genuine speech for 21.18 s, then the caller is swapped for a clone. Risk sits
negative, then climbs the moment the splice lands: **alert at 23.0 s — 1.8 s
after the swap.**

It reaches tier 3 — the fraud desk — before the clip ends. Worth knowing why
that is not guaranteed: escalation is rate-limited by a 10 s cooldown per step,
deliberately, so a detector that flickers cannot storm a fraud desk. Detection at
23.0 s against a clip ending at 42.4 s leaves room for the climb; on a shorter
call after the splice it would end lower, with the same detection time. If a
judge asks: the score is +79 nats against a tier-3 threshold of 6.0 — the
evidence is overwhelming and the *pace* of response is capped on purpose.

Say plainly: **a single-shot classifier cannot express this attack at all.** Ask
it "is this call fake?" and the honest answer for the first 21 seconds is no.
The change-point detector fires on this scenario and not on a call that was
synthetic from the first window — because that call contains no *change*.

---

## 2. Prevention — the half of the problem statement most demos skip

PS 26104 says "Detection **and Prevention**". Detection that does not become an
action is a dashboard.

Show `/health`'s `prevention` block: seven transaction contexts, each carrying a
prior, a miss cost, and **reversibility**. The point to make in one sentence:

> *A statement request and a new-payee addition at the same risk score get
> different responses, because one can be undone and the other cannot.*

If asked about the liveness challenge: the risk adjustment is deliberately
**asymmetric** — passing subtracts 1.5 nats, timing out adds 4.0. A clone that
fails a challenge is strong evidence; a clone that passes one is weak evidence
of innocence, because passing is exactly what a competent real-time system does.
Symmetric scoring would let an attacker buy their score down.

---

## 3. The numbers to quote — and the one not to

| Claim | Number | Where |
|---|---|---|
| Per-call false positives | **0 / 40** | 40 streamed bona fide calls |
| Per-call detection | **40 / 40** | 40 streamed spoof calls |
| Mid-call swap latency | **2.3 s** median after splice | 40-call run |
| Concurrency inside SLO | **64 calls** on one MIG slice | declared before the run |
| Per-chunk latency | **p99 ~104 ms** | flat from N=1 to N=16 |
| Channel detection | **98.8%** per call | held-out labelled audio |
| Concurrent calls in SLO | **64** | real WebSockets, 64/64 complete |

**Do not lead with 2.48%.** That is the in-domain EER and it is the number that
gets systems deployed and then embarrassed. If you are asked for accuracy, give
the honest shape: *in-domain 2.48%, cross-corpus roughly 30%, and on
RTC-transmitted Mandarin the ASVspoof-trained model was worse than guessing until
we retrained it.*

Say "0/40", not "0% false positives". The upper 95% bound on a 0/40 observation
is about 7%, and a judge who knows that will trust everything else you said more.

---

## 4. Questions you will get, and honest answers

**"Does it work on Indian languages?"**
Not yet, and we measured it rather than assuming. At a 5% false-positive budget
the best model catches **13.9%** of Hindi voice-cloning attacks. We also cannot
give you a Hindi EER at all: IndicSynth publishes synthetic speech only, and
pairing it with some other corpus's real Hindi would let the model separate them
by recording conditions instead of by synthesis. That number would look good and
mean nothing, so we did not produce it. This is the biggest gap in the project.

**"What happens on a real phone line?"**
Branch A degrades 2.0× on G.711 and 4.2× on a noisy line. The phase branch
degrades 10.5×, which is why it carries its own per-channel calibration. The
gateway now **detects the channel per call at 98.8% accuracy** and picks the
matching calibration bucket; that cuts the phase branch's calibration loss by
94%. The honest limit: it detects the channel profiles we *simulate*. Real
telephony adds adaptive noise suppression and comfort noise, and that split needs
re-fitting against real transmitted audio before production.

**"Will it generalise to a cloning tool you have not seen?"**
Yes, and that turned out to be the easy half. Holding out entire vocoder
families costs essentially nothing (30.29% unseen vs 31.43% seen). What breaks it
is changing *corpus*, not changing *generator* — which inverts the assumption the
system was originally designed around.

**"How many calls can it handle?"**
**64 concurrent calls** inside every declared SLO on one 18 GB MIG slice —
64/64 complete, chunk p99 22 ms against a 250 ms budget, first score in 1.0 s.
Server-side, excluding transport, it holds real time to **N=160**. That was 4
until we found that torch was sizing its thread pool from the host core count:
112 threads on 16 cores, invisible at one request and the entire problem under
concurrency.

**"What is the weakest part?"**
Cross-corpus generalisation. Retraining on RTCFake removed a total failure
(Mandarin 79% EER, anti-correlated, → 9.5%) but the retrained model is then at
chance on the corpora it used to handle. No configuration we tested is good on
more than one corpus at once. We built a domain guard that detects the specific
failure, and it fires on exactly the cases where it should.

---

## 5. When it breaks on stage

| Symptom | Cause | Fix |
|---|---|---|
| `verify_demo.py`: connection refused | gateway not up yet | `grep "Uvicorn running" logs/serve.log`; it takes 60–90 s |
| Gateway "starts" and exits immediately, status 0 | you ran `python -m vericall.serving.app` | that has no `__main__`; use `scripts/serve.sh` |
| `./scripts/pod.sh name` → no running pod | pod rescheduled | wait and retry; never recreate it, we are a co-tenant |
| `notes` mentions `NOT CALIBRATED` | `calib/` missing | risk numbers are meaningless — **stop, do not demo** |
| `calibration_pairing: "mismatch"` | heads and calibration from different corpora | unset `VERICALL_CKPT_*` / `VERICALL_CALIB_DIR` and restart |
| `notes` mentions a missing bucket | codec detection on, calibration for that channel absent | harmless for the demo (clips are clean); re-run `scripts/fit_channels.py` to fix properly |
| `fusion_weights_fitted: false` | `calib/weights.json` missing | falls back to the hand-set defaults — works, but quote 3.04% not 2.79% |
| Dashboard blank | port-forward died | restart `kubectl port-forward` |
| Dashboard says `unauthorized` | gateway has `VERICALL_AUTH_TOKEN` set | open the page as `?token=…`, or set `localStorage.vericall_token` |
| `branches_loaded` missing `C` | speechbrain could not fetch ECAPA | demo still works on A/A2/B; identity conflict is off |
| Scores look like noise | stale `.pyc` over NFS | `pod.sh` exports `PYTHONDONTWRITEBYTECODE=1`; restart the gateway |

**The one rule:** if `verify_demo.py` does not print `DEMO PATH READY`, do not go
on. Every failure above is silent from the dashboard's point of view — it will
happily draw a beautiful line through meaningless numbers.

---

## 6. Serving a different deployment profile

Not part of the 4-minute demo, but worth knowing if asked "could you deploy this
for a bank in Mumbai?" — the heads and calibration are selectable, and the
pairing between them is checked:

```bash
VERICALL_CKPT_A=checkpoints/branch_A_rtc_seed0.pt \
VERICALL_CKPT_A2=checkpoints/branch_A2_rtc_seed0.pt \
VERICALL_CALIB_DIR=calib_rtc \
  ./scripts/serve.sh
```

Verified: loads clean, `calibration_pairing: "ok"`, and reports honestly that
`calib_rtc` has no per-channel bucket so it fell back to pooled. Point the same
variables at a mis-matched pair and `/health` says `mismatch` and names both
branches. The honest framing: **refitting calibration per deployment is cheap
and works; retraining per deployment is neither.** A procedure that fits
calibration on a sample of the tenant's own traffic is the obvious next step and
is not built.
