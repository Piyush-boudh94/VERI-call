# VeriCall — session continuity

**Purpose:** so a new session (different model, different effort level) can pick
up without reading the full chat transcript.

**Read order:** this file → `CLAUDE.md` (rules + current state) → whichever of
`docs/` you need. This file is the *narrative*: what was decided, what was
already tried, what keeps going wrong. It deliberately does **not** repeat the
measured numbers or the rules — those live in `CLAUDE.md` and `docs/RESULTS.md`
and would go stale here.

---

## 1. What the user actually decided

Their answers, so these are not re-litigated:

| Question | Answer | When |
|---|---|---|
| Deadline | **Internal hackathon, ~2–3 weeks** (asked Sep 4 → roughly **Sep 18–25**) | Sep 4 |
| GPU quota | **Co-tenant** — run inside the existing `wmamba-dev` pod, never modify or restart it | Sep 5 |
| Datasets | **Tier 1 + Tier 2** (ASVspoof LA, WaveFake, In-the-Wild, RTCFake, IndicSynth subset) | Sep 4 |
| Collateral | **Technical whitepaper + annotated research bibliography** | Sep 4 |
| Storage/compute | "use full potential of dgx gpu… nothing should be on cpu" | Sep 4 |
| Demo staging | Pre-recorded clips through a **real** call path, not live cloning on stage | Sep 4 |
| **Deployment target** | **Export to their own laptop and run on an RTX 4050 (6 GB).** Demo runs locally. Cloudflare hosting is a "maybe later", explicitly not now | **Sep 17** |
| **Pod/hosting worries** | **Explicitly off the table** — "there will be no problem of rotating pod, or hosting it on server issues". Stop designing around the pod wipes for the PRODUCT; they still bite development | **Sep 17** |
| Time budget | **"whole lot of time"** — depth is preferred over speed now | Sep 17 |
| Free hand | Given, twice: find issues and fix each one specifically, retrain if needed, use internet sources | Sep 16–17 |

### Standing preferences they have expressed
- Wants conflicts and errors **found and named**, not smoothed over. Every spec
  they have handed over has been given with "find the issues in this".
- Wants honest numbers. Reacted well to retractions and to "this is a ceiling,
  not a result".
- Treats STOP gates literally; does not want work proceeding on inferred consent.
- Wants `CLAUDE.md` / audit docs kept current.

### Still unconfirmed — do not assume settled
- **Deployment mode.** Mode B (server-side) was *assumed* and built on. The
  question was asked but their reply answered a different question (demo
  staging). Never explicitly confirmed.
- The three open questions at the end of `docs/PLAN.md` (demo mode, whether to
  spend a day improving cross-domain EER, whether to consolidate sessions) were
  never answered.

---

## 2. The arc so far

**Phase 0–4 (Sep 4–5, uncommitted — git did not exist yet).** Fusion engine and
calibration first, deliberately, because they are pure float math and provable
before any model exists. Then GPU features, dual-rate chunker, Branch A, the
corpora and leakage-safe splits, training, calibration, first end-to-end run.

**Section 9 (Sep 5).** A production-readiness spec the user supplied; nine
further conflicts found (C11–C19), two severe enough to change the design.

**The incident (Sep 5).** A parallel agent session overwrote much of the tested
implementation with stubs, broke the venv, and left the demo serving random
numbers. Recovery, then **git was finally initialised** — its absence was the
root cause. See `docs/INCIDENT_2026-09-05.md`.

**Sep 11.** Second pod wipe. Re-fetched, rebuilt cache, fitted Branch C and the
per-subset table, and ran the four-branch end-to-end for the first time.

---

### Sep 16–18 — the product layer, and a negative result worth having

Given a free hand twice, the work split three ways. **Research**: read the 2026
literature and implemented what it rated highest — OC-Softmax, domain
adversarial training, RawBoost, SSL layer selection. The cross-corpus
experiment then returned a clean negative (section 20) and it is recorded as
prominently as any success. **Auditing**: probed the fusion engine with values
a broken branch can emit and found three fail-OPEN paths, quarantined a dead
loader that returned labelled silence on any error, and found EchoFake shipping
1,986 duplicate utt_ids that silently corrupted a split. **Product**: live
microphone, a call-history database with an API and UI, and `run_local.py` so
it runs on a laptop.

The target changed on Sep 17: the demo now runs on the user's own 4050, not on
the cluster. VRAM was measured first and is a non-issue (0.96 GiB at batch 32
against ~5 GB usable); the constraint is compute, so the knob is concurrency.

---

## 3. Things already tried that did not work — do not redo

| Tried | Outcome | What to do instead |
|---|---|---|
| Downloading corpora from canonical hosts (Edinburgh DataShare, Zenodo, Fraunhofer) | 25–180 KB/s. ASVspoof alone was a ~10 h download | **HuggingFace mirrors** (13–23 MB/s), already wired into `scripts/fetch_datasets.py` |
| Zenodo over default DNS | `Network is unreachable` — pod has no IPv6 route | IPv4 pinning (already in the fetcher) |
| Deep-SVDD collapse fix by removing biases | Insufficient. Spread 0.0057 → 0.0116, still collapsed | Collapse is only half the story — prosody's supervised ceiling on this corpus is ~39% EER (`docs/RESULTS.md` §6) |
| Naive dynamic batching across calls | Collects ~2 requests per 20 ms, not 32 | Global tick alignment (built) |
| Per-call score normalisation | Subtracts off exactly the shift being detected on a call spoofed from t=0 | Per-channel-bucket calibration (built) |
| n-of-m binomial error model | 9.3–17.8× optimistic on real correlated scores | Measure the rate empirically; `effective_m()` gives the honest window |
| A2 architecture recovery from checkpoint shapes | Loaded with zero missing keys and scored **at chance** — activations carry no parameters and are invisible in a `state_dict` | Retrain (~3 min) rather than guess |

---

## 4. Recurring traps in this environment

These have each cost real time more than once.

1. **Pod scratch is wiped without warning.** Twice so far (Sep 4→5, and again by
   Sep 11). `/tmp/vericall-{data,cache}` vanish; everything on NFS survives.
   Recovery: `./scripts/rehydrate.sh`, then `scripts/extract_asvspoof.sh`.
2. **NFS staleness.** The pod is a separate NFS client and can serve stale
   source or a stale `.pyc` (observed: 7 min older than its `.py`). `pod.sh`
   handles it now. If a fix "doesn't work", suspect this before rewriting it.
3. **Head-node memory pressure kills background waiters.** The *waiters* die;
   detached pod work (`setsid nohup`) survives. Check the log, not the task.
4. **A second agent session may be editing this folder.** Has happened twice.
   Diff before trusting; `git status` is now meaningful.

- **The NFS home filesystem fills up and it is SHARED.** Hit 100% on Sep 16
  with 156K free — could not write a single byte, which blocks training
  (checkpoints), manifests and even a log file. `~/research` is 136 GB of the
  user's WMamba project; other users hold 921 GB, 142 GB, 96 GB on the same
  5.3 TB volume. **Safe reclaim: `rm -rf ~/.cache/pip`** (7.3 GB, just
  downloaded wheels). Never touch `~/research` without asking.
- **Scratch has now gone TEN times**, twice mid-job. Any long script must
  fetch and rebuild what it needs rather than assume scratch is warm — see
  `scripts/stage_layer_sweep.sh` for the pattern.

### Mistakes this assistant has made — worth not repeating

**Sep 22 — two bugs in a measuring instrument, one of which nearly became a
finding.** Measuring the Opus channel, the script (a) fed posteriors to a Platt
mapping fitted on logits, reporting Cllr 0.93 against a recorded 0.083, and
(b) zero-padded 4 s crops when 95% of dev clips are under 4 s. The padding
drives `const_frame_ratio` to 0.25, which trips the channel detector's
packet-loss test, and **clean audio classified as `noisy_line` 53% of the
time.** That was reproducible, looked like a real defect in the detector, and
the obvious write-up would have sent the next work item entirely the wrong way.
The tell: 53% is suspiciously close to a number already in the detector's own
docstring describing a *previously fixed* bug. When a measurement indicts a
component that was recently fixed, suspect the harness first.

**Sep 22 — overstated an impact before checking the weights.** Wrote that the
missing Branch B cohort would shift the served operating point "because the
fusion weights were fitted with B contributing". `calib/weights.json` fits only
A and A2, so w[B]=0 and the engine discards B events entirely. Corrected in the
same session, but it was committed first.

**Sep 22 — `pkill -f` / `pgrep -f` matched the running shell three times**,
killing the command mid-script (exit 144). The pattern appears in the shell's
own command line. Use a bracket class (`[u]vicorn`) and never put the kill and
the launch in one invocation.

- Swept a stub into a recovery commit via `git add -A` without reading the diff
  (`rehydrate.sh`, undetected for 6 days).
- Wrote a `pgrep`-based wait that races the process it waits for.
- Labelled a default value as "MEASURED" in a log line because the print
  statement was written before the value was wired in.
- Sampled a cohort with a positional prefix on speaker-grouped data (3 of 20
  speakers).
- Set a diagnostic's pass threshold at 40% EER, which duly passed at 39.4% and
  printed a reassuring verdict about a near-chance result.

Pattern worth noticing: **most of these were verification bugs, not logic bugs.**
The check itself was wrong, so the failure looked like success.

---

- **Wrote a docstring asserting a data relationship that did not exist.**
  Claimed EchoFake's `source` pairs a bona fide clip with its attacks, by
  analogy with WaveFake. Measured: 19,964 rows, 19,964 distinct sources, zero
  repeats. **Check the grouping before documenting it.**
- **Referenced functions that did not exist** (`handleMessage`, `reset()` in the
  dashboard; `balanced=` on `CachedFold`). The JS ones would have thrown on the
  first live frame as a silent console error; the Python one killed the layer
  sweep AFTER the whole extraction finished.
- **Wrote a check that printed "ok" after its own FAIL** (`check_deps` in
  run_local.py). A check that reports success alongside failure trains you to
  skim it.
- **Seeded a max at 0.0 for a signed quantity.** Genuine calls score about
  -94 nats, so every clean call recorded `+0.0` peak risk — which reads as "no
  evidence either way" when the truth was the opposite.
- **Added a 4th experiment run without adding it to the evaluation list**, so
  it trained and was never scored. A four-row table where five runs happened
  looks entirely normal.

## 5. Findings that changed the design

Full list in `docs/CONFLICTS_RESOLVED.md` (24 items). The ones that would be
expensive to rediscover:

- **Branch C is an identity signal, never a spoof score.** A good clone *raises*
  voiceprint similarity. Adding it to the spoof score makes the best attacks
  look safest.
- **The domain gap dwarfs the vocoder gap** (12× vs ~0). Effort spent
  anticipating new cloning tools is close to wasted; corpus and channel
  robustness is where accuracy lives. Generator diversity must be specified by
  *synthesis paradigm*, not family.
- **η is per branch, not global**: A = 0.207, A2 = 0.928, B = 0.565. Call-level
  consistency is real for A (correlation decays only +0.38 -> +0.23 over eight
  lags) but A2's phase residual decorrelates within one hop. A single global η
  would have been wrong for two of the three.
- **An earlier η = 0.080 was an artifact, and the artifact is the lesson.** The
  chunker emitted byte-identical windows for a push spanning several ticks, so
  the "autocorrelation" was float non-determinism (score std 1.8e-15). It
  survived because the one chunker test counted windows and checked timestamps
  and never looked at the audio inside them. **Test what the data IS, not just
  how much of it arrives.**
- **A checkpoint that loads cleanly is not evidence the architecture is right.**
  Only reproducing the measured metric is.
- **Branch B cannot be "the main defence"** the v3 blueprint claimed. Retracted
  on measurement, then tested on real RTC transmission and failed there too
  (42–53% EER). Now carried at **zero weight**.
- **The trivial baseline is a property of the corpus, not a constant.** ASVspoof
  has a silence artifact worth 36.4% EER; RTCFake has none at all (leading
  silence is at chance on both languages). Quoting one corpus's baseline beside
  another corpus's EER compares a model against a stopwatch on different data.
  `scripts/audit_silence.py --dataset <name>` re-measures it per corpus.
- **A corpus can be single-class.** IndicSynth publishes spoof only. An EER
  cannot be computed from it, and pairing it with an unrelated real corpus
  would let a detector separate the two by *recording conditions* instead of by
  synthesis. `build_manifests.py` now declines to invent a split plan for a
  single-class corpus rather than producing one that looks fine.
- **An accumulator that scores below its own inputs is not an accumulator.**
  The channel detector latched on the first narrowband window: per-window 94.5%,
  per-call 88.4%, clean calls 53.9% (= 1 - 0.9^6). A majority vote gave 98.8%.
  Whenever a call-level rule underperforms its per-window rule, suspect a latch.
- **Fixing one error can expose another that was cancelling it.** Per-channel
  calibration fixed A2's mapping and made fusion look worse, because the
  hand-set weight 0.45 had been compensating for the bad mapping. Fitted, it is
  0.87. Two wrongs had been making a right.
- **`torch.set_num_threads(1)` in any serving process.** torch sizes its
  intra-op pool from the host core count (112 on this 16-core pod). Harmless at
  one request, catastrophic under concurrency: N forwards each try to claim the
  whole machine. This one line was worth ~20x on tail latency and took the
  server from 4 concurrent calls to 64.
- **Never trust a benchmark whose generator shares the box.** The load
  generator starved the server it was measuring; the same batch took 296 ms
  under load and 9 ms in isolation. Measure server-side and end-to-end
  separately and report both.
- **A control that was never exercised is not a control.** Every security check
  added this session was verified against the live server, and the call_id
  regex passed its unit test while accepting a trailing newline -- Python's `$`
  matches before one. `\A..\Z`.
- **Every fitted component carries the domain it was fitted on** — branch heads,
  Platt mappings, subset tables, fusion weights. Retraining fixes the heads;
  refitting fixes the table *for the language it was refitted on*. There is no
  layer at which this stops being true, so stop expecting one.
- **`python -m vericall.serving.app` exits 0 without starting a server** —
  app.py has no `__main__` block. `scripts/serve.sh` ran exactly that for
  weeks. Use `python -m uvicorn vericall.serving.app:app`. The load tests were
  never affected because they invoke uvicorn directly; only the one-command
  path a person would type on stage was broken.
- **fp16, not bf16, for THIS model — measured against the standard advice.**
  Industry guidance says bf16 for production inference because it keeps fp32's
  exponent range. That reasoning is about large models with large activations.
  Probed with forward hooks on real audio, the peak activation anywhere in
  either branch is **463.5** against fp16's 65504 ceiling — 141x headroom — so
  overflow is not a live risk and fp16's extra mantissa bits (10 vs 7) make it
  markedly more faithful: max |delta P(spoof)| 2.2e-4 vs bf16's 2.1e-3. 2.00x
  throughput at batch 32, every end-to-end decision unchanged. **Measure the
  activation range before accepting either default.**
- **All three published fixes for cross-corpus collapse FAILED** (section 20).
  Balanced sampling made it slightly worse; the domain adversary worked
  mechanically (domain loss 0.689 against ln(2)=0.693 — corpus genuinely
  unpredictable) and still left WaveFake at 46.81%; OC-Softmax did not separate.
  The lesson is narrow and reusable: **making a representation invariant to the
  corpora you HAVE does not make it invariant to one you have never seen.** The
  adversary can only remove what it can name, and it saw two domains.
- **Replay evades detection 1.2–1.5x** (section 21), best number 33.68% — not
  usable. Expected: a replayed recording of a real person is not synthetic
  speech, so branches trained on vocoder artifacts have no mechanism. Needs a
  different mechanism (recapture detection), not better training.
- **The ASVspoof model INVERTS on modern TTS** — 57.74% on EchoFake's
  CosyVoice2/F5-TTS/XTTS-v2 attacks. That is the section 13 inversion on a
  THIRD axis: not channel, not language, but **generator generation**. The
  2.48% in-domain number describes a 2019 threat landscape.
- **VRAM was never the constraint.** All four branches are 0.44 GiB of weights
  and batch 32 peaks at 0.96 GiB — about 4x headroom on a 6 GB card. On a
  laptop the binding constraint is COMPUTE, so the knob is concurrency
  (`--max-calls`), not batch size, and accuracy is identical.
- **Softmax saturation breaks quantile thresholds.** Every FPR target came back
  at exactly 1.0000 with the realised FPR off by 3x. Score in log-odds: same
  ranking, same EER, usable dynamic range. Same tie problem as the EER fix,
  wearing a different hat.

---

## 6. What is left

Items 1–4 of the original list are **done**. What replaced them, Sep 14–15:

- **Whitepaper + annotated bibliography — DONE Sep 14.**
  `docs/VERICALL_V4.md` and `docs/RESEARCH.md`. Every source verified against
  origin; the v3 blueprint's fabricated citation ("Pindrop's OCCAS pattern")
  is replaced with Pindrop Pulse and the substitution is *recorded* in the
  bibliography rather than quietly corrected.
- **RTCFake retraining pipeline — DONE Sep 15.** Manifest builder for a
  plain-WAV corpus, `plan_rtcfake` (offline/online pairs share a fold; zh
  dropped from training so Mandarin stays a real cross-lingual test),
  `--channels` on the extractor, `ConcatFold` for mixed-corpus training,
  `eval_folds.py` so every checkpoint meets every fold through one code path.
- **Branch A retrained on RTCFake-en.** The section 13 inversion is gone.
- **Hindi measured.** IndicSynth is spoof-only, so no EER exists for it; what
  is measurable is recall at a threshold set on bona fide we have.

Still open:

1. **Multilingual is the biggest remaining gap.** At a 5% false-positive
   budget the ASVspoof head catches 5.3% of Hindi attacks (chance) and the
   RTCFake head 13.9%. Neither is usable. An Indian-language *bona fide*
   corpus paired to IndicSynth's own source recordings is what would unblock
   a real EER — borrowing bona fide from an unrelated corpus would measure the
   pairing, not the model, so it is refused rather than caveated.
1b. **Cross-corpus generalisation is unsolved, only relocated.** Retraining on
   RTCFake killed the Mandarin inversion (79.08% → 9.51%) but the retrained
   model is at chance on ASVspoof and WaveFake, and mixing the two corpora
   destroys WaveFake outright even at 64% ASVspoof rows. Next things to try,
   none tested: balanced sampling, domain-adversarial training, per-domain
   heads. **Do not simply add more corpora and expect it to compose.**
1c. **A2 retrained and fusion refitted — DONE Sep 15.** Fusion now helps 1.28x
   on eval_rtc_en (16.07% vs 20.55% best single) and ties on eval_rtc_zh.
   Retraining A2 barely moved its RTCFake numbers, confirming it is the
   corpus-robust branch; the problem was downstream of it. **Calibration
   artifacts live in `calib_rtc/` — never point one head at another head's
   Platt file, that is confident nonsense rather than a degraded number.**
1d. **Per-deployment profiles — BUILT Sep 15.** `VERICALL_CKPT_A/_A2/_B` and
   `VERICALL_CALIB_DIR` select heads and calibration; both profiles verified
   end to end. `fit_subsets.py` writes `provenance.json` and the gateway
   reports `calibration_pairing` in `/health` -- verified by deliberately
   mis-pairing. What is still unbuilt is the *procedure*: fitting calibration
   on a sample of a tenant's own traffic.
1e. **Demo runbook — `docs/DEMO_SCRIPT.md`, Sep 15.** Every command in it was
   executed. Pre-flight gate is `scripts/verify_demo.py`; if it does not print
   DEMO PATH READY, do not demo. The demo needs **no scratch** — the clips are
   in `web/demo/` and tracked in git, so pod wipes cannot touch it.
2. **η — done Sep 15.** All three measured (0.207 / 0.928 / 0.565) and adopted
   in FusionConfig; the end-to-end result holds under them.
3. **Replay — DONE Sep 16** (section 21). EchoFake integrated, replay evades
   by 1.2–1.5x, best 33.68%. Needs a different mechanism, not more training.
4. **Runtime codec detection — DONE Sep 15.** 98.8% per call; all three
   buckets fitted for A and A2; `/health` reports the live bucket.
5. **Fusion weights — DONE Sep 15.** Fitted (A2 0.45 → 0.87) and stored in the
   calibration directory, not FusionConfig, because they are only valid against
   the calibration they were fitted for.

### Built Sep 16–17 (the product layer)

- **EchoFake + CodecFake integrated.** Manifests, split planners, all guards
  clean. EchoFake's replay/synthesis folds share their negatives so the gap is
  a measurement.
- **Cross-corpus experiment machinery**: `vericall/train/dann.py`
  (gradient reversal), `ocsoftmax.py`, `rawboost.py`, balanced sampling,
  `scripts/train_crossdomain.py`. All tested; the experiment's answer was
  negative (section 20) and that is recorded.
- **Serving: fp16 (2x), all four branches now actually loaded** (Branch C was
  fitted and documented but never served — identity_conflict could never fire),
  non-finite/negative-variance/extreme-LLR guards, per-client fair share,
  drain, audit rotation, Prometheus metrics, artifact integrity manifest.
- **Web/product**: live microphone capture (verified exactly against the clip
  path — identical risk, zero sample drift), SQLite call history with
  `/api/calls`, a history panel with tier timelines, and `run_local.py`.
- **Local deployment**: `docs/LOCAL_DEPLOYMENT.md`, all shell scripts made
  path-relative, VRAM measured for a 6 GB card.

### Closed Sep 18–22

1. **Cold export — verified, and it was not fine.** `scripts/verify_export.sh`
   clones the repo, boots it, streams all three clips and asserts the verdicts.
   Four defects, three of which were *invisible from inside the working tree*
   because the bug IS the difference between the tree and a clone:
   `.gitignore`'s blanket `*.npy` dropped both calibration cohorts; the five
   "vendored" ECAPA files were **symlinks into this machine's home**, which git
   stores as target text, so every clone got dangling links — and the preflight
   reported them "cached locally" because it tested the *directory*;
   `requirements.txt` pinned a torch nine releases from the one every number
   was measured on. Run the script, do not reason about the tree.
2. **SSL layer sweep — done, and it closes the question rather than opening
   one** (§22). Depth is monotonically harmful, layer 5 was already right, and
   the 2026 readout-layer result does not replicate here. No headroom in the
   layer index. Also: layer 3 *inverts* on Mandarin (56.63%, CI excludes 50%).
3. **Browser-to-browser calling — built.** `/call`, signalling on
   `/rtc/{room}`, each peer scores its own decoded copy of the remote track.
   The architectural fact to keep stating: WebRTC media is SRTP peer-to-peer,
   so **the server never sees the audio** and VeriCall scores what a
   *participant* hears. It is not a network-side interceptor.
4. **The Opus channel — measured and fitted** (§23). This is the one that
   mattered. Branch A survives WebRTC (1.18 → 1.33% EER); **Branch A2 is
   destroyed by it** (1.77 → 18.07%, monotone in bitrate) because A2 reads
   phase and Opus is a transform codec that does not preserve phase. At 12 kbps
   A2's *spoof* evidence goes negative — it argues the attacker is genuine.
   Fixed by refitting the mapping **and** the weight (0.87 → 0.42), because
   min_Cllr 0.607 vs 0.052 means the information is gone and no calibration
   recovers it. The channel is **declared, not detected**: Opus is wideband and
   loses no frames, so the detector calls it clean 97–98% of the time.
5. **Confidence was capped at 0.833 for every call ever scored.** `_confidence`
   divided by all of `SPOOF_BRANCHES` while `st.available()` deliberately
   excludes zero-weight branches, and w[B]=0. Fixed; also delayed every action
   by ~0.5 s via the `min_confidence` gate.

### Still open, in the order that matters

1. **Multilingual is still the biggest ML gap.** Hindi recall 13.9% at a 5% FP
   budget, and no Indian-language bona fide corpus exists to compute an EER at
   all. Not fixable by engineering — only data and evaluation close it.
2. **Replay needs a different mechanism** (§21). Channel/recapture detection,
   not synthesis-artifact detection. Nothing in the architecture does that.
3. **`identity_conflict` is dead code.** It needs `enrolled and identity_llr >
   0`; nothing sets `enrolled`, and C events are emitted with a hardcoded
   `llr=0.0`, so `0.0 > 0` is False regardless. `BranchC.llr_target()` has no
   call site. This is why `calib/C@asv.json` has no effect on serving either.
   An earlier comment claimed loading Branch C had fixed this; it had not.
4. **In-the-Wild downloaded, not integrated** (no manifest builder yet).
5. Accounts/login, TLS, and the per-tenant calibration *procedure*.
6. A TURN relay, if the demo ever has to cross a real NAT. STUN-only is fine
   for two tabs or one LAN and will fail behind symmetric/corporate NAT.

---

## 7. Orientation for a new session

```bash
cd "/home/dgx-s-bmu-cse-240577/SIH final"
git log --oneline | head            # ~90 commits as of Sep 22
python3 -m pytest tests/ -q         # expect 323 passed
bash scripts/verify_export.sh       # THE gate: clones, boots, streams, asserts
python3 run_local.py --check        # what is present vs degraded

# on the cluster (development):
./scripts/pod.sh name               # pod names change on every reschedule
./scripts/pod.sh exec 'ls /tmp/vericall-cache'   # empty = wiped again

# to demo:
./scripts/pod.sh exec 'setsid nohup scripts/serve.sh > logs/serve.log 2>&1 &'
./scripts/pod.sh py 'scripts/verify_demo.py'     # gate: must say DEMO PATH READY
```

**Read order for the numbers:** `docs/RESULTS.md` is the record, and sections
3 and 12 carry RETRACTION banners — do not quote them without reading the
banner. `docs/THREAT_MODEL.md` labels everything measured / argued / unknown.

**The honest one-line summary**, for when someone asks what this is: a
well-measured streaming detector that is excellent in-domain (2.48%), does not
transfer across corpora, with a production-hardened serving layer (64
concurrent calls), a verified cold export, and a real browser-to-browser call
demo. The gap is the model's behaviour on the population it would actually
meet, and no amount of engineering closes that — only data and evaluation do.

Git history starts at the incident recovery (Sep 5) — everything before that was
built but never committed, so the repo is a recovery log, not a project log.
`docs/` carries that earlier narrative in prose instead.
