# Running VeriCall on your own machine

Written for an RTX 4050 (6 GB) laptop, and correct for any CUDA GPU with 2 GB
or more. Everything here was verified on this project's artifacts rather than
assumed.

---

## It fits, with room to spare

The headline worry with a 6 GB card is memory. Measured
(`scripts/vram_budget.py`, results in `docs/vram_budget.json`):

| | VRAM |
|---|---:|
| All four branch weights resident | **0.44 GiB** |
| Peak at batch 16 | 0.72 GiB |
| Peak at batch 32 | **0.96 GiB** |
| Peak at batch 48 | 1.21 GiB |

On a 6 GB card with ~5 GB usable after the desktop, that is roughly **4×
headroom at the default batch size**. VRAM is not the constraint.

**Compute is.** A 4050 laptop GPU is a fraction of the H200 slice this was
developed on, so the number that will be lower locally is *concurrent calls*,
not batch size or accuracy. Accuracy is identical — same weights, same
calibration, same fp16 path.

Expect a **single live call to work comfortably in real time**. For a concert of
concurrent calls, measure rather than guess:

```bash
python scripts/capacity_probe.py --calls 1 4 8 16
```

That reports the largest N that holds real time on *your* hardware, using the
same criterion as the cluster measurements.

---

## Setup

```bash
# 1. a virtual environment
python -m venv .venv
.venv/bin/activate            # Windows: .venv\Scripts\activate

# 2. torch WITH CUDA for your driver -- install this BEFORE requirements.txt,
#    because the plain `pip install torch` on PyPI is often the CPU build and
#    a CPU-only install is the single most common reason this "runs but cannot
#    keep up".  Pick the index URL matching your CUDA version from
#    https://pytorch.org/get-started/locally/
pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# 3. everything else
pip install -r requirements.txt

# 4. check before you run
python run_local.py --check
```

`--check` verifies the interpreter, every required package, CUDA and its VRAM,
the branch checkpoints, the calibration files, the fitted fusion weights, the
demo clips, and whether the pretrained encoders are cached. **It reports what is
degraded rather than failing silently**, because each of these otherwise
surfaces later as something harder to read: a missing checkpoint becomes an
untrained head scoring at chance, a missing calibration becomes confident
numbers on the wrong scale, and a CPU-only torch becomes a demo that mysteriously
lags.

### First run needs the internet, once

WavLM (~360 MB) and ECAPA (~80 MB) are **downloaded on first use, not vendored**
— they are pretrained encoders, not our weights. After the first run they are
cached (`~/.cache/torch` and `./.cache/ecapa`) and the system works offline.

If you will demo somewhere without network, run it once at home first. Without
ECAPA, Branch C is disabled and the other three branches still work — `/health`
says so in `branches_loaded`.

---

## Running

```bash
python run_local.py
```

Then open **http://127.0.0.1:8001** and either press **🎤 Live microphone** or
click one of the three bundled clips.

Binds `127.0.0.1` by default on purpose: a detector with no auth token on
`0.0.0.0` is reachable by anything on the network, and the laptop default should
not be the insecure one. To expose it deliberately:

```bash
VERICALL_AUTH_TOKEN=pick-a-secret python run_local.py --host 0.0.0.0
```

and open the page as `?token=pick-a-secret`.

### Serving to another device — you need HTTPS

Browsers expose `getUserMedia` only on a **secure origin**, and `localhost` is
the single http exception. So the live microphone and all of `/call` work over
plain http on this machine and **silently do nothing from any other** — the page
loads, the button does nothing, and no error worth reading appears. Two tabs on
one laptop are fine. A laptop and a phone are not.

```bash
python run_local.py --tls --host 0.0.0.0        # generates .tls/ on first use
```

That writes a self-signed certificate with SANs for `localhost`, `127.0.0.1`
and this machine's LAN address, so the other device can reach it by IP. Every
browser will warn that the certificate is not trusted — **that is expected and
correct**: a self-signed certificate encrypts the link and authenticates
nothing. Click through on a network you control; do not do this on one you do
not. A real deployment terminates TLS at a reverse proxy with a certificate
someone actually trusts, and `--cert`/`--key` take one if you have it.

`.tls/` is gitignored. A private key in git history is a key you cannot
un-publish.

Bind `0.0.0.0` only with `VERICALL_AUTH_TOKEN` set, and open the page as
`?token=…`.

### Two-party calling — /call

`/` is the single-stream detector: one clip, one file, or your own microphone.
`/call` places a **real browser-to-browser call** and scores the person on the
other end.

Open `/call` in two tabs (or on two machines on the same network), type the
same room name in both, press **Join**. **Use headphones** — the remote stream
starts muted precisely because browser echo cancellation is left switched off,
and two unmuted tabs on one laptop will howl.

Three things worth knowing before you show it:

* **The server never sees this audio.** WebRTC carries SRTP directly between
  the two browsers, so the gateway relays only the offer/answer handshake on
  `/rtc/{room}` and could not decrypt the media if it wanted to. Each browser
  taps its own decoded copy of the remote track and streams *that* to `/ws/`.
  VeriCall scores what a participant hears; it is not a network-side
  interceptor, and on this transport it cannot be one.

* **NAT.** The default ICE config is a public STUN server and no TURN. Two
  tabs on one machine need neither. Two machines on one network are fine. Two
  peers behind symmetric or corporate NAT **will fail to connect**, and that
  needs a TURN relay, which cannot be run free at useful bandwidth because it
  carries the media. Point at one with
  `VERICALL_ICE_SERVERS='[{"urls":["turn:host:3478"],"username":"u","credential":"p"}]'`.

* **The channel is declared, not detected.** WebRTC encodes with Opus. The
  runtime channel detector cannot see that — Opus is wideband and loses no
  frames, so it reads as clean audio 97-98% of the time — so the page sends
  `channel=opus` and the server applies the opus bucket. Branch A is barely
  affected by Opus; **Branch A2 loses nine tenths of its information**, because
  it reads phase and Opus does not preserve phase, so its fusion weight is cut
  from 0.87 to 0.42 on this channel. See `docs/RESULTS.md` section 23. The
  verdicts are calibrated; they rest mostly on one branch.

### The knob that matters locally

```bash
python run_local.py --max-calls 8      # default is 64
```

64 is right for the cluster. On a laptop, set it to whatever
`capacity_probe.py` says holds real time, so the service **refuses** the call it
cannot serve rather than accepting it and falling behind — a late fraud
decision is worse than a refused connection.

---

### Which calibration profile to run

`--profile` picks one. A profile is a trained head set *plus the calibration
fitted for it*, selected together, because a Platt mapping describes one model
and mixing them produces confident numbers that mean nothing.

```bash
python run_local.py --list-profiles      # what is available
python run_local.py --profile hindi-telephony
```

The profiles are `baseline` (English), `rtc` (English + Mandarin),
`hindi-clean`, `hindi-telephony` (**the default**), and `indic5`, which is
marked experimental and refused — see the end of this section.

For Hindi there are **two** that look interchangeable and are not:

| profile | clean | G.711 | noisy line |
|---|---:|---:|---:|
| `calib_hindi` | **7.88%** | *no bucket* | *no bucket* |
| `calib_hindi_ch` | 9.24% | **13.93%** | **23.21%** |

`calib_hindi` wins on clean audio by 1.36 points. It also has **no calibration
at all** for a telephone channel: a G.711 call scored under it falls back to a
mapping fitted on wideband studio audio, so the score is not merely worse, it
is wrong in a way the number does not show. `calib_hindi_ch` pays those 1.36
points to be calibrated when the call is actually a call.

**Run `calib_hindi_ch` unless you know the audio is clean wideband.** This is a
fraud detector for phone calls, so telephony is the operating condition and
clean is the special case. `scripts/verify_export.sh` already gates on `_ch`;
this paragraph is what was missing, not the choice.

Read the third column honestly: **23.21% on a noisy line** is the real ceiling
in bad conditions, and it is poor. The profile does not fix a hard channel, it
stops the system from being confidently wrong about one.

```bash
python run_local.py --profile hindi-telephony      # or just: python run_local.py
```

### The five-language model, and why it will not start

`indic5` is a single Branch A head trained on Hindi, Bengali, Marathi, Tamil
and Telugu. It is real and it is measured — including **6.85% on Gujarati**, a
language it has never seen, which is the best evidence in this project that
pooling generalises.

It is still refused:

```
profile error: profile 'indic5' is not servable: status=experimental,
calibration=none.
```

No calibration was ever fitted for it and it has no A2 counterpart, so it has
no likelihood-ratio scale. It would still *rank* clips correctly, which is
exactly why this refuses instead of warning — every number downstream (the
accumulated LLR, the risk score, the escalation threshold) is in nats and
assumes calibration, so it would look like it was working.

Fitting that calibration needs the five languages' paired corpora, which a
namespace wipe took. `docs/ADDING_A_LANGUAGE.md` is the recipe.

### Adding a language later

Six steps and one entry in `profiles.json` — no code change. The whole
procedure is in **`docs/ADDING_A_LANGUAGE.md`**, with every command checked
against the real CLIs. Budget about an hour per language on a cluster GPU.
Twelve Indian languages are buildable; six are done.

---

## What exports and what does not

| | Size | Notes |
|---|---:|---|
| Code, tests, docs | ~2 MB | everything needed to run |
| `checkpoints/` | 47 MB | the trained branch heads — **required** |
| `calib/` | 1.2 MB | Platt mappings, subset table, fitted weights — **required** |
| `web/demo/*.wav` | 2.7 MB | the three demo clips |
| Corpora | ~100 GB | **do not export.** Re-fetchable with `scripts/fetch_datasets.py`, and only needed for *training*, never for running |

So the working export is roughly **50 MB**. The git repository already contains
all of it.

Retraining locally is possible but not the point: a 4050 would take
substantially longer than the cluster, and the trained heads are already in the
repo.

---

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `VERICALL_PORT` / `VERICALL_HOST` | 8001 / 127.0.0.1 | where to listen |
| `VERICALL_AUTH_TOKEN` | unset | required token; unset accepts everything |
| `VERICALL_MAX_CALLS` | 64 | concurrency cap |
| `VERICALL_MAX_CALLS_PER_CLIENT` | ¼ of the cap | fair share |
| `VERICALL_PRECISION` | fp16 | `fp32` / `fp16` / `bf16` |
| `VERICALL_BATCHING` | 1 | cross-call batching |
| `VERICALL_CALIB_DIR` | `calib` | which calibration profile to load — see *Which calibration profile to run* |
| `VERICALL_CHANNEL` | clean | calibration bucket before detection is confident |
| `VERICALL_CODEC_DETECT` | 1 | runtime channel detection (a call that *declares* `?channel=` pins its bucket and stands the detector down) |
| `VERICALL_ICE_SERVERS` | one public STUN | JSON array of RTCIceServer entries for `/call`; set this to add a TURN relay |
| `VERICALL_DB` | logs/vericall.db | call history |
| `VERICALL_REQUIRE_SEALED` | 0 | refuse to serve unverified artifacts |
| `VERICALL_TORCH_THREADS` | 1 | **leave at 1.** torch otherwise sizes its pool from the core count and concurrent forwards fight for the machine — this one line was worth ~20× on tail latency |

---

## If something is wrong

| Symptom | Cause | Fix |
|---|---|---|
| `--check` says cuda not available | CPU-only torch | reinstall torch from the CUDA index URL |
| Microphone button does nothing | browsers block `getUserMedia` off `https://` | `127.0.0.1` is exempt — use it, not your LAN IP |
| Page shows `unauthorized` | `VERICALL_AUTH_TOKEN` is set | open as `?token=…` |
| `branches_loaded` missing `C` | ECAPA not downloaded | run once with internet; three branches still work |
| Risk numbers look meaningless | `calibrated: false` in `/health` | `calib/` is missing — **stop, do not demo** |
| Server falls behind real time | concurrency above what the GPU can serve | lower `--max-calls` |
| First run hangs at startup | downloading WavLM | wait; it is ~360 MB, once |
| Mic button does nothing from another device | http:// is not a secure origin | `--tls`; localhost is the only http exception |
| Browser warns the certificate is untrusted | it is self-signed | expected — it encrypts, it does not authenticate |
| `/call` says *peer connection failed* | STUN alone cannot cross both NATs | two tabs on one machine always work; otherwise set `VERICALL_ICE_SERVERS` to a TURN relay |
| `/call` howls / echoes | remote audio unmuted without headphones | leave it muted, or use headphones — browser echo cancellation is off on purpose |
| `/call` joins but no verdict appears | the far side is not speaking, or the tap is muted at source | check `audio scored` climbing in the Detector state panel |
| `/call` says *that room already has two people in it* | a stale tab still holds a seat | close it, or use a different room name; rooms free their seats on disconnect |
