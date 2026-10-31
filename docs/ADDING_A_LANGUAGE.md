# Adding a language

The construction is not Hindi-specific and nothing in the serving code knows a
language name. Adding one is six steps and **one entry in `profiles.json`** —
no code change. This file is the recipe; it was written by doing it six times.

---

## What a language needs before you start

Two corpora that share speakers:

- **Kathbath** (IndicSuperb) — real recordings, CC0, one 3.3 GB tar covering
  all twelve languages.
- **IndicSynth** — clones *of Kathbath speakers*, on HuggingFace.

The shared speakers are the whole point. Pairing bona fide and spoof from
unrelated corpora lets a detector separate them by recording conditions instead
of by synthesis, and it will report a spectacular EER that means nothing.

**Twelve languages are possible, and that is the ceiling**: Kannada, Malayalam,
Odia, Punjabi, Sanskrit, Urdu, Hindi, Bengali, Marathi, Tamil, Telugu,
Gujarati. The other ten of India's twenty-two scheduled languages have no
paired corpus in any dataset. That is a corpus gap, not a backlog item.

| built | state |
|---|---|
| hindi | trained, calibrated, **served** |
| bengali, marathi, tamil, telugu | trained (pooled), corpora since wiped |
| gujarati | **frozen holdout** — do not train on it |
| kannada, malayalam, odia, punjabi, sanskrit, urdu | downloaded, nothing built |

---

## The six steps

Run them in the GPU pod. `$L` is the language, `$C` its two-letter code.

```bash
L=kannada; C=kn; CACHE=/scratch/vericall-cache-$C
DATA=/scratch/vericall-data

# 1. fetch -- the shard stride and size are already in scripts/fetch_datasets.py
python scripts/fetch_datasets.py --datasets kathbath_valid indicsynth_$C \
    --dest $DATA --state data/fetch_state_$C.json

# 2. the SPOOF-side manifest, which step 3 takes as input
python scripts/build_manifests.py --data-root $DATA --datasets indicsynth_$C

# 3. pair: reunite each IndicSynth clone with its Kathbath source speaker.
#    Equalises silence (removes the duration shortcut AT SOURCE, not by
#    caveat) and AAC round-trips the bona fide so both classes share a codec.
python scripts/build_indic_paired.py --language $L \
    --kathbath $DATA/kathbath_valid \
    --indicsynth-manifest data/manifests/indicsynth_$C.jsonl \
    --out $DATA/paired_$L \
    --report docs/paired_${L}_build.json

# 4. the paired manifest and the speaker-disjoint, generator-stratified split
python scripts/build_manifests.py --data-root $DATA \
    --datasets ${L}_paired ${L}_paired_unmatched

# 5. features, three channels cycled across four crops
python scripts/extract_features.py --dataset ${L}_paired \
    --folds train dev calib eval_$L --crops 4 \
    --channels clean g711_mulaw noisy_line --workers 8 --out $CACHE

# 6. train, then fit the calibration FOR those heads
python scripts/train_crossdomain.py --cache $CACHE --datasets ${L}_paired \
    --train-fold train --dev-fold dev --eval-on ${L}_paired:eval_$L \
    --epochs 18 --seeds 0 1 2 --balanced --tag ${C}ch
python scripts/train.py --branch A2 --cache $CACHE --dataset ${L}_paired \
    --eval-on ${L}_paired:eval_$L --epochs 18 --seeds 0 1 2 --tag ${C}ch

for B in A A2; do
  python scripts/fit_channels.py --branch $B \
      --ckpt checkpoints/branch_${B}${C}ch_seed0.pt \
      --cache $CACHE --dataset ${L}_paired --fold calib --out calib_$C
done
python scripts/fit_subsets.py --cache $CACHE --dataset ${L}_paired --fold calib \
    --a-ckpt checkpoints/branch_A${C}ch_seed0.pt \
    --a2-ckpt checkpoints/branch_A2${C}ch_seed0.pt \
    --calib-dir calib_$C --out calib_$C/subsets.json --per-channel
```

Budget roughly an hour per language on a `mig-2g.35gb` slice, most of it in
steps 1 and 3.

Step 2 is easy to miss and the reason this file exists: `BUILDERS` carried
only `indicsynth_hi` until October, while all twelve `<lang>_paired` entries
were already registered. The gap was invisible until a new language was
actually attempted, because the half that was complete was the half you read
first.

---

## Then register it

Add to `profiles.json`:

```json
"kannada": {
  "display": "Kannada over telephony",
  "languages": ["kannada"],
  "calib_dir": "calib_kn",
  "ckpt_a": "branch_Aknch_seed0.pt",
  "ckpt_a2": "branch_A2knch_seed0.pt",
  "status": "ready",
  "measured": {"clean_eer": 0.0, "g711_eer": 0.0, "noisy_line_eer": 0.0},
  "note": "..."
}
```

```bash
python run_local.py --list-profiles
python run_local.py --profile kannada
```

**`status` stays `experimental` until `calib_dir` exists.** An uncalibrated head
still ranks, so it looks fine, but every number downstream — the accumulated
LLR, the risk score, the escalation threshold — is in nats and assumes
calibration. The gateway refuses to serve such a profile rather than warn,
because a warning next to a plausible number gets ignored.

---

## Three things that will bite you

**Check the silence cue.** `equalise_silence` runs in step 2, but verify it per
language rather than trusting it:

```bash
python scripts/audit_silence.py --dataset ${L}_paired     # want ~50%
```

A figure well above 50% means the detector can win on duration alone. On
ASVspoof that shortcut is worth 36.4% EER by itself.

**Never pool a new language with Gujarati.** It is the project's only untouched
holdout. In-the-Wild has been scored by six checkpoints and CodecFake by five,
so neither is virgin any more. Gujarati is the one honest out-of-sample number
left, and training on it once destroys that permanently.

**Pooling is usually the better model, but it needs its own calibration.** One
head over five languages cost Hindi 0.76 points (8.30% vs 7.54%) and bought
four languages plus 6.85% on an unseen one. `branch_Aindic5_seed0.pt` is that
head. It is in `profiles.json` as `indic5`, marked experimental, because no
calibration was ever fitted for it and it has no A2 counterpart — the honest
state, recorded rather than quietly skipped.
