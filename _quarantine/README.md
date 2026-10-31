# Quarantine

Files moved here on 2026-09-05, not deleted. Each duplicates a tested module
already in the tree, with fewer capabilities, and having two implementations of
the same thing invites the wrong one being imported.

| File | Duplicates | Why quarantined |
|---|---|---|
| `fusion_state.py` | `vericall/fusion/engine.py` | Naive LLR sum keyed on `time.time()`. A streaming detector needs an **audio** clock: wall-clock decay is non-deterministic, untestable by replay, and corrupted by network jitter. Also lacks calibration, per-subset scaling, quality gating, ESS deflation, regime tracking and tier policy. |
| `features_torch_ops.py` | `vericall/features/spectral.py` | Subset of the tested STFT/LFCC path. |
| `features_f0.py` | `vericall/features/prosody.py` | Subset; lacks the YIN local-minimum fix (a naive threshold crossing biases pitch ~3% high) and the incremental F0 cache (3x saving on the slow lane). |
| `train_branch_a2.py` | `scripts/train.py --branch A2` | Bypasses the dev-EER early stopping, prior-weighted loss, overfit-100 sanity check and duration-baseline gate. |
| `calibrate.py` | `scripts/fit_calibration.py` | Generated a **mock** calibrator; see docs/INCIDENT_2026-09-05.md. |

Nothing here is imported by the live tree. Delete once the incident report has
been read and the decisions confirmed.

## `audio_dataset_hf.py` — dead HuggingFace-streaming loader

Was `vericall/audio/dataset.py`. **Nothing imported it**, and it predates the
cached-feature pipeline (`extract_features.py` + `CachedFold`) that replaced it.

Quarantined rather than deleted because it is a record of an earlier plan, and
removed from the package because it was actively dangerous to leave importable:

* On any load failure it returned **`torch.zeros(...)` with the original
  label** — silence, labelled. No raise, no log, no counter. Wired in by
  mistake, it would teach a model that silence is bona fide or spoof depending
  on which file happened to be corrupt.
* It imports `datasets`, which is not in `requirements.txt`.
