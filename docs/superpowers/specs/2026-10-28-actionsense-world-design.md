# ActionSense real-data world — design

**Date:** 2026-10-28 · **Status:** proposed, awaiting PI approval · **Refs:** #45, #35, #38

## 1. Why

The paper's real-data evidence rests on the Studyforrest phase-2 bridge
(ADR-0012), and that bridge has two problems. Its tensors are lost and
the public chain cannot rebuild them (#45). Even when it existed, only
one modality was a real recording of the same scene: force, from
cardiac and respiration physio. Audio was a LibriSpeech substitute.
Vision came from a 12 s flickering field-of-view localizer clip. Tactile
was derived from scene cuts, and gravity was zeros.

A cross-modal plasticity benchmark needs a world where all five
modalities are measured together, on the same people, during the same
actions. ActionSense (DelPreto et al., NeurIPS 2022 Datasets and
Benchmarks; MIT CSAIL) records exactly that in a kitchen: wearable
tactile gloves, forearm EMG and IMU, an egocentric eye-tracking camera,
and microphones, all time-stamped on a common clock, with
human-labelled activities.

## 2. Goals

1. A new world `ActionSenseWorld` that serves the same six aligned
   tensors as `StudyforrestRealWorld`, at 10 Hz, with the unchanged
   `WorldSimulator` contract (`audio (N,128)`, `vision (N,16,16)`,
   `tactile (N,32)`, `gravity (N,3)`, `force (N,6)`, `labels (N,)`
   with 4 classes). The encoders, nerve, head and CLI stay unchanged.
2. Every modality comes from a real sensor recorded during the same
   activity. No substitute signal, and no zero channel.
3. A byte-reproducible extraction: pinned raw-file URLs and sha256,
   a deterministic script, and the published tensors with their
   manifest.
4. The paper's real-data experiment replayed on this world under a
   new, git-dated local pre-registration amendment (v0.7), filed
   before any grid runs.

## 3. Non-goals

- Rescuing or re-deriving the Studyforrest numbers. Those 17 blocks
  stay historical, with their erratum (#45).
- Changing the locked thresholds (B-1 ≥ 0.05, B-2 ≥ 0.10, B-3 ≥ 0.02)
  or the Me1/Me3/Me6/Me7 definitions.
- Changing the encoder architecture or the modality dimensions to fit
  the richer sensors. Extraction reduces each sensor to the existing
  contract.
- Activity recognition as a goal in its own right. Labels only feed
  the integration head.

## 4. Anti-goals (failure even if it runs)

- Any attempt to identify subjects, or any derived artefact that could
  help do so. Vision is reduced to a 16×16 intensity grid, and audio
  to a 128-bin log-mel envelope: no raw frames and no waveform leave
  the extraction.
- Any commercial use. The data licence is CC BY-NC-SA 4.0. Derived
  tensors carry that licence, separate from the Apache-2.0 code, and
  L'Électron Rare's commercial work must not use them.
- Quoting a B-3 verdict before the paired-null calibration
  (ADR-0021/0022) has been run on this world.
- Running grids before the v0.7 amendment is committed.

## 5. Source data (facts checked 2026-10-28)

| Item | Value |
|---|---|
| Licence | CC BY-NC-SA 4.0 (project page); parsing code MIT |
| Host | `https://data.csail.mit.edu/ActionNet/`, per-subject links, no registration |
| Wearables | one HDF5 per recording: `wearable_data/<date>_experiment_S##/…/<datetime>_streamLog_actionNet-wearables_S##.hdf5` |
| Egocentric video | same folder, `…_eye-tracking-video-world_frame.mp4` (no gaze overlay) |
| Audio | `external_data/…/…_microphone_S##_{overhead,sink}_chunked_data.wav` + timing HDF5 |
| Streams used | `tactile-glove-{left,right}/tactile_data`, `myo-{left,right}/emg`, `myo-{left,right}/acceleration_g`, `experiment-activities/activities`, eye-tracking world video, sink microphone |
| Subjects with all five modalities | **S00, S02, S03, S04, S05** (S01: wearables lost; S06–S09: microphones and RGB marked "coming soon") |

File sizes are not published. The pilot measures them with
`HEAD` requests before any download, and macM1 has about 70 GB free.

## 6. Modality mapping (decided)

Every stream is resampled onto a common 10 Hz grid on the wearable
clock (`time_s`, seconds since epoch). A frame at time *t* aggregates
the window [*t*, *t* + 100 ms).

| bouba_sens | Source | Reduction to the contract |
|---|---|---|
| audio (128) | sink microphone, 48 kHz, cardioid (closest to hand–object sounds) | 128-bin log-mel, window 100 ms, hop 4800 samples, aligned through a least-squares line of the microphone HDF5 chunk timestamps against sample index (changed 2026-10-29, see §7) |
| vision (16,16) | eye-tracking world camera (egocentric), frame nearest to each 10 Hz tick | grayscale, then 16×16 mean-intensity patch grid. Same reduction as `extract_ccfilm_features.py`, with no pretrained network and no weight download |
| tactile (32) | both gloves, 32×32 pressure matrices | per glove, 4×4 average pooling of the matrix to 16 values; left and right concatenated to 32 |
| gravity (3) | right Myo `acceleration_g` (3 axes, in g) | mean over the window, then a 0.5 Hz low-pass so the vector tracks forearm orientation relative to gravity |
| force (6) | Myo EMG, 8 channels per arm | 100 ms RMS envelope, then per arm the mean of 3 electrode sectors (channels 0–2, 3–5, 6–7), giving 2 × 3 = 6 |
| labels (4) | `experiment-activities/activities` | 4 activity families, below. Frames outside a labelled segment are dropped |

**Label families (decided):** (0) cutting: peeling, slicing;
(1) spreading and jar handling: spreading, opening and closing a jar;
(2) cleaning: wiping pans or plates, cleaning tableware; (3) pouring and
tableware: pouring water, setting the table, loading the dishwasher. The
exact mapping from the 20 raw label strings is a table in the
extraction script, published in the manifest. Class frequencies and the
majority baseline are reported with the grids.

**Normalisation:** each modality is z-scored per subject, with
statistics computed on that subject's own labelled frames and stored in
the manifest. Rows from the five subjects are concatenated in subject
order, and each row records its subject id in a seventh tensor,
`subject.pt`. That tensor is ignored by the world contract and used
only for analyses.

## 7. Alignment checks (pilot gate)

The pilot runs on S00 only and must pass every check before the full
extraction:

1. All six tensors have the same length. The extraction script aborts
   otherwise, using the check added for #45.
2. Clock agreement: the labelled span falls inside the video and audio
   spans, and all timestamp series are monotonic; offsets between media
   and wearable starts/ends are reported.
3. Physical sanity: tactile pressure onsets and audio energy onsets
   cross-correlate with a peak within ±200 ms on at least one
   contact-heavy activity (slicing). The figure is saved.
4. Determinism: two extractions give identical sha256 for all tensors.
5. One smoke cell (`train` 20 steps, `lesion` T2 20 steps, `eval`) runs
   on `ActionSenseWorld`, and two runs give identical sha256 (#35).

*Changed 2026-10-29 (PI decision, after the S00 pilot,
`docs/pilot/actionsense-s00.md`):* check 2 previously required the
start and end of the video and audio to fall inside the wearable
recording. On S00 the video starts 9.0 s and the microphone 36.5 s
before the first wearable sample (and the video ends 2.9 s after the
last one), margins that only concern unlabelled time. The check now
tests what the extraction relies on, the labelled span covered by the
media, and reports the offsets instead of failing on them.

*Changed 2026-10-29 (PI decision, same pilot):* the S00 microphone
chunk timestamps arrive in bursts (steps of ~0, 43 and 85 ms instead of
a steady 2048 / 48000 = 42.67 ms) while the WAV is continuous. Mapping
each frame through a single chunk timestamp, with a 20 ms per-span drift
rejection, dropped 42.3 % of the S00 frames and could place a window
40–85 ms off its tick. Audio samples are now placed on a per-file linear
fit of chunk timestamp against sample index; a file is rejected only if
the implied rate deviates from the WAV rate by more than 2e-3 or a
residual exceeds 0.5 s (`MIC_RATE_TOL`, `MIC_RESIDUAL_MAX_S` in
`scripts/extract_actionsense.py`; fit statistics in `MANIFEST.md`).

## 8. Experiment (to be pre-registered, amendment v0.7)

- Same grid as ADR-0012: 5 seeds × 5 lesioned modalities × T1/T2 ×
  3 SNR floors = 150 cells, run twice: no lock and `LOCK_AFTER=200`.
  These are the direct analogues of `v05_sf45_{nolock,lock}`.
- Same metrics and thresholds. B-3 is reported only together with the
  paired sign-flip null of ADR-0022 run on this world.
- The amendment states the world, the subjects, the modality mapping,
  the label families and the grid, and that Studyforrest numbers are
  not compared as a replication target, only described.
- The Amedi dose-response and the Sprint 11–14 mechanism blocks are
  **not** part of v0.7. They are a later amendment, decided after the
  v0.7 verdicts.

## 9. Engineering

- `scripts/fetch_actionsense.py`: downloads only the needed files for
  the five subjects, records URL, size and sha256 in
  `data/actionsense_raw/MANIFEST.json`, and stops when free disk would
  fall below 20 GB.
- `scripts/extract_actionsense.py`: implements §6 and §7, writing
  `data/actionsense/{audio,vision,tactile,gravity,force,labels,subject}.pt`
  and `MANIFEST.md`.
- `src/bouba_sens/world/actionsense.py`: `ActionSenseWorld`, a copy of
  the `StudyforrestRealWorld` contract with `mode = "actionsense"`. It
  is registered in `cli._build_world` as `--world actionsense` and
  reads `BOUBA_SENS_ACTIONSENSE_DATA`.
- Tests: world contract and shapes on a tiny synthetic fixture, the
  alignment check, label-family mapping, and z-score determinism. No
  network and no large data in CI.
- **New dependency:** `h5py`, needed to read the wearable HDF5. Per the
  supply-chain policy it goes through human diff review and is pinned
  by hash before merge. Video decoding uses the system `ffmpeg`
  already on macM1, through a subprocess, so no new Python video
  library is needed.
- Compute: macM1 CPU, the same as the current regeneration. Two
  150-cell grids are expected to take a few hours, judging by the
  synthetic blocks (about 18 min per 150 cells on the Gaussian world).
- Publication: tensors, manifest and licence notice as a GitHub release
  asset under CC BY-NC-SA 4.0, with attribution to DelPreto et al.
  2022.

## 10. Milestones

1. **M0, approval:** the PI approves this document.
2. **M1, pilot on S00:** fetch, extract, the five checks of §7, one
   smoke cell. Ends at a go/no-go gate.
3. **M2, amendment v0.7:** committed, and tagged before any grid runs.
4. **M3, full extraction:** five subjects, tensors published.
5. **M4, grids:** two 150-cell grids plus the paired null. The ADR
   records the verdicts and provenance, as ADR-0012 did.
6. **M5, paper:** a new section on the ActionSense world, with the
   Studyforrest section kept as historical behind its erratum.

## 11. Calls made in this design (check these)

These were chosen by the design author, not by the PI:

- **Subjects.** S00 and S02–S05 only, the five with all five
  modalities.
- **Sensors.**
  - Egocentric eye-tracking video rather than the five fixed cameras.
  - Sink microphone rather than the overhead one.
  - Right-arm accelerometer for gravity.
  - EMG sectors 0–2, 3–5, 6–7 for force.
- **Reductions.** 4×4 pooling per glove for tactile, and a 16×16
  intensity grid for vision, not VGG features.
- **Labels and normalisation.** The four activity families, z-score
  per subject, and unlabelled frames dropped.
- **Scope of v0.7.** Only the no-lock and lock-200 grids. Dose-response
  and the mechanism blocks come later.

## 12. Left to the builder

- The exact 20-label to 4-family table, written from the label strings
  found in the HDF5 and listed in the manifest.
- The low-pass filter design for gravity and the mel parameters beyond
  those fixed above, documented in the manifest.
- How to handle short gaps or dropped samples inside a labelled segment
  (interpolate gaps ≤ 200 ms, otherwise drop the frame), with counts
  reported.
