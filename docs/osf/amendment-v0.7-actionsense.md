# OSF pre-registration amendment — bouba_sens v0.7 (ActionSense world)

> **Status note (2026-10-29).** Drafted as an amendment to the
> dream-of-kiki OSF registration. Like v0.4 to v0.6 (issue #20), it is
> **not filed on OSF**. It counts as a local, git-dated
> pre-registration only. Its date of record is the commit that tags it
> `bouba_sens/v0.7-actionsense-prereg`, after PI sign-off and before
> any ActionSense grid runs.

**Status:** Draft, PI decisions of 2026-10-30 recorded (§8); awaiting PI sign-off. Not filed on OSF (local, git-dated pre-registration)
**Parent pre-registration:** `dream-of-kiki` OSF DOI `10.17605/OSF.IO/Q6JYN`
**Previous amendment:** `amendment-v0.6-sprint10-17.md` (Sprint 10–17 hyperparameters, MINE estimator)
**Amendment tag to create after PI sign-off:** `bouba_sens/v0.7-actionsense-prereg`
**Amendment date:** 2026-10-29 (draft); PI decisions 2026-10-30
**Design and plan:** `docs/superpowers/specs/2026-10-28-actionsense-world-design.md`
(§6, §7 as amended 2026-10-29, §8),
`docs/superpowers/plans/2026-10-28-actionsense-world.md` (Task 6)
**Pilot reports:** `docs/pilot/actionsense-s00.md` (run 1),
`docs/pilot/actionsense-s00-run2.md` (run 2 and "Null test (2026-10-29)")
**Extraction code version:** `main` after the fix of issue #52 is
merged. The commit is identified at tag time and written into the tag
message and into ADR-0025.

> **Numbering (PI decision 2026-10-30).** Draft PR #9
> (`docs/osf-prereg-2026-10`, not merged) holds other drafts numbered
> v0.7 to v0.9 (TOST equivalence, subspace shift, decorrelation
> window). ActionSense keeps the number v0.7. The PR #9 draft
> `amendment-v0.7-tost-equivalence` is to be renumbered v0.8 (a comment
> to that effect is posted on PR #9; PR #9 itself is not modified).

> **Code prerequisite.** The decisions of 2026-10-30 below need code
> changes that are in PR #56 (`fix/v07-prereg-gates`): the `label_family`
> jar rule, the `>=` threshold in `aggregate_grid.py`, and `EMISSION` /
> `--probe-v2` in `run_grid.sh`. PR #56 must be merged before the tag,
> and the extraction commit recorded at tag time must contain it.

## 1. What is amended and why

The v0.5 amendment registered `StudyforrestRealWorld`, the 4.5-modal
Studyforrest phase-2 bridge, and the ADR-0012 grids
`v05_sf45_{nolock,lock}` ran on it. Those real tensors are lost:
`data/sf_phase2_adapted` cannot be rebuilt from the public chain, and
the public extraction is misaligned (vision 120 rows vs 1800). See
issue #45. Even when it existed, only force was a real recording of the
scene. Audio was a LibriSpeech substitute, vision came from a localizer
clip, tactile was a scene-cut proxy and gravity was zeros.

On 2026-10-28 the PI chose ActionSense to replace it as the real-data
world (spec §1, PR #46). In ActionSense all five modalities are
recorded together, on the same people, during the same kitchen
activities. This amendment registers:

1. the world `ActionSenseWorld` and its extraction, fixed in full below;
2. the replay of the paper's real-data experiment on it: two 150-cell
   grids (`nolock` and `LOCK_AFTER=200`) and the paired null of ADR-0022;
3. the analysis plan and the decision rules, stated before any grid runs.

It adds no metric, no threshold and no hyperparameter. The Me1, Me3,
Me6 and Me7 definitions and the B-1 / B-2 / B-3 thresholds are those
already on file.

## 2. World

### 2.1 Source

- **Dataset:** ActionSense (DelPreto et al., *ActionSense: A Multimodal
  Dataset and Recording Framework for Human Activities Using Wearable
  Sensors in a Kitchen Environment*, NeurIPS 2022 Datasets and
  Benchmarks; MIT CSAIL). Host: `https://data.csail.mit.edu/ActionNet/`.
- **Licence:** CC BY-NC-SA 4.0 for the raw data and for every derived
  tensor, with attribution to DelPreto et al. 2022. The code stays
  Apache-2.0. `data/` stays gitignored. The derived tensors are
  published as a separate asset (`actionsense-tensors-v0.7.tar.gz`, with
  `MANIFEST.md` and `LICENSE-DATA`), and their sha256 goes into ADR-0025.
  No commercial use.
- **Subjects:** **S00, S02, S03, S04, S05**, concatenated **in that
  order**. These are the five subjects with all five modalities. S01
  lost its wearables, and S06–S09 have their microphones and RGB marked
  "coming soon".
- **Raw files:** per subject, the wearable HDF5
  (`*_streamLog_actionNet-wearables_S##.hdf5`), the egocentric world
  video (`*_S##_eye-tracking-video-world_frame.mp4`, no gaze overlay), the
  microphone timing HDF5 (`*_streamLog_actionNet-microphone_S##.hdf5`)
  and the sink WAV (`*_sink_chunked_data.wav`). `fetch_actionsense.py`
  records their URLs, sizes and sha256 in `MANIFEST.json` and re-checks
  them with `--verify`. The S00 hashes are in
  `docs/pilot/actionsense-s00.md`. The S02–S05 hashes are recorded when
  they are fetched (plan Task 7 Step 1) and copied into ADR-0025.
- **No raw frame and no waveform** leaves the extraction. Vision is
  reduced to a 16×16 intensity grid and audio to a 128-bin log-mel
  vector, and no artefact is used to identify subjects.

### 2.2 Common grid

Every stream is placed on a common **10 Hz** grid on the wearable clock
(`time_s`, seconds since epoch; `HZ = 10.0`). Frame *t* aggregates the
window [*t*, *t* + 0.1 s) (`WIDTH = 0.1`). Frames are laid only inside
labelled segments (§2.4): a segment [*t0*, *t1*) gives
⌊(*t1* − *t0*) · 10⌋ full frames starting at *t0*.

### 2.3 Sensors and reductions (world contract)

| bouba_sens | Sensor (stream) | Reduction |
|---|---|---|
| audio (128) | sink microphone, 48 kHz (`sink/chunk_timestamp` + sink WAV) | samples placed on the clock fit of §2.6; one 100 ms window (4800 samples) per frame, Hann window, power spectrum, 128 triangular HTK-mel filters from 0 Hz to sr/2, `log(power + 1e-6)` |
| vision (16,16) | egocentric eye-tracking world camera (`eye-tracking-video-world/frame_timestamp` + MP4) | video frame nearest to the tick, kept only if within 0.1 s of it (`VISION_TOL_S`); decoded by ffmpeg (≥ 5.1) to 64×64 grayscale (`scale=64:64:flags=area`), then a 16×16 grid of mean intensities in [0, 1]. Same reduction as `extract_ccfilm_features.py`. No pretrained network |
| tactile (32) | both gloves, `tactile-glove-{left,right}/tactile_data`, 32×32 pressure matrices | per glove, means of the 8×8 blocks (a 4×4 grid, 16 values), then the mean over the window; left (16) then right (16) |
| gravity (3) | right Myo, `myo-right/acceleration_g` (3 axes, g, 50 Hz) | mean over the window, then a 2nd-order zero-phase Butterworth low-pass at 0.5 Hz (fs = 10 Hz, `sosfiltfilt`) on each contiguous run of valid frames within a segment |
| force (6) | Myo EMG, `myo-{left,right}/emg`, 8 channels per arm, 200 Hz | per-channel RMS over the window, then per arm the mean of 3 electrode sectors (channels 0–2, 3–5, 6–7); left (3) then right (3) |
| labels (4) | `experiment-activities/activities` | activity family, §2.4 |

A seventh tensor, `subject.pt` (long, the subject number: 0, 2, 3, 4,
5), and the side file `rows.pt` (`time_s`, `segment_id`,
`raw_label_idx`) are written for analysis only. Both are outside the
world contract and are ignored by `ActionSenseWorld`.

**Short gaps.** If a window holds no sample of a stream, the value is
linearly interpolated at the window centre from the two neighbouring
samples when they are at most 0.2 s apart (`MAX_GAP = 0.2`). Otherwise
the frame is invalid for that stream. Non-finite samples are masked and
counted (`nonfinite_samples`).

**Normalisation.** Each modality is z-scored **per subject**, with mean
and standard deviation computed on that subject's kept frames. A channel
with std < 1e-8 (for example a dead taxel) takes the value 0, not NaN.
The statistics are written to `MANIFEST.md`. Labels are not normalised.

### 2.4 Label families

Activities are read from `Start` / `Stop` rows paired by label. A
family is assigned by **prefix match** on the lower-cased, stripped
label (`FAMILY_RULES` in `src/bouba_sens/world/actionsense_features.py`):
`peel`, `slice` → 0; `spread`, `open/close a jar`, `open a jar` → 1; `clean`, `wipe` → 2;
`pour`, `set table`, `load dishwasher`, `unload dishwasher` → 3.
Matching leading words, not substrings, keeps "Spread … on a bread
slice" out of cutting.

| Family | Name |
|---|---|
| 0 | cutting (peel, slice) |
| 1 | spreading and jars |
| 2 | cleaning (wipe, clean) |
| 3 | pouring and tableware (pour, set table, dishwasher) |

The full table for the 20 activity strings of the ActionNet recording
code (`recording_data/sensor_streamers/ExperimentControlStreamer.py`,
`self._activities`) is unit-tested in
`tests/unit/test_actionsense_features.py::UPSTREAM_LABELS`. The PI
confirmed "Unload dishwasher" → family 3 on 2026-10-29, and on
2026-10-30 decided that "Open a jar of almond butter" is family 1 (the
jar family; rule `open a jar`, added in PR #56, with a test).

| Raw label | Family |
|---|---|
| Get/replace items from refrigerator/cabinets/drawers | — (dropped) |
| Clear cutting board | — (dropped) |
| Peel a cucumber | 0 |
| Slice a cucumber | 0 |
| Peel a potato | 0 |
| Slice a potato | 0 |
| Slice bread | 0 |
| Spread almond butter on a bread slice | 1 |
| Spread jelly on a bread slice | 1 |
| Open/close a jar of almond butter | 1 |
| Open a jar of almond butter (string found in the S00 HDF5) | 1 |
| Pour water from a pitcher into a glass | 3 |
| Clean a plate with a sponge | 2 |
| Clean a plate with a towel | 2 |
| Clean a pan with a sponge | 2 |
| Clean a pan with a towel | 2 |
| Get items from cabinets: 3 each large/small plates, bowls, mugs, glasses, sets of utensils | — (dropped) |
| Set table: 3 each large/small plates, bowls, mugs, glasses, sets of utensils | 3 |
| Stack on table: 3 each large/small plates, bowls | — (dropped) |
| Load dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils | 3 |
| Unload dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils | 3 |

Any label string found in the HDF5 that matches no prefix is dropped,
counted as `unmapped:<label>` and listed in `MANIFEST.md`. The
extraction does not guess. The strings recorded in the HDF5 can differ
from the upstream list. On S00 the unmapped pairs were "Get items from
refrigerator/cabinets/drawers" (5), "Open a jar of almond butter" (3),
"Get items from cabinets: …" (1) and "Stack on table: …" (1). Under
the prefix rules on main (ca3bcd8) alone, "Open a jar of almond butter"
would be unmapped; with the `open a jar` rule decided on 2026-10-30
(§8, item 4) it is family 1 and its 3 S00 segments are kept.

### 2.5 Frames dropped

A frame is kept only if all five modalities are valid for it. It is
dropped, and counted in `MANIFEST.md`, in each of these cases:

1. **Unlabelled:** it lies outside every labelled segment, or its
   segment has an unmapped label (§2.4). Frames are never laid there.
2. **Invalid segment:** the `Start` or the `Stop` row is not marked
   `Good` (counter `bad`), or a `Start` is never closed by a `Stop`
   before the next `Start` of the same label or the end of the file
   (counter `unclosed`). The segment is skipped, and the run does not
   crash.
3. **Stream gap:** a wearable stream has a gap longer than 0.2 s at the
   frame (§2.3), or a gravity run is shorter than 10 frames
   (`LOWPASS_MIN_FRAMES`, needed by `sosfiltfilt`). Counters
   `drop:tactile`, `drop:force`, `drop:gravity`.
4. **Outside the media:** no video frame lies within 0.1 s of the tick,
   or the audio window is not covered by a microphone file. These frames
   are dropped, never padded. Counters `drop:{vision,audio}:outside_span`
   and `drop:{vision,audio}:gap`. A session whose decoded video frame
   count differs from its `frame_timestamp` count, or whose ffmpeg run
   fails, contributes no vision (counters
   `session_vision_frame_count_mismatch`, `session_vision_ffmpeg_failed`).
5. **Near an audio clock breakpoint:** the audio window lies within 1 s
   of a breakpoint of the clock fit, or in the clock gap between two fit
   segments, or is claimed by two segments (counter
   `audio_breakpoint_drop`; §2.6).
6. **Rejected audio file or segment** (§2.6): its frames count as audio
   gaps, not as outside the audio span.

A session whose activity timestamps step backwards is skipped
(`session_skipped:nonmonotonic`), and so is a session that lacks a
required stream (`session_skipped:missing_stream`). If a subject ends
with zero kept frames, or if the tensors differ in length, the
extraction aborts (`check_aligned`, #45). When a subject has several
microphone files, the first one (in sorted file order) that covers a
frame supplies its audio.

### 2.6 Audio clock fit

The sink chunk timestamps arrive in bursts (S00 steps of about 0, 43 and
85 ms, against a steady 2048 / 48000 = 42.67 ms), while the WAV is
continuous. Audio samples are therefore placed on a **per-file piecewise
linear least-squares fit** of chunk timestamp against sample index
(spec §7, changed 2026-10-29; PR #51, `b0814ab`):

- Chunk *k* is stamped at its last sample, sample (*k* + 1) · 2048 − 1
  (`MIC_CHUNK = 2048`, fixed upstream in `MicrophoneStreamer.py`).
- **File gate.** A file is rejected if its WAV length is not exactly
  2048 × the number of chunk timestamps, or if its timestamps are not
  monotonic (counter `audio_file_inconsistent`).
- **Breakpoints.** A global fit is made first. Then
  *d*[*k*] = median(residuals of the `MIC_STEP_WINDOW = 25` chunks
  (~1 s) after *k*) − median(residuals of the 25 chunks before *k*).
  Each run of |*d*| > `MIC_STEP_DETECT_S = 0.05` s gives one breakpoint.
  It sits at the split that assigns each chunk near the run's peak to
  the nearer of the two median levels on either side.
- **Segments.** Each segment between breakpoints is fitted on its own.
  A segment is rejected (counter `audio_segment_bad_clock_fit`) if its
  implied rate deviates from the WAV rate by more than
  `MIC_RATE_TOL = 2e-3` (relative), or if it fails the residual gate. On
  `b0814ab` the residual gate is a maximum |residual| above
  `MIC_SEGMENT_RESIDUAL_MAX_S = 0.2` s. Issue #52 changes this gate to
  the p99 of the residuals, or to the residual minus its rolling median,
  so that an isolated ~0.3 s stall cannot reject a whole segment. A
  segment of fewer than 2 chunks is always rejected.
- **Placement.** In a segment fitted as *a* + *b*·*j*, the window of
  frame start *s* begins at sample rint((*s* − *a*) / *b*). A frame is
  covered only if its whole window lies inside one accepted segment, at
  least `MIC_BREAKPOINT_MARGIN_S = 1.0` s from any breakpoint, and in no
  other segment.
- **Limit.** A constant host latency between a sound and its timestamp
  is absorbed into the intercept, and this fit cannot detect it.
- **Reporting.** `MANIFEST.md` reports, per file, the segments, the
  breakpoints (time and step), and the rate and residual statistics of
  each segment.

**Extraction code version.** The full extraction (plan Task 7) runs on
`main` **after the fix of #52 is merged**. The commit is identified at
tag time. If the #52 fix changes any constant or rule stated in §2,
this text is updated before the tag. The S00 pilot numbers of §3 were
obtained with `b0814ab`, before #52.

Extraction command (macM1):

```
uv run python scripts/fetch_actionsense.py --out data/actionsense_raw --verify
uv run python scripts/extract_actionsense.py --raw data/actionsense_raw \
    --out data/actionsense --subjects S00 S02 S03 S04 S05
```

## 3. Pilot evidence (S00 only)

These reports were filed before this amendment. Both runs used the
same raw files (8 files, 37 487 809 070 bytes), macM1, nerve-wml v1.8.1
and ffmpeg 9.0.2. The first S00 recording (`17-18-46`) has an empty
activity table, so all S00 rows come from `18-11-37`.

| Check (spec §7) | Run 1 (`06525dd`) | Run 2 (`b0814ab`, clock fit) |
|---|---|---|
| 1. no abort, equal lengths | PASS, 7165 rows | PASS, 12 597 rows |
| 2. clock agreement, monotonic (amended) | PASS (caveat on the literal wording) | PASS: labelled span 0.0–2967.8 s lies inside the video (−290.9–3532.1 s) and microphone `18-11-04` (−318.4–3341.9 s) spans; 0 backwards steps |
| 3. tactile/audio onset xcorr, slicing | formal PASS, not evidential (peak −100 ms on 20 pairs) | PASS, weak: peak +100 ms, 0.063 on 2655 pairs |
| 4. extraction determinism | PASS, 8/8 sha256 | PASS, 8/8 sha256 |
| 5. smoke cell determinism | PASS, 4/4 sha256 | PASS, 4/4 sha256 |

Run 2 counters (out of 13 192 frames): `drop:tactile` 448 (3.4 %),
`drop:audio` 147 (1.1 %, down from 5579 or 42.3 % in run 1), no vision,
force or gravity drops, 12 597 kept (95.5 %). There are 29
`audio_breakpoints` across both microphone files (24 in `18-11-04`, 16
of them inside the labelled span) and 0 rejected segments. Class shares
are cutting 0.4465, spreading and jars 0.0834, cleaning 0.0917, pouring
and tableware 0.3784, so the **majority-class baseline is 0.4465**.

**Null test for check 3 (2026-10-29).** The PI fixed the rule before any
computation. The script `scripts/pilot_actionsense_null.py` was
committed in `fcd957e` before it was run (open PR #54). The statistic S
is the maximum cross-correlation over lags in [−200, +200] ms. The null
uses N = 1000 within-segment circular shifts of the audio-onset series,
each by at least 2 s, with seed 0. Results: **S_obs = 0.0634, null mean
0.0236, null p95 = 0.0482, null maximum 0.0956, 11 of 1000 draws ≥
S_obs, p = 0.012**. The decision under the rule was GO.

**Caveat.** This evidence rests on **one subject** (S00) and three
slicing labels. The effect is small in absolute terms. The trough of
−0.106 at +400 ms and the 0.056 at +1000 ms are not tested by the rule.
The pilot does not re-run on the post-#52 code. The full extraction
re-extracts S00 together with the other four subjects.

## 4. Experiment

### 4.1 Grids

There are four grids (decision of 2026-10-30): the two lock settings,
the direct analogues of ADR-0012's `v05_sf45_{nolock,lock}`, each run
under **both** emission modes, `argmax-bridge` and `straight-through`.
Both modes are pre-registered; a verdict is reported per mode.

| Grid | `LOCK_AFTER` | `EMISSION` | Cells |
|---|---|---|---|
| `nolock` / `argmax-bridge` | empty (no lock) | `argmax-bridge` | 5 seeds × 5 lesioned modalities × T1/T2 × 3 SNR floors = **150** |
| `lock` / `argmax-bridge` | `200` | `argmax-bridge` | **150** |
| `nolock` / `straight-through` | empty (no lock) | `straight-through` | **150** |
| `lock` / `straight-through` | `200` | `straight-through` | **150** |

Total: **600 cells**.

- Seeds **0, 1, 2, 3, 4** (`run_grid.sh` default; `SEEDS` unset).
  Phase-1 train seed = `seed`. Lesion seed = `seed * 10000 + count`,
  where `count` is the 1-based cell index in the loop order of
  `run_grid.sh` (seed, then modality, then timing, then SNR).
- Lesioned modalities: audio, vision, tactile, gravity, force.
- Timings: T1 (congenital, no Phase-1 checkpoint) and T2 (late,
  restoring `phase1_seed<seed>`).
- SNR floors: −20 dB (`floor`), −10 dB (`minus10`), +10 dB (`plus10`).
- Schedule: `STEPS_TRAIN=200`, `STEPS_LESION=100`, batch size 16 in
  both phases. B-3 probe: 64 items, `world.sample(batch_size=64,
  seed=seed + 777)`, which on this world is one contiguous window shared
  by the five queries of a cell.
- With `LOCK_AFTER=200` = `STEPS_TRAIN`, T2 cells inherit a constellation
  that was locked at the end of Phase 1. T1 cells never lock, since 200
  is more than `STEPS_LESION` = 100.
- The other `run_grid.sh` settings keep their defaults:
  `TRANSDUCER_GATING=hard`, `GUMBEL_TAU=1.0`, and `CODEBOOK_LOCK`,
  `GATING_SCHEDULE`, `GATING_TARGET` empty. `METRICS=Me1,Me2,Me3`.
  With PR #56, `run_grid.sh` reads `EMISSION` (default `argmax-bridge`)
  and passes `--emission` and `--probe-v2` (per-trio probe seeds as in
  `run_grid_probe_v2.sh`), so each cell writes both the v1 and the v2
  B-3 probe (see §8, items 1–2).
- Host: macM1, in the background under `nice`, with logs in
  `~/hypneum-lab/actionsense/logs/`.

### 4.2 Exact commands

```
export BOUBA_SENS_ACTIONSENSE_DATA=data/actionsense

# for E in argmax-bridge straight-through:
# nolock
WORLD=actionsense EMISSION=$E OUT_ROOT=runs/v07_actionsense_nolock_$E \
  STEPS_TRAIN=200 STEPS_LESION=100 METRICS=Me1,Me2,Me3 \
  bash scripts/run_grid.sh
uv run python scripts/aggregate_grid.py --root runs/v07_actionsense_nolock_$E \
  --out reports/v0.7_actionsense_nolock_${E}_aggregate.json

# lock
WORLD=actionsense EMISSION=$E OUT_ROOT=runs/v07_actionsense_lock_$E \
  LOCK_AFTER=200 STEPS_TRAIN=200 STEPS_LESION=100 METRICS=Me1,Me2,Me3 \
  bash scripts/run_grid.sh
uv run python scripts/aggregate_grid.py --root runs/v07_actionsense_lock_$E \
  --out reports/v0.7_actionsense_lock_${E}_aggregate.json

# ADR-0022 paired null, on each of the 4 grids, v2 probe (primary)
uv run python scripts/me6_paired_null.py --root runs/v07_actionsense_<grid>_$E \
  --replicates 20000 --seed 25102026 --block 10 \
  --successes-file per_query_successes_v2.json \
  --out reports/v0.7_actionsense_<grid>_${E}_me6_paired_null.json
# same with the default per_query_successes.json (v1 probe, comparison)
```

The run directory names are not part of the pre-registration. The
report names follow plan Task 7, which also wires the same commands
into `scripts/reproduce_paper_v01.sh` as `CHOOSE=v0.7_actionsense`. The block
length is fixed to 10 (decision of 2026-10-30, §8, item 3).

### 4.3 Metrics and thresholds (unchanged)

| Invariant | Statistic (as `scripts/aggregate_grid.py`) | Threshold |
|---|---|---|
| B-1 | median over the 75 (seed, modality, SNR) pairs of Me7 = Me1(T1) − Me1(T2) | **0.05** |
| B-2 | median over the 150 cells of the Me3 delta (`me3_delta.mean`, Kraskov MI in bits) | **0.10** |
| B-3 | median over the 30 (seed, timing, SNR) trios of Me6 = max over q ≠ m of \|P[q, m] − P[m, q]\| on the 5×5 per-query matrix | **0.02** |

These are the thresholds on file (B-1 ≥ 0.05, B-2 ≥ 0.10, B-3 ≥ 0.02)
and the Me1 / Me3 / Me6 / Me7 definitions. None of them changes. The
"≥" of the text is authoritative (decision of 2026-10-30): a median
exactly equal to the threshold passes. Before PR #56, `aggregate_grid.py`
tested `median > threshold`; PR #56 changes it to `>=` and adds a test on
the exact-threshold value. This is a deviation of the code from the
earlier amendments' implementation, not of the text; earlier ADR
verdicts are not re-run.

- **B-3 is reported only together with the ADR-0022 paired sign-flip
  null** with block length 10 (10 consecutive frames = 1 s at 10 Hz),
  run on the saved `per_query_successes_v2.json` (primary probe, ADR-0023;
  decides the B-3 verdict) and, for comparison, on
  `per_query_successes.json` (v1), with B = 20 000 replicates and
  seed 25102026. The report gives the observed
  median, the null mean, median, P95 and P99, P(null ≥ observed), p, and
  P(B-3 passes | H0) under the fixed rule, together with the iid
  binomial reference of ADR-0021. A B-3 number without its null is not
  quoted.
- **Class baseline.** The class frequencies and the majority-class
  baseline, pooled over the five subjects and per subject, are taken
  from `MANIFEST.md` and reported with the grids.

## 5. Analysis plan and decision rules (fixed before any grid)

1. **Completeness.** Verdicts are computed only on complete grids (four
   in all, §4.1): 150 of 150 `eval_report.json` files, 75 Me7 pairs and 30 complete trios.
   A failed cell may be re-run with the same seed, since `run_grid.sh`
   is idempotent. No cell is replaced by another seed. Any cell that is
   still missing is reported, and that grid gets no verdict.
2. **Verdict per grid and per invariant.** The verdict is the median
   rule of `aggregate_grid.py`, as in every previous amendment: B-k
   **PASS** if the grid median is greater than or equal to its
   threshold, and **FAIL** otherwise. B-3 is decided on the v2 probe;
   the v1 B-3 value is reported for comparison and does not decide. There is no bootstrap, no per-seed vote and no
   multiple-comparison correction. The four grids are judged
   separately, and one verdict per emission mode is reported.
3. **B-3 reading.** The fixed rule (step 2) is reported as is. B-3 is
   described as *distinguishable from the paired null* only if
   P(null ≥ observed) < 0.05 at the pre-registered block length
   (ADR-0022 reading rule). Otherwise it is described as *compatible
   with the paired null*, whatever the fixed rule returns. The
   per-item null (block 1) is reported as descriptive only, because the
   probe items are contiguous windows (ADR-0022, limit 1). H0 is the
   symmetry of expected performances. That symmetry is violated by
   unequal query-modality main effects even without any lesion effect
   (ADR-0022, limit 4 as corrected 2026-10-26). A rejection is
   therefore not read as a lesion effect.
4. **Lock vs no lock, and emission.** The difference between the lock
   settings, and between the two emission modes, is
   reported descriptively (per-invariant medians side by side, and
   per-seed values). No test of it is pre-registered.
5. **Descriptive reports, not verdicts.** These are per-seed medians,
   the Me1 means per timing and per lesioned modality against the
   majority baseline, the Phase-1 loss (`pretrain_losses.json`), and
   the per-trio Me6 values with their own null tails.
6. **Studyforrest.** The ADR-0012 numbers on the 4.5-modal bridge are
   B-1 0.0000 / +0.0063, B-2 −0.0376 / −0.0391 and B-3 0.1016 / 0.1250
   (no lock / lock). They were measured in the non-learning regime of
   ADR-0024, on tensors that are now lost (#45). They are **described**
   next to the ActionSense results, and they are **not a replication
   target**: no agreement or disagreement test against them is run or
   claimed.
7. **Regime qualification.** For the two `argmax-bridge` grids,
   every verdict carries the ADR-0024 qualification:
   *measured in a regime where the network does not learn the task*.
8. **Report.** The verdicts, the provenance (extraction commit, tag,
   nerve-wml commit, raw and tensor sha256) and the class baseline go
   into ADR-0025, `docs/adr/0025-actionsense-world-verdicts.md`.

## 6. Out of scope

- The **Amedi dose-response** curve (ADR-0013) on ActionSense.
- The **Sprint 11–14 mechanism blocks** (Gumbel gating, τ scans,
  codebook freeze, gating schedule) on ActionSense.

Both are left to a later amendment, decided after the v0.7 verdicts.
Also out of scope: the Studyforrest numbers as a replication target
(§5, item 6), activity recognition as a goal in itself, and any change
to the encoders or the modality dimensions.

## 7. Deviations policy

Once tagged, this text is frozen. Any change to the world, the
extraction, the subjects, the grids, the commands, the metrics, the
thresholds or the decision rules, whether made before or after the
grids, is a **new dated amendment** committed before the change takes
effect. Every deviation is listed in ADR-0025, and results produced
under a deviation are marked as such. A bug fix that changes any tensor
or any cell output counts as a deviation.

## 8. Decisions of the PI (2026-10-30) and fields still open

The points below were open in the draft of 2026-10-29. The PI decided
them on 2026-10-30; they are recorded as decided.

1. **Emission regime: both modes.** `argmax-bridge` and
   `straight-through` are both pre-registered: 2 modes × 2 lock settings
   (T1/T2 within each) = 4 grids, 600 cells, one verdict per mode.
   `run_grid.sh` takes `EMISSION` (default `argmax-bridge`) in PR #56.
2. **B-3 probe: v2 primary.** The ADR-0023 v2 probe decides the B-3
   verdict. The v1 probe is reported for comparison. `--probe-v2`
   stays active in `run_grid.sh` (PR #56).
3. **Paired-null block length: 10.** Blocks of 10 frames, 1 s at 10 Hz
   (ADR-0022, `me6_paired_null.py --block 10`; the option already exists).
4. **"Open a jar of almond butter": family 1.** Rule `open a jar` added
   to `FAMILY_RULES`, with a test (PR #56).
5. **Thresholds: "≥" is authoritative.** `aggregate_grid.py` is
   corrected from `>` to `>=`, with a test on the exact-threshold value
   (PR #56). §4.3 notes the discrepancy with the earlier code.
6. **Numbering.** ActionSense stays v0.7; the TOST-equivalence draft of
   PR #9 is to be renumbered v0.8 (comment posted on PR #9, PR #9
   untouched).

**Still pending before the tag:**

- the extraction commit after the fix of issue #52 (to be written into
  the tag message and ADR-0025; it must include PR #56);
- the sha256 of the S02–S05 raw files;
- the PI signature.

## 9. What is **not** amended

- No change to the three thresholds (0.05 / 0.10 / 0.02); the
  inequality is `>=`, as written (§4.3).
- No change to the Me1 / Me3 / Me6 / Me7 definitions. The aggregation
  code changes only by the `>` to `>=` comparison (PR #56).
- No change to the shape of one grid (5 seeds × 5 modalities × 2 timings × 3
  SNRs = 150 cells; four grids in all, §4.1) or to the (T1, T2) pairing for B-1.
- No change to the encoders, the nerve or the head. The ActionSense
  extraction reduces each sensor to the existing contract.
- The Studyforrest ADRs and their erratum (#45, ADR-0024) stay as they
  are.

## 10. Filing

- After PI sign-off, commit this file and tag it
  `bouba_sens/v0.7-actionsense-prereg`. Put the extraction code commit
  (post-#52) in the tag message, and push the tag **before** plan
  Task 7.
- It is not uploaded to OSF (status note). The git tag on
  `github.com/hypneum-lab/bouba_sens` is the record.
