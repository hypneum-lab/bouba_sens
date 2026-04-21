# OSF pre-registration amendment — bouba_sens v0.5 (5-modal Studyforrest)

**Status:** Draft, ready to file
**Parent pre-registration:** `dream-of-kiki` OSF DOI `10.17605/OSF.IO/Q6JYN`
**Previous amendment:** `amendment-v0.4-studyforrest.md` (2-modal ECG bridge)
**Amendment tag on file:** `bouba_sens/v0.5-studyforrest-5modal`
**Amendment date:** 2026-04-21

## What is being amended

The v0.4 amendment added a generic `StudyforrestWorld` to the pre-
registered world set but was exercised only with the 2-modality ECG
stub. ADR-0009 and ADR-0011 documented that the stub's 3 zeroed
modalities (tactile, gravity, force) made the plasticity-lock
mechanism structurally unobservable on that bridge.

This v0.5 amendment adds a `StudyforrestRealWorld` with **5 real
biological modalities** drawn from Studyforrest phase 2 (Hanke et al.
2016). The modality mapping (fixed before any grid runs) is:

| bouba_sens modality | Studyforrest source | Feature |
|---------------------|---------------------|---------|
| audio | film soundtrack | 128-bin mel-spectrogram @ 10 Hz |
| vision | film frames | VGG16 conv4 pool, PCA-256, reshape 16×16 |
| tactile | motion annotations | scene-id + motion-type embedding |
| gravity | fMRI rigid-body rotations | pitch, yaw, roll (3 dim) |
| force | ECG + respiration | 6-dim: signal + Δ + Δ² |

## Protocol

Two grid runs at the pre-registered thresholds (0.05 / 0.10 / 0.02,
unchanged):

1. `WORLD=studyforrest` no-lock — baseline.
2. `WORLD=studyforrest` LOCK_AFTER=200 — matches ADR-0011 condition.

Expected artefacts:
- `reports/v0.5_studyforrest_5modal_nolock_aggregate.json`
- `reports/v0.5_studyforrest_5modal_lock200_aggregate.json`
- `docs/adr/0012-real-5modal-studyforrest-verdicts.md`

## What is NOT being amended

- Thresholds 0.05 / 0.10 / 0.02 (frozen since spec §1.2).
- Metric implementations `me1_accuracy`, `me2_recovery_auc`,
  `me3_delta`, `me6_asymmetry`, `me6_max_abs_off_diag`,
  `me7_congenital_gap`, `me9_bootstrap`.
- The 5-seed × 5-modality × 2-timing × 3-SNR = 150-cell grid structure.

## Decision rule for paper v0.2

| B-3 on 5-modal real | Paper v0.2 claim |
|---------------------|------------------|
| PASS at ≥ 10× threshold | "B-3 is an architectural invariant across synthetic AND real biological 5-modal input." (strong) |
| PASS at 1-10× threshold | "B-3 persists but is attenuated under biological input complexity." (moderate) |
| FAIL | "B-3 is a synthetic-cluster artefact; the unlocked Studyforrest ECG-2 result was driven by zeroed modalities." (retraction of v0.1 headline) |

All three outcomes are publishable. No threshold change, no
metric-math change — only the world is new.

## Timeline

| Step | ETA |
|------|-----|
| File on OSF (this doc) | same-day |
| Task 9.6 grid runs | same-day (~40 min parallel) |
| ADR-0012 + paper §5.5 | same-day |
| v0.5.0 tag + release | same-day |

Grid runs strictly post-filing.
