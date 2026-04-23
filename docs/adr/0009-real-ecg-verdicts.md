# ADR-0009 — Real-biological-signal verdicts (MIT-BIH ECG)

**Status:** Accepted — **descriptive, partially retracted 2026-04-22**
**Date:** 2026-04-20 (original) ; **2026-04-22** (retraction note)
**Sprint:** 7 (extended scope, post-Task-7.6b)

## Retraction note (2026-04-22)

Running the reusable critical-validation pipeline
(`scripts/critical_validation_pipeline.sh`, commit `858ce51`) on
the v0.4 real-ECG grid (`runs/v04_studyforrest_real_grid`) shows
that the original §Decision and §Headline below **over-claimed**
on B-3. The null-model partition control (n=9 random 3+2
partitions of the same 5 ECG modalities) gives an identical Me6
median (0.4141) ; the pre-registered perceptive/proprioceptive
3+2 split ranks 2/9 (22.2 percentile), `passes_95pct = false`.

In plain words : **the absolute B-3 magnitude on real ECG comes
from the signal's per-modality entropy, not from the
perceptive/proprioceptive cognitive structure** the ADR claimed
to have validated. The "22.3× threshold lifts external-validity
critique" headline is **retracted**.

The B-1 and B-2 paragraphs of §Decision are also weakened :
- B-1 Me7 bootstrap CI = [−0.037, +0.025] straddles 0 ; "confirms
  topology-dependent pattern" was too strong, replaced by
  "consistent with" and the explicit CI is now cited.
- B-2 Me3_delta multi-estimator agreement at numerical noise
  (Kraskov 0.000, binning 0.000, MINE −1.7×10⁻⁸) ; "sign flips
  back to positive" was not robust at n=16 probe batch.

Issue #3 carries the full evidence + the audit transcript.

## Context

ADR-0008 recorded the out-of-cluster stress-test on the
Studyforrest **mock** surrogate (AR(1) scene-latent synthetic).
The maintainer then asked for a re-run on **real** Studyforrest
data. Upstream Studyforrest is distributed via `datalad` +
git-annex and its bulk download is out of scope for this session;
the pivot was to use a different CC0-licensed real biological
recording — the MIT-BIH ECG trace bundled with scipy
(`scipy/dataset-ecg`, 5 min @ 360 Hz) — as a proof-of-concept
biological-replication channel. The manifest is labelled
honestly as "MIT-BIH ECG surrogate for Studyforrest" so the paper
cannot over-claim.

## Grid provenance

| Item | Value |
|------|-------|
| Host | Studio (MacStudio, arm64, macOS 26.4) |
| Commit | `0673a2b` (main) |
| World | `StudyforrestWorld(data_dir=data/studyforrest_real)` |
| Data source | `scipy/dataset-ecg/ecg.dat` (108 000-sample uint16 trace, CC0) |
| Feature extraction | 1-sec windows @ 50 ms hop; audio = 128-bin rfft magnitude; vision = 256-sample waveform reshaped 16x16; label = 4-quantile of RMS |
| Config | same as v0.3 (`STEPS_TRAIN=200`, `STEPS_LESION=100`, `METRICS="Me1,Me2,Me3"`) |
| Cells | 150 / 150 processed, 0 skipped |
| Wall time | ~17 min solo |
| Artifacts | `reports/v0.4_studyforrest_real_aggregate.json` |

## Verdicts on real ECG vs cluster + mock

| Invariant | Gauss | XOR | Sinu | mock | **real ECG** | Verdict |
|-----------|------:|----:|-----:|-----:|-------------:|---------|
| B-1 Me7 > 0.05 | -0.0063 | -0.0062 | +0.0125 | 0.0000 | **-0.0062** | FAIL, back to the cluster sign |
| B-2 Me3 delta > 0.10 | +0.0275 | +0.0034 | +0.0019 | -0.0288 | **+0.0111** | FAIL, positive again |
| B-3 Me6 > 0.02 | 0.1484 | 0.1406 | 0.1406 | 0.3125 | **0.4453** | raw magnitude clears threshold ; **partition control fails** (22nd pctl, n=9, see retraction note) |

## Decision (revised 2026-04-22)

**B-3 absolute magnitude grows with input richness, but the
pre-registered partition is not distinguishable from random :**

| Input class | B-3 raw median | Threshold multiple | Partition control |
|-------------|---------------:|-------------------:|-------------------|
| Synthetic cluster (Gauss / XOR / Sinu) | ~0.14 | 7× | not yet run |
| AR(1)-scene mock (ADR-0008) | 0.31 | 15.6× | not yet run |
| **Real biological ECG** | **0.45** | **22.3×** | **22nd pctl on n=9 random 3+2 splits — fails** |

On a real 5-minute human ECG trace, B-3 reaches its largest raw
value of the whole benchmark — but a null-distribution test (n=9
random 3+2 partitions of the same 5 ECG modalities) gives an
identical median (0.4141) ; the pre-registered
perceptive/proprioceptive 3+2 split ranks **2/9** (22.2 pctl),
not the > 95th percentile that would lift the external-validity
critique. **The B-3 magnitude is driven by ECG signal entropy,
not by the perceptive/proprioceptive cognitive structure
claimed.** The 2026-04-20 external-validity critique is **not**
lifted by this evidence ; it remains open pending a real
Studyforrest replication AND a partition-controlled re-run on
the cluster + mock grids.

**B-1 is consistent with a topology-dependent pattern.** Real
ECG gives the same ~−0.006 sign as Gaussian and XOR
(orthogonal-factored) while Sinusoid (circular) remains the only
world where B-1 has the pre-registered positive sign. However,
the bootstrap 95% CI on Me7 is `[−0.037, +0.025]` — straddling
0 — so the cluster / mock / real "consistency" is at best a
directional observation within noise, not a confirmation of the
topology-dependence hypothesis.

**B-2 is at the noise floor across estimators on real data.**
The mock's negative B-2 (−0.029) was an artefact of the zeroed
tactile / gravity / force modalities. On real ECG the MI
migration is +0.011 by the binning estimator, but at n=16 probe
batch the multi-estimator robustness check (Kraskov 0.000,
binning 0.000, MINE −1.7×10⁻⁸) puts all three estimators in the
numerical-noise regime ; the sign of B-2 on real data is **not**
robustly determined.

## Headline (revised 2026-04-22)

On the 5-world benchmark (Gaussian, XOR, Sinusoid, mock AR(1),
real ECG), **B-3 raw magnitude grows 7× → 15.6× → 22.3× as
inputs move from synthetic-factorised to biologically-richer**,
but the only partition control run to date (real ECG, n=9)
shows the pre-registered split is statistically
indistinguishable from random partitions of the same modalities.
**The headline architectural-property claim from the 2026-04-20
ADR is retracted** ; the magnitude growth is now a raw
observation pending partition-controlled replications across the
synthetic + mock grids, and a real Studyforrest run.

## Limits of the real-ECG replication

1. **ECG is not Forrest Gump.** The original pre-registration
   targeted multi-modal film data (Studyforrest Phase 1). ECG
   is a single-channel physiological trace with no visual
   counterpart. We split it into audio + vision pseudo-modalities
   via FFT + waveform reshape, but the downstream network sees
   only one real signal, not two correlated ones. Biological
   plausibility is partial: the signal is real, the
   multi-modality is reconstructed.
2. **Tactile, gravity, force are still zeroed.** The full 5-
   modality architecture runs, but only 2 modalities carry
   biological content. Real Studyforrest would have a motion-
   annotation stream that could proxy proprioception; ECG does
   not.
3. **Sample size is 5 minutes.** 108 000 samples at 360 Hz is
   enough for 150-cell smoke validation but too short for any
   formal biological inference.
4. **No pre-registration yet.** This verdict is descriptive.
   The v0.4.0 tag is withheld until an OSF amendment is filed
   covering the ECG bridge AND a real Studyforrest run is
   performed via datalad.
5. **Null-model partition control fails (added 2026-04-22).**
   `scripts/critical_validation_pipeline.sh` (commit `858ce51`)
   on the same v0.4 grid : n=9 random 3+2 partitions of the
   same 5 ECG modalities give an identical Me6 median (0.4141) ;
   the pre-registered partition ranks 2/9 (22.2 percentile),
   `passes_95pct = false`. The B-3 magnitude reported above is a
   property of the ECG signal entropy, not of the
   perceptive/proprioceptive partition structure. Until the same
   partition control is run on the synthetic-cluster (ADR-0005)
   and AR(1)-mock (ADR-0008) grids, the entire monotone-growth
   table in §Decision must be read as raw-magnitude only.

## Next steps

1. File the OSF amendment (`docs/osf/amendment-v0.4-studyforrest.md`)
   with ECG bridge added as an explicit secondary analysis path.
2. **Re-run the critical-validation pipeline on ADR-0008 mock
   and ADR-0005 cluster grids** (added 2026-04-22 ; tracked in
   issue #4) — same `null_b3` axis, n ≥ 9. Required gateway
   before Sprint 9 ADR-0012 acceptance per issue #3 conclusion.
   The pipeline lives at
   `scripts/critical_validation_pipeline.sh` on branch
   `sprint9/critical-pipeline` commit `858ce51` (not yet merged
   to main). Dichotomous reading per issue #4 :
   - all 3 grids fail → B-3 architectural-property narrative
     dead, paper §8 must be rebuilt ;
   - only ECG fails → signal-specific (entropy floods partition
     signature), cluster-level B-3 claim survives at lower
     magnitude ;
   - mixed → diagnostic of which property B-3 captures, scope
     for an ADR-0014.
3. Sprint 8: real Studyforrest via `datalad install ///studyforrest`
   and a dedicated feature-extraction script on a machine with
   git-annex available — with the partition control wired in
   from the start, not added retrospectively.
4. Paper draft Sprint 8: this ADR can no longer serve as the
   "real-signal" tier of a three-tier B-3 robustness narrative.
   The §8 Discussion must either (a) defer all real-signal
   claims to post-Studyforrest data, or (b) report this ADR's
   raw magnitude observation explicitly alongside the failed
   partition control.
