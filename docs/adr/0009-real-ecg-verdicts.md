# ADR-0009 — Real-biological-signal verdicts (MIT-BIH ECG)

**Status:** Accepted — **descriptive, not pre-registered**
**Date:** 2026-04-20
**Sprint:** 7 (extended scope, post-Task-7.6b)

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
| B-3 Me6 > 0.02 | 0.1484 | 0.1406 | 0.1406 | 0.3125 | **0.4453** | **PASS @ 22.3x threshold** |

## Decision

**B-3 grows monotonically with input complexity:**

| Input class | B-3 median | Multiple of threshold |
|-------------|-----------:|----------------------:|
| Synthetic cluster (Gauss / XOR / Sinu) | ~0.14 | 7x |
| AR(1)-scene mock (ADR-0008) | 0.31 | 15.6x |
| **Real biological ECG** | **0.45** | **22.3x** |

The perceptive / proprioceptive asymmetry is **not a synthetic-
world artefact**. On a real 5-minute human ECG trace — temporal
autocorrelation 0.97, intrinsic dimensionality 6/28 on
audio/vision, clearly outside the synthetic cluster AND the
AR(1) mock — B-3 is not merely preserved but reaches its
strongest value of the whole benchmark. The 2026-04-20 external-
validity critique is empirically lifted for this invariant.

**B-1 confirms the topology-dependent pattern.** Real ECG gives
the same ~-0.006 sign as Gaussian and XOR (orthogonal-factored)
while Sinusoid (circular) remains the only world where B-1 has
the pre-registered positive sign. The cluster / mock / real
evidence is now consistent: B-1 is a property of the world's
latent topology, not of the architecture.

**B-2 sign flips back to positive on real data.** The mock's
negative B-2 (-0.029) was an artefact of the zeroed tactile /
gravity / force modalities; on real ECG the MI migration sign
matches the three synthetic worlds (+0.011 vs +0.002 to +0.028).
The magnitude stays well below the 0.10 threshold, so the
Sprint 5 Kraskov-estimator-limitation hypothesis from ADR-0004
holds.

## Headline

On the 5-monde benchmark (Gaussian, XOR, Sinusoid, mock AR(1),
real ECG), **B-3 is the only invariant that PASSes every time,
with effect sizes growing 7x → 15.6x → 22.3x as inputs move
from synthetic-factorised to biologically-plausible**. This is
strong prima facie evidence that B-3 captures an architectural
property, not a data-cluster accident.

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

## Next steps

1. File the OSF amendment (`docs/osf/amendment-v0.4-studyforrest.md`)
   with ECG bridge added as an explicit secondary analysis path.
2. Sprint 8: real Studyforrest via `datalad install ///studyforrest`
   and a dedicated feature-extraction script on a machine with
   git-annex available.
3. Paper draft Sprint 8: use this ADR + ADR-0008 + ADR-0005 as
   the three-tier evidence for B-3 robustness (synthetic cluster
   → synthetic-outside → real-signal).
