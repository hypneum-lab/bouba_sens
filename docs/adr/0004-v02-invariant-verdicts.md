# ADR-0004 — v0.2 empirical invariant verdicts (B-1 / B-2 / B-3)

**Status:** Accepted
**Date:** 2026-04-20
**Sprint:** 5 (close)

## Context

Sprint 5 closed the three CLI coverage gaps flagged in ADR-0003:

- Me3 delta is now populated per cell from probe codes captured
  pre- and post-lesion (Task 5.1).
- Me6 is computed aggregation-side from a 5x5 per-query accuracy
  matrix stacked across the 5 lesioned-modality cells of each
  (seed, timing, SNR) trio (Task 5.2).
- Me7 is computed aggregation-side by pairing `(seed, modality,
  SNR)` across T1 and T2 runs (Task 5.2).

Thresholds are unchanged (0.05 / 0.10 / 0.02) — same values as in
the OSF pre-registration, so the v0.1 and v0.2 verdicts are
directly comparable.

## Grid provenance

| Item | Value |
|------|-------|
| Host | Studio (MacStudio, arm64, macOS 26.4) |
| Commit | `242b292` (main) |
| Config | `STEPS_TRAIN=200`, `STEPS_LESION=100`, `METRICS="Me1,Me2,Me3"` |
| Cells | 150 / 150 processed, 0 skipped |
| Wall time | ~17 min |
| Artifacts | `reports/v0.2_aggregate.json` (30 aggregated cells, 13.5 kB) |

## Verdicts

| Invariant | Threshold | Median | Cells counted | Passes |
|-----------|-----------|--------|---------------|--------|
| B-1 (Me7 congenital gap > 0.05) | 0.05 | **-0.0063** | 75 | **No** |
| B-2 (Me3 delta > 0.10) | 0.10 | **0.0275** | 30 | **No** |
| B-3 (Me6 max-abs off-diag > 0.02) | 0.02 | **0.1484** | 30 | **Yes** |

### Direct v0.1 → v0.2 comparison

| Invariant | v0.1 cells | v0.1 median | v0.2 cells | v0.2 median | Diagnosis |
|-----------|-----------:|------------:|-----------:|------------:|-----------|
| B-1 Me7 | 10 | 0.0000 | 75 | -0.0063 | CLI gap closed; hypothesis directionally falsified |
| B-2 Me3_delta | 0 | 0.0000 | 30 | 0.0275 | CLI gap closed; effect positive but ~3.6x below threshold |
| B-3 Me6 max-abs | 0 | 0.0000 | 30 | 0.1484 | CLI gap closed; hypothesis confirmed (7.4x threshold) |

## Decision

**The verdict is partial: 1 / 3 invariants pass with empirical data.**
This corresponds to state 2 of the draft trichotomy (some pass, some
fail) and is a genuine scientific outcome — every invariant is now
backed by data (cells_counted >= 30 by construction).

### B-3 PASS (perceptive / proprioceptive asymmetry)

The 5x5 query-modality performance matrix exhibits a pronounced
off-diagonal asymmetry (median max-abs 0.148, 7.4x the 0.02
threshold). The pre-registered claim that the three perceptive
modalities (audio, vision, tactile) and the two proprioceptive
modalities (gravity, force) form a structurally asymmetric group
is supported on GaussianWorld across 5 seeds x 2 timings x 3 SNR.

### B-1 FAIL (congenital gap, directional falsification)

Me7 median is slightly **negative** (-0.0063), i.e. late-acquired
lesions (T2) recover marginally better than congenital ones (T1)
at SNR floor. This falsifies the **direction** of the
pre-registered hypothesis, not just its magnitude. On
GaussianWorld, the adaptive codebook and cross-modal transducers
appear to compensate lesions at least as well when the lesion is
applied post-convergence as when it is applied pre-training.
Candidate interpretations: (a) the intact 5-modality codebook
learnt during Phase 1 is more plastic than the "congenital"
restricted codebook; (b) `STEPS_LESION=100` is sufficient for the
recovery curve to plateau for both T1 and T2 on a simple world.

### B-2 FAIL (MI migration, effect present but under-threshold)

Me3 delta is positive (0.0275) but only about one-third of the
0.10 pre-registered threshold. Mutual information between fused
codes and labels **does** migrate post-lesion, just less strongly
than the pre-registration expected. Plausible causes: (a) the
Kraskov k-nearest estimator is noisy on 16-dimensional mean-pooled
probe codes with batch B=16; (b) GaussianWorld's 32-dim latent
carries less structured information than the threshold was cal
ibrated against.

## Pre-registration fidelity

- No threshold changes vs ADR-0003.
- No metric-implementation changes (same `me3_delta`, `me6_*`,
  `me7_congenital_gap` functions as Sprint 3).
- Only orchestration + CLI wiring changed between v0.1 and v0.2,
  so no p-hacking vector was introduced. The v0.1 NO-GO was a
  coverage artefact; the v0.2 verdict is the scientific one.

## Next steps

1. **Sprint 6 (paper draft)** consumes this ADR as its Results
   section backbone. Headline: *"On GaussianWorld the symbolic /
   topological asymmetry (B-3) is robustly measurable, while the
   congenital-gap (B-1) and MI-migration (B-2) claims are not
   confirmed at the pre-registered thresholds."*
2. Consider an **XOR-world and Sinusoid-world replication pass**
   before Sprint 6 freezes the paper — both are already in
   `bouba_sens.world`, and a 1-world-per-world 30-seed scan
   would triangulate whether the B-1 reversal is GaussianWorld-
   specific or world-agnostic.
3. **Revisit Me3 estimator calibration** in a follow-up ADR if
   Sprint 6 reviewers flag the 3.6x gap as methodologically load
   -bearing.
4. Nothing in this ADR warrants bumping the pre-registered
   thresholds. A future OSF amendment can extend them (e.g. add
   an XOR-world invariant) but must not weaken them.
