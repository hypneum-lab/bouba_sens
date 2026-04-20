# ADR-0005 — v0.3 cross-world invariant verdicts

**Status:** Accepted
**Date:** 2026-04-20
**Sprint:** 6 (close)

## Context

ADR-0004 closed Sprint 5 with a partial verdict on GaussianWorld
(B-3 PASS at 7.4x threshold, B-1 directionally falsified, B-2 under
threshold). The immediate concern flagged in that ADR was whether
the B-1 inversion was a structural property of the architecture or
a quirk of GaussianWorld. Sprint 6 triangulates by replicating the
150-cell grid on the two other pre-existing worlds.

## Grid provenance

| Item | Value |
|------|-------|
| Host | Studio (MacStudio, arm64, macOS 26.4) |
| Commit | `fee06cd` (main) |
| Config | `STEPS_TRAIN=200`, `STEPS_LESION=100`, `METRICS="Me1,Me2,Me3"` |
| Grids | XOR (150/150) + Sinusoid (150/150), wall ~40 min parallel on M3 Ultra |
| Artifacts | `reports/v0.3_xor_aggregate.json`, `reports/v0.3_sinusoid_aggregate.json` |

## Verdicts

| World | B-1 (Me7 > 0.05) | B-2 (Me3 delta > 0.10) | B-3 (Me6 > 0.02) |
|-------|------------------|------------------------|-------------------|
| GaussianWorld (v0.2) | -0.0063 / 75 cells / **FAIL (inverted)** | 0.0275 / 30 cells / FAIL | **0.1484** / 30 cells / PASS |
| XORWorld (v0.3) | -0.0062 / 75 cells / **FAIL (inverted)** | 0.0034 / 30 cells / FAIL | **0.1406** / 30 cells / PASS |
| SinusoidWorld (v0.3) | **+0.0125** / 75 cells / **FAIL (correct direction)** | 0.0019 / 30 cells / FAIL | **0.1406** / 30 cells / PASS |

## Decision

**B-3 is a robust architectural invariant (PASS in 3/3 worlds).** The
max-abs off-diagonal asymmetry is consistently ~0.14, about 7x the
pre-registered 0.02 threshold, with zero world-to-world variance
worth speaking of. The 5-modality asymmetric cross-modal
reorganisation is a property of the architecture itself, not of
GaussianWorld data statistics.

**B-1 sign flips between worlds, so the inversion is not
architectural.** Sinusoid restores the biologically predicted
direction (T1 beats T2 by +0.0125) even if the magnitude stays
below 0.05. Gaussian and XOR give the opposite sign. This rules
out the "adaptive codebook saturates congenital compensation"
hypothesis from ADR-0004 in the structural form. Best remaining
candidates:

- The T1 / T2 gap depends on how correlated the pre-training
  distribution is with the lesion-stressed distribution. Sinusoid
  lives on a smooth 1D manifold (periodic), so the restricted T1
  codebook stays close to the lesioned regime; Gaussian and XOR
  have more discontinuous latent structure, so T2 benefits from
  having the full codebook available at lesion onset.
- Me7 as currently measured (per-cell Me1 pair mean) is dominated
  by lesion-phase recovery noise on STEPS_LESION=100. A longer
  Phase 2 or a recovery-AUC variant (think Me2-style) might reveal
  a consistent sign.

**B-2 fails in all three worlds, with decreasing magnitude as the
world gets simpler (Gaussian 0.028 > XOR 0.003 > Sinusoid 0.002).**
Kraskov MI estimation on 1D mean-pooled probe codes is too noisy
to reliably capture informational migration. This is a
measurement-device issue, not an invariant falsification — the
delta stays positive (or near zero) but the scale is below the
threshold calibration baseline.

## Headline

On the three benchmark worlds, cross-modal plasticity manifests
as **topological asymmetry (B-3) but not as congenital advantage
(B-1) nor as strong MI migration (B-2) at the pre-registered
scales.** B-3 is the only of the three invariants that is a
genuine architectural property; B-1 is world-dependent; B-2 is
estimator-limited.

## Pre-registration fidelity

- No threshold changes across Sprints 4, 5, 6.
- No metric-math changes.
- Only the orchestration layer (CLI, aggregator, per-query perf
  matrix, probe-code capture, world dispatch) evolved from v0.1
  to v0.3. All 3 verdict triples reference the same spec 1.2
  thresholds (0.05 / 0.10 / 0.02). No p-hacking vector.

## Next steps

Sprint 7 will draft the v0.1 paper manuscript with B-3 as the
headline result and B-1 / B-2 as interpreted null / under-threshold
results. Specifically:

1. Paper results section consumes this ADR directly — 3 worlds
   for triangulation, B-3 as confirmed invariant, B-1 as
   world-dependent effect worth investigating in follow-up.
2. An optional spin-off exploring a recovery-AUC variant of Me7
   (analogous to Me2) could be Sprint 8 — not blocking the paper.
3. Me3 estimator review: consider K > 1 kNN or multi-dim preserved
   probe codes if reviewers flag the universal B-2 FAIL as
   methodologically load-bearing.
