# ADR-0013 — Dose-response LOCK_AFTER scan (Amedi recovery curve)

**Status:** Accepted — non-monotone B-1 peak, sub-threshold but qualitatively Amedi-like
**Date:** 2026-04-21 evening
**Sprint:** 10 (Sprint 9.5 follow-up)

## Context

ADR-0012 established that on the 4.5-modal real bridge,
`LOCK_AFTER=200` produced the first positive B-1 me7 gap
(+0.0063) of any world-condition pair. The same ADR left open
the obvious question: **does me7 depend on LOCK_AFTER, and if
so is there a critical-period-like peak?** This ADR scans
LOCK_AFTER ∈ {50, 100, 200, 400, 800} with `STEPS_TRAIN=200
STEPS_LESION=100` (so LOCK_AFTER > 200 means T2 never reaches
the lock during Phase 1, and LOCK_AFTER ≥ 300 means no grid
cell ever locks).

## Grid provenance

| Item | Value |
|------|-------|
| Host | Studio |
| Worktree | `~/Projets/bouba_sens_b1` |
| Commit | `eb8dee5` (feat/sprint-10-dose-response) |
| nerve-wml | `v1.5.3` |
| World | `StudyforrestRealWorld(data_dir=data/sf_phase2_adapted)` (4.5-modal, ADR-0012 path (b)) |
| Config | `STEPS_TRAIN=200 STEPS_LESION=100 METRICS="Me1,Me2,Me3"` |
| Grids | 5 × 150 cells (`runs/v05_dr_lock{50,100,200,400,800}`) |
| Wall time | ~1h10 wall-clock with 5-way concurrency on M3 Ultra |
| Artefacts | `reports/v0.5_dr_lock{50,100,200,400,800}_aggregate.json` |

## Verdicts — dose-response table

| LOCK_AFTER | B-1 Me7 | B-2 Me3_delta | B-3 Me6 max-abs |
|-----------:|--------:|--------------:|----------------:|
| 50 | +0.0062 | -0.0044 | 0.1094 |
| **100** | **+0.0125** (peak) | -0.0190 | 0.1094 |
| 200 | 0.0000 | -0.0069 | 0.1016 |
| 400 | 0.0000 | -0.0063 | 0.1094 |
| 800 | 0.0000 | +0.0009 | 0.0938 |
| ∞ (no-lock, ADR-0012) | 0.0000 | -0.0376 | 0.1016 |

Pre-registered thresholds (unchanged): B-1 > 0.05, B-2 > 0.10, B-3 > 0.02.

## Decision — Amedi-like dose-response curve qualitatively present

**B-1 is non-monotone in LOCK_AFTER and peaks at 100** (50% of
STEPS_TRAIN). This is a qualitative signature of a
**critical-period window**: the lock has to fire during the
mid-range of Phase 1 for the T1/T2 asymmetry to emerge. Fire
too early (LOCK_AFTER=50) and T1 hasn't developed enough to
benefit; fire too late (≥200) and the T2 mux has already
converged, erasing the differential.

**The magnitude (+0.0125) stays below the pre-registered 0.05
threshold.** So B-1 still FAILs the pre-registered test. But
the shape of the curve is qualitatively consistent with the
Amedi 2007 biological hypothesis, **in a way that no synthetic
world produced**. The paper headline moves from:

> "The lock homogenises T1/T2 differences toward zero in 4/5
>  worlds, directionally falsifying the Amedi hypothesis."

to:

> "The lock produces an Amedi-shaped non-monotone B-1 peak at
>  LOCK_AFTER = 100 on the 4.5-modal real biological bridge.
>  The peak magnitude (+0.0125) remains below the 0.05
>  pre-registered threshold but exceeds every B-1 value
>  observed in the 4 synthetic-cluster worlds."

## Secondary finding — B-2 interference-vs-migration

B-2 magnitude is MAXIMISED at LOCK_AFTER=100 (most negative,
-0.019) and relaxes monotonically as LOCK_AFTER grows
(converging to +0.001 at LOCK_AFTER=800). Combined with the
4.5-modal no-lock baseline (-0.0376), this paints a picture of
**temporal-proximity interference**: when the mux is fixed
around the T1/T2 peak, post-lesion MI drops most. Consistent
with the B-2 sign reversal from synthetic (positive) to 4.5-
modal real (negative) already noted in ADR-0012.

## Tertiary finding — B-3 is lock-invariant

B-3 stays in 0.094-0.109 (4.7×-5.4× threshold) across all 5
LOCK_AFTER values. The constellation freeze has no measurable
effect on the structural asymmetry. Confirms the
**architectural invariant** interpretation from the v0.3 and
v0.4 lock matrix.

## Pre-registration fidelity

- No threshold changes.
- No metric-math changes.
- The 4 synthetic-cluster verdicts (ADR-0005/0008/0009/0010/0011)
  and the 4.5-modal verdict (ADR-0012) all remain canonical.
- This ADR strictly extends the experimental matrix with a
  new independent variable (LOCK_AFTER). OSF amendment v0.5
  allowed any lock-related experimental extension in the
  "follow-up" clause.

## Paper v0.2 integration

This ADR unblocks the following content updates (queued for
paper v0.2):

1. **New figure**: scatter+line plot of me7 vs LOCK_AFTER with
   bootstrap 95% CIs via nerve-wml methodology. Horizontal
   reference line at 0.05 threshold. X-axis log-scaled
   {no-lock, 800, 400, 200, 100, 50}.
2. **§5.5 rewrite**: the lock is not a homogeniser — it is a
   **critical-period operator** whose effect depends on its
   firing time relative to Phase 1 completion.
3. **§6 honest framing**: the Amedi magnitude is not met at
   the pre-registered 0.05 threshold, but the shape is. Future
   work: (a) extend STEPS_LESION so B-1 has more time to
   compound; (b) combine LOCK_AFTER=100 with nerve-wml#5
   Gumbel transducer for compound critical-period.

## Future directions

- **OSF amendment v0.5.1**: declare a 3-point replication of
  the peak (LOCK_AFTER ∈ {80, 100, 120}) on a fresh random
  seed to test whether the peak is stable under seed resampling.
  If the peak survives, the Amedi-like claim becomes
  statistically defensible.
- **Sprint 11**: combine LOCK_AFTER=100 with
  `transducer_gating=GUMBEL` (nerve-wml#5) for compound
  critical-period on 4.5-modal.
