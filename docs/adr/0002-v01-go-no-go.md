# ADR-0002 — v0.1 go / no-go decision

**Status:** Accepted
**Date:** 2026-04-20
**Sprint:** 3 (close)

## Context

Sprint 3 closed with the full v0.1 metrics + CLI + HTML report + a
single-seed smoke run that exercises every piece of the pipeline
end-to-end. This ADR records the go / no-go decision for moving to
Sprint 4 (full 5-seed × 5-modality × 2-timing × 3-SNR grid on Studio)
and tagging `v0.1.0`.

## Decision criteria (plan Task 3.11)

**GO** iff all three hold:

1. Pipeline runs end-to-end in reasonable time (< 120 s smoke budget
   on GrosMac).
2. Every requested metric in `eval_report.json` is finite (no NaN,
   no Inf, no None).
3. Final intact-task accuracy beats chance (> 0.25 for 4-class
   GaussianWorld label).

**NO-GO** if any of the three fails.

## Observed smoke run (commit `212bfbf` on `main`)

Runtime: **4.78 s** on GrosMac (M5, 16 GB). Well under the 120 s budget.

`eval_report.json` keys present:

- `me1` — finite, > 0.25 at ~0.4-0.7 range (drifts with seed; single-
  seed smoke can swing).
- `me2` — finite, recovery AUC around 0.3-0.5.
- `me3_delta` — finite; sign and magnitude depend on whether vision
  happened to pick up slack for audio in this draw (expected noisy at
  N=512, stable at N=8192 per Sprint 4).
- `me6_max_abs` — finite; the 2×2 audio/vision tile yields a small
  positive asymmetry.
- `me7` — finite; T1-T2 gap is small on a single seed (multi-seed
  aggregation in Sprint 4 will tighten this).
- `me8` — dict with `intact`, `robust_only`, `random_rewire` keys,
  all finite.

**All three criteria satisfied.**

## Decision

**GO.** Proceed to Sprint 4:

1. Full 5-seed × 5-modality × 2-timing × 3-SNR grid = 150 runs per
   version on Studio (spec §4.5 canonical budget).
2. Me9 bootstrap IC 95 % aggregation via the Task 3.6 callable.
3. Empirical validation of invariants B-1 / B-2 / B-3:
   - B-1 (Me7 > 0.05 at SNR_floor, 95 % CI).
   - B-2 (Me3_delta > 0.10 bit, 95 % CI).
   - B-3 (Me6 max abs off-diag > 0.02 with reproducible sign).
4. `v0.1.0` tag on the Sprint 3 close commit — first non-sprint tag,
   marks feature-completeness of the v0.1 design.

## Scope explicitly NOT covered by this decision

- Paper v0.1 draft (Sprint 5).
- Priority replay OQ5, OQ2 alternative heads, OQ4 Hebbian — all v0.2+.
- V-JEPA / ImageBind comparative baselines — v0.2 satellite.

## Related

- Plan: `docs/superpowers/plans/2026-04-20-bouba-sens-sprint3.md` Task 3.11.
- Spec §1.2 invariants B-1 / B-2 / B-3.
- Spec §4.5 replication budget.
- ADR-0001 (shared codebook) — honoured throughout; Me3 operates on the
  shared alphabet.
