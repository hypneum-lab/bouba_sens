# Q3 — bouba_sens dose-response seed robustness

**Date pre-registered:** 2026-05-10 (BEFORE any seed sweep run)
**Spec source:** HYPNEUM-PLANS/specs/2026-05-10-three-innovation-experiments-design.md
**Plan source:** HYPNEUM-PLANS/2026-05-10-niveau8-three-experiments.md (Task 1)
**Status:** Pre-registered, not yet executed.

## H0 (to refute)

The B-1 peak observed at LOCK_AFTER=100 in `reports/v0.5_amedi_curve.png`
(single-seed) is seed-stable: the non-monotone B-1 reproduces with ≥4/5
seeds, peak position drifts ≤25% (peak ∈ [LOCK_AFTER=75, LOCK_AFTER=125]).

## Methodology

- 5 seeds: 0, 17, 42, 73, 101 (deliberate prime/non-prime mix to avoid
  hidden RNG correlations)
- Per-seed: re-run dose-response generator (`scripts/plot_amedi_curve.py`
  adapted), measure peak position + height + full curve
- Aggregate: median peak position across 5 seeds, IQR (25th-75th
  percentile)
- Statistical test: Jonckheere-Terpstra one-tailed for non-monotonicity
  multi-seed
- Multiple-comparisons correction: Bonferroni for 5 seeds (matches OSF
  pre-reg style)

## Decision criteria (pre-stated)

- **Survives:** peak stable @ LOCK_AFTER=100 ± 25% AND Jonckheere p<0.05
  → §5.5 claim survives, add "median over 5 seeds, IQR=[X,Y]" caveat
  to figure caption + prose
- **Reframe:** peak drifts >25% but Jonckheere still p<0.05 →
  reformulate §5.5 from "peak at LOCK_AFTER=100" to "peak in [50%, 75%]
  of STEPS_TRAIN, position seed-dependent" ; figure becomes "median
  curve with seed-IQR shading"
- **Retract:** Jonckheere p≥0.05 → claim Amedi non-monotonicity
  rétractable, ADR-0019 needed, §5.5 reframes to "non-monotone
  signature requires more N to confirm"
