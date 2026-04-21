# ADR-0006 — Critical validation of v0.3 findings (B-3 null / B-1 bootstrap / B-2 estimator robustness)

**Status:** Accepted
**Date:** 2026-04-21
**Sprint:** 7 (close)

## Context

ADR-0005 recorded preliminary cross-world verdicts on the three pre-registered invariants (B-1 sign-flip, B-2 decay, B-3 cross-world PASS). This sprint stress-tested each against the obvious reviewer objections:

- **Task 7.1** — is B-3 a tautology of 3+2 partition size?
- **Task 7.2** — is the B-1 topology-dependent sign flip above sampling noise?
- **Task 7.3** — is the B-2 Gaussian > XOR > Sinusoid decay Kraskov-specific?

## Verdict table

| Test | Result | Narrative change |
|------|--------|------------------|
| **7.1** B-3 null-model (partition tautology) | **INSUFFICIENT DATA + STATISTIC MISMATCH** (see F1 below) | v0.3 B-3 claim **suspended** pending rerun |
| **7.2** B-1 bootstrap 95 % CI | **DOWNGRADED** — all 3 CIs straddle 0, pairwise non-disjoint | F2 (topology-dependent sign flip) demoted to null result |
| **7.3** Me3 estimator robustness | **DOWNGRADED** — Kraskov + Binning medians = 0.0 ; MINE ordering in noise floor (1e-4 magnitudes) | F3 (Gaussian > XOR > Sinusoid decay) demoted to Kraskov-specific artefact at n=16 |

## F1 — B-3 null-model : insufficient data and statistic mismatch

**Design.** Plan called for 10 random 3+2 partitions × 150-cell Gaussian grid on Studio, comparing pre-reg Me6 median against the empirical null distribution. Target : prereg ≥ 95th percentile.

**Execution.** Only **1 / 10 partitions** (index=0 : `{force, tactile, vision}` vs `{audio, gravity}`) completed before Studio's git checkout drifted to a sibling branch (`feat/b1-plasticity-recovery`) that deleted `scripts/run_null_b3.sh`. Partitions 1-9 exited immediately with *"No such file or directory"*. Part_1 additionally crashed mid-run on a `state_dict` mismatch (new `plasticity_step` buffer not present in sprint7 checkpoint). Artefact : `reports/v0.3_critical_validation/null_b3_part_0_partial.json`.

**Complicating issue — statistic mismatch.** The null-partition grid was aggregated with `scripts/aggregate_grid.py --partition-seed 0 --partition-index 0`, which routes through `me6_max_abs_off_diag_partitioned` (cross-block-only statistic). The v0.2 pre-reg baseline (median Me6 = 0.1484) was aggregated **without** partition flags, which routes through the plain `me6_max_abs_off_diag` (max over **all** off-diagonals). These are different statistics ; the null's 0.1562 cannot be directly compared to the pre-reg's 0.1484.

**Verdict.** F1 is **suspended** — neither confirmed nor falsified. A rerun with :
1. All 10 null partitions on a stable branch with `run_null_b3.sh` present,
2. The pre-reg value re-aggregated via a new `--partition-prereg` flag using the same `me6_max_abs_off_diag_partitioned` statistic,

is required before any B-3 headline claim can survive review. Deferred to Sprint 8.

## F2 — B-1 bootstrap : DOWNGRADED

**Design.** `scripts/bootstrap_me7.py` loads each v0.2 aggregate's `raw_me7_pairs` (75 values per world = 5 seeds × 5 modalities × 3 SNR, minus per-modality-timing attrition) and resamples 10 000 × with scipy's percentile bootstrap to derive a 95 % CI on the median Me7.

**Results** (`reports/v0.3_critical_validation/me7_bootstrap.json`) :

| World | Median Me7 | 95 % CI |
|-------|-----------:|:--------|
| gaussian | -0.0062 | [-0.0125, +0.0125] |
| xor | -0.0063 | [-0.0188, 0.0000] |
| sinusoid | +0.0125 | [-0.0063, +0.0188] |

**Pairwise disjoint matrix : all FALSE.** All three 95 % CIs overlap each other and all three straddle 0. The apparent "Sinusoid positive vs Gaussian / XOR negative" pattern reported in ADR-0005 is below the sampling-noise floor at this grid scale.

**Verdict.** F2 (B-1 topology-dependent sign flip) is **DOWNGRADED to null result**. The hypothesis H-B1 ("critical-period ordering depends on world manifold topology") is not supported by the v0.2 grid.

## F3 — Me3 estimator robustness : DOWNGRADED

**Design.** `scripts/compare_mi_estimators.py` re-aggregated the v0.2 grids under three estimators (Kraskov k-NN — the Sprint 5 default, quantile-binning with Gaussian NB fallback for d>1, and a DV-bound MINE neural estimator at 300 epochs). Median Me3 delta per world, decay ordering check Gaussian > XOR > Sinusoid.

**Results** (`reports/v0.3_critical_validation/mi_estimator_comparison.json`) :

| World | Kraskov | Binning | MINE |
|-------|--------:|--------:|------:|
| gaussian | 0.0 | 0.0 | 1.6e-5 |
| xor | 0.0 | 0.0 | -5.8e-6 |
| sinusoid | 0.0 | 0.0 | -1.8e-4 |

- **Kraskov + Binning : medians = exactly 0.0** across all 3 worlds. The majority of 150 cells return 0 MI at n = 16 probe samples (scalar codes, shape `(16,)`).
- **MINE** : `decay_ordering_holds : true`, but magnitudes are in the **1e-5 to 1e-4 numerical-noise regime**, not a real signal.

**Verdict.** F3 (Gaussian > XOR > Sinusoid decay) is **DOWNGRADED** : the pattern reported in ADR-0004 / ADR-0005 (0.028 / 0.004 / 0.002) is a Kraskov-specific artefact on an n=16 sample × 1-D scalar probe. **No estimator can measure Me3 robustly at this probe scale.** Paper 1 must either increase `probe_batch_size` to > 16 (ideally ≥ 128 per cell) or drop Me3 from the core invariant set.

## Decision

**The v0.3 headline narrative "B-3 world-agnostic PASS + B-1 directional falsification + B-2 decay pattern" is no longer defensible as three independent findings.** Effective post-Sprint-7 state :

- **F1 (B-3)** — suspended pending rerun with apples-to-apples statistic.
- **F2 (B-1)** — null result at the v0.2 grid scale.
- **F3 (B-2)** — estimator artefact at the v0.2 probe scale.

Paper 1 draft (Sprint 8) must lead with **Sprint 7's methodological findings** rather than the suspended B-3 claim :

1. **Probe-batch-size matters for Me3.** At n=16, the Kraskov kNN estimator masks zero signal as a non-zero measurement. This is a reproducible, actionable recommendation for the field.
2. **Bootstrap CIs are mandatory for B-1 claims.** Single-point medians with effect sizes ~0.01 on a 5 × 5 × 3 grid are indistinguishable from noise.
3. **Partition-tautology controls need to precede any asymmetry claim.** The partitioned vs full Me6 statistic mismatch (F1) must be resolved before any B-3 publication.

These methodological findings are **more valuable** than the original three findings because they force a correction in how the field measures cross-modal plasticity invariants.

## Pre-registration fidelity

- **No threshold changes** vs ADR-0003 / 0004 / 0005. The 0.05 / 0.10 / 0.02 pre-reg thresholds are unchanged.
- **No metric-implementation changes** to `me3_delta`, `me6_*`, `me7_congenital_gap`. Only validation was added.
- The downgrades of F2 and F3 are honest null / artefact results, not p-hacking.

## Artefact SHA256s (Sprint 7 reproducibility)

| File | SHA256 |
|------|--------|
| `reports/v0.3_critical_validation/me7_bootstrap.json` | `0473a9c1eb4ab5a27e28598ed01e123e40afc593dcb812386b653148670a787d` |
| `reports/v0.3_critical_validation/mi_estimator_comparison.json` | `0699b6c23dad9bf5f923c454ce22c15c31c8f6fa2ee0f64e3a9b0b24924821fe` |
| `reports/v0.3_critical_validation/null_b3_part_0_partial.json` | `c3cf33a0c6251888e43be1a3eef47d40d618ea52e422d6a148246acc81529612` |

Reproduction commands in `reports/v0.3_critical_validation/MANIFEST.md`.

## Next steps (Sprint 8)

1. **Complete F1 validation** — rerun the 9 remaining null-partition grids on a clean sprint7 checkout ; add `--partition-prereg` flag to the aggregator for apples-to-apples comparison.
2. **Increase probe batch size** from 16 to ≥ 128 in `AdaptationLoop.lesion_phase` ; rerun v0.2 grid ; re-do Task 7.3 comparison.
3. **Paper 1 v0.1 draft** — lead with methodological findings (Probe batch, Bootstrap CIs, Partition-tautology) ; move the B-3 claim to Discussion pending F1 rerun.
4. **Defer Sprint 8 tag `v0.5.0`** until F1 either confirms or definitively falsifies B-3.

**This ADR closes Sprint 7 with honest null findings. The 2026-04-20 narrative that the programme had "three replicable empirical findings" is retracted ; the programme has one methodology paper's worth of findings, pending a rerun.**
