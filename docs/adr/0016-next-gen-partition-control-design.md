# ADR-0016 — Next-generation partition-control test design (3 candidates)

**Status:** Proposed (Sprint 10 spec gateway)
**Date:** 2026-04-24
**Sprint:** 9 → 10 transition
**Companion to:** ADR-0014 (verdict), ADR-0015 (unreachability lemma).

## Context

ADR-0015 proved that the current max-statistic partition control on
n=5 modalities has a structural ceiling on the mid-rank percentile
of pre-reg at ≈72.2 %, well below the binary 95 % threshold. The
test cannot return a positive verdict by design. Replacing it is a
prerequisite for any future B-3-style claim to have empirical
content.

This ADR specifies three candidate replacement designs, scores them
against five criteria, and recommends one for Sprint 10 prototype.

## Scoring axes

| Axis | Why it matters |
|---|---|
| **Theoretical max percentile** | Must reach > 95 % under positive control, otherwise we re-create the ADR-0015 problem. |
| **Implementation cost** | Scope-bounded ; Sprint 10 has a budget of ~1 week dev. |
| **Backward compatibility** | Can the new test re-aggregate the existing 4 grids' raw cell outputs, or does it require fresh runs ? Fresh runs cost Studio compute. |
| **Scientific interpretability** | Reviewers should be able to read the test outcome as a clean partition vs no-partition statement. |
| **OSF amendment cost** | Pre-registered statistics are part of the OSF lock ; changing the statistic requires an amendment with explanation. |

## Candidate A — Increase modality count to n ≥ 12

Replace the 5-modality benchmark with a 12+-modality benchmark
(e.g., 12 = `audio` + 5 vision sub-bands + 3 tactile sub-types
+ `gravity` + `force` + 1 vestibular). Partition becomes (k, m)
with k + m = 12, e.g., (7, 5) or (8, 4). Number of unique
alternatives explodes from 9 (n=5) to C(12, 7) − 1 = 791.

**Theoretical max percentile** : the structural ceiling formula
α(n=12, k=7) = (C(10, 5) + C(10, 3)) / (C(12, 7) − 1) =
(252 + 120) / 791 = 0.470. Ceiling = (1 + 0.470) / 2 = **73.5 %**
under max-statistic — STILL well below 95 %. **Increasing n alone
does not fix the structural problem** ; the ceiling is a
property of the max statistic on the partition graph, not of n.

**Implementation cost** : extreme. Requires rebuilding the
benchmark architecture, the lesion module, all 9 grids, and the
cell aggregator. Multi-month effort.

**Backward compatibility** : zero ; all existing grids become
non-comparable.

**OSF amendment cost** : major (changing the modality set
fundamentally restructures the pre-registered claims).

**Verdict** : ❌ Not recommended. Doesn't fix the ceiling
problem and costs the most.

## Candidate B — Replace max with mean (or sum) over cross pairs

Define `Me6_mean(P; A) = mean_{(i,j) ∈ K(B,S)} |A_{ij}|` and use
this as the partition test statistic instead of `max`.

**Theoretical max percentile** : the ceiling argument from
ADR-0015 relied on the max being achieved at a single pair, which
forces ≥ 5 alternatives to tie. The mean averages over all km
cross pairs, so it is sensitive to the *distribution* of high-
asymmetry pairs across the partition, not just the single hottest
one. Numerical verification required, but the ceiling under mean
should be substantially closer to 100 % when A has multiple high-
asymmetry pairs concentrated in K(B*, S*).

Quick analytical sketch : if A has all 6 pre-reg cross pairs at
value M and 0 elsewhere, mean(P*) = M. For an alternative P
sharing s pre-reg cross pairs in its own cross set (s ∈ 0..6),
mean(P) = s × M / km. The s value varies from 0 to 6 across
alternatives — pre-reg dominates strictly when s < 6, which can
hold for many alternatives. **Likely passes 95 % with a sharp
positive control.**

**Implementation cost** : low. Add `me6_mean_off_diag_partitioned`
to `src/bouba_sens/metrics/asymmetry.py` (≈30 LOC including
docstring + tests). Update `aggregate_grid.py` to optionally
report `me6_mean` alongside `me6_max_abs`. Update
`critical_validation_pipeline.sh` to invoke a `--statistic mean`
flag.

**Backward compatibility** : full. Existing grids' per-cell
outputs contain enough info to re-aggregate under the new
statistic without re-running cells.

**OSF amendment cost** : moderate. Add `Me6_mean` as a new
pre-registered statistic alongside the old `Me6_max` ; the old
statistic stays for continuity.

**Verdict** : ✅ **Recommended primary**. Cheap, backward-
compatible, theoretically promising, low-friction OSF
amendment.

## Candidate C — Permutation test on modality labels (not partition)

Instead of testing pre-reg vs random partitions, test the full
matrix A vs the matrices obtained by **permuting the modality
labels** themselves. Under the null hypothesis that no specific
modality grouping matters, modality labels are exchangeable, and
the observed Me6_partitioned should not stand out vs the
distribution under permutations.

**Theoretical max percentile** : permutation tests have well-
established statistical properties — under H0, the observed
statistic is uniformly distributed in rank, so percentile ≤ 100 %
is naturally achievable with enough permutations (e.g., 1000).

**Implementation cost** : moderate. Need to permute the 5
modality labels and re-compute the per-cell asymmetry matrix
under each permutation, which requires re-running each cell's
adaptation loop with permuted modality assignments. Substantially
more expensive than candidates A or B.

**Backward compatibility** : zero ; permutation needs cell-level
re-execution with shuffled modality assignments.

**OSF amendment cost** : substantial. The pre-registered protocol
runs the canonical 5 modalities ; permutation is a different
experimental setup entirely.

**Verdict** : ⚠️ Defensible methodology but expensive. Reserve
for Sprint 11+ if Candidate B's empirical results suggest more
power is needed.

## Recommendation

**Sprint 10 implementation order** :

1. **Candidate B (mean statistic)** — first.
   Implement `me6_mean_off_diag_partitioned` + pipeline flag,
   re-aggregate the 4 existing grids under the new statistic,
   produce ADR-0017 verdict. Decision gate: does B pass the
   95 % threshold under positive-control validation ? If yes,
   B becomes the primary partition control test. If no, escalate
   to Candidate C.

2. **Candidate C (permutation test)** — Sprint 11 conditional.
   Only if B fails the positive-control validation. Larger spec,
   needs OSF amendment, needs cell-level re-execution.

3. **Candidate A (n≥12)** — explicit no-go. Doesn't fix the
   ceiling problem ; costs the most. Documented for completeness.

## Acceptance criteria for Candidate B (Sprint 10 ADR-0017)

- [ ] `me6_mean_off_diag_partitioned(P, modalities, partition)` in
  `src/bouba_sens/metrics/asymmetry.py` with unit tests covering
  empty partition, full partition, and the 9 unique 3+2 alternatives.
- [ ] `aggregate_grid.py` extended with `--statistic {max,mean}`
  flag (default `max` for backward compat).
- [ ] `critical_validation_pipeline.sh` extended with `--statistic`
  flag.
- [ ] Re-aggregation of 4 v0.X exhaustive grids under mean ;
  results table in ADR-0017.
- [ ] Positive-control validation : single-hot-pair design + uniform-
  cross design + sparse-block design. Report mid-rank percentile
  for each ; verify > 95 % achievable on at least one design.
- [ ] OSF amendment filed (`docs/osf/amendment-v0.6-mean-statistic.md`)
  before Sprint 10 release tag.

## Cross-references

- ADR-0014 — verdict that motivated the redesign.
- ADR-0015 — Lemma proving the redesign is mathematically required.
- Issue #1 — open Sprint 5 v0.2 closeout (broader context).
- Issue #4 (closed) — gateway tracking for ADR-0014.
