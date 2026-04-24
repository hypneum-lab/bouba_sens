# ADR-0018 — Partition-test framework exhausted on n=5 ; new directions

**Status:** Accepted — **methodological closure**
**Date:** 2026-04-24
**Sprint:** 11
**Companion to:** ADR-0014, 0015, 0016, 0017.

## Context

ADR-0017 rejected Candidate B (mean statistic) on real data and
escalated to Candidate C (permutation test on modality labels) per
ADR-0016's escalation clause. Before implementing C, this ADR
performs a sanity check on the proposal — and finds that
**Candidate C as defined is mathematically equivalent to the
partition control already executed in ADR-0014**, contributing no
new statistical information.

## The equivalence (formal)

Setup. Let A ∈ ℝ^{n×n} be the asymmetry matrix, n=5. Let
P* = (B*, S*) with |B*|=k=3, |S*|=m=2 be the pre-registered
partition. Let Sym(B*) × Sym(S*) be the partition stabilizer of
size k!·m! = 12.

**Method 1 (partition control, ADR-0014)** — for each P ∈
{all C(n,k) partitions} = 10 distinct partitions, compute T(A, P).

**Method 2 (Candidate C — permute modality labels)** — for each
π ∈ Sym([n]) (size 5! = 120), compute T(π·A·π^T, P*) at the fixed
pre-reg partition.

**Lemma**. The multiset of values {T(π·A·π^T, P*) : π ∈ Sym([n])}
equals the multiset {T(A, P) : P ∈ {C(n,k) partitions}} with each
value repeated exactly |Stab| = k!·m! = 12 times.

**Proof**. Permuting rows+cols of A by π and then evaluating T at
fixed (B*, S*) reads A's entries at positions
(π^{-1}(B*), π^{-1}(S*)) — this is T(A, π^{-1}(P*)). The orbit
of P* under Sym([n]) is the full set of (k, m) partitions ; the
orbit-stabilizer theorem gives |orbit| = 120/12 = 10 = C(n,k).
Each partition is hit exactly 12 times. ∎

**Numerical verification** (`/tmp/perm_equivalence_check.py`,
random seed 2026): for both max-stat and mean-stat,
all 10 distinct partition values appear in the 120-permutation
distribution with exact ×12 multiplicity ; ✓ checks across both
statistics.

## Implication

The partition-test framework on n=5 modalities with any
permutation-symmetric statistic (max, mean, median, sum, ...)
collapses to **at most C(n,k) = 10 distinct values** in its null
distribution. With n=5, this gives a coarse 10-bin null
regardless of how many "permutations" the test nominally takes.
Combined with the ADR-0015 ceiling lemma, this means :

- The structural ceiling (1 + α)/2 ≈ 72.2 % is invariant under
  any choice of permutation-symmetric statistic.
- Adding more samples (re-runs, bootstrap, more cells) does not
  increase the resolution of the null distribution beyond 10
  bins.
- The pre-reg partition gets at most 1 of those 10 bins ; the
  test is information-theoretically capped at log_2(10) ≈ 3.3
  bits regardless of empirical effort.

**The partition-test framework on n=5 is exhausted.** Sprint 11
cannot make the partition reading of B-3 statistically defensible
without leaving the framework.

## Real paths forward (4 candidates)

### Path D — Bootstrap CI on per-cell percentile estimates (cheap, narrow value)

Instead of computing a single percentile per grid (which inherits
the 10-bin coarseness), compute a percentile **per cell** (each
of the 30 trios) and bootstrap a 95 % CI on the per-cell
percentile distribution. Gives a confidence interval rather than
a point estimate.

**Cost**: ~1 day implementation + analysis. Low Studio compute
(re-aggregation only).
**Value**: addresses the discrete-percentile noise (ADR-0014 v2)
but does NOT escape the (1+α)/2 ceiling. Useful for paper §8 to
report "pre-reg percentile = 58.3 % ± CI [...]" rather than
"pre-reg percentile = 58.3 %". Strictly improves rigor.

### Path E — Abandon partition test, double down on entropy reframing (zero cost, full closure)

Accept that the partition framework cannot resolve the question on
n=5 ; commit fully to the entropy-proxy framing already installed
in paper §3.3 / §4.4 (commit `6599c99`). The B-3 magnitude is a
per-modality entropy diagnostic, not a partition-distinguishing
statistic. Paper §8 stops apologising for the failed partition
test and just reports : "we ran the partition control under max,
mean, and 120-permutation framings ; all are equivalent on n=5
and all give pre-reg in the 22-72 % percentile range, well below
the structural ceiling of 72 % derived in ADR-0015. The signal
captured by Me6 raw magnitude is per-modality entropy, not
partition structure."

**Cost**: zero new compute, ~30 min paper edit to add the
"framework exhausted" note as a methodological boundary.
**Value**: maximum honesty + minimum residual debt. Accepts the
empirical reality.

### Path F — Bayesian model comparison (different framework, expensive)

Posterior odds of "partition matters" vs "doesn't" given the
observed asymmetry matrix and a prior over partition structures.
Sidesteps the ceiling because it doesn't use percentiles.

**Cost**: substantial — needs prior elicitation, likelihood
specification, posterior computation. Multi-week effort to
design properly. Reviewer bait if not done by someone who
publishes regularly in Bayesian model comparison.
**Value**: genuinely new framework but high risk-reward.

### Path G — Increase modality count to n ≥ 12 (ruled out)

Already ruled out in ADR-0016 Candidate A (cost too high, doesn't
fix the structural problem per ADR-0015 numerical verification).
Listed here for completeness.

## Recommendation

**Combo Path D + Path E.**

- Path E first (~30 min) : update paper §3.3 / §4.4 with the
  "framework exhausted" methodological boundary note.
  Acknowledges the limit cleanly without inventing new tests.

- Path D second (~1 day) : bootstrap CI on percentile estimates.
  Goes into the paper as a rigour upgrade : "pre-reg percentile
  CI [low, high]" instead of point estimates. Useful rebuttal
  ammunition against reviewers who ask for confidence intervals.

- Path F deferred to a future cycle when the broader research
  program has appetite for a Bayesian methodology paper as a
  separate output.

- Path G permanently shelved.

## Sprint 11 deliverables (concrete)

1. ADR-0018 (this file) — committed, closes Sprint 11 methodology
   investigation.
2. Paper §8 addendum (Path E) — short paragraph in §8 noting the
   partition-test framework exhaustion, citing ADR-0015 +
   ADR-0017 + ADR-0018 as the convergent evidence.
3. Bootstrap CI implementation (Path D) — separate sub-sprint,
   tracked as Sprint 11.D.

## Cross-references

- ADR-0014 — verdict from partition control (4/4 fail).
- ADR-0015 — Lemma proving max-stat ceiling.
- ADR-0016 — design proposing Candidates B + C.
- ADR-0017 — Sprint 10 verdict rejecting B.
- This ADR (0018) — Sprint 11 closure on partition framework.
- Verification script `/tmp/perm_equivalence_check.py` — formal
  numerical proof of Method 1 ↔ Method 2 equivalence.
