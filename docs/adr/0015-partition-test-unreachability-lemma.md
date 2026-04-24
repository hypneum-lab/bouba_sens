# ADR-0015 — Unreachability of the 95% threshold for the partition-control test (formal lemma)

**Status:** Accepted
**Date:** 2026-04-24
**Sprint:** 9 (companion to ADR-0014)
**Type:** Methodological — formal result for paper appendix.

## Context

ADR-0014 (Axe 2) established numerically that the pipeline's
`passes_95pct` flag is unreachable on the v0.4 grids even under
positive-control matrix designs. This ADR upgrades that observation
to a **formal lemma** : the unreachability is structural, not
empirical, and holds for **any** modality count n ≥ 4 under the
max-statistic partition test.

This is a paper-appendix-level contribution. It transforms a
methodological caveat into a published combinatorial bound that any
future work using a similar partition-control design can cite.

## Setup

Let n be the number of modalities, k + m = n with k ≥ m the
partition sizes (for v0.4 : n=5, k=3, m=2). Let
P* = (B*, S*) be the pre-registered partition with |B*|=k, |S*|=m.

For an asymmetry matrix A ∈ ℝ^{n×n} (zero diagonal), and a
partition P = (B, S), define the **max-statistic** :

$$\mathrm{Me6}(P; A) = \max_{(i, j) \in K(B, S)} |A_{ij}|$$

where K(B, S) = (B × S) ∪ (S × B) is the cross-block edge set.

Define the **alternative set** 𝒜 = { all (k, m) partitions of [n] }
∖ {P*}, so |𝒜| = C(n, k) − 1.

Define the **mid-rank percentile** of P* (standard convention for
ties) as

$$\rho(P^*; A) = \frac{|\{P \in \mathcal{A} : \mathrm{Me6}(P) < \mathrm{Me6}(P^*)\}| + \tfrac{1}{2} |\{P \in \mathcal{A} : \mathrm{Me6}(P) = \mathrm{Me6}(P^*)\}|}{|\mathcal{A}|}.$$

The pipeline's `passes_95pct` flag fires iff ρ(P*; A) > 0.95.

## Lemma (Mid-rank percentile ceiling)

For all A ∈ ℝ^{n×n} (zero diagonal) and all n ≥ 4 with k ≥ m ≥ 2 :

$$\rho(P^*; A) \;\le\; \frac{1 + \alpha(n, k)}{2}$$

where

$$\alpha(n, k) = \frac{C(n-2, k-2) + C(n-2, m-2)}{C(n, k) - 1}.$$

Numerical verification (script attached as comment in this ADR):

| n  | k  | m  | α      | ceiling |
|---:|---:|---:|-------:|--------:|
| 5  | 3  | 2  | 0.4444 | **0.7222** |
| 6  | 3  | 3  | 0.4211 | 0.7105  |
| 7  | 4  | 3  | 0.4412 | 0.7206  |
| 8  | 4  | 4  | 0.4348 | 0.7174  |
| 10 | 5  | 5  | 0.4462 | 0.7231  |
| 15 | 8  | 7  | 0.4667 | 0.7334  |
| 20 | 10 | 10 | 0.4737 | 0.7368  |

The ceiling stabilises around 72-74 % across modality counts and **never reaches 95 %** for any practical (n, k, m).

## Proof sketch

Step 1 (single hot-pair upper bound is achievable but tight).
The max of |A| over K(B*, S*) is achieved at some pair p* = (i*, j*)
with i* ∈ B*, j* ∈ S*. Set M* = |A_{p*}| = Me6(P*).

Step 2 (which alternatives are forced to tie or exceed P*).
For any alternative P = (B, S), if i* and j* lie on different sides
of P, then p* ∈ K(B, S), so Me6(P) ≥ |A_{p*}| = M* = Me6(P*).
Such alternatives **never** strictly fall below pre-reg.

Step 3 (combinatorial count of "safe" alternatives).
The number of alternatives placing i* and j* on the same side
equals (i* and j* both in B) + (i* and j* both in S) =
C(n−2, k−2) + C(n−2, m−2) — the rest of the (k−2)-subset of B (resp.
the (m−2)-subset of S) is chosen freely from the remaining n−2
modalities.

Step 4 (single-hot-pair design realises this bound).
For the design A with a single non-zero pair |A_{p*}| = 1 and zeros
elsewhere :
- alternatives placing i*, j* on same side achieve Me6(P) = 0
  (their cross set excludes p*) — strictly below P*. Their count
  equals the numerator of α.
- alternatives placing i*, j* on different sides achieve
  Me6(P) = 1 (their cross set includes p*) — tied with P*.

So the strict-less-than fraction = α, the tied fraction = 1 − α
(no alternative is strictly above P* for this design), and the
mid-rank percentile attains ρ = α + (1 − α)/2 = (1 + α)/2.

Step 5 (no design beats the single-hot-pair bound).
Multi-pair designs (any pair set S ⊂ K(B*, S*) with |S| > 1) can
only **decrease** ρ(P*) :
- The "missing" set of P (alternatives that catch zero pairs from S)
  is a strict subset of any single-pair miss set.
- By inclusion-exclusion on the constraint "P places (i, j) on same
  side for all (i, j) ∈ S", this set shrinks monotonically with |S|.
- Therefore the fraction strictly below P* is monotonically
  non-increasing as more pairs are added to S.

Single hot-pair is optimal ; the bound is tight.

QED.

## Corollary

The `passes_95pct` binary flag is **structurally unreachable** by
this max-statistic partition test for any practical modality
count. ADR-0014's verdict that 4/4 grids fail this flag is
**uninformative as evidence against the partition** — a true
architectural partition would also fail because the test design
itself cannot return a positive verdict on small modality sets.

## Implications for the paper

1. **Replace the binary `passes_95pct` reading with a quantitative
   percentile reporting** in the §Results section. Pre-reg's
   mid-rank percentile across the 4 grids — 61.1, 72.2, 55.6,
   44.4 % — should be compared to the **per-(n,k,m) ceiling**
   (72.2 % for n=5), not to the unreachable 95 %.
2. **Cite this lemma in the §Methods section** when introducing
   the partition control test, as the rationale for using
   percentile-relative-to-ceiling rather than absolute 95 %.
3. **The empirical reading of ADR-0014 stands** : pre-reg averages
   58.3 % across grids, against a structural ceiling of 72.2 %,
   which is *not* "near the ceiling" — pre-reg lives in the middle
   of the random-equivalent distribution, not near the top.
4. **Designing the next-generation test (ADR-0014 Axe 10)** : any
   replacement test must either (a) use a non-max statistic (mean,
   sum, weighted sum) that does not have this combinatorial
   ceiling, or (b) work on a strictly larger modality set (e.g.,
   n ≥ 12 with finer partitions), or (c) test partition-resolution
   via a continuous quantity (e.g., variance of Me6 across
   alternatives) rather than a percentile of a single statistic.

## Cross-references

- ADR-0014 Axe 2 — numerical positive-control demonstration of
  the same ceiling.
- ADR-0014 Axe 6 — mid-rank tie-handling correction (without
  which the strict-less-than rank dramatically underestimates ρ).
- Issue #3 (closed) — original ECG null-model finding that
  motivated the whole investigation.
- Issue #4 (closed) — gateway tracking issue, dichotomous
  framework.
- Branch `sprint9/critical-pipeline` commit `858ce51` — pipeline
  on which the empirical results were generated.

## Future work pointers

The bound α(n, k) + (1 − α − 1/(C−1))/2 should appear as Figure
N in the paper appendix, plotted over (n, k) grid with the 95 %
threshold marked as a horizontal line that the surface never
crosses. This visual is the strongest possible argument for
abandoning binary `passes_95pct` reporting in favor of either
(a) ceiling-relative percentile or (b) a re-designed test.
