# OSF pre-registration amendment — bouba_sens v0.8 (subspace shift vs LOCK_AFTER)

**Status:** DRAFT — not yet filed on OSF (to be filed by the PI)
**Parent pre-registration:** `dream-of-kiki` OSF DOI `10.17605/OSF.IO/Q6JYN` (unchanged by this draft)
**Previous amendment on file:** `amendment-v0.6-sprint10-17.md`
**Sibling drafts (same date, filed independently):** `amendment-v0.7-tost-equivalence-2026-10-23.md`, `amendment-v0.9-decorrelation-window-2026-10-23.md`
**Amendment tag on file:** `bouba_sens/v0.8-subspace-shift` (to be created at filing)
**Draft date:** 2026-10-23
**Pre-registration id:** to be assigned by the PI (next free N-id in the HYPNEUM-PLANS series)

> **Filing note.** This is the machine-readable draft. Nothing in this
> file has been run. The metric does not exist in the code base yet; its
> implementation and unit tests must be merged and frozen (commit hash on
> the OSF form) before any confirmatory seed runs. Pilot seeds (§7) may
> run before filing.

---

## 1. What is being amended

The amendment adds **one new outcome metric**, a representation
subspace shift after late lesion, and **one new confirmatory
hypothesis**: the shift shows an increasing ordered trend with the
critical-period lock time `LOCK_AFTER` (higher levels tend to have
higher within-seed ranks). The test does not establish strict
monotonicity (§6). No existing threshold, metric or
verdict changes.

This is **not** a rescue of the retracted §5.5 claim. §5.5 claimed a
*peaked* (inverted-U) dose-response of the *performance* gap Me7. Here
the outcome is a *geometric* quantity, the prediction is an
*increasing ordered trend*, not a peak,
and Me7 plays no role in the decision. The ADR-0019 retraction stands
whatever the outcome.

## 2. Motivation

Sui, Tan, Zhang, Khan, Hu & Chen, "Do More Modalities Always Help? A
Geometric Perspective on Missing-Modality Robustness", NeurIPS 2026
(arXiv:2610.04792), report that multimodal training induces structured
rotations of principal parameter subspaces, mainly in cross-modal
interaction layers. These rotations go with larger representation harm
when a modality is missing at inference. In bouba_sens, `LOCK_AFTER` is
the step at which the cross-modal constellation (the interaction
component) is frozen during full-modality Phase 1 training. A later
lock leaves the interaction component plastic for longer under
multimodal training. Sui et al. would therefore predict stronger
cross-modal dependency and a larger representational reorganisation
when one modality is later lesioned.

Two lessons from Q3/Q3+ shape the design:

- A Jonckheere-type test was the wrong instrument for a peaked
  hypothesis (critic v2, ADR-0019). Here the hypothesis is an
  increasing ordered trend, and the test is a repeated-measures ordered
  trend test.
- A single seed was misleading. The design uses 20 seeds as blocks.

The direction is derived by analogy: Sui et al. study parameter
subspaces in other architectures, and bouba_sens has not been tested.
A significant trend in the opposite direction is reported as an
exploratory contrary finding (§6).

## 3. Metric definition (pre-registered)

For a T2 cell c = (seed s, lock ℓ, lesioned modality m, SNR level f):

1. **Probe.** A fixed clean probe X of N = 512 samples, drawn as in
   `AdaptationLoop.lesion_phase` (`world.sample(512, seed = s − 99_999)`).
   The probe is decoupled from the training batch size, which stays 16.
2. **Codes.** Z_pre ∈ ℝ^{512×896}: full fused representation
   (`probe_pool="full"`) of X through the Phase-1 checkpoint at Phase-2
   entry. Z_post: the same probe, presented clean on all channels,
   through the network after the 100-step lesion phase. Both are already
   captured as `pre_lesion_codes` / `post_lesion_codes`.
3. **Subspaces.** Column-centre each matrix. V_pre, V_post ∈ ℝ^{896×k}
   are the top-k right singular vectors, i.e. the top-k principal
   directions in feature space, with **k = 12**. This is the effective
   dimension of the Me3 probe measured in ADR-0019 (participation ratio
   11.7, two-NN 12.4).
4. **Principal angles.** θ₁…θ_k = arccos of the singular values of
   V_preᵀ V_post.
5. **Shift.** S(c) = (1/k) Σᵢ sin² θᵢ = 1 − ‖V_preᵀ V_post‖²_F / k ∈
   [0, 1]. This is the normalised squared projection (chordal) distance
   on the Grassmannian Gr(k, 896).
6. **Training-drift control.** C(s, ℓ) is S computed for the *same*
   Phase-1 checkpoint after 100 Phase-2 steps with **no lesion** (all
   channels clean, same replay settings, same data seeds). This is arm
   (a) "intact" of the design spec. It removes the shift that continued
   training alone produces.
7. **Lesion-specific shift.** ΔS(c) = S(c) − C(s, ℓ).
8. **Per-block outcome.** Y(s, ℓ) = median of ΔS over the 15 T2 cells of
   (s, ℓ) (5 lesioned modalities × 3 SNR levels).
9. **Sampling-noise floor.** S_split(s, ℓ) is S between the top-k
   subspaces of two disjoint random halves (256 / 256, split seed = s)
   of Z_pre. It is shared by the 15 cells of (s, ℓ), reported, and used
   in the magnitude guard (§6).
10. **Rank guard and exclusion rules** (§3.1).

### 3.1 Rank guard, exclusion denominators and incomplete blocks

A matrix *passes the rank check* if its column-centred numerical rank
(singular values ≥ 1e-6 × the largest) is at least k. The check
applies to every matrix that enters S: Z_pre (one per (s, ℓ)), Z_post
(one per cell), the no-lesion control's end matrix (one per (s, ℓ)) and
both 256-row split halves of Z_pre.

- **Non-finite codes** (NaN/inf in any matrix) or a crash are
  *technical failures*: the whole seed is replaced from the reserve
  list (§7). Everything else below is an *outcome-dependent exclusion*
  and is **never** replaced, because rank deficiency is a property of
  the trained network, not of the infrastructure.
- **Cell validity.** A T2 cell c is valid if Z_pre, Z_post(c), the
  control matrix of its (s, ℓ) and both split halves pass. If Z_pre or
  the control fails, all 15 cells of that (s, ℓ) are invalid.
- **Block value.** Y(s, ℓ) is defined only if at least 12 of its 15
  cells are valid; it is then the median over the valid cells.
- **Complete seed block.** A seed is a complete block if Y(s, ℓ) is
  defined at all 7 levels. The Page test, the magnitude guard and S-2/S-3
  use complete blocks only, so both endpoint levels (ℓ = 25 and 175) are
  defined for every analysed seed.
- **Protocol failure.** If fewer than 80 % of the N confirmatory seeds
  are complete blocks (fewer than 16 of 20, or 24 of 30), the
  experiment is reported as a protocol failure with no verdict.
- **Reporting.** The number of invalid cells and of undefined Y is
  reported per level ℓ, with denominators (N × 15 cells and N blocks
  per level). If exclusions differ across levels, the paper states that
  the selection of complete blocks may itself depend on ℓ.

T1 cells are excluded from the primary metric: their pre-lesion codes
come from an untrained network, so S would measure training from
scratch, not lesion reorganisation.

**Secondary metric (exploratory, closer to Sui et al.):** a
parameter-subspace rotation. For every 2-D weight matrix of
`CrossModalNerve` (the cross-modal interaction module), take principal
angles between the top-k right singular subspaces at Phase-2 entry and
Phase-2 end, with the same no-lesion control and median aggregation.
k = min(12, rank). The exact list of parameters is fixed in code before
filing (§10).

## 4. Conditions

- World: `StudyforrestRealWorld` (4.5-modal real bridge), as in Q3/N12.
- `STEPS_TRAIN=200`, `STEPS_LESION=100`, batch 16, HARD gating, no
  codebook lock, no gating schedule.
- **LOCK_AFTER ℓ ∈ {25, 50, 75, 100, 125, 150, 175}** (7 ordered
  levels). All are below `STEPS_TRAIN`, so every T2 cell enters Phase 2
  with a frozen constellation, and only the time of freezing varies.
- Reference condition outside the trend test: LOCK_AFTER unset (never
  locked). It is reported descriptively.
- Per seed: 7 Phase-1 checkpoints, 7 × 15 T2 lesion runs, 7 no-lesion
  control runs.

## 5. Hypotheses

| Id | Statement | Test |
|----|-----------|------|
| **S-1** (primary) | Y(s, ℓ) shows an increasing ordered trend in ℓ | Page's L trend test, one-sided (increasing), blocks = seeds, 7 ordered treatments, α = 0.05 |
| S-2 (exploratory) | Lesion-specific reorganisation exists: mean over seeds of Ȳ(s) = mean_ℓ Y(s, ℓ) is > 0 | one-sample t-test, one-sided; nominal p, no decision |
| S-3 (exploratory) | Per-seed rank association | per-seed Spearman ρ(ℓ, Y); Wilcoxon signed-rank on ρ > 0, one-sided; nominal p, no decision |
| S-4 (exploratory) | ΔS tracks performance harm | Spearman between cell ΔS and the cell's T2 Me1 drop; no decision |
| S-5 (exploratory) | Parameter-subspace rotation shows the same trend | same Page test on the secondary metric; no decision |

Only S-1, with its magnitude guard, is confirmatory. S-2 to S-5 are
exploratory: they are reported with nominal p-values, labelled
exploratory, and cannot support a claim on their own or replace S-1.

**Page test, frozen method.** L = Σ_ℓ j(ℓ) · R_ℓ, where R_ℓ is the sum
over complete blocks of the within-seed rank of Y(s, ℓ) among the 7
levels, j(ℓ) = 1…7 in increasing ℓ, and ties get mid-ranks. The p-value
is a **within-block permutation** test: the 7 ranks of each seed are
permuted independently, 100 000 permutations, generator seed 20261023,
p = (1 + #{L_perm ≥ L_obs}) / (1 + 100 000). Validity rests on the null
hypothesis that, within each seed, the 7 values are exchangeable
across levels (no level effect); seeds are independent blocks. Under
that assumption the test is exact up to Monte Carlo error and valid with
ties.

*What a significant L shows.* A small p-value shows that higher levels
tend to receive higher within-seed ranks. It does not show that every
adjacent step increases: a profile rising over six levels and falling
at the last can still give a very small p. Claims are therefore worded
as an "increasing ordered trend", never as "monotone". To describe the
shape, the paper reports (no decision) the median over seeds of each
adjacent difference Y(s, ℓ_{j+1}) − Y(s, ℓ_j) with its 90 % bootstrap
CI. Reference implementation:
`page_perm_pvalue` in `scripts/power_prereg_2026_10.py`. The
asymptotic and (if there are no ties) exact `scipy.stats.page_trend_test`
p-values are reported alongside; the permutation p-value decides.

Sensitivity analyses (reported, no decision): k ∈ {6, 24}; SNR floor
cells only; per lesioned modality.

*Subspace stability (reported, no decision).* For every Z_pre and Z_post
the eigengap ratio λ₁₂ / λ₁₃ of the centred covariance is reported. A
ratio below 1.1 marks a top-12 subspace that is unstable to small
perturbations; the share of such matrices is reported per level, and
the S-1 test is repeated on blocks without them as a sensitivity
analysis.

## 6. Decision criteria (pre-stated)

The verdicts are exhaustive and mutually exclusive, evaluated in
this order:

- **Protocol failure:** fewer than 80 % complete blocks (§3.1). No
  verdict.
- **Supported:** S-1 permutation p < 0.05 **and** the median over
  complete blocks of [Y(s, 175) − Y(s, 25)] exceeds the median S_split(s, ℓ)
  over the complete blocks and all 7 levels. This is a **heuristic
  magnitude threshold**, not a calibrated noise test: the half-sample
  pre/pre distance is not the sampling distribution of the full-sample,
  control-subtracted endpoint contrast. The guard is part of one
  conjunctive rule; the Page p-value alone is not advertised as a
  separate finding.
- **Detectable but sub-floor:** S-1 p < 0.05, magnitude guard fails.
  Reported as "an increasing ordered trend was detected, but its
  endpoint size does not exceed the heuristic half-sample threshold"; no
  substantive geometric claim.
- **Not demonstrated:** S-1 p ≥ 0.05. Reported as "an increasing
  ordered trend of the subspace shift with LOCK_AFTER was not demonstrated at N"; this
  is not evidence that no trend exists.

Independently of the verdict, the mirrored (decreasing) Page test is
reported as an exploratory result. If its p < 0.05, the paper calls it
an exploratory contrary trend; the confirmatory direction was
increasing.

Paper consequence: if Supported, the paper may state a pre-registered
increasing ordered trend of a geometric quantity with lock timing,
above a heuristic magnitude threshold. It is a new result, not a
B-1 rescue: §5.5 stays retracted and Me7 is not re-interpreted. All
outcomes are publishable.

## 7. Seeds, power and stopping rule

- **Pilot seeds:** 2101–2105, extendable in order (2106, 2107, …) under
  the rule of the power paragraph below. They are used to validate the pipeline,
  check the rank guard and estimate the within-seed SD of Y. Pilot data
  are excluded from every confirmatory test.
- **Confirmatory seeds:** the first N of 2001, 2002, …, 2040 in order
  (N = 20: 2001–2020), disjoint from every seed used so far (0–4,
  Q3/Q3+, N12) and from the v0.7 / v0.9 drafts. **Reserve:** the four
  seeds that follow the last confirmatory seed (N = 20: 2021–2024), for
  technical replacement only.
- **Power (no pilot data yet).** `scripts/power_prereg_2026_10.py`,
  one-sided Page test (asymptotic for the component table; the
  complete-rule table uses the permutation critical value,
  `page_perm_crit`), 7 levels, linear trend
  with a seed random intercept of 1 SD. Results are in units of the
  within-seed SD of Y.

  *Component-test power only* (Page test alone):

  | N | range 0.5 SD | range 1.0 SD | range 1.5 SD |
  |--:|-------------:|-------------:|-------------:|
  | 10 | 0.36 | 0.80 | 0.98 |
  | 20 | 0.58 | 0.98 | 1.00 |
  | 30 | 0.72 | 1.00 | 1.00 |

  For the Page test alone, N = 20 reaches 0.80 at a total trend of about
  **0.70 within-seed SD** (0.75 at 0.65 SD, 0.81 at 0.70 SD); N = 30
  reaches it at about 0.60 SD.

  *Complete rule* (P(Supported): Page p < 0.05 **and** median endpoint
  difference > guard g, with g = median S_split expressed in
  within-seed SD units; each seed block independently lost to rank
  exclusion with probability q, no replacement; protocol failure if
  < 80 % complete blocks):

  | N | q | range | g = 0 | g = 0.25 | g = 0.5 | g = 1.0 | P(protocol failure) |
  |--:|--:|------:|------:|---------:|--------:|--------:|--------------------:|
  | 20 | 0 | 0.75 SD | 0.84 | 0.81 | 0.69 | 0.25 | 0.000 |
  | 20 | 0 | 1.0 SD | 0.97 | 0.96 | 0.90 | 0.50 | 0.000 |
  | 20 | 0 | 1.5 SD | 1.00 | 1.00 | 1.00 | 0.90 | 0.000 |
  | 20 | 0.10 | 1.0 SD | 0.92 | 0.90 | 0.84 | 0.48 | 0.042 |
  | 30 | 0 | 1.0 SD | 1.00 | 0.99 | 0.94 | 0.51 | 0.000 |
  | 30 | 0.10 | 1.0 SD | 0.97 | 0.96 | 0.91 | 0.49 | 0.024 |

  These numbers are **conditional on the simulation model**: a linear
  trend, Gaussian noise, a guard g treated as a fixed known constant,
  and rank exclusions independent of the outcome and of ℓ. None of these
  is verified. The guard dominates: when g is as large as the expected
  endpoint difference, P(Supported) is about 0.5 whatever N. Exclusions
  also cap the attainable power: with a per-block exclusion probability
  of 0.20, the protocol-failure probability stays near 0.4 even at
  N = 40, so 0.80 unconditional power is not reachable.

  *Drafted rule for N.* From the pilot seeds (§7, at least 5): σ̂ =
  residual SD of Y from the two-way ANOVA without replication (seed ×
  level) fitted on the **complete** pilot blocks only (df = (n_c − 1) ×
  6 for n_c complete blocks; at least 3 complete blocks are needed,
  otherwise the pilot is extended with seeds 2106, 2107, … until 3 are
  complete), inflated to its one-sided UCL80 (χ², same df); ĝ = pilot
  median S_split / σ̂. The planning exclusion rate q̂ is the exact
  (Clopper–Pearson) one-sided **80 % upper bound** of the pilot
  incomplete-block fraction, not the point estimate: 0/5 incomplete
  gives q̂ = 0.275, 0/10 gives 0.149, 0/20 gives 0.077. Because a large
  q̂ makes the target unreachable (q = 0.20 already gives a 0.41
  protocol-failure probability at N = 40), the PI may extend the pilot
  to tighten q̂ before computing N. N is the smallest N ≥ 20 for which
  the simulated P(Supported), **minus two Monte-Carlo SEs**, is ≥ 0.80,
  with at least 100 000 draws per candidate, the permutation-calibrated
  Page test and a total trend of ĝ + 0.70 SD units.
  If no N ≤ 40 qualifies (for example because q̂ is large), the
  amendment is not filed as drafted: the PI either files N = 40 with its
  computed power stated on the form as below target, or revises the
  rank rule or k before filing. N is fixed on the OSF form; it is never
  below 20.
- **Fixed-N design.** There is no interim look at outcomes and no
  optional stopping. A technical failure (crash, non-finite codes,
  missing capture) is replaced from the reserve list in order, blind to
  the outcome. Rank exclusions are not replaced (§3.1). If more than 4
  replacements are needed, the run halts and a deviation note is filed.
  Extending N after an inconclusive result needs a new amendment filed
  before the new data.
- **Runtime.** Per seed: 7 Phase-1 runs (200 steps), 7 × 15 = 105 T2
  lesion runs and 7 no-lesion control runs (100 steps each, probe 512),
  plus 112 × 3 SVDs of 512 × 896 or 256 × 896 matrices. At N = 20 this
  is 2,240 Phase-2 runs and 140 Phase-1 runs (3,360 / 210 at N = 30).
  No timing exists for probe-512 captures; the pilot measures the
  runtime per seed on the filing host, and the estimate is recorded on
  the OSF form.

## 8. What is NOT being amended

- Thresholds 0.05 / 0.10 / 0.02 and every existing metric.
- ADR-0019 verdicts and the §5.5 FINAL text.
- The Q3/N12 LOCK_AFTER grid {50, …, 150}. This amendment adds levels
  25 and 175 for its own outcome only.

## 9. Pre-registration fidelity checklist

- [ ] Filed on OSF by the PI before any confirmatory seed (2001–2044) is run.
- [ ] Metric code + unit tests merged, commit hash on the OSF form.
      Required tests: S = 0 for identical subspaces, S = 1 for
      orthogonal subspaces, invariance to basis rotation.
- [ ] Probe-size parameter (512, decoupled from batch 16) and the
      no-lesion control arm implemented and frozen.
- [ ] Rank checks of §3.1 implemented with unit tests (rank-deficient
      control invalidates its 15 cells; 12-of-15 rule; complete-block
      selection), and the Page permutation test frozen.
- [ ] Pilot within-seed SD of Y, median S_split, rank-exclusion rate and
      runtime recorded on the form; N fixed on the form.
- [x] No threshold or existing-metric change.
- [x] Hypothesis is an increasing ordered trend; no peaked-shape test.
- [x] Seeds disjoint from all previous and sibling-draft seeds.

## 10. Open decisions (PI, before filing)

Every item below must be resolved, and the text edited to a single
protocol, before filing. This draft is not fileable while any item is
open; the metric code, the pilot and a regenerated power analysis for
the final configuration are also filing prerequisites.

1. Magnitude guard: median S_split (drafted) vs a fixed SESOI on ΔS
   chosen after the pilot. The guard sets the power of the complete
   rule (§7).
2. Rank-exclusion thresholds: 12 of 15 cells per block and 80 %
   complete blocks (drafted).
3. k = 12 (from ADR-0019 D_eff on gaussian/Studyforrest) vs
   re-estimating D_eff on the pilot.
4. N: 20 (drafted minimum) vs a larger N from the pilot, by the drafted
   rule of §7 or another rule the PI states on the form.
5. Parameter list for the secondary metric (which `CrossModalNerve`
   weights count as the interaction layer).
6. Whether to also run the CC-film bridge as a secondary world.
7. Pre-registration id (N-id) and final amendment number.
