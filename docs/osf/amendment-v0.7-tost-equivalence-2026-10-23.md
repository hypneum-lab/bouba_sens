# OSF pre-registration amendment — bouba_sens v0.7 (TOST equivalence for B-1 / B-2)

**Status:** DRAFT — not yet filed on OSF (to be filed by the PI)
**Parent pre-registration:** `dream-of-kiki` OSF DOI `10.17605/OSF.IO/Q6JYN` (unchanged by this draft)
**Previous amendment on file:** `amendment-v0.6-sprint10-17.md`
**Sibling drafts (same date, filed independently):** `amendment-v0.8-subspace-shift-2026-10-23.md`, `amendment-v0.9-decorrelation-window-2026-10-23.md`
**Amendment tag on file:** `bouba_sens/v0.7-tost-equivalence` (to be created at filing)
**Draft date:** 2026-10-23
**Pre-registration id:** to be assigned by the PI (next free N-id in the HYPNEUM-PLANS series; see Open decisions)

> **Filing note.** This is the machine-readable draft. Nothing in this
> file has been run. Confirmatory seeds (below) must not be run before
> the PI files this text on OSF. The pre-filing code tasks of §4.3 must
> be merged and frozen first; the commit hash of the analysis code is
> recorded on the OSF form at filing.

---

## 1. What is being amended

Nothing in Q6JYN is changed. The B-1 and B-2 thresholds (0.05 and
0.10 bit) stay frozen, and so do the ADR-0019 verdicts (B-1 FINAL
Retract, B-2 Null). This amendment adds **a confirmatory equivalence
test** for the two invariants that ended null, so that a null can be
reported as a bounded statement about a precisely defined estimand:

- B-1: "the population mean of the E-1 estimand (§3) lies inside
  ±0.05; equivalence at α = 0.05 by TOST, i.e. the 90 % two-sided CI
  lies inside the bounds";
- B-2: "the population mean of an operational statistic, the
  primary-estimator ΔMI on the fused code (§3), lies inside ±0.10 bit;
  equivalence at α = 0.05 by TOST, i.e. the 90 % two-sided CI lies
  inside the bounds". This bounds the estimated statistic only, not the
  true MI change and not the original surviving-channel B-2 invariant.

The original tests ask whether the effect is *above* threshold. A failure
to reject there ("below the noise floor", §5.5 FINAL) is absence of
evidence. Two one-sided tests (TOST) against the same thresholds give
evidence that the tested estimand is smaller than the registered effect
size. The claim is limited to that estimand (§3, §9).

This amendment does **not** rescue the retracted §5.5 dose-response
claim (ADR-0019). Whatever the outcome, the retraction stands, and
ADR-0019 remains the historical verdict on the data it analysed. New
outcomes are recorded in a new ADR (§9).

## 2. Motivation

- §5.5 FINAL and ADR-0019 "Open items" both say B-1 is "below the noise
  floor" and name N≥30 at α=0.0033 as the route to a confirmatory test.
  That route was framed for detecting a subgroup *peak*, which N12 did
  not replicate (0/2). An equivalence test answers the question a
  benchmarks-track reviewer will ask: is there a gap of at least the
  registered size?
- The existing 10-seed Q3/Q3+ data
  (`reports/v0.5_amedi_curve_multiseed_10seed_final.json`) already give
  a descriptive TOST that passes on the per-seed mean over LOCK_AFTER
  (p_lower = 1.5e-5, p_upper = 4.1e-7; `scripts/power_prereg_2026_10.py`).
  Those seeds informed the hypothesis and the planning variance, so they
  cannot be confirmatory. This amendment re-tests the claim on **new
  seeds only**, following the N12 training/test split.

## 3. Hypotheses and estimands

Bounds are set from the frozen thresholds, never from data (§5).

**B-1 estimand.** For one seed s and one LOCK_AFTER ℓ, the grid outcome
is `invariants.b1.median_me7`: the median, over the 15 (modality, SNR)
pairs of the grid (5 modalities × 3 SNR levels), of Me7 = mean Me1(T1)
− mean Me1(T2) (`scripts/aggregate_grid.py`). The per-seed E-1 value is
the arithmetic mean of the five grid medians (ℓ ∈ {50, 75, 100, 125,
150}). E-1 is therefore an **average over five lock conditions of
medians over modalities and SNR levels**. It is not the original B-1
operationalisation at floor SNR per modality (design spec §1.2), and an
E-1 pass does not exclude a gap of ≥ 0.05 at a single lock, modality or
SNR level: effects of opposite sign can cancel in the aggregate. The
all-lock claim needs E-1b. No per-modality or floor-SNR claim is made
(those values are reported descriptively, §6).

**B-2 estimand.** For one seed, ΔMI-hat = MI-hat(Z_post; y) −
MI-hat(Z_pre; y) in bits, computed by the primary estimator of §4.2 on
the full fused probe representation (512 × 896), presented clean on all
channels including the lesioned audio, in the configuration of §4.2.
This is a **new, fused-code operational statistic**. It differs from the
original B-2 invariant (design spec §1.2), which concerns MI of the
*surviving* modalities: in the fused code, a gain on surviving channels
can cancel a loss on the lesioned one. E-2 therefore bounds the
estimated statistic only. No E-2 outcome is read as a bound on the true
MI change, on surviving-channel MI or on the original B-2 invariant,
and no E-2 outcome is described as "absence of MI migration". The
known-signal calibration of §4.3 is reported to describe how the
estimator responds to known MI changes; it does not license a reading
in true-MI units.

| Id | Quantity (population mean over seeds) | H0 (to reject) | H1 (equivalence) | Status |
|----|---------------------------------------|----------------|------------------|--------|
| **E-1** | μ₁ = mean of the per-seed E-1 value (above) | μ₁ ≤ −0.05 or μ₁ ≥ +0.05 | −0.05 < μ₁ < +0.05 | primary |
| E-1b | μ₁(ℓ) at each ℓ separately (grid medians) | any ℓ with \|μ₁(ℓ)\| ≥ 0.05 | all five inside ±0.05 (IUT) | confirmatory, step 2 of a fixed sequence |
| E-1c | μ at the no-lock default (LOCK_AFTER unset) | \|μ\| ≥ 0.05 | \|μ\| < 0.05 | confirmatory, step 3 of a fixed sequence |
| **E-2** | μ₂ = mean of the per-seed ΔMI, primary estimator | μ₂ ≤ −0.10 or μ₂ ≥ +0.10 | −0.10 < μ₂ < +0.10 | primary |
| E-2b | same quantity, secondary estimators and the CC-film bridge | — | — | exploratory, no decision |

**Multiplicity.** E-1 and E-2 are two separate claims about two separate
invariants, each tested at α = 0.05; no cross-family correction is
applied, and each is reported as its own claim. Within B-1, the three
equivalence tests form a **fixed sequence** E-1 → E-1b → E-1c, each at
α = 0.05: a later step is tested only if every earlier step concluded
equivalence. This controls the family-wise error rate of the
equivalence claims at 0.05. E-1b is an intersection-union test: the
claim "equivalent at every tested LOCK_AFTER" needs all five TOSTs to
pass, and the IUT exemption covers that conjunction only; a single
lock's TOST is not advertised as a separate finding. E-2b is
exploratory and supports no claim.

## 4. Protocol

### 4.1 B-1 (E-1, E-1b, E-1c)

Exact replication of the Q3/Q3+ protocol
(`docs/milestones/q3-amedi-seeds-2026-05-10.md`) on new seeds:

- World: `StudyforrestRealWorld` (4.5-modal real bridge, as in Q3).
- `STEPS_TRAIN=200`, `STEPS_LESION=100`, batch 16,
  `TRANSDUCER_GATING=hard`, no codebook lock, no gating schedule.
- LOCK_AFTER ∈ {50, 75, 100, 125, 150}, plus one no-lock grid per seed
  for E-1c.
- 30 cells per grid (5 modalities × 2 timings × 3 SNR), `run_grid.sh`
  unchanged.
- Outcome per (seed, LOCK_AFTER): `invariants.b1.median_me7` from
  `v0.1_aggregate.json`, read exactly as `scripts/analyse_amedi_quadratic.py`
  reads it.
- Per-seed summary for E-1: arithmetic mean of the five values.

### 4.2 B-2 (E-2, E-2b)

ADR-0019 E1 protocol, with the configuration frozen as follows:

- World: `--world studyforrest` (primary), `--world ccfilm` (E-2b).
- LOCK_AFTER = 100, HARD gating, `steps_train = 200`,
  `steps_lesion = 100`, lesion as in the E1 runner (audio, mode M2, SNR
  schedule 0 → −20 dB over the lesion steps).
- **Batch sizes, frozen separately:** Phase-1 training batch **16**,
  Phase-2 lesion batch **16**, probe **N = 512** (`world.sample(512,
  seed = s − 99_999)`). These match B-1 and v0.8. The current runner
  uses one `--batch-size` for all three (default 512, as in the
  ADR-0019 E1.3 runs); see §4.3 and Open decision 3.
- **Primary estimator:** Kraskov-Stögbauer-Grassberger (KSG,
  algorithm 1) on the full (512, 896) probe, as implemented by the
  multi-D path of `bouba_sens.metrics.mi_migration.me3_mi`, which calls
  `nerve_wml.methodology.mi_kraskov_ksg_continuous` with k = 3,
  Chebyshev distance, the class label passed as a float column (no
  jitter), and each MI term clipped at 0 before the difference.
  ΔMI-hat = `me3_delta(Z_pre, Z_post, labels)` in bits. The `nerve_wml`
  version is pinned by commit hash on the OSF form. This estimator
  assumes two continuous variables; the label is discrete (4 values)
  and its many ties violate that assumption. Its output is therefore
  registered as an **operational statistic** (continuity with v0.6 and
  ADR-0019), not as a consistent MI estimate. A mixed discrete–continuous
  estimator is not drafted; adopting one is Open decision 3.
- Secondary estimators (E-2b, exploratory): MINE on the full probe
  (`me3_mi_mine`, as in the E1 runner); KSG on the 1-D mean-pool
  (legacy). Adaptive binning on the full probe is **not** used
  (ADR-0019 E1.3b shows it collapses in 896-D).
- The effective dimension D (participation ratio and two-NN) is reported
  per seed, as ADR-0019 requires.

### 4.3 Required pre-filing code tasks (B-2)

The E1 runner (`scripts/spikes/e1_b2_dimensional_mine.py`) does not yet
implement §4.2. Its Kraskov path (line 210) runs on the **1-D
mean-pooled scalar**, its `--batch-size` sets training, lesion and probe
size together, and `--seeds` takes a count. The following changes are
authorised, and only these:

1. `--seed-list` (explicit seeds) replacing the seed count.
2. Three separate options `--train-batch`, `--lesion-batch`,
   `--probe-n`, frozen at 16 / 16 / 512 for confirmatory runs.
3. A full-probe primary output `ksg_full_delta_bits` computed exactly as
   in §4.2. The existing 1-D and MINE outputs are kept unchanged as
   secondary outputs.
4. A known-signal calibration harness (below), run before filing.

Unit tests: the three batch sizes are honoured; the full-probe output
equals `me3_delta` on the captured matrices; seed lists reproduce.

**Known-signal calibration (pre-filing, synthetic data, descriptive).**
Codes of the confirmatory shape (N = 512, d = 896, 4 equiprobable
classes) are generated as a 12-D Gaussian class mixture (class means
a·m_y, fixed seeded unit vectors m_y, identity covariance) embedded in
ℝ^896 by a fixed random orthonormal map, plus independent isotropic
noise of SD 0.1. The true MI is computed by Monte Carlo (10⁵ draws) and
the scale a is solved for a target MI. Baselines are fixed, not taken
from the pilot: true MI_pre ∈ {0.25, 0.50, 1.00} bit (all ≥ 0.25 bit,
so every target below is feasible). At each baseline, twenty replicate
(pre, post) pairs are drawn at each true ΔMI ∈ {−0.20, −0.10, 0,
+0.10, +0.20} bit. Reported for KSG and MINE: the mean and SD of the
estimated ΔMI per cell, and the least-squares slope and intercept of
estimated on true ΔMI per baseline. The calibration is descriptive.
It is not a pass/fail gate and does not convert E-2 into a statement
about true MI: an estimator that returns half the true change would
calibrate visibly (slope ≈ 0.5), but E-2 still bounds the estimated
statistic only (§3). The paper reports the slopes next to the E-2
result so that readers can judge the estimator's resolution.

**B-2 pilot (pre-filing).** Pilot seeds 1101–1105 run the frozen B-2
configuration once, to measure the per-seed SD of the primary
ΔMI-hat and the runtime. Pilot data are
excluded from every confirmatory test. No pilot is needed for B-1, whose
planning data exist (§7).

## 5. Equivalence bounds and their justification

**Primary bounds: ±0.05 (B-1, accuracy units) and ±0.10 bit (B-2).**

1. *The bounds equal the frozen OSF thresholds.* Q6JYN defines 0.05
   and 0.10 bit as the smallest effects that count as a congenital gap
   and as MI migration. Using them as the smallest effect size of
   interest keeps the registered effect scale. No new quantity is
   introduced. The estimands differ from the original operationalisation
   as stated in §3.
2. *They are fixed before any new data and are not derived from the
   observed effects.* This avoids the circularity of bounds fitted to
   the 10-seed data.
3. *Symmetric lower bound.* B-1 and B-2 are directional. A reverse
   effect of the same size would also be substantive: a late-lesion
   advantage of 0.05, or an MI *loss* of 0.10 bit after lesion. TOST
   must therefore reject both sides. The one-sided alternative (H1:
   μ < threshold only) is weaker and is listed as an open decision.
4. *Resolution.* In the existing data, Me7 grid medians are quantised
   in steps of 0.00625, so ±0.05 spans 8 quanta on each side. The bound
   is well above measurement resolution.

**Candidate alternatives (reported, not decision-relevant unless the PI
adopts them at filing):**

- Tight bounds: ±0.025 for B-1 (half threshold, 4 quanta) and ±0.05 bit
  for B-2. A pass here is a stronger claim; a fail does not change the
  primary verdict.
- One-sided non-superiority: H1 μ₁ < 0.05 (B-1) and μ₂ < 0.10 (B-2).

## 6. Analysis plan

- Unit of analysis: the seed. Cells within a seed are not treated as
  independent.
- TOST: two one-sided one-sample t-tests at α = 0.05 each. Equivalently,
  the **90 % two-sided CI** of the mean lies inside the bounds. Report
  the 90 % CI (decision-relevant), the 95 % CI (descriptive), both
  one-sided p-values, the mean and the SD.
- Robustness (reported, not decision-relevant): the same TOST on a
  10 000-resample percentile bootstrap of the mean, and a
  Wilcoxon-based TOST. The data are quantised and may be non-normal. If
  parametric and non-parametric verdicts disagree, both are reported and
  the disagreement is stated in the paper.
- Descriptive only: per-modality Me7 at floor SNR, per ℓ; per-seed D
  for B-2; E-2b estimators.
- Exclusions: a seed is excluded only for a technical failure, i.e. a
  crash, a missing aggregate or a non-finite metric (§8). Outliers are
  not removed. A KSG value clipped at 0 is a valid value, not a failure.

## 7. Power analysis and sample size

Computed by `scripts/power_prereg_2026_10.py` (Monte Carlo, α = 0.05,
script seed 20261023, about 20 s) from data already in the repo. The
first tables give **component-test power only** (one TOST in
isolation). The complete decision rule is simulated below them.

**B-1 (Q3/Q3+ 10 seeds × 5 LOCK_AFTER).**

| Quantity | Value |
|----------|------:|
| Per-seed mean Me7 over LOCK: mean / SD | −0.0106 / 0.0161 |
| Per-LOCK SD range | 0.0196 – 0.0468 |
| Upper 80 % confidence limit of the per-seed SD (df = 9) | 0.0208 |

Component-test power only:

| Scenario | N for 80 % power | N for 90 % power |
|----------|-----------------:|-----------------:|
| E-1, true μ = 0, SD 0.0161 | 3 | 4 |
| E-1, true μ = −0.0106, SD UCL80 0.0208 | 4 | 5 |
| E-1, true μ = 0, worst per-LOCK SD 0.0468 | 10 | 11 |
| E-1b (IUT, 5 LOCKs), true μ = 0, observed covariance | 10 | 12 |
| E-1b (IUT, 5 LOCKs), true μ = observed per-LOCK means | 18 | 23 |

**B-2 (ADR-0019 per-seed tables, 5 seeds each).** No full-probe KSG SD
exists for the frozen configuration of §4.2. The available per-table
SDs come from other configurations (mostly the gaussian world, one batch
size of 512 for training, lesion and probe, other estimators). The
planning SD is the largest of them, 0.0749 bit (gaussian MINE-full,
E1.3), with UCL80 (df = 4) 0.1166 bit. This is a *planning assumption*,
not evidence about the confirmatory SD; the B-2 pilot (§4.3) measures
the latter, and N is fixed from it at filing.

Component-test power only:

| Scenario | N for 80 % power | N for 90 % power |
|----------|-----------------:|-----------------:|
| true μ = 0, SD 0.0749 | 7 | 8 |
| true μ = −0.04, SD 0.0749 | 12 | 15 |
| true μ = 0, SD UCL80 0.1166 | 14 | 17 |
| true μ = −0.04, SD UCL80 0.1166 | 25 | 34 |

Normal-approximation formula used as a cross-check: for true μ = 0,
N ≈ ((z₁₋α + z₁₋β/₂) · σ / δ)²; for μ ≠ 0, N ≈ ((z₁₋α + z₁₋β) · σ /
(δ − |μ|))², with δ the bound.

**Complete decision rule (simulated).** Each scenario draws seeds,
applies the full §9 classification (90 % CI against {−δ, 0, +δ}) and,
for B-1, the fixed sequence E-1 → E-1b, with per-LOCK values drawn from
the observed 5 × 5 covariance and rounded to the 0.00625 quantum.
E-1c cannot be simulated (no no-lock planning data).

| Scenario (B-1) | N | P(E-1 equivalent) | P(E-1 and E-1b equivalent) |
|----------------|--:|------------------:|---------------------------:|
| observed per-LOCK means, observed covariance | 20 | 1.000 | 0.863 |
| same | 30 | 1.000 | 0.968 |
| μ = 0, observed covariance | 20 | 1.000 | 0.997 |
| observed means, covariance inflated to the UCL80 SD | 20 | 1.000 | 0.627 |
| same | 30 | 1.000 | 0.826 |

| Scenario (B-2, SD 0.1166) | N | equivalent | effect detected in the direction of μ, size unresolved | inconclusive |
|---------------------------|--:|-----------:|---------------------------------:|-------------:|
| μ = 0 | 20 | 0.960 | 0.029 (either direction) | 0.011 |
| μ = −0.04 | 20 | 0.713 | 0.260 | 0.027 |
| μ = −0.04 | 25 | 0.802 | 0.195 | 0.003 |
| μ = −0.04 | 30 | 0.860 | 0.140 | 0.000 |

(μ = +0.04 gives the mirror image. No scenario produces a
"beyond bound" verdict.)

**Sample size.** N = 20 gives ≈ 1.00 for E-1 and 0.86 for the E-1 → E-1b
sequence at the observed means (0.63 if the SD is at its UCL80). For
E-2, N = 20 gives 0.96 at μ = 0 and 0.71 in the worst planning case
(μ = ±0.04, SD 0.1166); **N = 25 is needed for 0.80** and N = 34 for 0.90
in that case. N = 20 meets the lab's multi-seed-first-class rule
(≥ 10 seeds). The choice between N = 20, 25 and 30 is an open PI
decision (§12), to be made with the B-2 pilot SD and fixed on the OSF
form. Drafted rule: the pilot SD is the sample SD (ddof = 1) of the 5
pilot ΔMI-hat values, inflated to its one-sided UCL80 (df = 4). N is the
smallest N ≥ 20 whose simulated complete-rule P(Equivalent) at
μ = ±0.04 is ≥ 0.80, **unrounded**, with at least 100 000 Monte-Carlo
draws per candidate N (Monte-Carlo SE ≤ 0.0013), using
`scripts/power_prereg_2026_10.py`. If no N ≤ 40 qualifies, the
amendment is not filed as drafted: the PI either files N = 40 with its
computed power stated on the form as below target, or redesigns E-2
(another estimator or world) before filing. The planning-SD values
above give N = 25 (P = 0.802).

**Runtime.** B-1 alone needs N × 6 grids × 30 cells: **3,600 lesion
cells at N = 20** (5,400 at N = 30), excluding B-2. The last measured
throughput (N12, `docs/milestones/n12-amedi-subgroup-replication-2026-05-11.md`:
1,500 cells in 2 h 37 min, 24-core host, parallel-4) gives about 6.3 h
at N = 20 and 9.4 h at N = 30. B-2 adds N single-seed runs of the
E1 runner whose runtime is not yet measured (probe 512, KSG on 896-D).
Measuring the B-2 runtime on the filing host, during the pilot, is a
pre-filing task; the estimate is recorded on the OSF form.

## 8. Seeds and stopping rule

- **Confirmatory seeds:** the first N seeds of 1001, 1002, …, 1040 in
  order (N = 20: 1001–1020; N = 30: 1001–1030). The same list serves
  B-1 and B-2. They are disjoint from every seed used so far: the
  default grid seeds 0–4, Q3/Q3+ {0, 17, 42, 73, 101, 7, 23, 31, 53,
  89} and N12 {3, 11, 19, 29, 37, 41, 47, 59, 67, 79}, and from the
  v0.8 / v0.9 drafts.
- **Reserve seeds (technical replacement only):** the four seeds that
  follow the last confirmatory seed (N = 20: 1021–1024), used in order.
- **B-2 pilot seeds:** 1101–1105 (§4.3), never confirmatory.
- **Fixed-N design.** N is fixed on the OSF form. There is no interim
  look at outcomes and no optional stopping or extension. Every analysis
  runs once, after all N seeds have finished.
- **Technical failure** (crash, NaN/inf metric, missing aggregate): the
  seed is replaced by the next reserve seed. The decision is made from
  logs, blind to the outcome. If more than 4 replacements are needed,
  the run halts and a deviation note is filed on OSF before any
  analysis.
- **Inconclusive result:** reported as inconclusive. Adding seeds needs
  a new amendment filed before the extra data are produced.

## 9. Decision rule for the paper

Each primary test (E-1, E-2) is classified from the 90 % two-sided CI
[L, U] of the mean against its bound δ (0.05 for B-1, 0.10 bit for
B-2). The six classes below are exhaustive and mutually exclusive.
"Registered direction" is a positive gap (T1 better) for B-1 and an MI
increase for B-2. Every claim names the estimand of §3.

| Class | Condition | Paper claim |
|-------|-----------|-------------|
| Equivalent | −δ < L and U < δ | "The [estimand] is bounded within ±δ (TOST, α = 0.05; 90 % CI inside the bounds) on N pre-registered new seeds." If 0 ∉ [L, U], add: "it is non-zero but smaller than the registered effect size." |
| Beyond bound, registered direction | L ≥ δ | "An effect of at least the registered size, in the registered direction, on the [estimand]." Reported as a new pre-registered finding in a new ADR. |
| Beyond bound, reverse direction | U ≤ −δ | "A reverse effect of at least the registered size on the [estimand]." New ADR. |
| Effect detected, size unresolved | 0 < L < δ ≤ U | "A non-zero effect in the registered direction; whether it reaches the registered size is unresolved at N." |
| Reverse effect detected, size unresolved | L ≤ −δ < U < 0 | "A non-zero reverse effect; size relative to the registered size unresolved at N." |
| Inconclusive | L ≤ 0 ≤ U and (L ≤ −δ or U ≥ δ) | "Neither an effect nor its absence is established at N." |

All labels are read from the same CI, so exactly one is reported per
test. Each label rests on one-sided tests at α = 0.05; the probability
that the reported label is false is at most 0.05 for each direction of
error and at most 0.10 overall.

Fixed-sequence add-ons for B-1: if E-1 is Equivalent, E-1b is tested;
if it passes, the paper adds "equivalence holds at each of the five
tested LOCK_AFTER values (grid medians)". If that passes, E-1c is
tested; if it passes, "and at the no-lock default". A failed step is
reported as "not shown at every tested LOCK_AFTER" (or at no-lock),
with no further claim.

For E-2, every claim names the estimated fused-code statistic (§3).
No E-2 outcome is reported as absence of MI migration, as a bound on
true MI or as a bound on the original surviving-channel B-2 invariant.
The §4.3 calibration slopes are reported alongside.

**Relation to ADR-0019 and §5.5.** ADR-0019 remains the historical
verdict on the data it analysed and is not edited. The new outcome is
recorded in a new ADR whatever it is, including a "beyond bound" or
"effect detected" outcome that conflicts with the ADR-0019 null; the
paper then reports both, dated, without overwriting the earlier
verdict. No outcome reinstates the §5.5 dose-response (peak) claim,
which concerned a different estimand. All outcomes are publishable.

## 10. What is NOT being amended

- Thresholds 0.05 / 0.10 / 0.02, frozen since spec §1.2.
- Metric implementations (`me1_accuracy`, `me3_delta`, `me7_congenital_gap`,
  `me6_*`, `me9_bootstrap`).
- ADR-0019 verdicts (B-1 FINAL Retract, B-2 Null) and the §5.5 FINAL
  text.
- B-3. A null-calibrated B-3 headline, as suggested in PR #8, is out of
  scope here.

## 11. Pre-registration fidelity checklist

- [ ] Filed on OSF by the PI before any confirmatory seed (1001–1044) is run.
- [ ] Pre-filing code tasks of §4.3 merged with unit tests; commit hashes
      of `run_grid.sh`, `aggregate_grid.py`, `analyse_amedi_quadratic.py`,
      the E1 runner, the calibration harness, the TOST analysis script
      and the pinned `nerve_wml` recorded on the OSF form.
- [ ] Calibration table and B-2 pilot SD / runtime recorded on the
      form; N fixed on the form by the rule of §7.
- [x] No threshold change.
- [x] No metric-math change.
- [x] Bounds derived from frozen thresholds, not from data.
- [x] Seeds disjoint from all previously used seeds.
- [x] Verdicts to be recorded in a new ADR; ADR-0019 untouched.

## 12. Open decisions (PI, before filing)

Every item below must be resolved, and the text edited to a single
protocol, before filing. This draft is not fileable while any item is
open; the §4.3 code tasks, the pilot and a regenerated power analysis
for the final configuration are also filing prerequisites.

1. Bounds: symmetric ±threshold (drafted) vs one-sided non-superiority
   vs tight ±half-threshold as primary.
2. N: 20, 25 or 30, fixed with the B-2 pilot SD (drafted rule in §7).
   N = 20 gives 0.71 complete-rule power for E-2 in the worst planning
   case; N = 25 gives 0.80 and N = 30 gives 0.86. N = 30 also raises
   E-1 → E-1b to 0.97.
3. B-2 primary estimator: full-probe KSG as an operational statistic
   (drafted, continuity with v0.6) vs full-probe MINE (the estimator
   ADR-0019 treats as load-bearing) vs a mixed discrete–continuous
   k-NN estimator suited to a discrete label (would need an
   implementation, tie handling and validation before filing). KSG in
   896 ambient dimensions at N = 512 has known bias, clips at 0 and
   treats the label as continuous.
   Also the batch sizes: 16 / 16 / 512 (drafted, matches B-1 and v0.8)
   vs 512 / 512 / 512 (the ADR-0019 E1.3 runs).
4. B-2 primary world: Studyforrest (drafted; same bridge as B-1, but the
   lesioned audio channel is an annotation proxy) vs CC-film (audio is
   the real signal).
5. Pre-registration id (N-id) and final amendment number. The numbering
   v0.7/v0.8/v0.9 assumes the drafts are filed in that order.
