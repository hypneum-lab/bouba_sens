# OSF pre-registration amendment — bouba_sens v0.7 (TOST equivalence for B-1 / B-2)

**Status:** DRAFT — not yet filed on OSF (to be filed by the PI)
**Parent pre-registration:** `dream-of-kiki` OSF DOI `10.17605/OSF.IO/Q6JYN` (unchanged by this draft)
**Previous amendment on file:** `amendment-v0.6-sprint10-17.md`
**Sibling drafts (same date, filed independently):** `amendment-v0.8-subspace-shift-2026-10-23.md`, `amendment-v0.9-decorrelation-window-2026-10-23.md`
**Amendment tag on file:** `bouba_sens/v0.7-tost-equivalence` (to be created at filing)
**Draft date:** 2026-10-23
**Pre-registration id:** to be assigned by the PI (next free N-id in the HYPNEUM-PLANS series; see Open decisions)
**PI decisions applied (2026-10-23):** N = 30; equivalence bounds ±half-threshold (B-1 ±0.025, B-2 ±0.05 bit); B-2 primary estimator = mixed discrete–continuous k-NN MI estimator (Ross 2014).

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
  ±0.025 (half the frozen 0.05 threshold); equivalence at α = 0.05 by
  TOST, i.e. the 90 % two-sided CI lies inside the bounds";
- B-2: "the population mean of an operational statistic, the
  primary-estimator ΔMI on the fused code (§3), lies inside ±0.05 bit
  (half the frozen 0.10 bit threshold); equivalence at α = 0.05 by
  TOST, i.e. the 90 % two-sided CI lies inside the bounds". This bounds
  the estimated statistic only, not the true MI change and not the
  original surviving-channel B-2 invariant.

The original tests ask whether the effect is *above* threshold. A failure
to reject there ("below the noise floor", §5.5 FINAL) is absence of
evidence. Two one-sided tests (TOST) against bounds set at half the
registered thresholds give evidence that the tested estimand is smaller
than half the registered effect size. The claim is limited to that
estimand (§3, §9).

This amendment does **not** rescue the retracted §5.5 dose-response
claim (ADR-0019). Whatever the outcome, the retraction stands, and
ADR-0019 remains the historical verdict on the data it analysed. New
outcomes are recorded in a new ADR (§9).

## 2. Motivation

- §5.5 FINAL and ADR-0019 "Open items" both say B-1 is "below the noise
  floor" and name N≥30 at α=0.0033 as the route to a confirmatory test.
  That route was framed for detecting a subgroup *peak*, which N12 did
  not replicate (0/2). An equivalence test answers the question a
  benchmarks-track reviewer will ask: is there a gap of at least half
  the registered size?
- The existing 10-seed Q3/Q3+ data
  (`reports/v0.5_amedi_curve_multiseed_10seed_final.json`) already give
  a descriptive TOST that passes on the per-seed mean over LOCK_AFTER
  at the decided bounds ±0.025 (p_lower = 0.010, p_upper = 3.2e-5;
  at ±0.05: 1.5e-5 and 4.1e-7; `scripts/power_prereg_2026_10.py`).
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
E-1 pass does not exclude a gap of ≥ 0.025 at a single lock, modality or
SNR level: effects of opposite sign can cancel in the aggregate. The
all-lock claim needs E-1b. No per-modality or floor-SNR claim is made
(those values are reported descriptively, §6).

**B-2 estimand.** For one seed, ΔMI-hat = MI-hat(Z_post; y) −
MI-hat(Z_pre; y) in bits, computed by the primary estimator of §4.2
(Ross 2014 mixed discrete–continuous k-NN estimator, k = 3) on
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
| **E-1** | μ₁ = mean of the per-seed E-1 value (above) | μ₁ ≤ −0.025 or μ₁ ≥ +0.025 | −0.025 < μ₁ < +0.025 | primary |
| E-1b | μ₁(ℓ) at each ℓ separately (grid medians) | any ℓ with \|μ₁(ℓ)\| ≥ 0.025 | all five inside ±0.025 (IUT) | confirmatory, step 2 of a fixed sequence |
| E-1c | μ at the no-lock default (LOCK_AFTER unset) | \|μ\| ≥ 0.025 | \|μ\| < 0.025 | confirmatory, step 3 of a fixed sequence |
| **E-2** | μ₂ = mean of the per-seed ΔMI, primary estimator (Ross 2014, k = 3) | μ₂ ≤ −0.05 or μ₂ ≥ +0.05 | −0.05 < μ₂ < +0.05 | primary |
| E-2b | same quantity, secondary estimators (full-probe KSG, MINE, 1-D KSG) and the CC-film bridge | — | — | exploratory, no decision |

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
  ADR-0019 E1.3 runs); see §4.3 and Open decision 1.
- **Primary estimator: mixed discrete–continuous k-NN MI estimator**
  of Ross (2014), *Mutual information between discrete and continuous
  data sets*, PLoS ONE 9(2): e87357, doi:10.1371/journal.pone.0087357
  (reference checked against Crossref and the PMC full text,
  PMC3929353, on 2026-10-23). The label y is discrete (4 classes); the
  code Z is continuous and multivariate, which the estimator allows.
  For one code matrix Z (the full 512 × 896 probe, pre or post) and
  its labels:
  - N = 512 probe points; N_y(i) = number of probe points whose label
    equals y_i (point i included).
  - **k = 3**, fixed. Ross reports that a low k works well and follows
    Kraskov et al. in suggesting k = 3; larger k lowers sampling error
    but raises coarse-graining error (bias).
  - Distance: **Euclidean** on the raw 896 coordinates (no
    standardisation, no jitter, no dimension reduction). Ross leaves
    the metric to the user; it is frozen here.
  - d_i = distance from z_i to its k-th nearest neighbour among the
    *other* points with the same label. m_i = number of points j ≠ i
    in the full probe with ‖z_j − z_i‖ ≤ d_i; this includes the k-th
    neighbour itself, as in Ross, so m_i ≥ k. Ties at distance d_i
    are all counted.
  - I_i = ψ(N) − ψ(N_y(i)) + ψ(k) − ψ(m_i), with ψ the digamma
    function; MI-hat = (1/N) Σ_i I_i in nats, divided by ln 2 for bits.
  - **No clipping**: MI-hat may be negative and is used as computed;
    ΔMI-hat = MI-hat(Z_post) − MI-hat(Z_pre). (The secondary KSG path
    clips each term at 0; that clipping biases a difference and is not
    used for the primary.)
  - Edge cases: a class with fewer than k + 1 probe points makes the
    seed a technical failure (§8); zero distances (duplicated codes)
    are handled by the same definitions and their count is reported.
  - **Bias behaviour.** Ross validated the estimator on scalar
    continuous variables with known MI. Its bias with 896 ambient
    dimensions and about 128 points per class is not characterised by
    that reference and may be material; it is measured by the
    calibration of §4.3. A bias common to Z_pre and Z_post cancels in
    ΔMI-hat; a bias that depends on the MI level or on the code
    geometry that the lesion changes does not. Because m_i ≥ k, each
    I_i ≤ ψ(N) − ψ(N_y(i)), so MI-hat is capped near the empirical
    label entropy (≈ 2 bits for balanced classes); a code that
    separates the classes almost perfectly before and after the lesion
    leaves little room for ΔMI-hat (ceiling effect).
  The `nerve_wml` and `bouba_sens` versions are pinned by commit hash
  on the OSF form. The estimator is new code (§4.3); it adds a
  function and changes no existing metric implementation (§10).
- Secondary estimators (E-2b, exploratory, no decision): KSG
  algorithm 1 on the full (512, 896) probe, as implemented by the
  multi-D path of `bouba_sens.metrics.mi_migration.me3_mi`, which calls
  `nerve_wml.methodology.mi_kraskov_ksg_continuous` with k = 3,
  Chebyshev distance, the class label passed as a float column (no
  jitter) and each MI term clipped at 0 (continuity with v0.6 and
  ADR-0019; it treats the discrete label as continuous); MINE on the
  full probe (`me3_mi_mine`, as in the E1 runner); KSG on the 1-D
  mean-pool (legacy). Adaptive binning on the full probe is **not**
  used (ADR-0019 E1.3b shows it collapses in 896-D).
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
3. The primary estimator of §4.2, implemented as a new function (for
   example `mi_ross_discrete_continuous(z, y, k=3)`), and a full-probe
   primary output `ross_full_delta_bits` computed exactly as in §4.2.
4. A secondary output `ksg_full_delta_bits` (full-probe KSG as in
   §4.2). The existing 1-D and MINE outputs are kept unchanged as
   secondary outputs.
5. An implementation verification and a known-signal calibration
   harness (below), both run before filing.

Unit tests: the three batch sizes are honoured; `ksg_full_delta_bits`
equals `me3_delta` on the captured matrices; seed lists reproduce.

**Implementation verification of the primary estimator (pre-filing,
pass/fail).** The new function must pass all of the following before
filing; a failure is fixed in code (the definition of §4.2 is not
changed) and the whole verification is rerun:

- agreement to 1e-10 with an independent brute-force O(N²) reference
  implementation of §4.2 on fixed random fixtures, including fixtures
  with duplicated rows and ties at d_i;
- invariance (to 1e-10) to relabelling of the classes, to permutation
  of the rows and to a translation plus an orthogonal rotation of Z;
- known MI in low dimension: 4 equiprobable classes, Z ∈ ℝ¹ with class
  means 0, a, 2a, 3a and unit variance, true MI computed by numerical
  integration, N = 512, 200 replicates at each a ∈ {0, 0.5, 1}: the
  mean estimate lies within ±0.02 bit of the true MI at each a
  (including a = 0, true MI = 0).

**Known-signal calibration (pre-filing, required, synthetic data,
descriptive).**
Codes of the confirmatory shape (N = 512, d = 896, 4 equiprobable
classes) are generated as a 12-D Gaussian class mixture (class means
a·m_y, fixed seeded unit vectors m_y, identity covariance) embedded in
ℝ^896 by a fixed random orthonormal map, plus independent isotropic
noise of SD 0.1. The true MI is computed by Monte Carlo (10⁵ draws) and
the scale a is solved for a target MI. Baselines are fixed, not taken
from the pilot: true MI_pre ∈ {0.25, 0.50, 1.00} bit (all ≥ 0.25 bit,
so every target below is feasible). At each baseline, twenty replicate
(pre, post) pairs are drawn at each true ΔMI ∈ {−0.20, −0.10,
−0.05, 0, +0.05, +0.10, +0.20} bit (±0.05 is the equivalence bound).
Reported for the primary estimator, and for full-probe KSG and MINE
as secondary: the mean and SD
of the estimated MI and ΔMI per cell (bias = mean estimate − truth),
and the least-squares slope and intercept of estimated on true ΔMI per
baseline. The calibration is required before filing and its table is
recorded on the OSF form, but it is descriptive.
It is not a pass/fail gate and does not convert E-2 into a statement
about true MI: an estimator that returns half the true change would
calibrate visibly (slope ≈ 0.5), but E-2 still bounds the estimated
statistic only (§3). The paper reports the slopes next to the E-2
result so that readers can judge the estimator's resolution.

**B-2 pilot (pre-filing).** Pilot seeds 1101–1105 run the frozen B-2
configuration once, to measure the per-seed SD of the primary
(Ross) ΔMI-hat and the runtime. The pilot does not change N (§7).
Pilot data are excluded from every confirmatory test. No pilot is
needed for B-1, whose planning data exist (§7).

## 5. Equivalence bounds and their justification

**Primary bounds (PI decision, 2026-10-23): ±0.025 (B-1, accuracy
units) and ±0.05 bit (B-2), i.e. ±half the frozen thresholds.**

1. *The bounds are fixed fractions of the frozen OSF thresholds.* Q6JYN
   defines 0.05 and 0.10 bit as the smallest effects that count as a
   congenital gap and as MI migration. Half of each is used as the
   equivalence margin, so that a pass excludes effects well below the
   registered size, not only effects at it. No new effect scale is
   introduced and the thresholds themselves are unchanged. The
   estimands differ from the original operationalisation as stated in
   §3.
2. *They are fixed before any new data and are not derived from the
   observed effects.* This avoids the circularity of bounds fitted to
   the 10-seed data.
3. *Symmetric lower bound.* B-1 and B-2 are directional. A reverse
   effect would also be substantive: a late-lesion advantage, or an MI
   *loss* after lesion. TOST must therefore reject both sides.
4. *Resolution.* In the existing data, Me7 grid medians are quantised
   in steps of 0.00625, so ±0.025 spans 4 quanta on each side. The
   bound is above measurement resolution.

The price of the tighter bounds is power (§7): at N = 30 several
components are under-powered, and the study's expected
informativeness is stated there.

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
  not removed. A negative primary MI-hat, and a secondary KSG value
  clipped at 0, are valid values, not failures.

## 7. Power analysis and sample size

Computed by `scripts/power_prereg_2026_10.py` (Monte Carlo, α = 0.05,
script seed 20261023, about 40 s) from data already in the repo. The
tables below are the script section "v0.7 DECIDED" (its own generator,
seed 20261023 + 707), at the decided bounds (±0.025 / ±0.05 bit) and
N = 30. The script still prints the earlier tables at ±0.05 / ±0.10
bit; they are superseded and kept only for reference. "Component-test
power" is one TOST in isolation; the complete decision rule is
simulated below it. N needed for 0.80 is found by bisection on
Monte-Carlo power (20,000 draws per N for single TOSTs, 2,000 per N for
the B-1 complete rule), so it is approximate.

**Sample size: N = 30 (PI decision, 2026-10-23).** N is fixed at 30
whatever the power computations below or the pilot show. Where power
at N = 30 is below 0.80, the N that would be needed is stated and the
expected informativeness of that component is labelled; N is not
changed.

**B-1 (Q3/Q3+ 10 seeds × 5 LOCK_AFTER).**

| Quantity | Value |
|----------|------:|
| Per-seed mean Me7 over LOCK: mean / SD | −0.0106 / 0.0161 |
| Per-LOCK means (ℓ = 50, 75, 100, 125, 150) | −0.0181, −0.0181, +0.0044, −0.0144, −0.0069 |
| Per-LOCK SD range | 0.0196 – 0.0468 |
| Upper 80 % confidence limit of the per-seed SD (df = 9) | 0.0208 |

Component-test power only (E-1, bound ±0.025):

| Scenario | Power at N = 30 | N for 80 % power |
|----------|----------------:|-----------------:|
| true μ = 0, SD 0.0161 | 1.000 | 6 |
| true μ = −0.0106, SD 0.0161 | 0.999 | 10 |
| true μ = −0.0106, SD UCL80 0.0208 | 0.979 | 15 |
| true μ = 0, worst per-LOCK SD 0.0468 | 0.774 | 32 |

Complete decision rule (simulated): seeds drawn from the observed
5 × 5 covariance of the per-LOCK values, rounded to the 0.00625 quantum,
full §9 classification of E-1 and the fixed sequence E-1 → E-1b. E-1c
cannot be simulated (no no-lock planning data).

| Scenario (B-1), N = 30 | P(E-1 equivalent) | P(E-1 and E-1b equivalent) | N for 0.80: E-1 / E-1 → E-1b |
|------------------------|------------------:|---------------------------:|-----------------------------:|
| observed per-LOCK means, observed covariance | 0.999 | 0.069 | 10 / ≈ 340 |
| μ = 0 at every lock, observed covariance | 1.000 | 0.689 | 6 / ≈ 37 |
| observed means, covariance inflated to the UCL80 SD | 0.979 | 0.028 | 15 / > 400 |

**B-2 (primary estimator: Ross 2014, k = 3).** No planning data exist
for the primary estimator: it has never been run on this model. The
only per-seed SDs available (ADR-0019 tables, 5 seeds each) come from
other estimators and other configurations; the largest is 0.0749 bit
(gaussian MINE-full, E1.3), with UCL80 (df = 4) 0.1166 bit. These are
*planning assumptions*, not evidence about the confirmatory SD. The
B-2 pilot (§4.3) measures the latter. The planning case is
μ = ±0.02 bit (0.4 × the bound) at SD 0.1166.

Complete decision rule (normal model, 90 % CI classes against
{−0.05, 0, +0.05}; for E-2, "Equivalent" is the TOST pass, so the
complete-rule and component powers coincide):

| Scenario (B-2), N = 30 | Equivalent | effect detected in the direction of μ, size unresolved | inconclusive | N for 0.80 |
|------------------------|-----------:|------------------------------:|-------------:|-----------:|
| μ = 0, SD 0.0749 | 0.945 | 0.044 (either direction) | 0.011 | 21 |
| μ = −0.02, SD 0.0749 | 0.692 | 0.282 | 0.025 | 40 |
| μ = −0.04, SD 0.0749 | 0.178 | 0.797 | 0.016 | ≈ 350 |
| μ = 0, SD 0.1166 | 0.481 | 0.099 (either direction) | 0.419 | 48 |
| μ = −0.02, SD 0.1166 (planning case) | 0.335 | 0.232 | 0.427 | ≈ 97 |
| μ = −0.04, SD 0.1166 | 0.114 | 0.553 | 0.314 | ≈ 840 |

(μ = +0.02 / +0.04 give the mirror image. The remaining probability
is "beyond bound" in the direction of μ: at most 0.019, at μ = −0.04,
SD 0.1166.) At N = 30, P(Equivalent) ≥ 0.80 needs a per-seed SD of
the primary ΔMI-hat ≤ 0.091 bit if μ = 0 and ≤ 0.065 bit if
μ = ±0.02.

**Expected informativeness at N = 30 (stated on the OSF form).**

- **E-1: high.** P(Equivalent) ≥ 0.97 in every planning scenario of the
  complete rule.
- **E-1 → E-1b: low.** P = 0.07 at the observed per-LOCK means (two
  locks sit at −0.018, close to −0.025), 0.69 if every per-LOCK mean
  is 0; ≈ 340 seeds would be needed for 0.80 at the observed means.
  A failed E-1b step at N = 30 is expected and is reported as "not
  shown at every tested LOCK_AFTER", not as evidence of a lock effect.
- **E-2: low under the planning assumptions.** P(Equivalent) = 0.335
  in the planning case (≈ 97 seeds needed for 0.80) and 0.48 at μ = 0
  with SD 0.1166. Whether it is better depends on the pilot SD of the
  Ross ΔMI-hat. At filing, the script is rerun with the pilot SD
  inflated to its one-sided UCL80 (df = 4), at least 100,000
  Monte-Carlo draws, μ = 0 and μ = ±0.02; the resulting power and the
  N needed for 0.80 are recorded on the form with one label: "adequately
  powered" (planning-case power ≥ 0.80), "under-powered" (0.50–0.80)
  or "low informativeness" (< 0.50). N stays 30 in every case, and an
  inconclusive E-2 is reported with the label and the N that would
  have been needed.

Normal-approximation formula used as a cross-check: for true μ = 0,
N ≈ ((z₁₋α + z₁₋β/₂) · σ / δ)²; for μ ≠ 0, N ≈ ((z₁₋α + z₁₋β) · σ /
(δ − |μ|))², with δ the bound (δ = 0.025 for B-1, 0.05 bit for B-2).

**Runtime.** B-1 needs N × 6 grids × 30 cells: **5,400 lesion cells at
N = 30**, excluding B-2. The last measured throughput (N12,
`docs/milestones/n12-amedi-subgroup-replication-2026-05-11.md`: 1,500
cells in 2 h 37 min, 24-core host, parallel-4) gives about 9.4 h. B-2
adds 30 single-seed runs of the E1 runner whose runtime is not yet
measured (probe 512, k-NN estimators on 896-D). Measuring the B-2
runtime on the filing host, during the pilot, is a pre-filing task;
the estimate is recorded on the OSF form.

## 8. Seeds and stopping rule

- **Confirmatory seeds:** 1001–1030 (N = 30), the same list for B-1
  and B-2. They are disjoint from every seed used so far: the default
  grid seeds 0–4, Q3/Q3+ {0, 17, 42, 73, 101, 7, 23, 31, 53, 89} and
  N12 {3, 11, 19, 29, 37, 41, 47, 59, 67, 79}, and from the v0.8
  (2001–2044, pilots from 2101) and v0.9 (3001–3044, pilots 3101–3105)
  seed pools.
- **Reserve seeds (technical replacement only):** 1031–1034, used in
  order.
- **B-2 pilot seeds:** 1101–1105 (§4.3), never confirmatory.
- **Fixed-N design.** N = 30 is fixed on the OSF form. There is no interim
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
[L, U] of the mean against its bound δ (0.025 for B-1, 0.05 bit for
B-2; half the registered thresholds, §5). The six classes below are exhaustive and mutually exclusive.
"Registered direction" is a positive gap (T1 better) for B-1 and an MI
increase for B-2. Every claim names the estimand of §3.

| Class | Condition | Paper claim |
|-------|-----------|-------------|
| Equivalent | −δ < L and U < δ | "The [estimand] is bounded within ±δ (TOST, α = 0.05; 90 % CI inside the bounds) on N pre-registered new seeds." If 0 ∉ [L, U], add: "it is non-zero but smaller than half the registered effect size." |
| Beyond bound, registered direction | L ≥ δ | "An effect of at least half the registered size (≥ δ), in the registered direction, on the [estimand]." Reported as a new pre-registered finding in a new ADR; this does not establish an effect of the full registered size. |
| Beyond bound, reverse direction | U ≤ −δ | "A reverse effect of at least half the registered size (≥ δ) on the [estimand]." New ADR. |
| Effect detected, size unresolved | 0 < L < δ ≤ U | "A non-zero effect in the registered direction; whether it reaches δ (half the registered size) is unresolved at N = 30." |
| Reverse effect detected, size unresolved | L ≤ −δ < U < 0 | "A non-zero reverse effect; size relative to δ unresolved at N = 30." |
| Inconclusive | L ≤ 0 ≤ U and (L ≤ −δ or U ≥ δ) | "Neither an effect nor its absence is established at N = 30." The expected-informativeness label of §7 is quoted. |

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
The §4.3 calibration slopes and biases of the primary estimator are
reported alongside.

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

- [ ] Filed on OSF by the PI before any confirmatory seed (1001–1034) is run.
- [ ] Pre-filing code tasks of §4.3 merged with unit tests; commit hashes
      of `run_grid.sh`, `aggregate_grid.py`, `analyse_amedi_quadratic.py`,
      the E1 runner, the calibration harness, the TOST analysis script
      and the pinned `nerve_wml` recorded on the OSF form.
- [ ] Primary-estimator implementation verification (§4.3) passed;
      calibration table, B-2 pilot SD / runtime and the recomputed
      E-2 power with its informativeness label (§7) recorded on the
      form; N = 30 on the form.
- [x] No threshold change.
- [x] No metric-math change.
- [x] Bounds derived from frozen thresholds (half of each), not from data.
- [x] Seeds disjoint from all previously used seeds.
- [x] Verdicts to be recorded in a new ADR; ADR-0019 untouched.

## 12. Open decisions (PI, before filing)

Every item below must be resolved, and the text edited to a single
protocol, before filing. This draft is not fileable while any item is
open; the §4.3 code tasks (including the implementation verification
and the calibration), the pilot and the regenerated E-2 power with the
pilot SD are also filing prerequisites.

Decided by the PI on 2026-10-23 and applied above (no longer open):
N = 30; bounds ±half-threshold (±0.025 / ±0.05 bit), symmetric; B-2
primary estimator = Ross (2014) mixed discrete–continuous k-NN, k = 3,
with full-probe KSG and MINE secondary.

1. B-2 batch sizes: 16 / 16 / 512 (drafted, matches B-1 and v0.8) vs
   512 / 512 / 512 (the ADR-0019 E1.3 runs).
2. B-2 primary world: Studyforrest (drafted; same bridge as B-1, but the
   lesioned audio channel is an annotation proxy) vs CC-film (audio is
   the real signal).
3. Pre-registration id (N-id) and final amendment number. The numbering
   v0.7/v0.8/v0.9 assumes the drafts are filed in that order.
