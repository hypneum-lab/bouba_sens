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
> the PI files this text on OSF; the commit hash of the analysis code is
> to be recorded on the OSF form at filing.

---

## 1. What is being amended

Nothing in Q6JYN is changed. The B-1 and B-2 thresholds (0.05 and
0.10 bit) stay frozen, and so do the ADR-0019 verdicts (B-1 FINAL
Retract, B-2 Null). This amendment adds **a confirmatory equivalence
test** for the two invariants that ended null, so that the null can be
reported as a positive bounded statement:

- B-1: "the congenital gap Me7 is smaller than 0.05 in absolute value,
  with 95 % confidence";
- B-2: "the MI migration Me3 delta is smaller than 0.10 bit in absolute
  value, with 95 % confidence".

The original tests ask whether the effect is *above* threshold. A failure
to reject there ("below the noise floor", §5.5 FINAL) is absence of
evidence. Two one-sided tests (TOST) against the same thresholds give
evidence of absence of an effect of at least the registered size.

This amendment does **not** rescue the retracted §5.5 dose-response
claim (ADR-0019). Whatever the outcome, the retraction stands. A
successful equivalence test strengthens it: B-1 is then shown to be
bounded below the registered effect size, not merely undetected.

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

## 3. Hypotheses

Bounds are set from the frozen thresholds, never from data (§5).

| Id | Quantity (population mean over seeds) | H0 (to reject) | H1 (equivalence) |
|----|---------------------------------------|----------------|------------------|
| **E-1** (primary) | μ₁ = mean over seeds of the per-seed mean Me7 across LOCK_AFTER ∈ {50, 75, 100, 125, 150} | μ₁ ≤ −0.05 or μ₁ ≥ +0.05 | −0.05 < μ₁ < +0.05 |
| E-1b (secondary, IUT) | μ₁(ℓ) at each LOCK_AFTER ℓ separately | any ℓ with \|μ₁(ℓ)\| ≥ 0.05 | all five ℓ inside ±0.05 |
| E-1c (secondary) | μ₁ at the no-lock default (LOCK_AFTER unset) | \|μ\| ≥ 0.05 | \|μ\| < 0.05 |
| **E-2** (primary) | μ₂ = mean over seeds of B-2 = MI_post − MI_pre (bits), primary estimator | μ₂ ≤ −0.10 or μ₂ ≥ +0.10 | −0.10 < μ₂ < +0.10 |
| E-2b (secondary) | same, secondary estimators and the CC-film bridge | \|μ\| ≥ 0.10 | \|μ\| < 0.10 |

E-1 and E-2 are two separate claims about two separate invariants. Each
is tested at α = 0.05 with no cross-family correction. E-1b is an
intersection-union test: the claim "equivalent at every LOCK_AFTER" is
made only if all five TOSTs pass. An IUT needs no multiplicity
correction.

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

ADR-0019 E1 protocol (`scripts/spikes/e1_b2_dimensional_mine.py`):

- World: `--world studyforrest` (primary), `--world ccfilm` (E-2b).
- LOCK_AFTER = 100, HARD gating, `steps_train=200`, `steps_lesion=100`,
  probe N = 512, full `(N, 896)` probe.
- Primary estimator: Kraskov k-NN on the full 896-D probe. This keeps
  the v0.6 choice of Kraskov as primary and applies it to the probe that
  ADR-0019 made the default. This is an open decision; see §12.
- Secondary estimators: MINE on the full probe; Kraskov on the 1-D
  mean-pool (legacy). Adaptive binning on the full probe is **not** used
  (ADR-0019 E1.3b shows it collapses in 896-D).
- The effective dimension D (participation ratio and two-NN) is reported
  per seed, as ADR-0019 requires.
- The runner currently takes `--seeds <count>` (seeds 0…n−1). An explicit
  seed-list option must be added before confirmatory runs. No other
  change to the runner is allowed.

## 5. Equivalence bounds and their justification

**Primary bounds: ±0.05 (B-1, accuracy units) and ±0.10 bit (B-2).**

1. *The bounds equal the frozen OSF thresholds.* Q6JYN defines 0.05
   and 0.10 bit as the smallest effects that count as a congenital gap
   and as MI migration. Using them as the smallest effect size of
   interest means the equivalence test answers exactly the registered
   question, in the opposite direction. No new quantity is introduced.
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
  the 90 % CI of the mean lies inside the bounds. Report the 90 % and
  95 % CIs, both one-sided p-values, the mean and the SD.
- Robustness (reported, not decision-relevant): the same TOST on a
  10 000-resample percentile bootstrap of the mean, and a
  Wilcoxon-based TOST. The data are quantised and may be non-normal. If
  parametric and non-parametric verdicts disagree, both are reported and
  the disagreement is stated in the paper.
- Exclusions: a seed is excluded only for a technical failure, i.e. a
  crash or a non-finite metric (§8). Outliers are not removed.

## 7. Power analysis and sample size

Computed by `scripts/power_prereg_2026_10.py` (Monte Carlo, α = 0.05,
script seed 20261023) from data already in the repo:

**B-1 (Q3/Q3+ 10 seeds × 5 LOCK_AFTER).**

| Quantity | Value |
|----------|------:|
| Per-seed mean Me7 over LOCK: mean / SD | −0.0106 / 0.0161 |
| Per-LOCK SD range | 0.0196 – 0.0468 |
| Upper 80 % confidence limit of the per-seed SD (df = 9) | 0.0208 |

| Scenario | N for 80 % power | N for 90 % power |
|----------|-----------------:|-----------------:|
| E-1, true μ = 0, SD 0.0161 | 3 | 4 |
| E-1, true μ = −0.0106, SD UCL80 0.0208 | 4 | 5 |
| E-1, true μ = 0, worst per-LOCK SD 0.0468 | 10 | 11 |
| E-1b (IUT, 5 LOCKs), true μ = 0, observed covariance | 10 | 12 |
| E-1b (IUT, 5 LOCKs), true μ = observed per-LOCK means | 18 | 23 |

**B-2 (ADR-0019 per-seed tables, 5 seeds each).** The planning SD is
the largest per-table SD, 0.0749 bit (gaussian MINE-full, E1.3). Its
UCL80 (df = 4) is 0.1166 bit. The real-bridge SDs are far smaller
(Studyforrest Kraskov 0.0022), so this choice is conservative.

| Scenario | N for 80 % power | N for 90 % power |
|----------|-----------------:|-----------------:|
| true μ = 0, SD 0.0749 | 7 | 8 |
| true μ = −0.04, SD 0.0749 | 12 | 15 |
| true μ = 0, SD UCL80 0.1166 | 14 | 17 |
| true μ = −0.04, SD UCL80 0.1166 | 25 | 34 |

Normal-approximation formula used as a cross-check: for true μ = 0,
N ≈ ((z₁₋α + z₁₋β/₂) · σ / δ)²; for μ ≠ 0, N ≈ ((z₁₋α + z₁₋β) · σ /
(δ − |μ|))², with δ the bound.

**Chosen N = 20 new seeds** for both E-1 and E-2. Power is ≥ 0.99 for
E-1, 0.86 for E-1b at the observed means, > 0.90 for E-2 at μ = 0 with
the UCL80 SD (N = 17 suffices for 90 %), and 0.72 for E-2 in the worst case (μ = −0.04 with the
UCL80 SD). N = 20 also meets the lab's multi-seed-first-class rule
(≥ 10 seeds, after the 5→10 seed collapse of Q3/Q3+). N = 30 is the
alternative listed in §12.

## 8. Seeds and stopping rule

- **Confirmatory seeds:** 1001–1020 (20 seeds). They are disjoint from
  every seed used so far: the default grid seeds 0–4, Q3/Q3+
  {0, 17, 42, 73, 101, 7, 23, 31, 53, 89} and N12 {3, 11, 19, 29, 37,
  41, 47, 59, 67, 79}.
- **Reserve seeds (technical replacement only):** 1021–1024, used in
  that order.
- **Fixed-N design.** There is no interim look at outcomes and no
  optional stopping or extension. Every analysis runs once, after all
  20 seeds have finished.
- **Technical failure** (crash, NaN/inf metric, missing aggregate): the
  seed is replaced by the next reserve seed. The decision is made from
  logs, blind to the outcome. If more than 4 replacements are needed,
  the run halts and a deviation note is filed on OSF before any
  analysis.
- **Inconclusive result:** reported as inconclusive. Adding seeds needs
  a new amendment filed before the extra data are produced.

## 9. Decision rule for the paper

| Outcome | Paper claim |
|---------|-------------|
| E-1 passes | "B-1 is bounded: \|Me7\| < 0.05 with 95 % confidence on 20 pre-registered new seeds (TOST). The §5.5 signature remains retracted (ADR-0019)." |
| E-1 fails, upper test rejects | "No congenital gap ≥ 0.05; a reverse (late-lesion) advantage cannot be excluded." |
| E-1 fails, lower test rejects | "A congenital gap of the registered size cannot be excluded at N = 20." Reported as is; this does **not** reinstate §5.5. |
| E-1b passes | "Equivalence holds at every tested LOCK_AFTER." |
| E-2 passes | "B-2 is bounded: \|ΔMI\| < 0.10 bit with 95 % confidence (primary estimator), with D reported." |
| E-2 fails | "B-2 remains null but is not shown equivalent at N = 20 (estimator-limited)." |

All outcomes are publishable. None reverses an ADR-0019 verdict.

## 10. What is NOT being amended

- Thresholds 0.05 / 0.10 / 0.02, frozen since spec §1.2.
- Metric implementations (`me1_accuracy`, `me3_delta`, `me7_congenital_gap`,
  `me6_*`, `me9_bootstrap`).
- ADR-0019 verdicts (B-1 FINAL Retract, B-2 Null) and the §5.5 FINAL
  text.
- B-3. A null-calibrated B-3 headline, as suggested in PR #8, is out of
  scope here.

## 11. Pre-registration fidelity checklist

- [ ] Filed on OSF by the PI before any confirmatory seed (1001–1024) is run.
- [ ] Commit hash of `run_grid.sh`, `analyse_amedi_quadratic.py`, the E1
      runner (with the seed-list option) and the TOST analysis script
      recorded on the OSF form.
- [x] No threshold change.
- [x] No metric-math change.
- [x] Bounds derived from frozen thresholds, not from data.
- [x] Seeds disjoint from all previously used seeds.
- [x] Verdicts to be recorded in a new ADR; ADR-0019 untouched.

## 12. Open decisions (PI, before filing)

1. Bounds: symmetric ±threshold (drafted) vs one-sided non-superiority
   vs tight ±half-threshold as primary.
2. N: 20 (drafted) vs 30. N = 30 raises E-1b power to 0.97 and E-2
   worst-case power to between 0.80 and 0.90 (N = 25 gives 80 %,
   N = 34 gives 90 %).
3. B-2 primary estimator: Kraskov on the full probe (drafted, continuity
   with v0.6) vs MINE on the full probe (the estimator ADR-0019 treats as
   load-bearing). Kraskov k-NN in 896 ambient dimensions at N = 512 has
   known bias; the PI may prefer MINE.
4. B-2 primary world: Studyforrest (drafted; same bridge as B-1, but the
   lesioned audio channel is an annotation proxy) vs CC-film (audio is
   the real signal).
5. Pre-registration id (N-id) and final amendment number. The numbering
   v0.7/v0.8/v0.9 assumes the drafts are filed in that order.
