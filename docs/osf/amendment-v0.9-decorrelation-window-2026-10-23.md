# OSF pre-registration amendment — bouba_sens v0.9 (decorrelation-window lesion)

**Status:** DRAFT — not yet filed on OSF (to be filed by the PI)
**Parent pre-registration:** `dream-of-kiki` OSF DOI `10.17605/OSF.IO/Q6JYN` (unchanged by this draft)
**Previous amendment on file:** `amendment-v0.6-sprint10-17.md`
**Sibling drafts (same date, filed independently):** `amendment-v0.7-tost-equivalence-2026-10-23.md`, `amendment-v0.8-subspace-shift-2026-10-23.md`
**Amendment tag on file:** `bouba_sens/v0.9-decorrelation-window` (to be created at filing)
**Draft date:** 2026-10-23
**Related audit:** ADR-0020 (SNR floors were nearly identical under default settings), draft PR #14 on `docs/adr-snr-floor-audit`. This draft corrects the SNR schedule of the D-4 lesion phase (§4).
**Pre-registration id:** to be assigned by the PI (next free N-id in the HYPNEUM-PLANS series)

> **Filing note.** This is the machine-readable draft. Nothing in this
> file has been run. The manipulation does not exist in the code base
> yet; it must be implemented, unit-tested and frozen (commit hash on the
> OSF form) before any confirmatory seed runs. Pilot seeds (§7) may run
> before filing.

---

## 1. What is being amended

The amendment adds **a new lesion type**, a *decorrelation window*,
and one confirmatory hypothesis family about it. All modalities stay
present during the window. Only their instance-level cross-modal
pairing is broken, for a bounded window of Phase-1 training, and then
restored. No existing lesion, threshold, metric or verdict changes.

This is **not** a rescue of the retracted §5.5 claim. The B-1 lesion
removes or degrades a channel. The manipulation here removes cross-modal
correlation, which the benchmark has never manipulated, and the outcome
is measured on a new quantity, the fusion gain (§3). ADR-0019 stands
whatever the outcome.

## 2. Motivation

Kleinman, Achille & Soatto, "Critical Learning Periods for
Multisensory Integration in Deep Networks", CVPR 2023
(arXiv:2210.04643), report that a network's ability to integrate
diverse sources depends on exposure to properly correlated signals
early in training, and that interfering during this initial stage can
permanently impair integration. They report the effect in deep
(including deep linear) networks but not in shallow ones. Their
critical period is defined by the *timing of a correlated-signal
disturbance*, not by removing a modality. bouba_sens has tested only
removal (B-1, T1 vs T2), and B-1 did not survive multi-seed validation.
A decorrelation window tests the precedent's own kind of manipulation.

The operationalisation below (within-class cross-modal resampling, the
window grid, the fusion-gain outcome, the controls) is ours. It is not
copied from Kleinman et al. The PI should check the correspondence
before filing (§10). Our claims are limited to what the design tests:
whether the deficit varies across the specified training schedules
(in which onset, total training and clean exposure co-vary), persistence
through the tested recovery horizon, and nothing about permanence
beyond it. The design does not identify a special early sensitivity: a
deficit that simply declines with total training could produce the same
main-grid trend.

## 3. Manipulation, controls and outcome (pre-registered)

### 3.1 Donor resampling

The Studyforrest world samples **contiguous** windows of its time
series (`StudyforrestRealWorld.sample`), so a training batch of 16 is
16 consecutive samples. Donors are drawn from the **whole dataset**,
not from a batch-sized pool, so a same-label donor always exists:

- For training sample i at Phase-1 step t, with dataset index k_i and
  label y_i, a donor index is drawn uniformly from
  {j : label_j = y_i, j ≠ k_i} over all dataset rows. Donors are drawn
  with replacement across samples, modalities and steps. If a class had
  a single row, the sample would be its own donor; this is logged and
  counted (not expected: every class has many rows).
- Generator: `torch.Generator` seeded with 10⁹ + 1000 · s + t (s =
  training seed, t = step), freshly re-seeded at each step with this
  formula (no stream carried across steps, so step t is reproducible on
  its own), modalities drawn in the
  fixed order audio, vision, tactile, gravity, force.
- No class weighting: the donor is drawn uniformly within the label of
  i, and the label of i is kept, so the label distribution of each batch
  is unchanged.
- The sampler must expose dataset indices (`k_i`); adding this to the
  world is part of the frozen implementation.

### 3.2 Arms

During Phase-1 steps t ∈ [o, o + w) (onset o, width w):

- **Decorrelated arm (decor).** For each sample i, each modality m gets
  the modality-m input of its **own independent donor** j_m. Each
  modality's class-conditional distribution is preserved; the
  instance-level cross-modal pairing beyond class identity is broken.
- **Joint-resampling control (joint, primary comparator).** For each
  sample i, *all* modalities get the inputs of a single donor j, namely
  the donor that the decor arm draws for audio at the same (s, t, i).
  Cross-modal pairing is preserved. Both arms lose the within-batch
  temporal contiguity of the original windows identically, so the
  contrast decor − joint isolates the cross-modal decorrelation from the
  temporal-structure change that any resampling causes.
- **Plain control (plain, descriptive).** No resampling at all.

Outside the window, all arms are identical (same seeds, same batches).

### 3.3 Manipulation check (pre-registered, gates the experiment)

The gate compares the two resampled arms under **matched resampling**,
so that temporal structure, effective sample size and the donor scheme
are identical and only the cross-modal pairing differs.

- **Check sample.** On each pilot seed, 512 anchor rows are drawn as 32
  contiguous windows of 16 (`world.sample(16, seed = s + 3·10⁶ + r)`,
  r = 0…31), to cover several labels.
- **Draws.** For b = 1…200, the donor rule of §3.1 is applied to the
  anchors with generator seed 10⁹ + 7·10⁸ + 1000 · s + b, producing one
  joint-arm sample (one donor per anchor, all modalities) and one
  decor-arm sample (independent donors per modality, the audio donor
  shared with the joint sample), exactly as in training.
- **Within-class linear CKA.** For each of the 10 modality pairs (a, b),
  features flattened, and each class c with n_c ≥ 10 rows: column-centre
  X_a and X_b within the class and compute
  CKA = ‖X_bᵀ X_a‖²_F / (‖X_aᵀ X_a‖_F · ‖X_bᵀ X_b‖_F) (linear kernel,
  biased HSIC estimator). A (pair, class) with a zero denominator (a
  modality constant within the class) is skipped. The pair value is the
  n_c-weighted mean over the remaining classes; the statistic K is the
  mean over pairs with a value. If no (pair, class) is valid in a draw,
  the check fails.
- **Gate statistic.** Δ_K = mean over pilot seeds and draws of
  (K_joint − K_decor).
- **Null calibration.** Under the null of no within-class cross-modal
  dependence, joint and decor draws are exchangeable within a seed. The
  null distribution of Δ_K is obtained by swapping the joint/decor
  labels independently within each (seed, draw) pair, 10 000 times
  (generator seed 20261023). The gate passes if the one-sided
  permutation p-value of the observed Δ_K is < 0.05.
- **Failure.** If p ≥ 0.05, the check failed: the experiment is
  reported as *uninformative in this world* with no verdict. No
  inference is drawn about the size of the manipulation.
- Reported (no decision): Δ_K with its permutation p-value, the mean
  K_joint and K_decor, and the CKA of the original contiguous anchors
  (which differs from both in temporal structure and is not used for
  the gate).

### 3.4 Outcome: fusion gain and deficit

At the end of Phase 1 (no lesion), on one clean probe of 512 rows,
`world.sample(512, seed = s + 2·10⁶)`, identical for all arms:

- acc_all: plain accuracy with all channels present.
- acc_m: `query_accuracy(m, batch_size = 512, seed = s + 2·10⁶)`, the
  accuracy when only modality m carries signal (same probe rows).
- **Fusion gain** FG = acc_all − max_m acc_m. Plain (unweighted)
  accuracy is used; the probe is the same for all arms of a seed, so
  class imbalance in the probe is common to both terms of a contrast.
- **Deficit (primary)** D(s, o) = FG_joint(s, o) − FG_decor(s, o).
- **Secondary contrasts (descriptive):** D_plain(s, o) = FG_plain −
  FG_decor, and the resampling effect FG_plain − FG_joint.

Positive D means the decorrelation window reduced fusion gain at the
end of the run, relative to a run that saw the same resampled inputs
with their cross-modal pairing intact.

## 4. Conditions

- World: `StudyforrestRealWorld` (4.5-modal real bridge; the real
  modalities carry temporally structured within-class covariance).
  Subject to the manipulation check.
- HARD gating, no constellation lock (LOCK_AFTER unset; a lock would
  confound the window with a freeze), no codebook lock, batch 16.
- Window width **w = 50** steps.
- **Main grid (fixed recovery):** onsets o ∈ {0, 50, 100, 150, 200}
  (5 ordered levels), recovery **R = 200** clean steps after the
  window, total Phase-1 length o + w + R = 250 … 450 steps. Arms decor,
  joint and plain at every onset.
- **Fixed-budget grid (control for training budget):** total length
  **T = 450** for every onset, R = 400 − o (400, 350, 300, 250, 200).
  Arms decor and joint. Its o = 200 cell coincides with the main grid,
  and its o = 0 cell is the R = 400 persistence arm.
- **Persistence arm:** o = 0, R = 400 (decor and joint), which is the
  o = 0 cell of the fixed-budget grid.
- Per seed: 15 main-grid runs, 2 persistence runs, 6 additional
  fixed-budget runs (o = 50, 100, 150 × decor, joint): 23 Phase-1 runs.
- **Lesion phase for D-4 (SNR schedule: changed versus the historical
  grids, ADR-0020).** The "standard T2 lesion phase" is 100 lesion steps
  with SNR(t) = max(−20, 20 − 40 · min(1, t / k)) and **`k_steps` = 50**.
  The floor −20.00 dB is reached at step 50 (generally, `k_steps` = N
  reaches the floor within the first N steps, N ≤ 99) and held to step
  99; mean SNR over the 100 steps is −9.80 dB. The historical default
  (`k_steps` = 5000) only reached 19.21 dB at the last step (ADR-0020,
  PR #14), so D-4 is not comparable with historical lesion results. The
  CLI already has `--k-steps`; `run_grid.sh` does not forward it, so the
  D-4 driver must pass `--k-steps 50` explicitly (code task, not done by
  this draft; code defaults unchanged).

**What the two grids do and do not separate.** In the main grid,
onset is crossed with a fixed recovery, but total training (250 → 450
steps) and clean exposure (200 → 400 steps) grow with onset. In the
fixed-budget grid, total training is constant, but the recovery shrinks
with onset (400 → 200), so onset remains coupled to recovery duration.
Neither grid identifies the separate effects of onset, budget and
recovery. If recovery reduced the deficit monotonically, a shorter
recovery at late onsets would work against a decreasing trend, but that
is an untested assumption, not a design property. The fixed-budget grid
is therefore reported only as a **replication at fixed total training
budget**, with the remaining recovery confound stated.

## 5. Hypotheses and tests

| Id | Statement | Test | Status |
|----|-----------|------|--------|
| **D-1** | Early decorrelation reduces fusion gain: mean D(s, 0) > 0 (main grid) | one-sample t, one-sided, α = 0.05; equivalently L > 0 for the 90 % CI | confirmatory, step 1 |
| **D-2** | The deficit shows a decreasing ordered trend over onset (main grid, 5 onsets) | Page test, decreasing, within-block permutation (§6), α = 0.05 | confirmatory, step 2 (tested only if D-1 rejects) |
| D-M | Magnitude of D(s, 0) relative to SESOI 0.05 | classification of the 90 % CI (§6) | confirmatory estimation statement |
| D-2b | The same decreasing ordered trend at fixed budget T = 450 | as D-2 | secondary, gated (§6) |
| D-3 | Persistence: mean D(s, 0) at R = 400 > 0 | one-sample t, one-sided | secondary, gated (§6) |
| D-4 | Effect on later lesion: after a standard T2 lesion phase (100 steps, floor SNR reached at step 50 with `k_steps` = 50, each modality; §4, ADR-0020) applied to the o = 0, R = 200 decor and joint runs, the post-lesion Me1 differs | descriptive | exploratory |
| D-5 | Representation geometry: S between decor and joint end-of-Phase-1 codes (v0.8 metric, k = 12) | descriptive | exploratory |

**SESOI = 0.05 accuracy units.** D is an accuracy difference, like Me7,
and 0.05 is the benchmark's registered smallest meaningful accuracy gap
(B-1 threshold, Q6JYN). Reusing it keeps one effect-size scale across
the benchmark.

**Multiplicity.** D-1 → D-2 is a fixed sequence, each at α = 0.05; the
headline (both reject) is their conjunction, and neither component is
advertised as a separate discovery outside the axes of §6. D-2b and D-3
form a secondary family tested only if the headline holds (serial
gatekeeping), with Holm's correction within the family at α = 0.05.
D-4 and D-5 are exploratory and support no claim.

## 6. Decision criteria (pre-stated)

**Page test, frozen method.** As in v0.8: within-seed mid-ranks of the
5 onsets, ordered so that the hypothesis predicts *decreasing* D;
within-block permutation p-value with 100 000 permutations, generator
seed 20261023, p = (1 + #{L_perm ≥ L_obs}) / (1 + 100 000)
(`page_perm_pvalue` in `scripts/power_prereg_2026_10.py`, applied to
D[:, ::-1]). D takes values on a grid of 1/512, so ties are possible;
mid-ranks handle them. Asymptotic and exact p-values are reported
alongside; the permutation p decides. Validity rests on within-seed
exchangeability of the 5 onset values under the null and on
independent seeds. A significant L shows that earlier onsets tend to
have larger within-seed ranks of D; it does not show that every
adjacent step decreases. Claims say "decreasing ordered trend", never
"monotone". The median over seeds of each adjacent difference
D(s, o_{j+1}) − D(s, o_j), with 90 % bootstrap CIs, is reported to
describe the shape (no decision).

The verdict is reported on **three separate axes**. Each axis has
mutually exclusive labels, and the paper sentence is the combination of
the three labels, so no outcome receives two conflicting verdicts.

**Gate.** If the manipulation check fails (§3.3): *uninformative in this
world*; no axis is reported as a finding.

**Axis A — detectable harm at onset 0 (D-1).** From the 90 % CI [L, U]
of mean D(s, 0):

- A+ "harm detected": L > 0.
- A0 "harm not detected": L ≤ 0 ≤ U.
- A− "reverse effect detected" (decorrelation *increased* fusion gain):
  U < 0. Reported as such; the hypothesis was harm.

**Axis B — magnitude at onset 0 (D-M, SESOI 0.05).** Same CI:

- B≥ "at least the SESOI": L ≥ 0.05.
- B≈ "practically negligible": −0.05 < L and U < 0.05 (TOST
  equivalence at α = 0.05; 90 % CI inside ±0.05).
- B≤ "reverse, at least the SESOI": U ≤ −0.05.
- B? "magnitude unresolved": all other cases. The point estimate is
  reported, but no magnitude claim is made.

**Axis C — timing (D-2, only if A+).**

- C+ "schedule dependence detected: a decreasing ordered trend of the
  deficit over the main-grid schedules, in which onset, total training
  and clean exposure co-vary": D-2 p < 0.05. It is not attributed to
  onset alone.
- C0 "timing (schedule) dependence was not demonstrated": D-2 p ≥ 0.05. This does
  not establish that the deficit is independent of timing.
- C– "not tested": A0 or A−.

The mirrored (increasing) Page test is reported as exploratory.

**Combined statements.**

| A | B | C | Paper statement |
|---|---|---|-----------------|
| A+ | B≥ | C+ | "Early decorrelation reduced fusion gain by at least the SESOI, and the deficit showed a decreasing ordered trend across the specified training schedules, in which onset, total training and clean exposure co-vary; the trend is not attributed to onset alone." |
| A+ | B? | C+ | Same, with "by an amount whose size relative to the SESOI is unresolved". |
| A+ | B≈ | C+ | "A detectable but practically negligible (< SESOI) deficit with a decreasing ordered trend across the specified training schedules (confounds as above)." |
| A+ | any | C0 | "Early decorrelation reduced fusion gain [magnitude label]; timing dependence was not demonstrated." |
| A0 | B≈ | C– | "No deficit of practical size: the onset-0 deficit is bounded within ±0.05 (informative null)." |
| A0 | B? | C– | "Inconclusive: neither a deficit nor its absence is established at N." |
| A− | any | C– | "Decorrelation increased fusion gain [magnitude label]; contrary to the hypothesis." |

(A+ with B≤, or A− with B≥, is impossible by construction.)

**Gated qualifiers (only if A+ and C+).** With Holm over {D-2b, D-3}:

- D-2b rejects: "the decreasing ordered trend replicates at fixed total
  training budget; in that grid, onset remains coupled to recovery
  duration, so the separate effects of onset, budget and recovery are
  not identified". Not rejected: "the trend was not shown to replicate
  at fixed total training budget".
- D-3 rejects: "the deficit persists through the tested recovery horizon
  of 400 steps". Not rejected: "persistence beyond 200 recovery steps
  was not demonstrated". No claim of permanence beyond the tested
  horizon is made in any case.

**Error rates.** Axis A and C form a fixed sequence at α = 0.05. Each
Axis B label is a one-sided α = 0.05 claim from the same CI; the
probability that the reported Axis B label is false is at most 0.05 per
direction. Under no effect (μ = 0), the simulated probability of the
headline A+ and C+ is 0.009–0.014 (§7).

Paper consequence: a supported result is a *new* pre-registered
finding that a correlated-signal manipulation (not channel removal)
reduces fusion gain and that the deficit varies across the specified
training schedules. It is not described as a critical period, because
the design does not separate onset from training budget and recovery. It does not reinstate B-1 or §5.5. All outcomes are
publishable.

## 7. Seeds, power and stopping rule

- **Pilot seeds:** 3101–3105, used to validate the pipeline, run the
  manipulation check, estimate the variance components of D (reported;
  the planning ρ stays 0) and measure the runtime. They are excluded from
  confirmatory tests.
- **Confirmatory seeds:** the first N of 3001, 3002, …, 3040 in order
  (N = 20: 3001–3020), disjoint from every seed used so far and from the
  v0.7 / v0.8 drafts. **Reserve:** the four seeds that follow the last
  confirmatory seed (N = 20: 3021–3024), for technical replacement
  only.
- **Planning SD.** With no pilot yet, the SD of D is borrowed from the
  largest per-LOCK SD of Me7 in Q3/Q3+ (0.0468, also an accuracy
  difference between two training regimes of the same seed) and from
  its UCL80 (0.0605). This is an assumption; the pilot replaces it.

  *Component-test power only* (`scripts/power_prereg_2026_10.py`):

  | Planning SD | N | D-1 power (true D = 0.05) | TOST ±0.05 power (true D = 0) |
  |-------------|--:|--------------------------:|------------------------------:|
  | 0.0468 | 10 | 0.93 | 0.86 |
  | 0.0468 | 20 | 1.00 | 1.00 |
  | 0.0605 | 10 | 0.78 | 0.56 |
  | 0.0605 | 20 | 0.97 | 0.94 |

  D-2 alone (Page, 5 levels): N = 20 gives power 0.94 for a total trend
  of 1 within-seed SD and 0.48 for 0.5 SD (N = 30: 0.98 / 0.59).

  *Complete rule* (SD 0.0605, D quantised to 1/512, D-2 decided at the
  permutation critical value of Page's L (`page_perm_crit`, no-tie
  calibration; ties from quantisation are rare at this SD), onsets correlated
  through a seed effect with intraclass correlation ρ; "linear" = true
  deficit μ₀ at o = 0 falling linearly to 0 at o = 200; "early" = only
  o = 0 affected):

  | N | profile | ρ | μ₀ | P(A+ and C+) | P(B≥) | P(B≈) |
  |--:|---------|--:|---:|-------------:|------:|------:|
  | 20 | any | any | 0 | 0.009–0.014 | 0.000 | 0.94 |
  | 20 | linear | 0 | 0.05 | 0.805 | 0.05 | 0.05 |
  | 20 | linear | 0.5 | 0.05 | 0.95 | 0.05 | 0.05 |
  | 20 | early | 0 | 0.05 | 0.618 | 0.05 | 0.05 |
  | 20 | early | 0.5 | 0.05 | 0.843 | 0.05 | 0.05 |
  | 20 | linear | 0 | 0.075 | 0.985 | 0.56 | 0.00 |
  | 20 | any | any | 0.10 | ≥ 0.98 | 0.96–0.97 | 0.00 |
  | 30 | linear | 0 | 0.05 | 0.93 | 0.05 | 0.05 |
  | 30 | early | 0 | 0.05 | 0.789 | 0.05 | 0.05 |
  | 30 | linear | 0 | 0.075 | 1.00 | 0.72 | 0.00 |

  Two consequences. First, B≥ can only be expected when the true
  deficit is well above the SESOI (about 0.10 at N = 20); at a true
  deficit equal to the SESOI it occurs about 5 % of the time by
  construction. A rule that required the point estimate to reach the
  SESOI (the earlier draft) would have power 0.50 at a true deficit of
  0.05 whatever N. Second, the headline power depends strongly on
  the onset profile and ρ; planning uses the least favourable ρ.
  These numbers are conditional on the simulation model (Gaussian
  seed and residual effects, the two onset profiles); the pilot
  replaces the planning SD. ρ = 0 gives the lowest headline power in
  every simulated scenario, so planning uses ρ = 0 and does not estimate
  ρ.
- **N.** N = 20 is the minimum. Drafted rule, from the 5 pilot seeds
  (decor vs joint, main grid, 5 seeds × 5 onsets): fit the two-way
  model D(s, o) = μ_o + u_s + e_so by ANOVA without replication;
  σ̂_e² = MS_residual (df = 16), σ̂_u² = max(0, (MS_seed − MS_residual)
  / 5), σ̂² = σ̂_u² + σ̂_e² (the marginal SD the planning model needs).
  Its planning value is the 80th percentile of σ̂ over 2 000 parametric
  bootstrap replicates of 5 seeds drawn from the fitted model (seed
  20261023), which accounts for the seed correlation that an ordinary
  χ² limit ignores. With ρ = 0 (above), N is the smallest N ≥ 20 for
  which the simulated P(A+ and C+) at μ₀ = 0.05 under the "early"
  profile, **minus two Monte-Carlo SEs**, is ≥ 0.80, using at least
  100 000 draws per candidate N and the permutation-calibrated Page
  test. With the planning SD 0.0605 this gives **N = 32** (estimates
  0.788 / 0.801 / 0.809 at N = 30 / 31 / 32; lower bounds 0.786 /
  0.799 / 0.807; `scripts/power_prereg_2026_10.py`).
  If no N ≤ 40 qualifies, the amendment is not filed as drafted: the PI
  either files N = 40 with its computed power stated on the form as
  below target, or redesigns (for example a larger SESOI justified
  independently of the data, or a different window) before filing. The
  PI fixes N on the OSF form; it is never lowered after the
  confirmatory runs start.
- **Fixed-N design.** There is no interim look at outcomes and no
  optional stopping. A technical failure (crash, non-finite accuracy,
  missing run) is replaced from the reserve in order, blind to the
  outcome; there are no outcome-dependent exclusions, so every analysed
  seed has all onsets. If more than 4 replacements are needed, the run
  halts and a deviation note is filed. Extending N needs a new amendment
  filed before the new data.
- **Runtime.** Per seed: 23 Phase-1 runs of 250–450 steps (about 8,850
  training steps at batch 16), 23 fusion-gain evaluations (6 accuracy
  calls on 512 rows each) and the D-4 lesion runs (2 arms × 5
  modalities × 100 steps). At N = 20, about 177,000 Phase-1 steps,
  roughly 590 B-1-cell equivalents of 300 steps. This is an estimate in
  steps, not time; the pilot measures wall-clock time on the filing
  host, recorded on the OSF form.

## 8. What is NOT being amended

- Thresholds 0.05 / 0.10 / 0.02, the M2 lesion, the T1/T2 arms and
  every existing metric.
- ADR-0019 verdicts and the §5.5 FINAL text.

## 9. Pre-registration fidelity checklist

- [ ] Filed on OSF by the PI before any confirmatory seed (3001–3044) is run.
- [ ] Decorrelation scheduler + unit tests merged, commit hash on the OSF
      form. Required tests: labels unchanged; donor ≠ own dataset index;
      donor label = sample label; joint arm shares the decor audio
      donor; identity outside the window; deterministic given seed;
      world exposes dataset indices.
- [ ] Fusion-gain probe and CKA check implemented as specified (§3.3,
      §3.4), with unit tests (CKA = 1 for identical features, zero
      denominators skipped).
- [ ] Manipulation check passed on pilot seeds and recorded on the form,
      with Δ_K, its permutation p-value, K_joint and K_decor.
- [ ] Pilot variance components of D, bootstrap planning SD and runtime
      recorded; N fixed on the form.
- [x] No threshold or existing-metric change.
- [x] Seeds disjoint from all previous and sibling-draft seeds.

## 10. Open decisions (PI, before filing)

Every item below must be resolved, and the text edited to a single
protocol, before filing. This draft is not fileable while any item is
open; the scheduler, the fusion-gain and CKA code, the pilot (including
the manipulation check) and a regenerated power analysis for the final
configuration are also filing prerequisites.

1. Correspondence with Kleinman et al. (2023): confirm that within-class
   cross-modal resampling is an acceptable analogue of their
   correlated-signal disturbance, or switch to their exact manipulation
   after reading the paper.
2. Window width w = 50 and recovery R = 200 (drafted) vs a second width.
   Each added width multiplies compute.
3. Fixed-budget grid: keep (drafted; 6 extra runs per seed, reported as
   a replication at fixed budget) or drop. Either way, every positive
   claim is limited to "the deficit varies across the specified
   training schedules"; separate identification of onset, budget and recovery
   effects would need a further design (e.g. a factorial over recovery
   length), not drafted.
4. Primary comparator: joint-resampling control (drafted) vs the plain
   control. With the plain control, the deficit also contains the
   effect of losing within-batch temporal contiguity, and the attribution
   must be narrowed to "resampling with decorrelation".
5. Manipulation-check gate: the matched joint-vs-decor permutation test
   (drafted), optionally plus a minimum Δ_K fixed before the pilot.
6. Whether the onset grid should also include LOCK_AFTER regimes (a
   cross with v0.8). Not drafted, to keep the family small.
7. Optional mitigation arm (cross-modal reconstruction objective, which
   Kleinman et al. report reduces critical-period sensitivity). Not
   drafted.
8. Pre-registration id (N-id) and final amendment number.
9. SNR schedule `k_steps` = 50 for the D-4 lesion phase (drafted, ADR-0020) vs another N.
