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
hypothesis**: the shift increases monotonically with the
critical-period lock time `LOCK_AFTER`. No existing threshold, metric or
verdict changes.

This is **not** a rescue of the retracted §5.5 claim. §5.5 claimed a
*peaked* (inverted-U) dose-response of the *performance* gap Me7. Here
the outcome is a *geometric* quantity, the prediction is *monotone*,
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
  hypothesis (critic v2, ADR-0019). Here the hypothesis is monotone by
  construction, and the test is a repeated-measures monotone trend test.
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
9. **Sampling-noise floor.** S_split(c) is S between the top-k subspaces
   of two disjoint random halves (256 / 256, split seed = s) of Z_pre.
   It is reported per cell and used in the magnitude guard (§6).
10. **Rank guard.** If the numerical rank of centred Z_pre or Z_post is
    below k (singular value < 1e-6 × the largest), the cell is flagged
    and excluded. If more than 20 % of cells are excluded, the
    experiment is reported as a protocol failure, with no verdict.

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
| **S-1** (primary) | Y(s, ℓ) increases with ℓ | Page's L trend test, one-sided (increasing), blocks = seeds, 7 ordered treatments, α = 0.05 |
| S-2 (secondary) | Lesion-specific reorganisation exists: mean over seeds of Ȳ(s) = mean_ℓ Y(s, ℓ) is > 0 | one-sample t-test, one-sided, α = 0.05 |
| S-3 (secondary) | Per-seed monotone association | per-seed Spearman ρ(ℓ, Y); Wilcoxon signed-rank on ρ > 0, one-sided |
| S-4 (exploratory) | ΔS tracks performance harm | Spearman between cell ΔS and the cell's T2 Me1 drop; no decision |
| S-5 (exploratory) | Parameter-subspace rotation shows the same trend | same Page test on the secondary metric; no decision |

Sensitivity analyses (reported, no decision): k ∈ {6, 24}; SNR floor
cells only; per lesioned modality.

## 6. Decision criteria (pre-stated)

- **Supported:** S-1 Page p < 0.05 **and** the median over seeds of
  [Y(s, 175) − Y(s, 25)] exceeds the median S_split over all cells
  (magnitude guard: the trend is larger than PCA sampling noise).
- **Detectable but sub-floor:** S-1 p < 0.05, magnitude guard fails.
  Reported as such, with no substantive claim.
- **Not supported:** S-1 p ≥ 0.05. Reported as a pre-registered null.
- **Contrary (exploratory):** the mirrored (decreasing) Page test has
  p < 0.05. Reported as exploratory only; the confirmatory direction was
  increasing.

Paper consequence: if Supported, the paper may state a pre-registered
monotone geometric correlate of lock timing. It is a new result, not a
B-1 rescue: §5.5 stays retracted and Me7 is not re-interpreted. All
outcomes are publishable.

## 7. Seeds, power and stopping rule

- **Pilot seeds:** 2101–2105. They are used to validate the pipeline,
  check the rank guard and estimate the within-seed SD of Y. Pilot data
  are excluded from every confirmatory test.
- **Confirmatory seeds:** 2001–2020 (N = 20), disjoint from every seed
  used so far (0–4, Q3/Q3+, N12) and from the v0.7 / v0.9 drafts.
  **Reserve:** 2021–2024, for technical replacement only.
- **Power (no pilot data yet).** `scripts/power_prereg_2026_10.py`,
  one-sided Page test, 7 levels, linear trend with a seed random
  intercept. Results are in units of the within-seed SD of Y:

  | N | range 0.5 SD | range 1.0 SD | range 1.5 SD |
  |--:|-------------:|-------------:|-------------:|
  | 10 | 0.36 | 0.80 | 0.98 |
  | 20 | 0.58 | 0.98 | 1.00 |
  | 30 | 0.72 | 1.00 | 1.00 |

  N = 20 gives ≥ 0.80 power when the total trend across the grid is
  ≥ about 0.75 within-seed SD, and 0.98 at 1 SD. The PI may raise N
  after the pilot, never lower it below 20, and must fix it on the OSF
  form at filing.
- **Fixed-N design.** There is no interim look at outcomes and no
  optional stopping. A technical failure (crash, non-finite S, missing
  capture) is replaced from the reserve list in order, blind to the
  outcome. If more than 4 replacements are needed, the run halts and a
  deviation note is filed. Extending N after an inconclusive result
  needs a new amendment filed before the new data.

## 8. What is NOT being amended

- Thresholds 0.05 / 0.10 / 0.02 and every existing metric.
- ADR-0019 verdicts and the §5.5 FINAL text.
- The Q3/N12 LOCK_AFTER grid {50, …, 150}. This amendment adds levels
  25 and 175 for its own outcome only.

## 9. Pre-registration fidelity checklist

- [ ] Filed on OSF by the PI before any confirmatory seed (2001–2024) is run.
- [ ] Metric code + unit tests merged, commit hash on the OSF form.
      Required tests: S = 0 for identical subspaces, S = 1 for
      orthogonal subspaces, invariance to basis rotation.
- [ ] Probe-size parameter (512, decoupled from batch 16) and the
      no-lesion control arm implemented and frozen.
- [x] No threshold or existing-metric change.
- [x] Hypothesis is monotone; no peaked-shape test.
- [x] Seeds disjoint from all previous and sibling-draft seeds.

## 10. Open decisions (PI, before filing)

1. Magnitude guard: median S_split (drafted) vs a fixed SESOI on ΔS
   chosen after the pilot.
2. k = 12 (from ADR-0019 D_eff on gaussian/Studyforrest) vs
   re-estimating D_eff on the pilot.
3. N: 20 (drafted) vs a larger N from the pilot SD.
4. Parameter list for the secondary metric (which `CrossModalNerve`
   weights count as the interaction layer).
5. Whether to also run the CC-film bridge as a secondary world.
6. Pre-registration id (N-id) and final amendment number.
