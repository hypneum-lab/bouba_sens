# OSF pre-registration (standalone) — bouba_sens v0.8 (subspace shift vs LOCK_AFTER)

**Status:** DRAFT — not yet filed on OSF (to be filed by the PI)
**Parent pre-registration:** none. **This draft is a standalone pre-registration** (corrected on 2026-10-25, PI decision of 19:04, after the pilots). bouba_sens has no filed pre-registration: OSF `10.17605/OSF.IO/Q6JYN` covers only the dream-of-kiki hypotheses H1–H4 (issue #20), and the bouba_sens amendments v0.4–v0.6 under `docs/osf/` were drafted but never filed. The thresholds 0.05 / 0.10 / 0.02 used here are those of the local, git-dated design spec `docs/superpowers/specs/2026-04-20-bouba-sens-design.md` (commit `87f723f`, §1.2); the file name keeps "amendment" for continuity only.
**Earlier local drafts (never filed):** `amendment-v0.4-studyforrest.md`, `amendment-v0.5-studyforrest-5modal.md`, `amendment-v0.6-sprint10-17.md`
**Sibling drafts (same date, filed independently):** `amendment-v0.7-tost-equivalence-2026-10-23.md`, `amendment-v0.9-decorrelation-window-2026-10-23.md`
**Tag:** `bouba_sens/v0.8-subspace-shift` (to be created at filing)
**Draft date:** 2026-10-23
**Redesign pending (PI decision 2026-10-25, 19:04, after the pilot):** this draft is **not fileable as drafted**. The pilot showed a problem with the measurement and guard (the guard is 25 within-seed SDs, k* cuts a flat spectrum, the codes are readout-noise dominated); a redesign is proposed in `docs/osf/redesign-proposals-v08-v09-2026-10-25.md` (section A) and will be piloted on fresh pilot seeds after the PI's review.
**Related audit:** ADR-0020 (SNR floors were nearly identical under default settings), `docs/adr/0020-snr-floors-nearly-identical-under-default-settings.md`, merged in PR #14. This draft corrects the SNR schedule (§4).
**Pre-registration id:** **N19** (confirmed by the PI on 2026-10-25, 19:04). Basis, read on 2026-10-24: the HYPNEUM-PLANS series ends at N14 (`preregistrations/n14_latin_hypercube_7dim.md`) and its roadmap reserves N15–N17 (`2026-05-11-roadmap-Q2-Q3-2026.md`), so N18 / N19 / N20 are proposed for v0.7 / v0.8 / v0.9 in filing order; the last bouba_sens ADR was then 0020, so the verdict ADRs were proposed as ADR-0021 / 0022 / 0023 (shifted on 2026-10-25 to ADR-0023 / 0024 / 0025, ADR-0021 and ADR-0022 having been used by PRs #19 and #28 / #40 for the Me6 nulls); amendment numbers v0.7 / v0.8 / v0.9 follow v0.6, the last amendment on file, and assume filing in that order.
**Verdict ADR:** **ADR-0024** (confirmed by the PI on 2026-10-25, 19:04; ADR-0021 and ADR-0022 are used by the Me6 null calibration and paired null, so v0.7 / v0.8 / v0.9 take ADR-0023 / 0024 / 0025)

> **Filing note.** This is the machine-readable draft. No confirmatory
> seed has been run; the pilot (seeds 2101–2110) is run with the
> driver below (`docs/osf/pilot-report-2026-10-25.md`). The metric, probe capture, control arm, rank rules,
> aggregation and Page test are merged with unit tests (PRs #10 and #13:
> `metrics/subspace.py`, `subspace_protocol.py`,
> `metrics/subspace_aggregate.py`, `metrics/page_trend.py`). The driver
> that trains the Phase-1 checkpoints and calls `capture_block` for
> every (seed, lock) block, and the pilot planning computations, were
> added on 2026-10-25 (`bouba_sens.v08_driver`,
> `scripts/run_v08_subspace.py`, `scripts/pilots/derive_v08_pilot.py`).
> Still missing: the S-2 to S-5 analyses, the secondary
> parameter-subspace metric in the driver and the CC-film arm (§9). All of it is frozen
> (commit hash on the OSF form) before any confirmatory seed runs. Pilot
> seeds (§7) may run before filing.

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
Geometric Perspective on Missing-Modality Robustness" (arXiv:2610.04792,
v2 revised 2026-10-13; the arXiv comment reads "NeurIPS 2026 Main
Conference"), report that multimodal training induces structured
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

The primary metric S is not Sui et al.'s quantity and is not a
replication of it. Sui et al. use the mean principal angle (in degrees)
between **weight** singular subspaces as their diagnostic. S is the
normalised chordal (squared sine) distance between **representation**
subspaces, the top-k principal directions of the fused codes (§3). The
secondary parameter-subspace metric of §3 is the closer analogue of
their measurement.
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
   **Gumbel probe readout (PI decision 2026-10-25, after the v0.7 B-2
   pilot; supersedes the hard-demodulation rule of 2026-10-23).** Both
   captures use the Gumbel-softmax demodulation used in training
   (`probe_demod="soft"`, `fuse(hard_demod=False)`, τ = 1), as in
   ADR-0019. The hard (argmax) path mapped every probe row to one code
   on every v0.7 pilot seed and on a seed-0 check of this protocol
   (LOCK_AFTER 25 and 175: Z_pre rank 0), which would make every rank
   check of §3.1 fail (`docs/osf/pilot-report-2026-10-25.md`). The
   captures are seeded as below, so Z_pre is bit-identical across the
   16 arms of a block (checked in code). S includes the remaining
   Gumbel sampling noise of the 8-draw means in both captures; S_split (item 9) is computed on the same Gumbel
   Z_pre.
   **K = 8 Gumbel draws (PI decision 2026-10-25, 13:12).** Each capture
   (Z_pre, every Z_post, the control end matrix) is the **mean of 8
   Gumbel captures** of the same network and probe: draw r = 0 … 7 of
   phase p (0 = Z_pre, 1 = end of Phase 2) is taken inside
   `torch.random.fork_rng` after `torch.manual_seed(2·10⁹ + 1000·s₂ +
   100·p + r)`, with s₂ the Phase-2 seed of item 11
   (`subspace_protocol.PROBE_GUMBEL_DRAWS`, `lesion_phase(...,
   probe_gumbel_draws=8)`), so the draws do not touch the training RNG
   stream and Z_pre is identical across the 16 arms. v0.8 has no MI
   estimate: the PI's "average each network's estimate over K draws"
   is implemented here by averaging the **codes** over the 8 draws
   before S, the rank checks, S_split and k* are computed (averaging S
   itself would need 8 rank-checked matrices per capture); code
   averaging was confirmed by the PI on 2026-10-25 (13:32). A deterministic soft
   (probability-based, non-sampled) readout may be added later as a
   declared secondary comparison; it is not part of this draft.
   Training and lesion-phase steps are unchanged.
3. **Subspaces.** Column-centre each matrix. V_pre, V_post ∈ ℝ^{896×k}
   are the top-k right singular vectors, i.e. the top-k principal
   directions in feature space. **k is fixed on the pilot (PI decision
   2026-10-24)** by this deterministic rule, frozen before any
   confirmatory seed: for every pilot block (s, ℓ) with s in 2101–2110,
   compute the plug-in participation ratio PR = (Σ λ)² / Σ λ² of the
   centred covariance of its Z_pre (Gumbel readout, 512 × 896);
   k* = the median of these PRs (mean of the two middle values for an
   even count), rounded half up, clamped to [1, 255] (255 = the centred
   rank bound of a 256-row split half). Non-finite Z_pre is a technical
   failure, as in §3.1, and is not silently dropped. Only pilot data
   enter the rule; k* and every PR are recorded on the OSF form before
   filing and reported in the paper. Implemented as
   `bouba_sens.metrics.subspace.select_k_from_pilot`, unit tests
   `tests/unit/test_subspace_k_selection.py`. The plain (plug-in)
   participation ratio was confirmed by the PI on 2026-10-24; the
   finite-sample-corrected ratio of Chun et al. (§9) is reported
   alongside only and never changes k*. Context: the ADR-0019
   Me3 probe had a participation ratio of 11.7 (two-NN 12.4), which
   motivated the earlier fixed k = 12 (`K_DEFAULT`, kept only as the
   code default until k* is fixed). Below, k denotes k*.
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
11. **Seed mapping (frozen; implemented by `capture_block`).** For
    seed s and lock ℓ: the global torch RNG is re-seeded with s
    immediately before the Phase-1 network is built (so initial weights
    and Gumbel noise depend on s only), and the Phase-1 checkpoint is
    trained with seed s and `LOCK_AFTER` = ℓ (batch 16, 200 steps;
    `v08_driver.train_checkpoint`). Every Phase-2 arm of
    the block (the no-lesion control and the 15 cells) restores that
    checkpoint into a fresh loop and uses one Phase-2 seed, s, for the
    probe (`world.sample(512, seed = s − 99_999)`) and for the fresh and
    replay batches, and re-seeds the global torch RNG with s before the
    arm, so lesion noise draws are common to the 16 arms. The 7 blocks
    of a seed share the same Phase-2 seed and probe; only the
    checkpoint (ℓ) differs. `run_grid.sh`, which gives every cell its
    own seed (seed × 10 000 + cell count), is not used for v0.8.

### 3.1 Rank guard, exclusion denominators and incomplete blocks

A matrix *passes the rank check* if its column-centred numerical rank
is at least k. The numerical rank is 0 if the largest singular value of
the centred matrix is 0 (a zero or constant code); otherwise it is the
number of singular values ≥ 1e-6 × the largest (which is then > 0).
Unit tests cover zero and constant-code fixtures (rank 0, check
failed). The check
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
  defined at all 7 levels. The primary analysis (Page test, magnitude
  guard) and S-2/S-3 use complete blocks, so both endpoint levels
  (ℓ = 25 and 175) are defined for every analysed seed.
- **Selection and exchangeability (added 2026-10-24 after the Codex
  round-2 review).** Rank exclusions are outcome-dependent: whether a
  block is complete can depend on the same codes that determine Y.
  Conditioning on completion can therefore break the within-block
  exchangeability that the Page permutation test assumes, and the
  complete-block analysis alone is **not** guaranteed to keep its
  nominal error rate. Example (Codex): N = 40, 7 i.i.d. levels, no
  trend, a block kept only if its endpoint difference exceeds its null
  14.9th percentile (about 85 % kept, above the 70 % threshold): 27 %
  of runs would be "Supported" (20,000 draws; 4.7 % with exclusions
  independent of the outcome).
- **Selection safeguard (frozen; exact worst case since Codex round 3,
  2026-10-24).** A second analysis on **all N blocks** is required for
  "Supported". For each block, the undefined values are completed in
  the way that **minimises** the block's Page contribution Σ_j j·R_j,
  over every strict ranking of the 7 levels that is consistent with
  the order of the observed values (ties among observed values are
  broken against the trend); a block with no defined value contributes
  the fully reversed minimum 84. The worst-case statistic L_wc is the
  sum of these minima, and its p-value is computed against the
  registered complete-data null (each block a uniform permutation of
  ranks 1–7; 100,000 permutations, seed 20261023). For the guard, a
  block whose endpoint Y(s, 25) or Y(s, 175) is undefined has an
  endpoint contrast of −∞. "Supported" needs the Page test and the
  guard to pass in **both** analyses (§6). With no excluded block and
  no tied values the two analyses coincide. With observed ties they
  need not: the registered complete-block test uses mid-ranks, while
  the worst-case statistic breaks each tie group against the trend
  (reversed ranks within the group). That choice minimises the
  group's contribution, so it is at most the contribution of every
  strict refinement of the ties, and hence at most the mid-rank
  statistic (the average of those refinements). The safeguard is
  therefore still conservative with ties; its reference is the
  strict-rank (unconditional) null, not the mid-rank null conditional
  on the observed tie pattern.

  *Why this is conservative under any exclusion mechanism.* Let L* be
  the Page statistic of the complete data that would have been
  observed without exclusions. Exclusion only removes values; it never
  changes the observed ones. Each block's minimum is taken over a set
  of completions that contains the true one, so L_wc ≤ L* for every
  realisation, whatever rule decided which values were removed (even
  one that looked at the data). Under the null of within-block
  exchangeability of the complete data (the registered assumption),
  L* has the registered permutation distribution, so
  P(L_wc ≥ c) ≤ P(L* ≥ c) ≤ α at the registered critical value. The
  only assumptions are this null and that a missing value is a removed
  value (Y is not altered by the exclusion process); ties in Y have
  probability zero for continuous S and, if they occur, are handled by
  the refinement argument above.

  *Earlier version withdrawn.* The first safeguard (round 2) imputed
  missing values in a fixed direction (high for ℓ ≤ 100, low above).
  Codex showed that this is not a worst case: imputing one missing
  interior value above the block maximum can raise a block's L by up
  to 6, and a selection that exploits it kept 7.3 % false positives.

  *Checks (unit tests).* The completion never exceeds the true L over
  all 5,040 orders × every 1- and 2-missing pattern, and equals an
  independent brute-force completion. Null false-positive rates at
  N = 40 (20,000 draws, script section "v0.8 selection safeguard"),
  complete blocks only → with the safeguard: Codex's outcome
  counterexample 26.9 % → 1.0 %; Codex's adversarial interior-deletion
  selection (the 6 lowest-L blocks among those where the old imputation
  would gain) 44.2 % → 4.1 %; a block-local version (every block with
  such a gain and L below the null 15th percentile) 35.9 % → 4.3 %. The tests require at most 5 % plus a
  stated tolerance of 1 percentage point plus 3 Monte-Carlo SEs.
  Implemented as `page_trend.worst_case_block_l` and
  `page_worst_case_test`, `subspace_aggregate.worst_case_page_test`,
  `worst_case_endpoint_contrast` and `verdict(...,
  p_value_worst_case=...)`.

  *Alternatives considered.* (a) A test of whether exclusion depends on
  level or arm cannot detect this failure, because the selection acts
  through the outcome. (c) Inverse-probability weighting needs a model
  of the exclusion probability that cannot be estimated from the few
  excluded blocks. The worst-case bound needs no model and costs
  nothing when nothing is excluded.
- **Protocol failure.** If fewer than **70 %** of the N confirmatory
  seeds are complete blocks (fewer than 14 of 20, 21 of 30 or 28 of
  40), the experiment is reported as a protocol failure with no
  verdict. (Changed on 2026-10-24 from 80 %; rationale in §10.)
- **Reporting.** The number of invalid cells and of undefined Y is
  reported per level ℓ, with denominators (N × 15 cells and N blocks
  per level). If exclusions differ across levels, the paper states that
  the selection of complete blocks may itself depend on ℓ.

T1 cells are excluded from the primary metric: their pre-lesion codes
come from an untrained network, so S would measure training from
scratch, not lesion reorganisation.

**Secondary metric (exploratory, closer to Sui et al.):** a
parameter-subspace rotation. **Parameter list (frozen 2026-10-24,
`interaction_parameter_names`):** the shared codebook projection
`codebook.codebook` (64 × 128) and the `fc1.weight` and `fc2.weight`
matrices (128 × 128) of the 20 pairwise transducers of
`CrossModalNerve`, 41 matrices in the nerve's own order; biases, the
1-D gate vector and modules outside the nerve are excluded. For each
matrix, S between the top-k right singular subspaces at Phase-2 entry
and Phase-2 end (weights not centred; k = min(k*, rank at entry);
`parameter_subspace_shifts`). A matrix that is bit-identical before and
after received no update (under HARD gating a transducer fires only
when its gate condition holds) and is reported as **inactive**, not as
a shift of 0. Per arm: ΔS_param = S(lesion cell) − S(no-lesion
control) per active matrix; the block value is the median over the
cells and active matrices, reported with the number of active
matrices; the codebook value is also reported on its own, since the
codebook is updated at every step. Exploratory, no decision.

## 4. Conditions

- World: `StudyforrestRealWorld` (4.5-modal real bridge), as in Q3/N12.
- **Secondary world (exploratory, decided 2026-10-24): CC-film.**
  `--world ccfilm` (`CCFilmWorld`: real audio and vision from a CC-BY
  film, synthetic proprioception; ADR-0019 E1.3d), run only after the
  confirmatory Studyforrest runs are complete, on the first 10
  analysed confirmatory seeds (2001–2010, or their reserve
  replacements), all 7 levels, same k*, protocol and analysis. Its
  Page test and endpoint difference are reported descriptively with
  their own exclusion counts; it supports no claim and never changes
  the verdict. If the CC-film cache is absent on the filing host, the
  arm is not run and this is stated. Rationale in §10.
- `STEPS_TRAIN=200`, `STEPS_LESION=100`, batch 16, HARD gating, no
  codebook lock, no gating schedule.
- **SNR schedule: changed versus the historical grids (ADR-0020).** The
  historical grids ran `m2_snr_schedule` with `k_steps` = 5000 over 100
  lesion steps and only reached 19.21 / 19.41 / 19.80 dB for the
  nominal floors −20 / −10 / +10 dB (ADR-0020, PR #14). Here every
  lesion run uses SNR(t) = max(floor, 20 + (floor − 20) · min(1, t / k)),
  t = 0 … 99, with **`k_steps` = 50** (the floor is reached at step
  t = k, 0-indexed, and is held until the last lesion step, so
  `k_steps` = N reaches the floor at step N, N ≤ 99). The floors −20.00 / −10.00 / +10.00 dB are reached at
  step 50 and held to step 99; mean SNR over the 100 steps is −9.80 /
  −2.35 / +12.55 dB. The three SNR levels f in the cell definition (§3)
  are therefore distinct severities, which they were not historically.
  Consequence: cell values are not comparable with the historical
  grids. Code: `capture_block` defaults to `k_steps` = 50 (constant
  `subspace_protocol.K_STEPS`), and `run_grid.sh` forwards `K_STEPS`
  to `--k-steps` if a grid is ever run for this amendment. The CLI and
  `m2_snr_schedule` defaults are unchanged.
- **World check.** `--world studyforrest` falls back, without an
  error, to a 2-modal stub when the real cache is absent
  (`cli.py:54-63`); the driver records the world class and a run on the
  stub is a technical failure.
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

Sensitivity analyses (reported, no decision): k ∈ {⌊k*/2 + 0.5⌋, 2k*}
(at least 1, at most 255; {6, 24} if k* = 12); nominal-floor (−20 dB, reached)
cells only; per lesioned modality. Also k ∈ {k* − 4, k* + 4} (at least 1; descriptive;
exploratory): plug-in empirical singular vectors leak out of the
population subspace, with a bias that grows for directions of smaller
singular value (Song, Yang & Ma, "Bias-Corrected Subspace Intersection:
Minimax-Optimal Shared Subspace Estimation in Multi-View Data",
arXiv:2609.05617, a different estimation problem, cited for the bias
mechanism only). Directions near the k* cutoff are the most
affected, so S could vary with ℓ through spectrum changes alone. The
S_split floor (§3, item 9) partly controls for this.

*Subspace stability (reported, no decision).* For every Z_pre and Z_post
the eigengap ratio λ_k* / λ_(k*+1) of the centred covariance is reported per
cell (arXiv:2609.05617 motivates this report; descriptive). A
ratio below 1.1 marks a top-k* subspace that is unstable to small
perturbations; the share of such matrices is reported per level, and
the S-1 test is repeated on blocks without them as a sensitivity
analysis.

## 6. Decision criteria (pre-stated)

The verdicts are exhaustive and mutually exclusive, evaluated in
this order:

- **Protocol failure:** fewer than 70 % complete blocks (§3.1). No
  verdict.
- **Supported:** S-1 permutation p < 0.05 on the complete blocks
  **and** the worst-case Page test on all blocks has p < 0.05 (§3.1,
  selection safeguard), **and** the guard passes in both analyses (in
  the worst-case analysis an undefined endpoint counts as −∞): the median over
  complete blocks of [Y(s, 175) − Y(s, 25)] exceeds the median S_split(s, ℓ)
  over the complete blocks and all 7 levels. **Magnitude guard (PI
  decision 2026-10-24: median S_split, as drafted).** This is a
  **heuristic guard**, not a calibrated noise test and not a SESOI: the half-sample
  pre/pre distance is not the sampling distribution of the full-sample,
  control-subtracted endpoint contrast. The guard is part of one
  conjunctive rule; the Page p-value alone is not advertised as a
  separate finding.
- **Not robust to exclusions (added 2026-10-24):** S-1 p < 0.05 on
  the complete blocks, but the worst-case Page test on all blocks
  (§3.1, selection safeguard) has p ≥ 0.05. Reported as "an
  increasing ordered trend among complete blocks that does not survive
  a worst-case treatment of the excluded values"; no claim.
- **Detectable but sub-floor:** S-1 p < 0.05 in both analyses, but the
  magnitude guard fails in at least one of them.
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

- **Pilot seeds (PI decision 2026-10-24): exactly 10, 2101–2110,
  fixed before the pilot runs; never extended.** They are used to
  validate the pipeline, fix k* (§3 item 3), check the rank guard and
  estimate the within-seed SD of Y, the median S_split and the
  exclusion rate. Order: k* is computed from the pilot Z_pre first,
  then every pilot quantity that depends on k (rank checks, S, S_split,
  Y) is computed with k*. Pilot data are excluded from every
  confirmatory test.
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
  < 70 % complete blocks):

  | N | q | range | g = 0 | g = 0.25 | g = 0.5 | g = 1.0 | P(protocol failure) |
  |--:|--:|------:|------:|---------:|--------:|--------:|--------------------:|
  | 20 | 0 | 0.75 SD | 0.84 | 0.81 | 0.69 | 0.25 | 0.000 |
  | 20 | 0 | 1.0 SD | 0.97 | 0.96 | 0.90 | 0.50 | 0.000 |
  | 20 | 0 | 1.5 SD | 1.00 | 1.00 | 1.00 | 0.90 | 0.000 |
  | 20 | 0.10 | 1.0 SD | 0.96 | 0.93 | 0.87 | 0.50 | 0.002 |
  | 30 | 0 | 1.0 SD | 1.00 | 0.99 | 0.94 | 0.51 | 0.000 |
  | 30 | 0.10 | 1.0 SD | 0.99 | 0.98 | 0.93 | 0.50 | 0.000 |

  (q = 0.10 rows regenerated on 2026-10-24 under the 70 % rule; the
  q = 0 rows do not depend on the threshold.)

  *At the planning exclusion rate of the fixed pilot* (q̂ = 0.149 at
  0/10, trend ĝ + 0.70 SD, 20,000 draws, section "v0.8 complete rule at
  q=0.149" of the script):

  | Threshold | g | N = 20 | N = 25 | N = 30 | N = 35 | N = 40 |
  |---|---|---:|---:|---:|---:|---:|
  | 70 % | 0 | 0.72 | 0.80 | 0.87 | 0.91 | 0.94 |
  | 70 % | 0.25 | 0.89 | 0.92 | 0.96 | 0.97 | 0.98 |
  | 70 % | 0.5 | 0.93 | 0.94 | 0.97 | 0.98 | 0.99 |
  | 80 % (earlier) | 0 | 0.62 | 0.69 | 0.75 | 0.79 | 0.82 |
  | 80 % (earlier) | 0.25 | 0.76 | 0.80 | 0.83 | 0.84 | 0.86 |
  | 80 % (earlier) | 0.5 | 0.79 | 0.82 | 0.84 | 0.85 | 0.86 |

  *With the selection safeguard (§3.1; Codex round 2, 2026-10-24).*
  `v08_rule_selection` in the script (section "v0.8 selection
  safeguard", 20,000 draws per cell, q = 0.149) compares the complete-
  block rule alone with the safeguarded rule under three exclusion
  models, all with about 85 % of seeds complete: **level** (each
  (s, ℓ) value undefined independently, so a seed is usually
  incomplete at one level; the rank checks act per (s, ℓ) matrix),
  **seed** (whole seed lost independently, pessimistic; `independent` in
  the script) and **outcome**
  (whole seed lost when its endpoint difference is below its null
  14.9th percentile, Codex's counterexample).

  | Exclusion | trend, g | N = 20 | N = 30 | N = 40 |
  |---|---|---:|---:|---:|
  | level | null (0, 0) | 0.047 / 0.018 | 0.044 / 0.015 | 0.048 / 0.013 |
  | level | 0.70, 0 | 0.73 / 0.61 | 0.87 / 0.76 | 0.94 / 0.86 |
  | level | 0.95, 0.25 | 0.89 / 0.83 | 0.96 / 0.93 | 0.99 / 0.97 |
  | seed | null | 0.047 / 0.004 | 0.050 / 0.002 | 0.052 / 0.001 |
  | seed | 0.70, 0 | 0.73 / 0.20 | 0.87 / 0.22 | 0.94 / 0.24 |
  | seed | 0.95, 0.25 | 0.89 / 0.37 | 0.96 / 0.44 | 0.99 / 0.50 |
  | outcome | null | 0.169 / 0.019 | 0.226 / 0.013 | 0.269 / 0.010 |
  | outcome | 0.70, 0 | 0.89 / 0.64 | 0.97 / 0.77 | 0.99 / 0.85 |

  (complete-block rule alone / safeguarded rule.) The safeguard keeps
  the false-positive rate below 5 % in all three models, including the
  outcome-dependent one where the complete-block rule alone reaches
  17–27 %. Its cost is small when exclusions are isolated levels and
  large when whole seeds are lost, because a wholly lost seed then
  counts as a fully reversed block (L = 84). With no exclusion it costs
  nothing. Calibration: the simulator compares L with the strict-rank
  permutation critical value (`page_perm_crit`); for tie-free data this
  is the registered mid-rank permutation test (`page_trend`), and the
  worst-case completion uses strict ranks, so no tie arises (a unit
  test checks that the two take the same decision on tie-free data).
  The table was regenerated after the Codex round-3 fix (exact worst
  case).

  These numbers are **conditional on the simulation model**: a linear
  trend, Gaussian noise, a guard g treated as a fixed known constant,
  and rank exclusions independent of the outcome and of ℓ. None of these
  is verified. The guard dominates: when g is as large as the expected
  endpoint difference, P(Supported) is about 0.5 whatever N. Exclusions
  also cap the attainable power: with a per-block exclusion probability
  of 0.20, the protocol-failure probability is 0.087 / 0.061 / 0.043 at
  N = 20 / 30 / 40 under the 70 % rule (0.370 / 0.393 / 0.407 under the
  earlier 80 % rule, which made 0.80 unconditional power unreachable).

  *Drafted rule for N.* From the 10 pilot seeds: σ̂ =
  residual SD of Y from the two-way ANOVA without replication (seed ×
  level) fitted on the **complete** pilot blocks only (df = (n_c − 1) ×
  6 for n_c complete blocks; at least 3 complete blocks are needed;
  with fewer, the pilot is **not** extended and the amendment is not
  filed as drafted), inflated to its one-sided UCL80 (χ², same df); ĝ =
  median of S_split(s, ℓ) over the **complete** pilot blocks and all 7
  levels, divided by the UCL80-inflated σ̂ (the same denominator as the
  guard of §6; `v08_driver.pilot_quantities`). The planning exclusion rate q̂ is the exact
  (Clopper–Pearson) one-sided **80 % upper bound** of the pilot
  incomplete-block fraction, not the point estimate: 0/5 incomplete
  gives q̂ = 0.275, 0/10 gives 0.149, 0/20 gives 0.077; with the fixed
  10-seed pilot, q̂ = 0.149 at 0/10. A large q̂ can still make the
  target unreachable (q̂ = 0.275 gives a protocol-failure probability of
  about 0.29 at every N ≤ 40 under the 70 % rule); because the pilot size is fixed, the
  Clopper–Pearson bound keeps its nominal coverage (no data-dependent
  extension). N is the smallest N ≥ 20 for which
  the simulated P(Supported) **of the safeguarded rule**
  (`v08_rule_selection`), **minus two Monte-Carlo SEs**, is ≥ 0.80,
  with exactly 100 000 draws per candidate, the permutation-calibrated
  Page test, a total trend of ĝ + 0.70 SD units and the simulation
  generator `numpy.random.default_rng(20261023 + 808)`, candidates
  N = 20, 21, … run in order from one generator
  (`scripts/pilots/derive_v08_pilot.py`). Exclusion model
  for planning (frozen, from the pilot pattern): **level** if no pilot
  seed has Y undefined at more than one level, **seed** otherwise
  (outcome-dependent selection cannot be estimated from a pilot and is
  covered by the safeguard, not by N).
  If no N ≤ 40 qualifies (for example because q̂ is large), the
  amendment is not filed as drafted: the PI either files N = 40 with its
  computed power stated on the form as below target, or revises the
  rank rule before filing (k* is fixed by the rule of §3 item 3 and is
  not tuned to reach power). N is fixed on the OSF form; it is never
  below 20.
  *Pilot result (2026-10-25, seeds 2101–2110, macM1, commit `aef0988`;
  `reports/pilots/v08_pilot_quantities.json`, `…_n_rule.json`,
  `docs/osf/pilot-report-2026-10-25.md`).* Gumbel readout, K = 8
  averaged codes. **k* = 13** (median plug-in PR 12.72 over the 70
  pilot Z_pre; range 10.74–13.39; ADR-0019 value 11.7). Rank checks:
  0 invalid cells of 1,050, **10 / 10 complete blocks**, so q̂ = 0.149
  (Clopper–Pearson 80 % bound of 0/10) and the exclusion model is
  **level**. σ̂ = 0.00293 (df 54), UCL80 **0.00320**; median S_split
  over the complete blocks **0.0739**; **ĝ = 23.09** within-seed SD.
  N rule: at N = 20 the safeguarded P(Supported) is 0.9070 (lower
  bound 0.9052; P(protocol failure) 0.020) with a total trend of
  ĝ + 0.70 = 23.79 SD, so **N = 20**. Runtime: about 365 s per seed
  (7 blocks, 51–59 s each), about 2.0 h at N = 20. Reported (no
  decision): Y ranged −0.0074 to +0.0159 (median +0.0008), an order
  of magnitude below the guard; 66 % of the Z_pre / Z_post matrices
  have an eigengap ratio λ_k* / λ_(k*+1) below 1.1. Neither number
  changes N or any rule.
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
- [x] Metric code + unit tests merged (PR #10,
      `tests/unit/test_metrics_subspace.py`: S = 0 for identical
      subspaces, S = 1 for orthogonal subspaces, invariance to basis
      rotation, rank 0 for zero/constant codes). Commit hash recorded on
      the OSF form at filing.
- [ ] Driver merged with unit tests: Phase-1 checkpoints per (seed,
      lock), `capture_block` per block with the seed mapping of §3
      item 11, world-class check, aggregation and Page test end to end;
      S-2 to S-5 analyses; secondary parameter-subspace metric (after
      the parameter list of §3.1); CC-film exploratory arm (§4).
- [x] Selection safeguard (§3.1, §6; Codex rounds 2–3) implemented with
      unit tests: `worst_case_block_l`, `page_worst_case_test`,
      `worst_case_page_test`, `worst_case_endpoint_contrast`,
      `verdict(..., p_value_worst_case=...)`, protocol failure at 70 %
      in code (`tests/unit/test_subspace_aggregate.py`), and the
      outcome-dependent counterexample
      (`tests/unit/test_power_planning_sd.py`), with the exact
      worst-case completion (`tests/unit/test_page_worst_case.py`). The
      aggregation API requires both p-values (`verdict`); the driver
      that computes them is pending (item above).
- [x] Secondary-metric parameter list and per-matrix shift implemented
      (`interaction_parameter_names`, `parameter_subspace_shifts`, unit
      tests `tests/unit/test_subspace_parameters.py`).
- [x] Pilot rule for k (§3 item 3, PI decision 2026-10-24) implemented
      with unit tests (`select_k_from_pilot`,
      `tests/unit/test_subspace_k_selection.py`: definition, equal
      spectrum, invariances, zero codes, round half up, even-count
      median, order-free, clamps, non-finite refused).
- [ ] k* computed on the 10 pilot seeds and recorded on the form with
      every pilot PR. Descriptive alongside it (does not change k*): the
      finite-sample-corrected participation ratio of Chun, Canatar,
      Chung & Lee, "Estimating Dimensionality of Neural Representations
      from Finite Samples" (arXiv:2509.26560), on the same pilot codes,
      next to the ADR-0019 value 11.7.
- [x] Probe-size parameter (512, decoupled from batch 16), probe
      readout option and the no-lesion control arm implemented (PR #13,
      `lesion_phase(probe_size, probe_seed, probe_demod)`,
      `subspace_protocol.capture_block`, default `k_steps` = 50);
      `capture_block` uses the Gumbel readout (`PROBE_DEMOD = "soft"`)
      since the PI decision of 2026-10-25.
- [x] Rank checks of §3.1 implemented with unit tests (rank-deficient
      control invalidates its 15 cells; 12-of-15 rule; complete-block
      selection; seed roster with reserve replacement), and the Page
      permutation test implemented (`metrics/page_trend.py`, PR #13).
- [ ] Pilot (exactly 10 seeds, 2101–2110) within-seed SD of Y, median
      S_split, rank-exclusion rate and runtime recorded on the form; N
      fixed on the form. (Pilot run 2026-10-25: k* = 13, σ̂ UCL80
      0.00320, median S_split 0.0739, 0 / 10 incomplete, N = 20, §7.)
- [x] No threshold or existing-metric change.
- [x] Hypothesis is an increasing ordered trend; no peaked-shape test.
- [x] Seeds disjoint from all previous and sibling-draft seeds.

## 10. Open decisions (PI, before filing)

Every item below must be resolved, and the text edited to a single
protocol, before filing. This draft is not fileable while any item is
open; the metric code, the pilot and a regenerated power analysis for
the final configuration are also filing prerequisites.

Decided by the PI on 2026-10-23 and applied above (no longer open):
hard-demodulation probe captures (§3 item 2); SNR ramp `k_steps` = 50
(§4, ADR-0020). **The hard-demodulation rule is superseded** by the PI
decision of 2026-10-25 (Gumbel readout, §3 item 2), taken after the
PI had seen the v0.7 B-2 pilot and before any v0.8 pilot seed was run;
it is stated on the OSF form as a post-pilot change.

Decided by the PI on 2026-10-24 and applied above (no longer open):
magnitude guard = median S_split, a heuristic guard (§6); k re-estimated
on the pilot by the frozen rule of §3 item 3 (k*); pilot fixed at 10
seeds, 2101–2110, never extended (§7). Pre-registration id **N19**,
verdict ADR **ADR-0024** (shifted twice on 2026-10-25: ADR-0021 and ADR-0022 are now used), amendment v0.8, as proposed in the header
(proposed — PI confirms at filing).

Decided on 2026-10-24 (second PI round, "if you find better, do it"):
each item below gives the decision, the rationale and what changed.

1. **Rank-exclusion thresholds.**
   - *12 of 15 valid cells per block: kept.* A block can lose up to
     three cells (for example the three SNR levels of one lesioned
     modality) and its median still rests on at least 12 cells of the
     15, so at least four of the five lesioned modalities are
     represented. A stricter rule would discard blocks for a single
     modality's collapse; a looser one (for example 9 of 15) would let a
     block's median rest on as few as three modalities. Unchanged.
   - *Protocol-failure threshold: changed from 80 % to 70 % complete
     blocks.* Under the planning exclusion rate of the fixed pilot
     (q̂ = 0.149 at 0/10), the 80 % rule alone fails the run with
     probability 0.167 / 0.149 / 0.131 at N = 20 / 30 / 40 and caps the
     complete-rule power below 0.86 even at N = 40 (P(Supported) at a
     trend of ĝ + 0.70 SD: 0.62–0.82 for g = 0 and 0.76–0.86 for
     g = 0.25, N = 20–40). With 70 % the failure probability is
     0.021 / 0.009 / 0.004, and the same targets give 0.72–0.94 and
     0.89–0.98 (table in §7). *Correction (Codex round 2,
     2026-10-24):* an earlier version of this item said that a lower
     threshold does not change any test's validity because the
     analysis uses complete blocks. That is withdrawn: rank exclusions
     are outcome-dependent, so selecting complete blocks can bias the
     Page test whatever the threshold (§3.1). The threshold only bounds
     how many blocks may be lost; validity under outcome-dependent
     selection is now protected by the worst-case safeguard of §3.1,
     required for "Supported" (§6). 70 % keeps a clear majority of
     seeds.
     The threshold is fixed before any data (`V08_COMPLETE_MIN` in
     `scripts/power_prereg_2026_10.py`, section "v0.8
     protocol-failure threshold").
2. **N: rule kept (N ≥ 20, smallest N whose simulated P(Supported)
   minus two Monte-Carlo SEs is ≥ 0.80, cap 40), now under the 70 %
   threshold.** The rule is deterministic, uses pilot data only (σ̂,
   ĝ, q̂) and is fixed before the confirmatory run; nothing in the
   literature or the numbers favours a fixed N over it, because the
   guard size, the within-seed SD and the exclusion rate are all
   unknown before the pilot. Illustration at q̂ = 0.149 with a trend of
   ĝ + 0.70 SD, complete-block rule alone: under 70 %, N = 30 (g = 0;
   0.799 at N = 25) and N = 20 (g ≥ 0.25) reach 0.80 at the point
   estimate; under the earlier 80 % rule, N = 40 was needed at g = 0
   and N = 25 at g = 0.25. With the selection safeguard now required
   (§3.1) and the level exclusion model, about N = 35 is needed at g = 0
   (0.76 at N = 30, 0.86 at N = 40) and N = 20 at g = 0.25 (0.83). Cost: per seed 7 Phase-1 + 112 Phase-2 runs; at the
   cap N = 40, 280 Phase-1 and 4,480 Phase-2 runs. Changed: only the
   threshold used inside the simulation.
3. **Secondary-metric parameter list: fixed now, before data**
   (§3.1, `interaction_parameter_names`, unit tests
   `tests/unit/test_subspace_parameters.py`): the codebook projection
   and the 40 transducer weight matrices; inactive (never-updated)
   matrices reported as inactive, not as zero shift. Rationale: these
   are the parameters through which modalities interact in
   `CrossModalNerve`; the gate vector is 1-D (no subspace) and the
   encoders and head are not cross-modal. Changed: the list was
   "fixed in code before filing" and is now fixed and implemented.
4. **CC-film as a secondary world: yes, exploratory, on 10 seeds**
   (§4). Rationale: the Studyforrest audio channel that the lesion
   targets is an annotation proxy, while CC-film audio is the real
   signal (ADR-0019 E1.3d, v0.7 §4.2), so a same-direction trend on
   CC-film would show that the result does not hinge on the proxy.
   Ten seeds keep the cost at half of the N = 20 confirmatory run
   (70 Phase-1 + 1,120 Phase-2 runs); the Page test alone has about
   0.80 power at a trend of 1.0 within-seed SD at N = 10 (§7), so the
   arm is descriptive, not confirmatory. Changed: from undecided to
   this exploratory arm.

No design decision remains open. Done on 2026-10-25: the driver
(capture and pilot planning), the pilot and N by the rule of §7 (N =
20, k* = 13). Filing still needs the S-2 to S-5 analyses, the
secondary parameter-subspace metric and the CC-film arm in the driver,
the corrected participation ratio of Chun et al., the recording of the
pilot quantities, k* and N on the OSF form, and the PI's confirmation
of the proposed numbers.

