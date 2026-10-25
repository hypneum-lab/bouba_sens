# OSF pre-registration amendment — bouba_sens v0.9 (decorrelation-window lesion)

**Status:** DRAFT — not yet filed on OSF (to be filed by the PI)
**Parent pre-registration:** `dream-of-kiki` OSF DOI `10.17605/OSF.IO/Q6JYN` (unchanged by this draft)
**Previous amendment on file:** `amendment-v0.6-sprint10-17.md`
**Sibling drafts (same date, filed independently):** `amendment-v0.7-tost-equivalence-2026-10-23.md`, `amendment-v0.8-subspace-shift-2026-10-23.md`
**Amendment tag on file:** `bouba_sens/v0.9-decorrelation-window` (to be created at filing)
**Draft date:** 2026-10-23
**Related audit:** ADR-0020 (SNR floors were nearly identical under default settings), `docs/adr/0020-snr-floors-nearly-identical-under-default-settings.md`, merged in PR #14. This draft corrects the SNR schedule of the D-4 lesion phase (§4).
**Pre-registration id:** **N20** (proposed — PI confirms at filing). Basis, read on 2026-10-24: the HYPNEUM-PLANS series ends at N14 (`preregistrations/n14_latin_hypercube_7dim.md`) and its roadmap reserves N15–N17 (`2026-05-11-roadmap-Q2-Q3-2026.md`), so N18 / N19 / N20 are proposed for v0.7 / v0.8 / v0.9 in filing order; the last bouba_sens ADR was then 0020, so the verdict ADRs were proposed as ADR-0021 / 0022 / 0023 (shifted on 2026-10-25 to ADR-0023 / 0024 / 0025, ADR-0021 and ADR-0022 having been used by PRs #19 and #28 / #40 for the Me6 nulls); amendment numbers v0.7 / v0.8 / v0.9 follow v0.6, the last amendment on file, and assume filing in that order.
**Verdict ADR:** **ADR-0025** (proposed — PI confirms at filing; ADR-0021 and ADR-0022 are now used by the Me6 null calibration and paired null, so the proposals are ADR-0023 / 0024 / 0025 for v0.7 / v0.8 / v0.9)

> **Design change of 2026-10-25 (disclosed).** The PI replaced the
> tail held-out split by a class-stratified held-out probe (§3.4)
> after the manipulation check had been run on the pilot seeds and
> after a driver smoke run on seed 0 (outside every seed pool); no
> pilot arm or deficit D had been computed. Stated on the OSF form.

> **Filing note.** This is the machine-readable draft. Only pilot
> seeds (3101–3105) have been run; no confirmatory seed. The donor scheduler, the three arms, the CKA
> manipulation check and the fusion-gain probe are merged with unit
> tests (PR #12, `src/bouba_sens/decorrelation.py`, hook in
> `AdaptationLoop.pretrain`). The driver for the main-grid arms and the
> pilot D variance, with the learning-rate check of §4, was added on
> 2026-10-25 (`bouba_sens.v09_driver`, `scripts/run_v09_decorrelation.py`,
> `scripts/pilots/derive_v09_pilot.py`). Still missing: the
> fixed-budget grid, persistence and secondary cells in the driver, the
> train-side fusion gain of §3.4, and the D-4 / D-5 analyses (§9). All of it is frozen (commit
> hash on the OSF form) before any confirmatory seed runs. Pilot seeds
> (§7) may run before filing.

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

**Novelty statement.** Temporary decorrelation of modalities as a
critical-period intervention is prior art: Kleinman, Achille & Soatto
already decorrelated the sources during an early window and measured
the lasting deficit in integration. This amendment does not claim
decorrelation itself as new. What it adds rests on the design: the
multiplexed (mux) multi-channel fused code of bouba_sens, a grid of
window onsets, and the comparison with the channel-lesion paradigm
(B-1, T1 vs T2) on the same architecture and world. Whether that
combination yields a timing law is the open question (§6, §7).

Context, not a competitor (descriptive): Han et al., "Towards Physics
of Multimodal Pretraining: Knowledge Flow, Modality Synergy, Early
Unification, and Recipes" (arXiv:2608.05000, v2), sweep the onset of
joint training at a fixed total token budget and report that later
unification degrades vision. They manipulate delayed introduction of a
modality, not decorrelation, and share the confound declared below:
onset, total training and clean exposure co-vary. Cited as supporting
context only; it adds no hypothesis and no test.

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
16 consecutive samples. Donors are drawn from the **whole training
split** (every frame except the held-out probe frames of §3.4), not
from a batch-sized pool, so a same-label donor always exists:

- For training sample i at Phase-1 step t, with dataset index k_i and
  label y_i, a donor index is drawn uniformly from
  {j : label_j = y_i, j ≠ k_i} over all training-split rows. Donors are drawn
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
**Seeding (frozen, `v09_driver.run_cell`).** Before each Phase-1 run of
seed s the global torch RNG is re-seeded with s and the network is built
(module seeds s + 1 … s + 5 as in the B-2 runner), so all arms of a seed
start from the same weights and draw the same Gumbel noise up to the
window; batch t is `world.sample(16, seed = s + t)` and the donors of
step t use the generator seed of §3.1.

### 3.3 Manipulation check (pre-registered, gates the experiment)

The gate compares the two resampled arms under **matched resampling**,
so that temporal structure, effective sample size and the donor scheme
are identical and only the cross-modal pairing differs.

- **Check sample.** On each pilot seed, 512 anchor rows are drawn as 32
  contiguous windows of 16 (`world.sample(16, seed = s + 3·10⁶ + r)`,
  r = 0…31), to cover several labels; all anchors lie in the training
  split (§3.4).
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

**Held-out probe frames: class-stratified (PI decision 2026-10-25,
replacing the tail split of 2026-10-24).** The tail split (the last 900
frames held out, one contiguous 512-frame window per seed) contained
only classes 2 and 3 and 10 frames of class 0, never class 1; in a
smoke run of the driver on seed 0 (outside every seed pool) the
held-out accuracy was 0.26 at onset 0 and 0.00 at onset 200, so FG
was 0 by construction. The PI replaced it before any v0.9 pilot arm
was run. The world is built with
`StudyforrestRealWorld(..., stratified_probe = True)`
(`bouba_sens.probe.stratified_heldout_indices`):

- candidate blocks: every label run of the cache is tiled from its
  first frame into consecutive, non-overlapping blocks of **16**
  frames (a trailing remainder shorter than 16 is not used), so each
  block lies in one run and has one label (the v0.7 tiling);
- counts per class: **32 blocks** in all, apportioned to the four
  classes in proportion to their frame counts by largest remainder,
  with at least one block per class (`proportional_block_counts`). On
  the filing cache (class counts 180 / 3,480 / 4,280 / 1,060) this is
  **1 / 12 / 15 / 4 blocks = 16 / 192 / 240 / 64 rows**;
- draw: one `torch.Generator` seeded with the frozen seed
  **20261025** (`V09_HELDOUT_SEED`, the same for every seed and arm);
  for each class c = 0, 1, 2, 3 in that order, one `torch.randperm`
  over that class's blocks (in increasing start order), of which the
  first count_c are kept;
- the held-out set is the **512 rows** of the chosen blocks (5.7 % of
  the 9,000 frames); a class with too few blocks is a technical failure
  of the world (checked when the world is built);
- disjointness from training: every Phase-1 batch of every arm, every
  donor and every manipulation-check anchor avoids the held-out rows.
  A training window of 16 frames is drawn uniformly (generator seeded
  with the batch seed, as before) among the windows that contain no
  held-out frame and do not wrap around the end of the cache; donor
  pools see the held-out rows with label −1, a class no anchor has.
  Unit tests: `tests/unit/test_studyforrest_stratified_probe.py`.
  Residual: frames are a time series, so the frames next to a held-out
  block are temporally adjacent to it; no gap is inserted.

At the end of Phase 1 (no lesion), on one clean probe of 512 rows,
**all the held-out rows** (`world.probe_sample_with_indices(512, ·)`;
the seed argument is not used), identical for all arms and all
seeds:

- acc_all: plain accuracy with all channels present.
- acc_m: the accuracy on the same 512 probe rows when only modality m
  carries signal (the other four zeroed, as in `query_accuracy`;
  `decorrelation.fusion_gain`).
- **Fusion gain** FG = acc_all − max_m acc_m. Plain (unweighted)
  accuracy is used; the probe is the same for all arms of a seed, so
  class imbalance in the probe is common to both terms of a contrast.
- **Deficit (primary)** D(s, o) = FG_joint(s, o) − FG_decor(s, o).
- **Secondary contrasts (descriptive):** D_plain(s, o) = FG_plain −
  FG_decor, and the resampling effect FG_plain − FG_joint.

*Exploratory check, train vs held-out fusion gain (descriptive, no
test, no decision).* SynIB (Kontras et al., "SynIB: Informational
Bottleneck for Maximizing Synergy in Multimodal Learning",
arXiv:2606.09853) reports that synergistic cues are learned but then
overfit, which would compete with a lost-plasticity reading of a
deficit. To separate the two readings, FG and D are also computed at
the end of Phase 1 on 512 rows drawn from the Phase-1 training
samples (dataset indices recorded by the scheduler) in addition to the
held-out probe above, and the train-side and held-out D are reported
side by side per onset. The held-out probe is the one defined above: before the PI decision
of 2026-10-24 the world had no train/test split and about 36–55 % of
the probe rows were seen in training (9,000 frames, windows of 16,
T = 250–450 steps); the held-out frame range removes that overlap, so
the comparison can separate the two readings. A deficit that appears
only on the held-out side would fit overfitting of re-learned
cross-modal cues; one on both sides would fit reduced fusion itself.
This is not tested.

Positive D means the decorrelation window reduced fusion gain at the
end of the run, relative to a run that saw the same resampled inputs
with their cross-modal pairing intact.

### 3.5 Secondary arms (PI decision 2026-10-24)

Both arms are **secondary**: they are run on every confirmatory seed,
analysed with the pre-specified tests below and always reported, but
they never change the primary axes A, B, C, the headline or the gated
qualifiers of §6. Power (§7) is adequate only for large effects, which
is why neither is confirmatory. Within-class cross-modal resampling
(§3.1) stays the analogue of the Kleinman et al. disturbance in both.

**(a) Mitigation arm: cross-sensor reconstruction (D-6).** Kleinman,
Achille & Soatto (CVPR 2023, arXiv:2210.04643) report that
architectures trained with cross-sensor reconstruction objectives are
markedly more resilient to critical periods. *Mechanism tested.* The
deficit is read as the network settling, during the decorrelated
window, into modality-specific representations that later clean
training does not re-integrate. An auxiliary objective that rewards
each modality's encoder representation for predicting the other
modalities keeps cross-modal predictive structure in the encoders
throughout Phase 1; during the window it can only learn class-level
structure (pairing is broken), but after the window it pushes the
encoders back towards shared, mutually predictive features, so
integration can be re-learned and the deficit should shrink.
*Objective (frozen).* `CrossSensorReconstruction`
(`src/bouba_sens/decorrelation.py`): for every modality m, a linear
decoder maps the mean of the other four modalities' encoder
representations `core(input_proj(x))` (128-d) to m's input, flattened
and standardised within the batch per feature (ε = 1e-6); the loss is
the mean squared error averaged over features and over the five
modalities, added to the classification loss with weight **λ = 1.0**
(`RECON_WEIGHT`, fixed, not tuned) at every Phase-1 step of the run
(before, during and after the window). Decoders are initialised from
their own generator (seed s) and join the same Adam optimiser (learning
rate 1e-3, §4). With λ = 0 the training is bit-identical to no
objective (unit test). *Cells.* o = 0, R = 200 (the main-grid onset-0
cell), arms decor + reconstruction and joint + reconstruction: 2 runs
per seed. *Estimand.* D_rec(s) = FG_joint+rec − FG_decor+rec, and the
mitigation M(s) = D(s, 0) − D_rec(s); M > 0 means the objective reduced
the deficit. Also reported: FG of both reconstruction arms, and D_rec
itself.

**(b) LOCK_AFTER cross (D-7).** The primary design runs without a
constellation lock, because a lock would confound the window with a
freeze (§4). The cross asks whether freezing the constellation inside
or after the window changes the deficit. *Cells.* o = 0, R = 200, arms
decor and joint, `LOCK_AFTER` = ℓ ∈ {25, 100, 175} (25 inside the window
[0, 50); 100 and 175 in recovery; 25 and 175 are the v0.8 endpoint
levels): 6 runs per seed. *Estimand.* D_ℓ(s) = FG_joint,ℓ − FG_decor,ℓ,
and the contrast C_ℓ(s) = D_ℓ(s) − D(s, 0) against the unlocked
onset-0 deficit of the same seed. Expected direction (not tested
directionally): a lock inside the window freezes a constellation
shaped under decorrelated input and may enlarge the deficit (C_25 > 0).

## 4. Conditions

- World: `StudyforrestRealWorld` (4.5-modal real bridge; the real
  modalities carry temporally structured within-class covariance).
  Subject to the manipulation check.
- HARD gating, no constellation lock (LOCK_AFTER unset; a lock would
  confound the window with a freeze), no codebook lock, batch 16.
- **Learning-rate schedule (frozen).** The optimiser is Adam with a
  constant learning rate 1e-3 (the `AdaptationLoop` default in
  `src/bouba_sens/loop.py`; no scheduler exists there). The rate is the
  same before, during and after the window, with no warm-up, no cyclic
  schedule and no reset when the window ends, and the optimiser state
  is not reset at window start or end. The driver must not override it
  (checked in code before filing, §9). Pawlak, "On the Occurence of
  Critical Learning Periods in Neural Networks" (arXiv:2510.09687),
  reports that critical-period deficits can be largely avoided by a
  cyclic learning rate or a reset when the deficit ends, so any
  deficit or trend reported here is conditional on this schedule.
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
  fixed-budget runs (o = 50, 100, 150 × decor, joint): 23 Phase-1 runs
  for the primary and gated analyses; plus the secondary arms of §3.5,
  2 mitigation runs and 6 LOCK_AFTER runs (all o = 0, R = 200): **31
  Phase-1 runs** in total.
- **Lesion phase for D-4 (SNR schedule: changed versus the historical
  grids, ADR-0020).** The "standard T2 lesion phase" is 100 lesion steps
  with SNR(t) = max(−20, 20 − 40 · min(1, t / k)) and **`k_steps` = 50**.
  The floor −20.00 dB is reached at step 50 (generally, `k_steps` = N
  reaches the floor at step N, 0-indexed, N ≤ 99) and held until the
  last lesion step (step 99); mean SNR over the 100 steps is −9.80 dB. The historical default
  (`k_steps` = 5000) only reached 19.21 dB at the last step (ADR-0020,
  PR #14), so D-4 is not comparable with historical lesion results. The
  D-4 driver passes `k_steps` = 50 explicitly (`--k-steps 50` on the
  CLI; `run_grid.sh` forwards `K_STEPS` if a grid is used); code
  defaults are unchanged.
- **World check.** `--world studyforrest` falls back, without an
  error, to a 2-modal stub when the real cache is absent
  (`cli.py:54-63`); the driver records the world class and a run on the
  stub is a technical failure.

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
| D-6 | Mitigation: mean M(s) = D(s, 0) − D_rec(s) > 0 (§3.5 a) | one-sample t, one-sided, α = 0.05; 90 % CI reported | secondary, not gated, never changes the primary verdict |
| D-7 | LOCK_AFTER cross: mean C_ℓ(s) = D_ℓ(s) − D(s, 0) ≠ 0 for ℓ ∈ {25, 100, 175} (§3.5 b) | three two-sided one-sample t, Holm over the three at α = 0.05; mean D_ℓ with 90 % CI reported | secondary, not gated, never changes the primary verdict |
| D-5 | Representation geometry: S between decor and joint end-of-Phase-1 codes (v0.8 metric at k = 12, descriptive; not tied to the v0.8 pilot k*) | descriptive | exploratory |

**SESOI = 0.05 accuracy units.** D is an accuracy difference, like Me7,
and 0.05 is the benchmark's registered smallest meaningful accuracy gap
(B-1 threshold, Q6JYN). Reusing it keeps one effect-size scale across
the benchmark.

**Multiplicity.** D-1 → D-2 is a fixed sequence, each at α = 0.05; the
headline (both reject) is their conjunction, and neither component is
advertised as a separate discovery outside the axes of §6. D-2b and D-3
form a secondary family tested only if the headline holds (serial
gatekeeping), with Holm's correction within the family at α = 0.05.
D-6 and D-7 are two further secondary families, tested whatever the
headline (D-6 one test at α = 0.05; D-7 Holm over its three contrasts);
they are reported with their own wording and never change the primary
axes, the headline or the gated qualifiers.
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

**Secondary arms (reported, never change the statements above).**
D-6 rejects: "the cross-sensor reconstruction objective reduced the
onset-0 deficit by [M, 90 % CI]". Not rejected: "a reduction of the
deficit by the reconstruction objective was not demonstrated at N".
D-7, for each ℓ whose Holm-adjusted test rejects: "with the
constellation locked at step ℓ, the onset-0 deficit differed from the
unlocked deficit by [C_ℓ, 90 % CI]"; otherwise "no difference from the
unlocked deficit was demonstrated at ℓ". Neither is described as
evidence for or against a critical period.

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
  **Planning value (PI decision 2026-10-24): the conservative χ² UCL80
  with the least favourable df = 4**, σ̂_plan = σ̂ · √(4 / χ²₀.₂₀,₄) =
  1.558 · σ̂ (`sd_ucl(σ̂, 5)` in `scripts/power_prereg_2026_10.py`).
  df = 4 is the df of 5 independent seed values, the least favourable
  case (ρ = 1, all variation between seeds). Under a Satterthwaite
  approximation the effective df of σ̂² lies between 4 and 16, so the
  df = 4 multiplier is the largest and the bound is approximately
  conservative for any ρ ∈ [0, 1]. The earlier 80th percentile of a
  parametric bootstrap was not: the round-4 review simulated its
  coverage at 74 % / 66 % / 61 % for ρ = 0.25 / 0.5 / 1. With ρ = 0 (above), N is the smallest N ≥ 20 for
  which the simulated P(A+ and C+) at μ₀ = 0.05 under the "early"
  profile, **minus two Monte-Carlo SEs**, is ≥ 0.80, using at least
  100 000 draws per candidate N and the permutation-calibrated Page
  test; for the pilot-based N the generator is
  `numpy.random.default_rng(20261023 + 909)`, candidates N = 20, 21, …
  run in order from one generator (`scripts/pilots/derive_v09_pilot.py`).
  With the borrowed planning SD 0.0605 (not a pilot bound) this
  gives **N = 32** (estimates
  0.788 / 0.801 / 0.809 at N = 30 / 31 / 32; lower bounds 0.786 /
  0.799 / 0.807; `scripts/power_prereg_2026_10.py`).
  *Pilot result (2026-10-25, seeds 3101–3105, macM1, commit `849d0ef`,
  class-stratified held-out probe; `reports/pilots/v09_pilot_main_grid.json`,
  `…_n_rule.json`).* Manipulation check under the stratified split:
  Δ_K = 0.1206, p = 1 / 10,001, gate **passed** (K_joint 0.130,
  K_decor 0.010). Main grid (decor / joint / plain at 5 onsets):
  held-out acc_all 0.373–0.506 in 74 of 75 runs, i.e. mostly the
  majority-class rates of the probe (192 / 512 = 0.375 for class 1,
  240 / 512 = 0.469 for class 2), and FG = 0 in most runs. D(s, o) is
  0 in 14 of 25 cells, within ±0.008 elsewhere, and −0.111 once (3103,
  o = 200). ANOVA: MS_seed 0.000422, MS_residual 0.000511 (df 16),
  σ̂_u² = 0, **σ̂ = 0.0226**, planning SD **σ̂_plan = 0.0352**
  (× 1.558). N rule: at N = 20 P(A+ and C+) = 0.9499 (lower bound
  0.9485), so **N = 20**. Runtime: 404 s for the 75 main-grid runs
  (3.5–7.3 s each, about 81 s per seed). Reported (no decision): the
  networks are close to majority-class prediction on the held-out
  probe after 250–450 steps, which leaves fusion gain little room;
  the SD is driven by one cell. This changes no rule.
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
- **Secondary arms (§3.5): component power only, exact noncentral t,
  planning SD of D 0.0605** (`scripts/power_prereg_2026_10.py`, section
  "v0.9 secondary arms"). The SD of a within-seed contrast is
  0.0605 · √(2(1 − r)) for a correlation r between the two deficits of
  a seed (r = 0: 0.0856; r = 0.5: 0.0605).

  | Contrast | r | N = 20 | N = 32 | N = 40 |
  |---|---|---:|---:|---:|
  | D-6, M = 0.05 (full mitigation), one-sided | 0 / 0.5 | 0.81 / 0.97 | 0.94 / 1.00 | 0.98 / 1.00 |
  | D-6, M = 0.025 (half), one-sided | 0 / 0.5 | 0.35 / 0.55 | 0.49 / 0.74 | 0.57 / 0.82 |
  | D-6, M = 0.0125 (quarter), one-sided | 0 / 0.5 | 0.16 / 0.23 | 0.20 / 0.31 | 0.23 / 0.36 |
  | D-7, C = 0.05, two-sided at α/3 (Holm, worst step) | 0 / 0.5 | 0.51 / 0.85 | 0.78 / 0.98 | 0.88 / 1.00 |
  | D-7, C = 0.025, two-sided at α/3 | 0 / 0.5 | 0.12 / 0.25 | 0.21 / 0.43 | 0.27 / 0.55 |

  0.80 power for a half mitigation needs N = 74 (r = 0) or 38
  (r = 0.5), and N = 292 / 147 for a quarter. Only a full mitigation or
  a lock contrast of the SESOI size is resolvable at the planned N, so
  both arms are registered as secondary, not confirmatory; N is not
  changed for them.
- **Runtime.** Per seed: 31 Phase-1 runs (23 primary/gated + 8
  secondary) of 250–450 steps (about 8,850 + 2,000 = 10,850 training
  steps at batch 16; the secondary arms add about 23 %), 31 fusion-gain evaluations (6 accuracy
  calls on 512 rows each) and the D-4 lesion runs (2 arms × 5
  modalities × 100 steps). At N = 20, about 217,000 Phase-1 steps,
  roughly 720 B-1-cell equivalents of 300 steps (at N = 32, about
  347,000 steps). This is an estimate in
  steps, not time; the pilot measures wall-clock time on the filing
  host, recorded on the OSF form.

## 8. What is NOT being amended

- Thresholds 0.05 / 0.10 / 0.02, the M2 lesion, the T1/T2 arms and
  every existing metric.
- ADR-0019 verdicts and the §5.5 FINAL text.

**What is NOT claimed.** Results are conditional on the frozen
constant learning rate (§4); no claim is made about other schedules,
including a cyclic rate or a reset at the end of the window. The
decorrelation window itself is not claimed as a new intervention (§2).
The reconstruction arm tests one fixed objective at one weight; a null
D-6 is not evidence that cross-sensor objectives in general fail to
mitigate.

## 9. Pre-registration fidelity checklist

- [ ] Filed on OSF by the PI before any confirmatory seed (3001–3044) is run.
- [x] Decorrelation scheduler + unit tests merged (PR #12,
      `tests/unit/test_decorrelation.py`: donor ≠ own dataset index;
      donor label = sample label, label kept; joint arm shares the decor
      audio donor; identity outside the window; per-step determinism
      from the seed; world exposes dataset indices; default `pretrain`
      path unchanged). Commit hash recorded on the form at filing.
- [ ] Driver for the arms and grids merged with unit tests; learning
      rate confirmed constant at 1e-3 in it (no override, no scheduler,
      no optimiser reset); world built with `stratified_probe = True`;
      train-side fusion-gain output implemented. (Main grid and D
      variance done: `bouba_sens.v09_driver`,
      `scripts/run_v09_decorrelation.py`, unit tests
      `tests/unit/test_v09_driver.py`; the fixed-budget grid,
      persistence arm, secondary cells and train-side fusion gain are
      not yet in the driver.)
- [x] Mitigation objective (§3.5 a, PI decision 2026-10-24):
      `CrossSensorReconstruction` and the optional `reconstruction`
      argument of `AdaptationLoop.pretrain`, unit tests
      `tests/unit/test_cross_sensor_reconstruction.py` (λ = 0 is
      bit-identical to no objective, construction leaves the global RNG
      untouched, decoders and encoders are trained, loss matches its
      definition, decoders join the optimiser once).
- [ ] Driver runs the 8 secondary cells per seed (§3.5: 2 mitigation,
      6 LOCK_AFTER), with `constellation_lock_after` = ℓ set on the
      multiplexer for the lock cells only.
- [x] Class-stratified held-out probe (§3.4, PI decision 2026-10-25):
      `StudyforrestRealWorld(stratified_probe=True)`, unit tests
      `tests/unit/test_studyforrest_stratified_probe.py`.
- [x] (superseded) Held-out probe frames (§3.4, PI decision 2026-10-24):
      `StudyforrestRealWorld(probe_fraction=...)`, held-out
      `fusion_gain`, unit tests
      `tests/unit/test_studyforrest_probe_split.py` (no training draw,
      donor or check anchor touches a probe frame; default unchanged).
- [x] Fusion-gain probe and CKA check implemented as specified (§3.3,
      §3.4), with unit tests (CKA = 1 for identical features, zero
      denominators skipped, translation invariance, label-swap gate),
      PR #12.
- [ ] Manipulation check passed on pilot seeds and recorded on the form,
      with Δ_K, its permutation p-value, K_joint and K_decor. (Pilot
      2026-10-25, stratified split: Δ_K 0.1206, p 1e-4, passed, §7.)
- [ ] Pilot variance components of D, χ² UCL80 planning SD (df = 4,
      × 1.558) and runtime recorded; N fixed on the form.
- [x] No threshold or existing-metric change.
- [x] Seeds disjoint from all previous and sibling-draft seeds.

## 10. Open decisions (PI, before filing)

Every item below must be resolved, and the text edited to a single
protocol, before filing. This draft is not fileable while any item is
open; the scheduler, the fusion-gain and CKA code, the pilot (including
the manipulation check) and a regenerated power analysis for the final
configuration are also filing prerequisites.

Decided by the PI on 2026-10-23 and applied above (no longer open):
SNR ramp `k_steps` = 50 for the D-4 lesion phase (§4, ADR-0020).

Decided by the PI on 2026-10-24 (no longer open): §3.4 probe overlap,
resolved by holding out the last 10 % of frames (§3.1, §3.3, §3.4).
**Superseded on 2026-10-25** by the class-stratified held-out probe
(§3.4), decided by the PI after a driver smoke run on seed 0 (outside
every seed pool) showed that the tail probe never contains class 1
and gave FG = 0; no v0.9 pilot arm had been run. The manipulation check run on the pilot seeds
under the tail split (pilot report 2026-10-25) is re-run under the
stratified split.

Decided by the PI on 2026-10-24 (no longer open): the design is kept as
drafted, with the held-out probe as the only change to the primary design: within-class
cross-modal resampling as the analogue of the Kleinman et al.
disturbance (§2, §3.1); w = 50 and R = 200, no second width (§4);
fixed-budget grid kept, reported as a replication at fixed budget,
with every positive claim limited to "the deficit varies across the
specified training schedules"; joint-resampling control as primary
comparator (§3.2); the matched joint-vs-decor permutation gate with no
added minimum Δ_K (§3.3). Later the same day (second PI round): **add** the LOCK_AFTER
cross and a mitigation arm (cross-sensor reconstruction), both as
**secondary** analyses (§3.5, D-6 / D-7), because their power at the
planned N is adequate only for large effects (§7). Planning SD for N = the conservative χ² UCL80 with
df = 4 (× 1.558, §7). Pre-registration id **N20**, verdict ADR
**ADR-0025** (shifted twice on 2026-10-25: ADR-0021 and ADR-0022 are now used), amendment v0.9, as proposed in the header (proposed — PI
confirms at filing).

No design decision remains open. Filing still needs the prerequisites
stated above (driver, train-side fusion gain, D-4 / D-5 analyses,
pilot with the manipulation check, regenerated N) and the PI's
confirmation of the proposed numbers.
