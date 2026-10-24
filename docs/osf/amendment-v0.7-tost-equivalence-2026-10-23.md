# OSF pre-registration amendment — bouba_sens v0.7 (TOST equivalence for B-1 / B-2)

**Status:** DRAFT — not yet filed on OSF (to be filed by the PI)
**Parent pre-registration:** `dream-of-kiki` OSF DOI `10.17605/OSF.IO/Q6JYN` (unchanged by this draft)
**Previous amendment on file:** `amendment-v0.6-sprint10-17.md`
**Sibling drafts (same date, filed independently):** `amendment-v0.8-subspace-shift-2026-10-23.md`, `amendment-v0.9-decorrelation-window-2026-10-23.md`
**Amendment tag on file:** `bouba_sens/v0.7-tost-equivalence` (to be created at filing)
**Draft date:** 2026-10-23
**Related audit:** ADR-0020 (SNR floors were nearly identical under default settings), `docs/adr/0020-snr-floors-nearly-identical-under-default-settings.md`, merged in PR #14. This draft corrects the SNR schedule (§4.1, §4.2).
**Pre-registration id:** **N18** (proposed — PI confirms at filing). Basis, read on 2026-10-24: the HYPNEUM-PLANS series ends at N14 (`preregistrations/n14_latin_hypercube_7dim.md`) and its roadmap reserves N15–N17 (`2026-05-11-roadmap-Q2-Q3-2026.md`), so N18 / N19 / N20 are proposed for v0.7 / v0.8 / v0.9 in filing order; the last bouba_sens ADR is 0020, so the verdict ADRs are proposed as ADR-0021 / 0022 / 0023; amendment numbers v0.7 / v0.8 / v0.9 follow v0.6, the last amendment on file, and assume filing in that order.
**Verdict ADR:** **ADR-0021** (proposed — PI confirms at filing)
**PI decisions applied (2026-10-23):** N = 95 (smallest N with 0.80 power for E-2 in the planning case); equivalence bounds ±half-threshold (B-1 ±0.025, B-2 ±0.05 bit); B-2 primary estimator = mixed discrete–continuous k-NN MI estimator (Ross 2014).

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
| **E-2** | μ₂ = mean of the per-seed ΔMI, primary estimator (Ross 2014, k = 3), hard-demodulated probe | μ₂ ≤ −0.05 or μ₂ ≥ +0.05 | −0.05 < μ₂ < +0.05 | primary |
| E-2g | same quantity and estimator, Gumbel-sampled probe readout (ADR-0019) | — | — | registered secondary, reported alongside E-2, no decision |
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
lock's TOST is not advertised as a separate finding. E-2g is a
registered secondary analysis: it is specified here, always reported
next to E-2, and never changes the E-2 label. E-2b is exploratory and
supports no claim.

## 4. Protocol

### 4.1 B-1 (E-1, E-1b, E-1c)

Replication of the Q3/Q3+ protocol, except for the SNR schedule (below),
(`docs/milestones/q3-amedi-seeds-2026-05-10.md`) on new seeds:

- World: `StudyforrestRealWorld` (4.5-modal real bridge, as in Q3).
- `STEPS_TRAIN=200`, `STEPS_LESION=100`, batch 16,
  `TRANSDUCER_GATING=hard`, no codebook lock, no gating schedule.
- LOCK_AFTER ∈ {50, 75, 100, 125, 150}, plus one no-lock grid per seed
  for E-1c.
- 30 cells per grid (5 modalities × 2 timings × 3 SNR), `run_grid.sh`
  with the SNR schedule below (the only change to `run_grid.sh` behaviour).
- Outcome per (seed, LOCK_AFTER): `invariants.b1.median_me7` from
  `v0.1_aggregate.json`, read exactly as `scripts/analyse_amedi_quadratic.py`
  reads it.
- Per-seed summary for E-1: arithmetic mean of the five values.

**SNR schedule: changed versus the historical grids (ADR-0020).** The
historical Q3/Q3+/N12 grids ran `m2_snr_schedule` with its defaults
(`snr_init` = +20 dB, `k_steps` = 5000) over 100 lesion steps. The ramp
covered 99/5000 of its length, so the nominal floors −20 / −10 / +10 dB
only reached 19.21 / 19.41 / 19.80 dB at the last step, and the three
"SNR levels" were near-replicates (ADR-0020, PR #14). This amendment
does **not** reuse that schedule. Schedule in effect for every lesion
run of §4.1:

- SNR(t) = max(floor, 20 + (floor − 20) · min(1, t / k)), t = 0 … 99.
- The floor is reached at step t = k (0-indexed) and held until the
  last lesion step, so set **`k_steps` = N** to reach it at step N
  (valid for N ≤ 99, the last step index). Frozen: **N = 50**, `snr_init` = +20 dB.
- Reached values: the floors −20.00 / −10.00 / +10.00 dB are reached at
  step 50 and held for steps 50 … 99 (50 of the 100 lesion steps are at
  the floor). Mean SNR over the 100 steps: −9.80 / −2.35 / +12.55 dB.
- **Consequence:** E-1 is no longer an exact replication of Q3/Q3+/N12.
  It is a replication on new seeds with a stronger and genuinely
  three-level SNR factor. The equivalence bounds (±0.025) are unchanged
  because they derive from the frozen threshold, not from data.
  Estimates are not comparable cell by cell with the historical grids.
- **Code status:** `bouba-sens lesion` has `--k-steps` (`cli.py:267`),
  and `run_grid.sh` now forwards a `K_STEPS` environment variable to
  it (§4.3 item 7, done). Confirmatory grids run with `K_STEPS=50`.
  Code defaults stay as they are: with `K_STEPS` unset the command line
  of the historical grids is unchanged.
- **World check.** `--world studyforrest` loads `StudyforrestRealWorld`
  only if `BOUBA_SENS_STUDYFORREST_DATA` points to a cache containing
  `tactile.pt`; otherwise `_build_world` falls back, without an error, to
  the 2-modal stub (`cli.py:54-63`). The pilot and every confirmatory
  run record the world class in their logs; a run on the stub is a
  technical failure (§8).

### 4.2 B-2 (E-2, E-2b)

ADR-0019 E1 protocol, with the configuration frozen as follows:

- World (PI decision 2026-10-24): `--world studyforrest` (primary,
  E-2), `--world ccfilm` (secondary, E-2b). Under a preset the runner
  builds the world once before any seed and stops if the class is not
  `StudyforrestRealWorld` (or `CCFilmWorld` for E-2b), so a silent
  fallback to the 2-modal stub cannot reach the results.
- LOCK_AFTER = 100, HARD gating, `steps_train = 200`,
  `steps_lesion = 100`, lesion as in the E1 runner (audio, mode M2) except for the SNR
  schedule: `snr_init` = 0 dB, floor −20 dB, `k_steps` = 50, so −20.00 dB is reached at step 50 and held to step 99 (mean SNR −14.90 dB). **This differs from ADR-0019 E1**, whose runner sets k = `steps_lesion` = 100 and reaches only −19.80 dB at the last step (`capture_probe` in `e1_b2_dimensional_mine.py`; ADR-0020). The runner's `--k-steps 50` option sets it (§4.3 item 6).
- **Batch sizes, frozen separately:** Phase-1 training batch **16**,
  Phase-2 lesion batch **16**, probe **N = 512** (`world.sample(512,
  seed = s − 99_999)`). These match B-1 and v0.8. The runner's legacy
  `--batch-size` sets all three (default 512, as in the ADR-0019 E1.3
  runs); confirmatory runs pass `--train-batch 16 --lesion-batch 16
  --probe-n 512` (§4.3 item 2), set together with the rest of the
  frozen configuration by `--preset v07-e2` (E-2) or `--preset v07-e2b`
  (CC-film, E-2b), which refuse any conflicting option (PI decision
  2026-10-24).
- **Probe demodulation (PI decision 2026-10-24, "both").** Z_pre and
  Z_post are captured twice on the same checkpoint and the same 512
  probe rows: (i) **hard** argmax demodulation
  (`nerve.fuse(hard_demod=True)`, deterministic, no RNG use, as in
  v0.8), the **primary** readout for E-2; (ii) **Gumbel-softmax**
  sampling, as in the ADR-0019 E1 runner, the registered secondary
  readout E-2g. The Gumbel captures are taken where the legacy runner
  takes them and the hard captures draw no random numbers, so the
  training trajectory and the Gumbel readout are identical to a
  legacy run. The runner option is `--probe-demod both` (§4.3
  item 8). The primary/secondary split was confirmed by the PI on
  2026-10-24.
- **World check:** as in §4.1, the run log must show
  `StudyforrestRealWorld` (or the CC-film world for E-2b); a silent
  fallback to the stub is a technical failure.
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

When this draft was written, the E1 runner
(`scripts/spikes/e1_b2_dimensional_mine.py`) did not implement §4.2: its
Kraskov path ran on the **1-D mean-pooled scalar**, its `--batch-size`
set training, lesion and probe size together, and `--seeds` took a
count. The following changes are authorised, and only these (status
below the list):

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
6. A `--k-steps` option in the E1 runner (currently hard-coded to
   k = `steps_lesion`, line 98), frozen at 50 for confirmatory runs
   (SNR schedule of §4.2).
7. A `K_STEPS` environment pass-through in `scripts/run_grid.sh`
   (forwarded as `--k-steps` to `bouba-sens lesion`), frozen at 50 for
   confirmatory B-1 grids (SNR schedule of §4.1). The CLI flag exists;
   the script does not forward it. Defaults of the CLI and of
   `m2_snr_schedule` are not changed by this amendment (follow-up in
   ADR-0020).
8. A `--probe-demod both` option in the E1 runner capturing the hard
   (primary) and Gumbel (E-2g) probe readouts of §4.2 from one run
   (added after the PI decision of 2026-10-24).
9. Presets `--preset v07-e2` / `v07-e2b` that set the frozen
   configuration of §4.2 (world, lock, steps, batch sizes, `k_steps`,
   probe readouts), refuse a conflicting option and check the world
   class before any seed (added after the PI decisions of 2026-10-24).

Unit tests: the three batch sizes are honoured; `ksg_full_delta_bits`
equals `me3_delta` on the captured matrices; seed lists reproduce.

**Implementation status (2026-10-24, branch of this draft; hashes are
recorded on the form at filing).**

- Item 3: `bouba_sens.metrics.mi_ross.mi_ross_discrete_continuous`
  (merged, PR #11). The pass/fail verification below is implemented as
  `tests/unit/test_mi_ross.py` (brute-force agreement and invariances to
  1e-10, 1-D known-MI test with 200 replicates per a) and passes.
- Items 1, 2, 4, 6: `--seed-list` (each listed seed re-seeds the global
  torch RNG with s, so a seed's result does not depend on the seeds run
  before it), `--train-batch` / `--lesion-batch` / `--probe-n`,
  `--k-steps`, and the outputs `ross_full_delta_bits` and
  `ksg_full_delta_bits` (function `full_probe_b2`). The legacy options
  and outputs are unchanged. Unit tests: `tests/unit/test_e1_runner_v07.py`.
- Item 5: calibration harness `scripts/calibrate_mi_known_signal.py`
  (design below, unit tests `tests/unit/test_calibrate_mi_known_signal.py`).
  Not yet run; it is run on the frozen code before filing and its table
  is recorded on the form. Exactly balanced classes (128 rows each)
  are used for "4 equiprobable classes".
- Item 7: `K_STEPS` pass-through in `run_grid.sh`, unit test
  `tests/unit/test_run_grid_k_steps.py`.
- Analysis: `scripts/analyse_v07_tost.py` implements §6 and §9 (TOST,
  six labels, E-1 → E-1b → E-1c fixed sequence, reported bootstrap and
  Wilcoxon TOSTs), unit tests `tests/unit/test_analyse_v07_tost.py`.
  Its only confirmatory entry point, `analyse_confirmatory`, refuses
  any input that is not exactly the roster of §8 after documented
  reserve substitutions (in order, at most 4, each with a reason) and
  any B-2 record that is not from the hard primary readout or lacks
  the Gumbel value; it reports E-2g next to E-2 without a label (§6).
  Still to write: the collector that builds its input from the grid
  aggregates (`invariants.b1.median_me7` per seed and lock) and the E1
  runner outputs.
- Item 8 (probe demodulation, decided 2026-10-24): `--probe-demod
  both` in the E1 runner captures the hard (primary) and Gumbel
  (secondary, E-2g) readouts from one run (`capture_probe_readouts`);
  per-seed outputs `ross_full_delta_bits` / `ksg_full_delta_bits` are
  the hard readout and the same keys with suffix `_gumbel` are E-2g;
  the JSON records `primary_readout`. The default `--probe-demod soft`
  keeps the legacy Gumbel-only output. Unit tests in
  `tests/unit/test_e1_runner_v07.py` check that the Gumbel readout and
  training under `both` equal the legacy run bit for bit and that the
  hard capture is deterministic.
- Item 9 (presets and world check): `apply_preset` / `check_world` in
  the E1 runner, unit tests in `tests/unit/test_e1_runner_v07.py`
  (frozen values set, conflicting option refused, stub world refused,
  legacy defaults unchanged without a preset).

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
recorded on the OSF form, but it is descriptive. The ΔMI = 0 rows are
the known-null calibration and the ΔMI = ±0.05 bit rows are the
injected-effect calibration at the equivalence bound; both report bias
and resolution of the estimators in that weak-signal regime. This
follows the ground-truth benchmarking of information-theoretic
estimators by Andrews & Mengaldo, "When Does Text Inform? Benchmarking
Information-Theoretic Metrics for Multimodal Time-Series Forecasting"
(arXiv:2609.11282), whose abstract reports that estimator performance
differs on weak signals.
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

The price of the tighter bounds is sample size (§7): N = 95 is needed
for 0.80 power on E-2 in the planning case, and E-1b stays
under-powered at the observed per-LOCK means; the expected
informativeness of each component is stated there.

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
- E-2g (registered secondary): the same TOST, CIs and p-values as E-2,
  computed on the Gumbel-readout ΔMI-hat of the same 95 seeds, reported
  in the same table as E-2. It carries no label of its own and does not
  change the E-2 label; a disagreement between E-2 and E-2g is stated
  in the paper. The pilot records the per-seed SD of both readouts;
  the E-2 power is computed on the hard readout.
- Descriptive only: per-modality Me7 at floor SNR, per ℓ; per-seed D
  for B-2; E-2b estimators.
- Descriptive only, estimator noise (after Knopov & Shvai,
  arXiv:2610.23700, who separate a seed level from an item level): for
  B-2, report per seed the within-seed noise of ΔMI-hat, i.e. the SD of
  ΔMI-hat over a bootstrap over probe items (1,000 resamples of the 512
  probe rows, Ross estimator re-evaluated), separately from the
  between-seed SD of the 95 per-seed ΔMI-hat that TOST uses. The TOST
  itself is unchanged and still uses one ΔMI-hat per seed.
- Exclusions: a seed is excluded only for a technical failure, i.e. a
  crash, a missing aggregate or a non-finite metric (§8). Outliers are
  not removed. A negative primary MI-hat, and a secondary KSG value
  clipped at 0, are valid values, not failures.

## 7. Power analysis and sample size

Computed by `scripts/power_prereg_2026_10.py` (Monte Carlo, α = 0.05,
script seed 20261023, about 40 s) from data already in the repo. The
tables below are the script sections "v0.7 DECIDED" and "v0.7 N
selection" (own generator, seed 20261023 + 707), at the decided bounds
(±0.025 / ±0.05 bit) and N = 95. The script still prints the earlier
tables at ±0.05 / ±0.10 bit; they are superseded and kept only for
reference. "Component-test power" is one TOST in isolation; the
complete decision rule is simulated below it. Where stated, N needed
for 0.80 is found by bisection on Monte-Carlo power (20,000 draws per N
for single TOSTs, 2,000 per N for the B-1 complete rule), so it is
approximate; the selected N itself uses exact TOST power.

**Sample size: N = 95 (PI decision, 2026-10-23).** The PI set N to the
smallest value giving 0.80 power for E-2 in the planning case (below:
μ = ±0.02 bit, SD 0.1166). Exact TOST power (normal data, conditioning
on the sample SD; `tost_power_exact`) is 0.7968 at N = 94 and **0.8006
at N = 95**; a Monte-Carlo check with 400,000 draws gives 0.7997
(SE 0.0006) at N = 95. N = 95 is fixed on the OSF form. The B-2 pilot
does not change it (§4.3).

**B-1 (Q3/Q3+ 10 seeds × 5 LOCK_AFTER).**

| Quantity | Value |
|----------|------:|
| Per-seed mean Me7 over LOCK: mean / SD | −0.0106 / 0.0161 |
| Per-LOCK means (ℓ = 50, 75, 100, 125, 150) | −0.0181, −0.0181, +0.0044, −0.0144, −0.0069 |
| Per-LOCK SD range | 0.0196 – 0.0468 |
| Upper 80 % confidence limit of the per-seed SD (df = 9) | 0.0208 |

Component-test power only (E-1, bound ±0.025):

| Scenario | Power at N = 95 | N for 80 % power |
|----------|----------------:|-----------------:|
| true μ = 0, SD 0.0161 | 1.000 | 6 |
| true μ = −0.0106, SD 0.0161 | 1.000 | 10 |
| true μ = −0.0106, SD UCL80 0.0208 | 1.000 | 15 |
| true μ = 0, worst per-LOCK SD 0.0468 | 1.000 | 32 |

Complete decision rule (simulated): seeds drawn from the observed
5 × 5 covariance of the per-LOCK values, rounded to the 0.00625 quantum,
full §9 classification of E-1 and the fixed sequence E-1 → E-1b. E-1c
cannot be simulated (no no-lock planning data).

| Scenario (B-1), N = 95 | P(E-1 equivalent) | P(E-1 and E-1b equivalent) | N for 0.80: E-1 / E-1 → E-1b |
|------------------------|------------------:|---------------------------:|-----------------------------:|
| observed per-LOCK means, observed covariance | 1.000 | 0.255 | 10 / ≈ 340 |
| μ = 0 at every lock, observed covariance | 1.000 | 1.000 | 6 / ≈ 36 |
| observed means, covariance inflated to the UCL80 SD | 1.000 | 0.146 | 15 / > 400 |

**B-2 (primary estimator: Ross 2014, k = 3).** No planning data exist
for the primary estimator: it has never been run on this model. The
only per-seed SDs available (ADR-0019 tables, 5 seeds each) come from
other estimators and other configurations; the largest is 0.0749 bit
(gaussian MINE-full, E1.3), with UCL80 (df = 4) 0.1166 bit. These are
*planning assumptions*, not evidence about the confirmatory SD. The
B-2 pilot (§4.3) measures the latter. The planning case is
μ = ±0.02 bit (0.4 × the bound) at SD 0.1166.

*Provenance of SD 0.1166* (checked against
`scripts/power_prereg_2026_10.py`). `B2_TABLES` holds the per-seed
ΔMI values of the ADR-0019 E1.3 / E1.3b / E1.3c tables (5 seeds each).
The script takes the largest sample SD over those tables, 0.0749 bit,
and applies `sd_ucl(sd, 5)`: the one-sided upper 80 % confidence limit
of an SD, sd · √(df / χ²₀.₂₀,df) with df = 4, which gives 0.1166 bit.
N = 95 is therefore already sized on an **upper confidence bound** of
the planning variance, not on a point SD. This is the design rule of
Knopov & Shvai, "Equivalence testing and run-count design for
stochastic machine-learning experiments" (arXiv:2610.23700), who
recommend an upper confidence bound of the pilot variance in the
run-count formula, with seeds as the experimental unit. The same
bound is applied to the B-2 pilot at filing (see "Expected
informativeness" below):
if the pilot upper 80 % limit implies N > 95, the study is reported as
under-powered for E-2 (labels below); N is not changed.

Complete decision rule (normal model, 90 % CI classes against
{−0.05, 0, +0.05}; for E-2, "Equivalent" is the TOST pass, so the
complete-rule and component powers coincide; Monte Carlo, 20,000
draws per row):

| Scenario (B-2), N = 95 | Equivalent | effect detected in the direction of μ, size unresolved | inconclusive | N for 0.80 |
|------------------------|-----------:|------------------------------:|-------------:|-----------:|
| μ = 0, SD 0.0749 | 1.000 | 0.000 | 0.000 | 21 |
| μ = −0.02, SD 0.0749 | 0.987 | 0.013 | 0.000 | 40 |
| μ = −0.04, SD 0.0749 | 0.364 | 0.634 | 0.000 | ≈ 345 |
| μ = 0, SD 0.1166 | 0.988 | 0.012 (either direction) | 0.000 | 48 |
| μ = −0.02, SD 0.1166 (planning case) | **0.8006** (exact; Monte Carlo 0.806, 20,000 draws, SE ≈ 0.003) | 0.193 | 0.000 | 95 (exact) |
| μ = −0.04, SD 0.1166 | 0.215 | 0.778 | 0.000 | ≈ 845 |

(μ = +0.02 / +0.04 give the mirror image. The remaining probability
is "beyond bound" in the direction of μ: at most 0.007, at μ = −0.04,
SD 0.1166.) At N = 95, P(Equivalent) ≥ 0.80 needs a per-seed SD of
the primary ΔMI-hat ≤ 0.166 bit if μ = 0 and ≤ 0.117 bit if
μ = ±0.02. A true |μ| of 0.04 bit (0.8 × the bound) is not resolvable
as equivalence at this N.

**Expected informativeness at N = 95 (stated on the OSF form).**

- **E-1: high.** P(Equivalent) = 1.000 in every planning scenario of
  the complete rule.
- **E-1 → E-1b: still under-powered at the observed per-LOCK means.**
  P = 0.26 at the observed means (two locks sit at −0.018, close to
  −0.025) and 0.15 with the inflated covariance; ≈ 340 seeds would be
  needed for 0.80 at the observed means. It is 1.00 if every per-LOCK
  mean is 0. N is not changed for E-1b. A failed E-1b step is
  expected and is reported as "not shown at every tested LOCK_AFTER",
  not as evidence of a lock effect.
- **E-2: 0.80 in the planning case by construction** (0.99 at μ = 0
  with SD 0.1166). Its real power depends on the pilot SD of the Ross
  ΔMI-hat. At filing, the script is rerun with the pilot SD inflated
  to its one-sided UCL80 (df = 4), μ = 0 and μ = ±0.02, using exact
  TOST power; the resulting power and the N that would be needed for
  0.80 are recorded on the form with one label: "adequately powered"
  (planning-case power ≥ 0.80), "under-powered" (0.50–0.80) or "low
  informativeness" (< 0.50). N stays 95 in every case, and an
  inconclusive E-2 is reported with the label and the N that would
  have been needed.

Normal-approximation formula used as a cross-check: for true μ = 0,
N ≈ ((z₁₋α + z₁₋β/₂) · σ / δ)²; for μ ≠ 0, N ≈ ((z₁₋α + z₁₋β) · σ /
(δ − |μ|))², with δ the bound (δ = 0.025 for B-1, 0.05 bit for B-2).
It gives 94 for the planning case, against 95 from the exact power.

**Runtime.** B-1 needs N × 6 grids × 30 cells: **17,100 lesion cells
at N = 95**, excluding B-2. The last measured throughput (N12,
`docs/milestones/n12-amedi-subgroup-replication-2026-05-11.md`: 1,500
cells in 2 h 37 min, 24-core host, parallel-4) gives about **30 h**
(17,100 / 1,500 × 157 min ≈ 1,790 min). B-2 adds 95 single-seed runs
of the E1 runner whose runtime is not yet measured (probe 512, k-NN
estimators on 896-D). Measuring the B-2 runtime on the filing host,
during the pilot, is a pre-filing task; the total estimate is recorded
on the OSF form.

## 8. Seeds and stopping rule

- **Confirmatory seeds:** 1001–1095 (N = 95), the same list for B-1
  and B-2. They are disjoint from every seed used so far: the default
  grid seeds 0–4, Q3/Q3+ {0, 17, 42, 73, 101, 7, 23, 31, 53, 89} and
  N12 {3, 11, 19, 29, 37, 41, 47, 59, 67, 79}, and from the v0.8
  (2001–2044, pilots 2101–2110) and v0.9 (3001–3044, pilots 3101–3105)
  seed pools.
- **Reserve seeds (technical replacement only):** 1096–1099, used in
  order. They do not overlap the pilot seeds.
- **B-2 pilot seeds:** 1101–1105 (§4.3), never confirmatory.
- **Fixed-N design.** N = 95 is fixed on the OSF form. There is no interim
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
| Effect detected, size unresolved | 0 < L < δ ≤ U | "A non-zero effect in the registered direction; whether it reaches δ (half the registered size) is unresolved at N = 95." |
| Reverse effect detected, size unresolved | L ≤ −δ < U < 0 | "A non-zero reverse effect; size relative to δ unresolved at N = 95." |
| Inconclusive | L ≤ 0 ≤ U and (L ≤ −δ or U ≥ δ) | "Neither an effect nor its absence is established at N = 95." The expected-informativeness label of §7 is quoted. |

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

- [ ] Filed on OSF by the PI before any confirmatory seed (1001–1099) is run.
- [ ] Pre-filing code tasks of §4.3 merged with unit tests; commit hashes
      of `run_grid.sh`, `aggregate_grid.py`, `analyse_amedi_quadratic.py`,
      the E1 runner, the calibration harness, the TOST analysis script
      and the pinned `nerve_wml` recorded on the OSF form.
- [ ] Primary-estimator implementation verification (§4.3) passed;
      calibration table, B-2 pilot SD / runtime and the recomputed
      E-2 power with its informativeness label (§7) recorded on the
      form; N = 95 on the form.
- [x] No threshold change.
- [x] No metric-math change.
- [ ] SNR schedule differs from the historical grids (ADR-0020): `K_STEPS=50` / `--k-steps 50` pass-through merged (done on this branch, §4.3), effective SNR (−20 / −10 / +10 dB at step 50) checked in `metadata.json` of a pilot cell, and the change stated on the OSF form.
- [ ] World class (`StudyforrestRealWorld`) confirmed in the pilot logs (§4.1).
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
N = 95 (second-round decision: 0.80 power for E-2 in the planning
case); bounds ±half-threshold (±0.025 / ±0.05 bit), symmetric; B-2
primary estimator = Ross (2014) mixed discrete–continuous k-NN, k = 3,
with full-probe KSG and MINE secondary; SNR ramp `k_steps` = 50 for B-1
and B-2 (ADR-0020).

Decided by the PI on 2026-10-24 (no longer open): B-2 probe
demodulation, "both": hard argmax readout primary (E-2), Gumbel-sampled
readout as registered secondary E-2g (§3, §4.2, §6); the
primary/secondary split was confirmed by the PI the same day.

Decided by the PI on 2026-10-24 (no longer open): B-2 batch sizes
16 / 16 / 512 (train / lesion / probe), as drafted (§4.2); B-2 primary
world Studyforrest, CC-film as E-2b (§4.2). Pre-registration id and
amendment number: **N18**, verdict ADR **ADR-0021**, amendment v0.7, as
proposed in the header (proposed — PI confirms at filing).

No design decision remains open. Filing still needs the prerequisites
stated above (§4.3 code tasks, calibration, pilot, regenerated E-2
power) and the PI's confirmation of the proposed numbers.
