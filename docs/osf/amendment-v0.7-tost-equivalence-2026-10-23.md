# OSF pre-registration amendment — bouba_sens v0.7 (TOST equivalence for B-1 / B-2)

**Status:** DRAFT — not yet filed on OSF (to be filed by the PI)
**Parent pre-registration:** `dream-of-kiki` OSF DOI `10.17605/OSF.IO/Q6JYN` (unchanged by this draft)
**Previous amendment on file:** `amendment-v0.6-sprint10-17.md`
**Sibling drafts (same date, filed independently):** `amendment-v0.8-subspace-shift-2026-10-23.md`, `amendment-v0.9-decorrelation-window-2026-10-23.md`
**Amendment tag on file:** `bouba_sens/v0.7-tost-equivalence` (to be created at filing)
**Draft date:** 2026-10-23
**Related audit:** ADR-0020 (SNR floors were nearly identical under default settings), `docs/adr/0020-snr-floors-nearly-identical-under-default-settings.md`, merged in PR #14. This draft corrects the SNR schedule (§4.1, §4.2).
**Pre-registration id:** **N18** (proposed — PI confirms at filing). Basis, read on 2026-10-24: the HYPNEUM-PLANS series ends at N14 (`preregistrations/n14_latin_hypercube_7dim.md`) and its roadmap reserves N15–N17 (`2026-05-11-roadmap-Q2-Q3-2026.md`), so N18 / N19 / N20 are proposed for v0.7 / v0.8 / v0.9 in filing order; the last bouba_sens ADR was then 0020, so the verdict ADRs were proposed as ADR-0021 / 0022 / 0023 (shifted on 2026-10-25 to ADR-0023 / 0024 / 0025, ADR-0021 and ADR-0022 having been used by PRs #19 and #28 / #40 for the Me6 nulls); amendment numbers v0.7 / v0.8 / v0.9 follow v0.6, the last amendment on file, and assume filing in that order.
**Verdict ADR:** **ADR-0023** (proposed — PI confirms at filing; ADR-0021 and ADR-0022 are now used by the Me6 null calibration and paired null, so the proposals are ADR-0023 / 0024 / 0025 for v0.7 / v0.8 / v0.9)
**PI decisions applied (2026-10-23):** N = 95 (smallest N with 0.80 power for E-2 in the planning case; superseded by N = 150 on 2026-10-25, see below); equivalence bounds ±half-threshold (B-1 ±0.025, B-2 ±0.05 bit); B-2 primary estimator = mixed discrete–continuous k-NN MI estimator (Ross 2014).
**Post-pilot design changes (PI decisions 2026-10-25, disclosed).** Made after the PI had seen the B-2 pilot (round 1, `docs/osf/pilot-report-2026-10-25.md`), before filing: (i) the Gumbel-sampled readout becomes the primary E-2 readout and the hard readout becomes the descriptive E-2h, because the hard readout was input-independent on every pilot seed; (ii) the E-2 probe becomes a class-balanced multi-window probe (§4.2), because one contiguous window was single-label on 3 of 5 pilot seeds. A deterministic soft (probability-based, non-sampled) readout may be added later as a declared secondary comparison; it is not part of this draft. The pilot was re-run on the same pilot seeds with the changed protocol (round 2). N = 95, the bounds and the estimator are unchanged. **Second set (PI decisions 2026-10-25, 13:12, after pilot round 2 and the seed-1105 diagnosis):** (iii) each seed's primary MI estimate averages over **K = 8 Gumbel draws** with fixed seeds (§4.2); (iv) the E-2 decision uses a **robust TOST** (90 % Hodges–Lehmann CI, §6, §9) instead of the mean-based t-TOST, which is kept as a reported secondary analysis; (v) a registered secondary binary outcome per seed, **"class-separating switch learned"** (E-2s, rule fixed in §4.2 from the pilot, analysis in §6). The power of the robust rule is recomputed from the pilot (round 3, same seeds). **Third set (PI decisions 2026-10-25, 13:32, after round 3):** (vi) **N = 150** (robust-rule power target 0.80; §7, §8); (vii) E-1, E-1b and E-1c also use the robust TOST, with the mean-based t-TOST kept as a reported secondary analysis (§3, §6). These changes are stated on the OSF form.

> **Filing note.** This is the machine-readable draft. Only the B-2
> pilot (seeds 1101–1105, §4.3; rounds 1–3, `docs/osf/pilot-report-2026-10-25.md`)
> has been run; no confirmatory seed has been run. Confirmatory seeds (below) must not be run before
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

- B-1: "the pseudo-median (centre of symmetry) of the per-seed E-1
  value (§3) lies inside ±0.025 (half the frozen 0.05 threshold);
  equivalence at α = 0.05 by the robust TOST, i.e. the 90 %
  Hodges–Lehmann CI lies inside the bounds" (§6; the mean-based
  t-TOST is reported as a secondary analysis);
- B-2: "the pseudo-median of an operational statistic, the
  primary-estimator ΔMI on the fused code (§3), lies inside ±0.05 bit
  (half the frozen 0.10 bit threshold); equivalence at α = 0.05 by
  the robust TOST, i.e. the 90 % Hodges–Lehmann CI lies inside the
  bounds". For both, the nominal α = 0.05 of the robust TOST holds
  for a distribution symmetric about its centre; under skew (for
  example a switch tail) it is approximate, with the size at the
  bounds under the pilot planning law reported (§6, §7). This bounds
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

| Id | Quantity (over seeds; pseudo-median = centre of symmetry) | H0 (to reject) | H1 (equivalence) | Status |
|----|---------------------------------------|----------------|------------------|--------|
| **E-1** | θ₁ = pseudo-median of the per-seed E-1 value (above) | θ₁ ≤ −0.025 or θ₁ ≥ +0.025 | −0.025 < θ₁ < +0.025 | primary (robust TOST, §6) |
| E-1b | θ₁(ℓ) at each ℓ separately (grid medians) | any ℓ with \|θ₁(ℓ)\| ≥ 0.025 | all five inside ±0.025 (IUT) | confirmatory, step 2 of a fixed sequence (robust TOST) |
| E-1c | θ at the no-lock default (LOCK_AFTER unset) | \|θ\| ≥ 0.025 | \|θ\| < 0.025 | confirmatory, step 3 of a fixed sequence (robust TOST) |
| E-1 mean | the means of the same three steps | as above with means | as above with means | reported secondary, no label (the t-TOSTs of the earlier draft) |
| **E-2** | θ₂ = pseudo-median (centre of symmetry) of the per-seed ΔMI, primary estimator (Ross 2014, k = 3) averaged over K = 8 Gumbel draws of the probe readout (ADR-0019), class-balanced probe (§4.2) | θ₂ ≤ −0.05 or θ₂ ≥ +0.05 | −0.05 < θ₂ < +0.05 | primary (robust TOST, §6) |
| E-2 mean | μ₂ = mean of the same per-seed values | as E-2 with μ₂ | as E-2 with μ₂ | reported secondary, no label (the t-TOST of the earlier draft) |
| E-2h | same quantity and estimator, hard (argmax) probe readout | — | — | descriptive, reported alongside E-2, no decision (input-independent in the pilot) |
| E-2s | proportion π of seeds with a post-lesion class-separating switch (§4.2) | — | — | registered secondary, estimation with exact CI, no decision |
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
lock's TOST is not advertised as a separate finding. E-2h is a
descriptive analysis: it is specified here, always reported next to
E-2, and never changes the E-2 label. E-2b is exploratory and
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
  Phase-2 lesion batch **16**, probe **N = 512** (class-balanced
  probe, next item). The batch sizes match B-1 and v0.8. The runner's legacy
  `--batch-size` sets all three (default 512, as in the ADR-0019 E1.3
  runs); confirmatory runs pass `--train-batch 16 --lesion-batch 16
  --probe-n 512` (§4.3 item 2), set together with the rest of the
  frozen configuration by `--preset v07-e2` (E-2) or `--preset v07-e2b`
  (CC-film, E-2b), which refuse any conflicting option (PI decision
  2026-10-24).
- **Class-balanced probe (PI decision 2026-10-25, after the pilot).**
  The E-2 probe is built from short windows that cover all four
  classes equally (`bouba_sens.probe.class_balanced_probe_indices`,
  runner `--probe-scheme balanced`, set by `--preset v07-e2`):
  - candidate windows: every label run of the cache is tiled from its
    first frame into consecutive, non-overlapping blocks of **16**
    frames; a trailing remainder shorter than 16 is not used, so every
    block lies inside one run and has one label;
  - draw: one `torch.Generator` seeded with **s − 99_999** (the legacy
    probe seed); for each class c = 0, 1, 2, 3 in that order, one
    `torch.randperm` over that class's blocks (in increasing start
    order), of which the first **8** are kept, in permutation order;
  - probe rows: the frames of the 32 chosen blocks, class 0 first,
    each block in increasing frame order: **512 rows, exactly 128 per
    class**, no frame twice;
  - class-coverage rule: a class with fewer than 8 blocks is a
    technical failure of the world (checked before any seed); on the
    filing cache the block counts are 10 / 217 / 265 / 65 for classes
    0–3 (class 0 has 170 + 10 frames in two runs), so the rule holds;
  - the draw uses only its own generator, so training and the Gumbel
    readout draw the same random numbers as before.
  The probe rows are also training frames (v0.7 has no held-out
  split), as in ADR-0019. The earlier probe, one contiguous window
  `world.sample(512, seed = s − 99_999)`, was single-label on 3 of the
  5 pilot seeds (the cache has only 10 label runs), which makes MI
  zero by construction. **E-2b (CC-film)** keeps the contiguous probe:
  its 600 frames have 324 label runs, all shorter than 16 frames, so
  no block exists, and a 512-frame window covers 512 of the 600
  frames.
- **Probe demodulation (PI decision 2026-10-24, "both"; primary
  changed by the PI on 2026-10-25 after the pilot).** Z_pre and Z_post
  are captured twice on the same checkpoint and the same 512 probe
  rows: (i) **Gumbel-softmax** sampling, as in the ADR-0019 E1 runner,
  the **primary** readout for E-2; (ii) **hard** argmax demodulation
  (`nerve.fuse(hard_demod=True)`, deterministic, no RNG use), the
  descriptive readout E-2h. The Gumbel captures draw from the global
  torch RNG, which the runner re-seeds with s before the seed's
  training (`--seed-list`), so a seed's Gumbel codes are reproducible;
  they are taken where the legacy runner takes them and the hard
  captures draw no random numbers, so the training trajectory and the
  Gumbel readout are identical to a legacy run. The runner option is
  `--probe-demod both` (§4.3 item 8). In pilot round 1 the hard
  readout mapped all 512 probe rows to one code on every seed (pre
  and post), so every k-NN MI difference was 0 by construction; this
  is why it is no longer primary.
- **K = 8 Gumbel draws (PI decision 2026-10-25, 13:12).** The primary
  readout of each network averages the MI estimate over K = 8
  independent Gumbel captures of the same checkpoint and probe: draw
  r = 0 … 7 of phase p (0 = Z_pre, 1 = Z_post) is captured inside
  `torch.random.fork_rng` after `torch.manual_seed(2·10⁹ + 1000·s +
  100·p + r)` (`bouba_sens.readout.gumbel_draw_seed`), so the draws
  neither use nor change the training RNG stream; MI-hat_pre and
  MI-hat_post are the means over the 8 draws of the per-draw Ross
  estimates (draw r of Z_pre paired with draw r of Z_post) and the
  per-seed ΔMI-hat is their difference; KSG is averaged the same way.
  The single legacy Gumbel capture is still taken (unchanged training
  stream) and reported with suffix `_gumbel1`. Rationale: in the
  pilot, re-drawing the Gumbel readout of one network changed its
  MI-hat by SD 0.017–0.028 bit, the size of the between-seed
  differences of four of the five pilot seeds.
- **Class-separating switch, E-2s (PI decision 2026-10-25, 13:12; rule
  fixed before any confirmatory run).** For each of the K = 8 draws of
  Z_pre and Z_post: the top-1 share λ₁ / Σλ of the eigenvalues of the
  centred covariance, and the between-class share, the fraction of the
  total variance carried by the four class means
  (Σ_c n_c ‖z̄_c − z̄‖² / (N − 1) / Σλ); each averaged over the 8 draws
  (`bouba_sens.readout`). A switch is **detected** for a seed when the
  post-lesion top-1 share is **≥ 0.25** and the post-lesion
  between-class share is **≥ 0.10**; the same rule on Z_pre gives the
  pre-lesion state. Justification of the thresholds, not tuned on seed
  1105: with no class structure the expected between-class share is
  about (classes − 1) / (N − 1) = 3 / 511 ≈ 0.006, and the nine
  non-switch matrices of the pilot (round 2: all five pre-lesion codes
  and the post-lesion codes of four seeds) had top-1 shares 0.099–0.118
  and between-class shares 0.004–0.010; 0.25 is about twice the
  largest top-1 share and 0.10 about ten times the largest
  between-class share and seventeen times the no-structure value.
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
    seed a technical failure (§8; the balanced E-2 probe has 128
    points in every class); zero distances (duplicated codes)
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
8. A `--probe-demod both` option in the E1 runner capturing the
   Gumbel (primary) and hard (E-2h) probe readouts of §4.2 from one
   run (added after the PI decision of 2026-10-24; primary changed on
   2026-10-25).
9. Presets `--preset v07-e2` / `v07-e2b` that set the frozen
   configuration of §4.2 (world, lock, steps, batch sizes, `k_steps`,
   probe readouts), refuse a conflicting option and check the world
   class before any seed (added after the PI decisions of 2026-10-24).
10. The class-balanced probe of §4.2 (`bouba_sens.probe`, runner
    `--probe-scheme balanced`, set by `--preset v07-e2`; added after
    the PI decision of 2026-10-25).
11. K = 8 Gumbel draws in the primary readout (`bouba_sens.readout`,
    runner `--gumbel-draws`, set to 8 by the presets), the E-2s
    structure statistics and switch flags per seed; the robust TOST for
    E-1 / E-1b / E-1c / E-2 with exact signed-rank p-values, the mean
    t-TOST as a reported secondary, the switch analysis, the N = 150
    roster and stricter provenance checks in `analyse_v07_tost.py`
    (added after the PI decisions of 2026-10-25, 13:12 and 13:32). Unit
    tests: `tests/unit/test_readout.py`,
    `tests/unit/test_e1_runner_v07.py`,
    `tests/unit/test_analyse_v07_tost.py`.

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
  any B-2 record that is not from the Gumbel primary readout or lacks
  the hard value; it reports E-2h next to E-2 without a label (§6).
  Still to write: the collector that builds its input from the grid
  aggregates (`invariants.b1.median_me7` per seed and lock) and the E1
  runner outputs.
- Item 8 (probe demodulation, decided 2026-10-24, primary changed
  2026-10-25): `--probe-demod both` in the E1 runner captures the
  Gumbel (primary) and hard (E-2h) readouts from one run
  (`capture_probe_readouts`); per-seed outputs `ross_full_delta_bits`
  / `ksg_full_delta_bits` are the Gumbel readout and the same keys
  with suffix `_hard` are E-2h; the JSON records `primary_readout`. The default `--probe-demod soft`
  keeps the legacy Gumbel-only output. Unit tests in
  `tests/unit/test_e1_runner_v07.py` check that the Gumbel readout and
  training under `both` equal the legacy run bit for bit and that the
  hard capture is deterministic.
- Item 9 (presets and world check): `apply_preset` / `check_world` in
  the E1 runner, unit tests in `tests/unit/test_e1_runner_v07.py`
  (frozen values set, conflicting option refused, stub world refused,
  legacy defaults unchanged without a preset).
- Item 10 (class-balanced probe, decided 2026-10-25):
  `class_balanced_probe_indices` and `draw_probe`, unit tests
  `tests/unit/test_class_balanced_probe.py` (tiling, 512 rows with 128
  per class and no repeat, single-label blocks, determinism and seed
  dependence, agreement with the written definition, global RNG
  untouched, too few blocks refused) and
  `tests/unit/test_e1_runner_v07.py` (runner draws exactly those rows;
  presets set the scheme).

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

*Pilot run (2026-10-25, macM1, `docs/osf/pilot-report-2026-10-25.md`).*
Round 1 (protocol as drafted on 2026-10-24: hard primary, contiguous
probe) gave a primary ΔMI-hat of exactly 0 on all five seeds, because
the hard readout mapped every probe row to one code, and a
single-label probe on 3 of 5 seeds. After the PI decisions of
2026-10-25 (header), round 2 re-ran the same five pilot seeds with the
protocol of §4.2 (Gumbel primary, class-balanced probe, 128 rows per
class on every seed). Round 2, primary (Gumbel, Ross) ΔMI-hat per seed:
+0.0466, +0.0347, +0.0694, +0.0789, +0.6461 bit (mean +0.1751, SD
0.2639, UCL80 0.4110); hard readout (E-2h): 0, 0, 0, 0, +0.1859 bit.
Runtime: 54 s for the five seeds (about 11 s per seed). The §7
recomputation is below.

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

The price of the tighter bounds is sample size (§7): N = 150 is needed
for 0.80 power of the robust E-2 rule under the pilot planning law
(PI decision 2026-10-25); the expected informativeness of each
component, including E-1b, is stated there.

## 6. Analysis plan

- Unit of analysis: the seed. Cells within a seed are not treated as
  independent.
- Mean-based TOST (reported secondary for E-1, E-1b, E-1c and E-2, no
  label): two one-sided one-sample t-tests at α = 0.05 each, i.e. the
  90 % two-sided t-interval of the mean inside the bounds; reported
  with the 95 % CI, both one-sided p-values, the mean and the SD.
- **Robust TOST, E-1, E-1b, E-1c and E-2 (PI decisions 2026-10-25,
  13:12 for E-2 and 13:32 for B-1).** Each label
  is read from the **90 % distribution-free Hodges–Lehmann CI** of the
  per-seed values (E-1: the mean over the five locks; E-1b: each lock;
  E-1c: no-lock; E-2: ΔMI-hat): the Hodges–Lehmann estimate is the median of the
  M = n(n + 1)/2 Walsh averages (x_i + x_j)/2, i ≤ j, and the CI is
  [W₍c₎, W₍M−c+1₎] of the sorted Walsh averages with c the largest
  integer such that P(T⁺ ≤ c − 1) ≤ 0.05 under the exact null law of
  the Wilcoxon signed-rank statistic (`analyse_v07_tost.hl_ci`). This
  is the inversion of the two one-sided signed-rank tests at α = 0.05
  each. Its estimand is the pseudo-median (centre of symmetry); the
  CI has coverage ≥ 90 % for a continuous distribution symmetric about
  its centre. Reported: the estimate, the median, the 90 % and 95 %
  CIs and both one-sided signed-rank p-values, computed from the same
  exact law (`signed_rank_p`), with a frozen tie convention: a Walsh
  average equal to the bound never counts as beyond it (lower test:
  T = #{W > −δ}, p = P(T⁺ ≥ T); upper test: T = #{W ≥ +δ},
  p = P(T⁺ ≤ T)). Then, tied or not, p ≤ 0.05 on a side exactly when
  the 90 % CI lies strictly beyond that bound, so both p ≤ 0.05 exactly
  when the label is Equivalent.
  **Ties.** B-1 grid medians are quantised (steps of 0.00625), so Walsh
  averages tie; the CI is still read from the order statistics and the
  untied law is used, an approximation stated in the paper.
  **Inferential limitation (registered).** The coverage guarantee needs
  symmetry about the centre. A switch component (E-2s) makes the
  per-seed ΔMI-hat skewed; the CI then targets the pseudo-median only
  approximately, and the error rate at the bounds is not guaranteed to
  be 0.05. Its size under the pilot planning law is reported in §7
  (rejection frequency at θ = ±0.05) and quoted next to the E-2 label.
  Rationale: one pilot seed (1105) showed a genuine class-separating
  reorganisation with a ΔMI-hat about ten times the others, which
  dominates a mean and its SD.
- E-2 mean (reported secondary, no label): the t-TOST of the earlier
  draft on the same per-seed values, with its 90 % CI; a disagreement
  with the E-2 label is stated in the paper.
- E-2s (registered secondary, no decision): number and proportion of
  seeds with a post-lesion switch, with the exact (Clopper–Pearson)
  two-sided 95 % CI; the paired pre/post comparison by the exact
  McNemar test (two-sided binomial on the seeds whose switch state
  differs before and after the lesion). B-2 has one condition per
  world, so this pre/post contrast is the only within-design
  comparison; the CC-film E-2b proportion is reported descriptively.
- Robustness of the mean-based secondary (reported, not
  decision-relevant): the same t-TOST on a 10 000-resample percentile
  bootstrap of the mean, and SciPy's Wilcoxon-based TOST. The data are quantised and may be non-normal. If
  parametric and non-parametric verdicts disagree, both are reported and
  the disagreement is stated in the paper.
- E-2h (descriptive): the same TOST, CIs and p-values as E-2,
  computed on the hard-readout ΔMI-hat of the same 150 seeds, reported
  in the same table as E-2. It carries no label of its own and does not
  change the E-2 label; a disagreement between E-2 and E-2h is stated
  in the paper. The pilot records the per-seed SD of both readouts;
  the E-2 power is computed on the Gumbel (primary) readout.
- Descriptive only: per-modality Me7 at floor SNR, per ℓ; per-seed D
  for B-2; E-2b estimators.
- Descriptive only, estimator noise (after Knopov & Shvai,
  arXiv:2610.23700, who separate a seed level from an item level): for
  B-2, report per seed the within-seed noise of ΔMI-hat, i.e. the SD of
  ΔMI-hat over a bootstrap over probe items (1,000 resamples of the 512
  probe rows, Ross estimator re-evaluated), separately from the
  between-seed SD of the 150 per-seed ΔMI-hat that the TOST uses. The TOST
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

**Sample size: N = 150 (PI decision 2026-10-25, 13:32, after pilot
round 3; supersedes N = 95).** N is the value at which the robust E-2
rule (§6) reaches the 0.80 power target under the pilot planning law of
round 3 (pilot values resampled, deviations from their pseudo-median
inflated by the df = 4 UCL80 factor 1.558, law shifted to pseudo-median
μ; `scripts/pilots/derive_v07_pilot.py … 150`,
`reports/pilots/v07_b2_pilot_r3_derived_n150.json`): on the grid
95, 150, 200, … (2,000 draws per N) 150 is the first N with power
≥ 0.80; with 20,000 draws at N = 150 the power is **0.904 (μ = 0) and
0.905 (μ = −0.02)**, label "adequately powered". *Size at the bounds
under the same (asymmetric) law:* P(Equivalent) = 0.0035 at θ = +0.05
and 0.0028 at θ = −0.05, below α = 0.05 (§6, inferential
limitation). The mean-based secondary at N = 150: exact t-TOST power
2e-5, "low informativeness".
*B-1 at N = 150, robust rule* (`scripts/pilots/derive_v07_b1_robust.py`,
`reports/pilots/v07_b1_robust_power_n150.json`): planning law =
the Q3/Q3+ 10 seeds × 5 locks resampled as 5-vectors, deviations from
the per-lock pseudo-medians inflated by the df = 9 UCL80 factor 1.293,
rounded to the 0.00625 quantum, 2,000 draws: P(E-1 equivalent) = 1.000
and P(E-1 and E-1b) = **0.938** at the observed per-lock pseudo-medians
(−0.0125, −0.0063, +0.0047, −0.0125, −0.0063) and 0.951 with every
centre at 0. The per-lock pseudo-medians are closer to 0 than the
per-lock means (−0.0181 at two locks), which is why E-1b is far better
powered under the robust rule than under the mean rule below. E-1c has
no planning data. These are the planning numbers stated on the form.

*Earlier N (kept for the record).* **Sample size: N = 95 (PI decision, 2026-10-23; superseded).** The PI set N to the
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

**Expected informativeness at N = 95 (earlier draft, mean-based rules; superseded by the N = 150 robust numbers above).**

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
  *Pilot result (round 2, 2026-10-25,
  `reports/pilots/v07_b2_pilot_r2_derived.json`):* pilot SD 0.2639 bit,
  UCL80 (df = 4) 0.4110 bit; exact TOST power at N = 95 is 3.8e-7
  (μ = 0) and 3.4e-7 (μ = −0.02); N for 0.80 would be 581 (μ = 0) and
  1,162 (μ = −0.02). Label: **"low informativeness"**. N stays 95. The
  SD is dominated by one seed (1105, +0.646 bit); the rule uses the
  pilot SD as defined, without exclusion.
  *Pilot result (round 3, 2026-10-25, after the PI decisions of 13:12;
  `reports/pilots/v07_b2_pilot_r3_derived.json`).* Same seeds, K = 8
  Gumbel draws. Primary ΔMI-hat per seed: −0.0177, −0.0184, −0.0328,
  +0.0047, +0.6265 bit (Hodges–Lehmann −0.0068, median −0.0177);
  within-seed SD of the per-draw ΔMI-hat 0.033–0.059 bit. *Robust rule
  (registered):* planning model = resampling of the pilot values, their
  deviations from the pilot pseudo-median inflated by the df = 4 UCL80
  factor 1.558 and the law shifted to pseudo-median μ
  (`derive_v07_pilot.robust_power`, 20,000 draws at N = 95, a coarse
  grid of N with 2,000 draws each, generator 20261025): power at N = 95
  is **0.744 (μ = 0) and 0.746 (μ = −0.02)**; 0.80 is reached at N =
  150 on the grid (95, 150, 200, …). Label: **"under-powered"**. *Mean
  rule (secondary):* SD 0.2877, UCL80 0.4481, exact t-TOST power at
  N = 95 ≈ 0 (N for 0.80: 690 / 1,381), "low informativeness". N stays
  95. E-2s in the pilot: switch after the lesion on 1 of 5 seeds
  (1105; exact 95 % CI 0.005–0.716), on 0 of 5 before it.

Normal-approximation formula used as a cross-check: for true μ = 0,
N ≈ ((z₁₋α + z₁₋β/₂) · σ / δ)²; for μ ≠ 0, N ≈ ((z₁₋α + z₁₋β) · σ /
(δ − |μ|))², with δ the bound (δ = 0.025 for B-1, 0.05 bit for B-2).
It gives 94 for the planning case, against 95 from the exact power.

**Runtime at N = 150 (macM1, measured 2026-10-25).** One B-1 grid (a
Phase-1 run and 30 lesion cells, Studyforrest, `K_STEPS=50`,
`LOCK_AFTER=100`) took 182 s on macM1 (Apple M1 Max, CPU, one job,
`nice 10`; timing run on seed 0, outside every pool,
`reports/pilots/logs/b1_timing_seed0.log`). B-1 needs 150 × 6 grids =
900 grids: about **45.5 h** sequentially. B-2 (K = 8 draws) took about
22 s per seed in pilot round 3: about **0.9 h** for 150 seeds. Total
about **46.5 h** of sequential compute on macM1 (less with parallel
grids, if the host allows). *Earlier estimate (N = 95):* B-1 needs N × 6 grids × 30 cells: **17,100 lesion cells
at N = 95**, excluding B-2. The last measured throughput (N12,
`docs/milestones/n12-amedi-subgroup-replication-2026-05-11.md`: 1,500
cells in 2 h 37 min, 24-core host, parallel-4) gives about **30 h**
(17,100 / 1,500 × 157 min ≈ 1,790 min). B-2 adds 95 single-seed runs
of the E1 runner whose runtime is not yet measured (probe 512, k-NN
estimators on 896-D). Measuring the B-2 runtime on the filing host,
during the pilot, is a pre-filing task; the total estimate is recorded
on the OSF form.

## 8. Seeds and stopping rule

- **Confirmatory seeds (N = 150, PI decision 2026-10-25, 13:32):**
  1001–1100 and 1106–1155, the same list for B-1 and B-2; the B-2
  pilot seeds 1101–1105 are skipped (proposed by the drafter, PI
  confirms at filing). They are disjoint from every seed used so far: the default
  grid seeds 0–4, Q3/Q3+ {0, 17, 42, 73, 101, 7, 23, 31, 53, 89} and
  N12 {3, 11, 19, 29, 37, 41, 47, 59, 67, 79}, and from the v0.8
  (2001–2044, pilots 2101–2110) and v0.9 (3001–3044, pilots 3101–3105)
  seed pools.
- **Reserve seeds (technical replacement only):** 1156–1159, used in
  order. They do not overlap the pilot seeds.
- **B-2 pilot seeds:** 1101–1105 (§4.3), never confirmatory.
- **Fixed-N design.** N = 150 is fixed on the OSF form. There is no interim
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

Each primary test (E-1, E-2) is classified from its 90 % two-sided CI
[L, U] (E-1: the t-interval of the mean; E-2: the Hodges–Lehmann
interval of §6) against its bound δ (0.025 for B-1, 0.05 bit for
B-2; half the registered thresholds, §5). The six classes below are exhaustive and mutually exclusive.
"Registered direction" is a positive gap (T1 better) for B-1 and an MI
increase for B-2. Every claim names the estimand of §3.

| Class | Condition | Paper claim |
|-------|-----------|-------------|
| Equivalent | −δ < L and U < δ | "The [estimand] is bounded within ±δ (robust TOST, nominal α = 0.05 under symmetry, 90 % Hodges–Lehmann CI inside the bounds) on N pre-registered new seeds." For E-2 the size at the bounds under the pilot planning law (§7) is quoted. If 0 ∉ [L, U], add: "it is non-zero but smaller than half the registered effect size." |
| Beyond bound, registered direction | L ≥ δ | "An effect of at least half the registered size (≥ δ), in the registered direction, on the [estimand]." Reported as a new pre-registered finding in a new ADR; this does not establish an effect of the full registered size. |
| Beyond bound, reverse direction | U ≤ −δ | "A reverse effect of at least half the registered size (≥ δ) on the [estimand]." New ADR. |
| Effect detected, size unresolved | 0 < L < δ ≤ U | "A non-zero effect in the registered direction; whether it reaches δ (half the registered size) is unresolved at N = 150." |
| Reverse effect detected, size unresolved | L ≤ −δ < U < 0 | "A non-zero reverse effect; size relative to δ unresolved at N = 150." |
| Inconclusive | L ≤ 0 ≤ U and (L ≤ −δ or U ≥ δ) | "Neither an effect nor its absence is established at N = 150." The expected-informativeness label of §7 is quoted. |

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

- [ ] Filed on OSF by the PI before any confirmatory seed (1001–1100, 1106–1159) is run.
- [ ] Pre-filing code tasks of §4.3 merged with unit tests; commit hashes
      of `run_grid.sh`, `aggregate_grid.py`, `analyse_amedi_quadratic.py`,
      the E1 runner, the calibration harness, the TOST analysis script
      and the pinned `nerve_wml` recorded on the OSF form.
- [ ] Primary-estimator implementation verification (§4.3) passed;
      calibration table, B-2 pilot SD / runtime and the recomputed
      E-2 power with its informativeness label (§7) recorded on the
      form; N = 150 on the form.
- [ ] Post-pilot changes of 2026-10-25 stated on the OSF form as made
      after the PI saw the pilot: round 1 → Gumbel primary and the
      balanced probe; round 2 → K = 8 draws, robust E-2, E-2s; round 3
      → N = 150 and the robust E-1, E-1b, E-1c.
- [x] No threshold change.
- [x] No metric-math change.
- [ ] SNR schedule differs from the historical grids (ADR-0020): `K_STEPS=50` / `--k-steps 50` pass-through merged (done on this branch, §4.3), effective SNR (−20 / −10 / +10 dB at step 50) checked in `metadata.json` of a pilot cell, and the change stated on the OSF form.
- [x] World class (`StudyforrestRealWorld`) confirmed in the pilot logs (§4.1; preset world check passed and recorded, pilot report 2026-10-25).
- [ ] Post-pilot design changes of 2026-10-25 (Gumbel primary, class-balanced probe; header) stated on the OSF form as made after the PI saw pilot round 1.
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
**Superseded on 2026-10-25** (PI decision after pilot round 1, header):
Gumbel-sampled readout primary (E-2), hard readout descriptive (E-2h);
class-balanced multi-window probe for E-2 (§4.2). A deterministic soft
readout may be added later as a declared secondary comparison.

Decided by the PI on 2026-10-24 (no longer open): B-2 batch sizes
16 / 16 / 512 (train / lesion / probe), as drafted (§4.2); B-2 primary
world Studyforrest, CC-film as E-2b (§4.2). Pre-registration id and
amendment number: **N18**, verdict ADR **ADR-0023** (shifted twice on 2026-10-25: ADR-0021 and ADR-0022 are now used), amendment v0.7, as proposed in the
header (proposed — PI confirms at filing).

Decided by the PI on 2026-10-25 after the pilot (disclosed, header):
13:12 — K = 8 Gumbel draws (§4.2), robust E-2 with the mean t-TOST as
a reported secondary (§6), registered secondary E-2s (§4.2, §6);
13:32 — N = 150 (§7, §8) and the robust E-1, E-1b, E-1c (§6).

No design decision remains open. Filing still needs the prerequisites
stated above (§4.3 code tasks, calibration, pilot, regenerated E-2
power) and the PI's confirmation of the proposed numbers.
