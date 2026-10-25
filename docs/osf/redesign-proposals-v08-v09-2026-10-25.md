# Redesign proposals — v0.8 measurement and v0.9 probe (2026-10-25)

**Status:** PROPOSAL for PI review (PI decision 2026-10-25, 19:04:
"propose and justify first, then pilot on fresh pilot seeds; stop for
review before running"). No new pilot has been run. The drafts
`amendment-v0.8-*` and `amendment-v0.9-*` are not edited by this file;
after the PI's review the accepted design goes into the drafts, with
the pilot on fresh seeds, before anything is filed.

Evidence below comes only from the existing pilot data: v0.8 blocks
of seeds 2101–2110 (`reports/pilots/v08_pilot_quantities.json`, and a
read-only re-analysis of the stored code matrices,
`scripts/pilots/v08_redesign_evidence.py` →
`reports/pilots/v08_redesign_evidence.json`) and the v0.9 main grid of
seeds 3101–3105 (`reports/pilots/v09_pilot_main_grid.json`).

---

## A. v0.8 — subspace shift vs LOCK_AFTER

### A.1 What the pilot shows

1. **The guard cannot be reached.** The block outcome Y (median over
   the 15 cells of S(pre, post) − S(pre, control), k* = 13) spans
   −0.0074 to +0.0159 (median +0.0008, within-seed SD 0.0029, level
   means 0.0011–0.0031, endpoint contrast median −0.0005). The guard,
   median S_split = 0.0739, is **25 within-seed SDs** and about five
   times the largest Y seen. "Supported" needs the endpoint contrast
   to exceed it, which the pilot never approaches. The guard and Y are
   on different scales: S_split is a raw half-sample (256 / 256)
   distance, dominated by sampling and readout noise, while Y is a
   difference of two full-sample shifts in which common noise cancels.
2. **k* sits inside a flat spectrum, one below a cliff.** The Z_pre
   spectra have **14 nearly equal eigenvalues** (successive ratios
   λ_k / λ_(k+1), k ≤ 13: 1.00–2.44, mostly below 1.1) and then a drop
   by a factor of 117–1,160 (median 384) between the 14th and the 15th,
   in all 70 blocks. k* = 13 (median
   plug-in participation ratio 12.72) cuts the flat part: λ₁₃ / λ₁₄ has
   median 1.066 and is below 1.1 in **89 %** of the Z_pre (66 % over
   all Z_pre and Z_post, as reported). The top-13 subspace is therefore
   not identified: a small perturbation can swap directions 13 and 14.
   The participation ratio estimates "how many directions carry
   variance" (about 12.5 of 14 equal ones); it is not a cut-off for a
   stable subspace.
3. **The codes carry little stable, input-driven structure.** On the
   v0.7 round-3 pilot codes (same architecture and K = 8 Gumbel
   readout) the between-class share of the variance is 0.005–0.008
   (no-structure value 3 / 511 = 0.006),
   and the linear CKA between Z_pre and the no-lesion control end
   matrix, on the same 512 probe rows, is only 0.03 (1 − CKA = 0.973,
   range 0.968–0.990): the item geometry of the K = 8-averaged Gumbel
   code is not preserved over 100 Phase-2 steps without lesion. Most
   of the variation inside the 14-dimensional span is therefore
   readout (Gumbel) noise and not a representation of the input. A
   subspace or geometry metric on such codes mostly measures noise.
4. A k-free geometry measure on the same data, d_c = 1 − CKA(Z_post(c),
   Z_control), has a block median of 0.078 (levels 0.078–0.163) with
   one outlier block (seed 2110, lock 25: 0.897); it inherits the same
   readout noise.

### A.2 Proposed design (v0.8 rev. B)

**(1) Readout: deterministic soft readout, primary.** Replace the
Gumbel readout of the probe captures by its noise-free counterpart:
for each carrier, the demodulator's logits (negative squared distance
to the unit-normalised constellation, as in `GammaThetaMultiplexer.demodulate`)
are turned into probabilities by softmax(logits / τ), τ = 1 (the
training temperature), and fed to the codebook exactly as the Gumbel
sample is now. Same checkpoint → same code, no random draw, and,
unlike the argmax path (rank 0, PI finding 2026-10-25), the code varies
with the input. This is the "deterministic soft readout" the PI
reserved on 2026-10-25 as a later comparison; here it is proposed as
primary because finding 3 shows that sampling noise dominates the
Gumbel code even after 8 draws. Implementation: a function in
bouba_sens that recomputes the demodulator logits from the carrier
(a copy of the `demodulate` arithmetic up to the logits, tested to
reproduce `demodulate(hard=True)` as its argmax and the Gumbel path's
expectation as τ → its limits), so nerve-wml is not changed (no
dependency bump under the HITL policy). The K = 8 Gumbel readout
stays as a registered secondary readout.

**(2) Primary measurement: k-free, lesion-vs-control.** Per T2 cell,
d(c) = 1 − CKA_lin(Z_post(c), Z_control) on the same 512 probe rows
(both end states of Phase 2, same checkpoint, same data and noise
seeds; only the lesion differs). It needs no k, is invariant to any
rotation and to isotropic scaling, uses every direction weighted by
variance, and compares the two end states directly instead of
differencing two noisy distances from Z_pre. Per block Y(s, ℓ) =
median over the 15 cells. Page test, selection safeguard and protocol
failure stay as registered. The rank guard is replaced by a
non-degeneracy check (finite codes, non-zero centred variance of
both matrices).

**(3) Secondary measurement, the principal-subspace shift, with a
stable k.** S at k_gap, where k_gap is the k (1 ≤ k ≤ 255) with the
largest eigenvalue ratio λ_k / λ_(k+1) of the pilot Z_pre, median over
pilot blocks (frozen before the confirmatory run). Pilot: k_gap = 14
in all 70 blocks, ratio 117–1,000. If the pilot median ratio at k_gap
is < 2, S is reported descriptively only. Because the gap may be the
architectural span of the code, the pilot also reports whether S at
k_gap varies at all (if S(k_gap) ≈ 0 everywhere it is dropped).

**(4) Guard vs expected effect.** The S_split guard is withdrawn: it is
not on the scale of Y (finding 1) and with a noise-free readout its
noise interpretation disappears. Two options for the PI:

- **(a) standardised guard (recommended):** "Supported" additionally
  needs the median endpoint contrast Y(s, 175) − Y(s, 25) to exceed
  g = 0.5 σ̂ (σ̂ the UCL80 within-seed SD of the new Y from the new
  pilot). This is a declared smallest effect of interest in
  within-seed SD units, fixed before the data, and the N rule is then
  run with guard 0.5 and a total trend of 0.5 + 0.70 = 1.2 SD (the
  same "+0.70 SD" margin as now);
- **(b) no guard:** the Page test (with the safeguard) decides; the
  endpoint magnitude is reported with its CI.

Under (a), the earlier simulation (v0.8 §7, level model, q = 0.149)
gives a safeguarded P(Supported) of 0.83 at N = 20 and 0.93 at N = 30
for a trend of 0.95 SD with g = 0.25; with g = 0.5 and a trend of
1.2 SD N is expected in the same range. The new pilot fixes it by the
rule.

**(5) Eigengap handling.** For the primary d no subspace is cut. For
the secondary S the k_gap rule picks a well-separated subspace by
construction; every matrix's λ_k / λ_(k+1) at the chosen k is
reported, the share below 1.1 is reported per level, and S-1 is
repeated without the blocks below 1.1 (sensitivity, as now).

**(6) Pilot on fresh seeds.** 10 new pilot seeds **2201–2210** (outside
2001–2044 and 2101–2110), all 7 locks, same driver otherwise; outputs:
the distribution of Y under the new measurement, σ̂ and its UCL80,
non-degeneracy failures, q̂, k_gap and its ratio, S(k_gap), the
deterministic-vs-Gumbel comparison (CKA between the two readouts of
the same network), runtime; then N by the rule. Gates before the PI
accepts the design: (i) the deterministic code has an input-driven
structure (between-class share clearly above 0.006, and CKA between
Z_pre and Z_control well above the 0.03 of the Gumbel code); (ii) Y is
not degenerate (non-zero within-seed SD); (iii) N ≤ 40 by the rule.
If (i) fails, the Phase-2-entry codes carry no stable representation
and v0.8 as a whole needs rethinking (for example measuring encoder
representations instead of the fused code).

### A.3 Decisions needed (v0.8)

1. Deterministic soft readout as primary (K = 8 Gumbel secondary)?
2. Primary measurement d = 1 − CKA(post, control); S at k_gap
   secondary?
3. Guard option (a) 0.5 σ̂ or (b) none?
4. Fresh pilot seeds 2201–2210?

---

## B. v0.9 — fusion-gain probe

### B.1 What the pilot shows

1. **No headroom.** On the stratified held-out probe (class rates
   0.031 / 0.375 / 0.469 / 0.125), the network's own all-channel
   accuracy is 0.373–0.506 in 74 of 75 runs, i.e. the rates of
   always predicting class 1 (0.375) or class 2 (0.469); the best
   single-modality accuracy is equally at those rates (0.375–0.75,
   mostly 0.469). FG = acc_all − max_m acc_m is exactly 0 in **45 of 75
   runs** and D is 0 in 14 of 25 seed × onset cells; σ̂ = 0.0226 comes
   from one cell (−0.111).
2. **The zero-masking probe is off-distribution.** acc_m is computed by
   zeroing the four other channels. ADR-0022 (main, accepted
   2026-10-25) found that this probe carries no per-modality
   information decodable by the head: mean query accuracy 0.254 /
   0.240 = 4-class chance on the Gaussian world and the Studyforrest
   mock. Zero-filled inputs are never seen in training, so acc_m
   measures the head's response to an artificial input, not what a
   modality contributes.

### B.2 Proposed design (v0.9 rev. B)

**(1) Linear-probe fusion gain on frozen representations.** At the end
of Phase 1, freeze the network and read:

- the fused code h_all (896-d) on the real five-channel input, with
  the deterministic soft readout of A.2 (1) (K = 8 Gumbel mean as a
  secondary readout);
- each modality's own representation h_m = core_m(input_proj_m(x_m))
  (128-d, the encoder output that enters the carrier), on the real
  input of that modality: **no channel is zeroed**.

Fit a multinomial logistic regression on each representation (features
standardised on the probe-training rows; L2, C = 1.0; lbfgs, 2,000
iterations; scikit-learn, already a dependency) using a fixed
**probe-training set** drawn from the training split: 64 blocks of 16
label-run frames, class-stratified (the `bouba_sens.probe` block
tiling, generator seed 20261025 + 9, disjoint from the 512 held-out
rows; class 0 limited to its available blocks), and evaluate on the
512 stratified held-out rows. Outcome per run: **balanced accuracy**
(mean per-class recall; chance 0.25 whatever the class mix), so the
majority-class shortcut scores at chance:

FG_lin = bacc(h_all) − max_m bacc(h_m), D = FG_lin,joint − FG_lin,decor.

The SESOI stays 0.05 (balanced-accuracy units, the design-spec B-1
scale). Secondary (descriptive): the same contrast in held-out
log-loss (continuous, more sensitive than accuracy), and the old
network-head FG for continuity.

Rationale: a linear probe measures what the representation makes
linearly available, independent of the network's own head, which
collapses to the majority class; balanced accuracy removes the class
mix; per-modality representations are read on real inputs, which
answers the ADR-0022 objection to zero masking. Fusion gain is then
"what the fused code adds over the best single modality", the quantity
v0.9 intends.

**(2) Headroom gate (pilot, before the PI accepts).** On the fresh
pilot: mean bacc(h_all) ≥ 0.35 (chance + 0.10) in the plain arm, and
fewer than half of the D cells exactly 0. If bacc(h_all) is near
chance, Phase 1 does not learn a usable representation in 250–450
steps, and the training budget or the world has to change first (a
separate decision).

**(3) Unchanged.** Arms, windows, grids, held-out stratified split,
manipulation check (it passed under the stratified split), seeding,
learning-rate check, tests D-1 / D-2 / D-M / D-2b / D-3, N rule (with
the new pilot SD).

**(4) Pilot on fresh seeds.** 5 new pilot seeds **3201–3205** (outside
3001–3044 and 3101–3105): manipulation check, main grid with the new
outcome, D variance components, N by the rule, headroom gate, runtime
(the probes add 6 logistic fits per run).

### B.3 Decisions needed (v0.9)

1. Linear-probe FG on frozen representations, balanced accuracy, as the
   primary outcome (old head FG descriptive)?
2. Fused code read with the deterministic soft readout (as for v0.8)?
3. Probe-training set (64 class-stratified blocks of 16 training-split
   frames) and the logistic settings as above?
4. Headroom gate thresholds (bacc ≥ 0.35; < 50 % zero cells)?
5. Fresh pilot seeds 3201–3205?
